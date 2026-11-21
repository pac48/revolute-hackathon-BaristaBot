#include "src/teleop/quest_app/core/websocket_client.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

namespace qt {

namespace {

constexpr size_t kMaxMessage = 16 * 1024 * 1024;

std::string base64(const uint8_t* data, size_t n) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < n; i += 3) {
        uint32_t v = data[i] << 16;
        if (i + 1 < n) v |= data[i + 1] << 8;
        if (i + 2 < n) v |= data[i + 2];
        out += tbl[(v >> 18) & 63];
        out += tbl[(v >> 12) & 63];
        out += i + 1 < n ? tbl[(v >> 6) & 63] : '=';
        out += i + 2 < n ? tbl[v & 63] : '=';
    }
    return out;
}

bool send_all(int fd, const char* data, size_t n) {
    while (n > 0) {
        ssize_t w = ::send(fd, data, n, MSG_NOSIGNAL);
        if (w < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                pollfd p{fd, POLLOUT, 0};
                if (::poll(&p, 1, 1000) <= 0) return false;
                continue;
            }
            return false;
        }
        data += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

}  // namespace

bool parse_ws_url(std::string_view text, WsUrl& out) {
    if (text.starts_with("ws://")) text.remove_prefix(5);
    if (text.empty()) return false;
    out.path = "/";
    if (auto slash = text.find('/'); slash != std::string_view::npos) {
        out.path = std::string(text.substr(slash));
        text = text.substr(0, slash);
    }
    out.port = 9090;
    if (auto colon = text.rfind(':'); colon != std::string_view::npos) {
        int port = 0;
        for (char c : text.substr(colon + 1)) {
            if (c < '0' || c > '9') return false;
            port = port * 10 + (c - '0');
            if (port > 65535) return false;
        }
        if (port == 0) return false;
        out.port = static_cast<uint16_t>(port);
        text = text.substr(0, colon);
    }
    if (text.empty()) return false;
    out.host = std::string(text);
    return true;
}

WebSocketClient::~WebSocketClient() { close(); }

bool WebSocketClient::fail(std::string what) {
    error_ = std::move(what);
    close();
    return false;
}

void WebSocketClient::close() {
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    rx_.clear();
    fragments_.clear();
}

bool WebSocketClient::connect(const WsUrl& url, int timeout_ms) {
    close();
    error_.clear();

    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    std::string port = std::to_string(url.port);
    if (int rc = ::getaddrinfo(url.host.c_str(), port.c_str(), &hints, &res); rc != 0 || !res)
        return fail("resolve " + url.host + ": " + gai_strerror(rc));

    fd_ = ::socket(res->ai_family, res->ai_socktype | SOCK_CLOEXEC, res->ai_protocol);
    if (fd_ < 0) {
        ::freeaddrinfo(res);
        return fail("socket: " + std::string(strerror(errno)));
    }
    // Non-blocking connect so a wrong IP doesn't hang the thread for the TCP timeout.
    int flags = ::fcntl(fd_, F_GETFL, 0);
    ::fcntl(fd_, F_SETFL, flags | O_NONBLOCK);
    int rc = ::connect(fd_, res->ai_addr, res->ai_addrlen);
    ::freeaddrinfo(res);
    if (rc < 0 && errno != EINPROGRESS) return fail("connect: " + std::string(strerror(errno)));
    if (rc < 0) {
        pollfd p{fd_, POLLOUT, 0};
        if (::poll(&p, 1, timeout_ms) <= 0) return fail("connect: timeout");
        int err = 0;
        socklen_t len = sizeof(err);
        ::getsockopt(fd_, SOL_SOCKET, SO_ERROR, &err, &len);
        if (err != 0) return fail("connect: " + std::string(strerror(err)));
    }
    int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    uint8_t key_bytes[16];
    auto seed = static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    rng_ ^= seed;
    for (auto& b : key_bytes) {
        rng_ = rng_ * 1664525u + 1013904223u;
        b = static_cast<uint8_t>(rng_ >> 24);
    }
    std::string req = "GET " + url.path + " HTTP/1.1\r\nHost: " + url.host + ":" + port +
                      "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: " +
                      base64(key_bytes, sizeof(key_bytes)) + "\r\nSec-WebSocket-Version: 13\r\n\r\n";
    if (!send_all(fd_, req.data(), req.size())) return fail("handshake send failed");

    // Read the response headers (bounded by the timeout).
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    size_t end;
    while ((end = rx_.find("\r\n\r\n")) == std::string::npos) {
        int left = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                        deadline - std::chrono::steady_clock::now())
                                        .count());
        if (left <= 0 || rx_.size() > 8192) return fail("handshake: no response");
        pollfd p{fd_, POLLIN, 0};
        if (::poll(&p, 1, left) <= 0) continue;
        char buf[1024];
        ssize_t r = ::recv(fd_, buf, sizeof(buf), 0);
        if (r <= 0) return fail("handshake: connection closed");
        rx_.append(buf, static_cast<size_t>(r));
    }
    std::string status = rx_.substr(0, rx_.find("\r\n"));
    if (status.find(" 101") == std::string::npos) return fail("handshake rejected: " + status);
    rx_.erase(0, end + 4);
    return true;
}

bool WebSocketClient::send_frame(uint8_t opcode, std::string_view payload) {
    if (fd_ < 0) return false;
    std::string frame;
    frame.reserve(payload.size() + 14);
    frame += static_cast<char>(0x80 | opcode);  // FIN
    size_t n = payload.size();
    if (n < 126) {
        frame += static_cast<char>(0x80 | n);  // MASK bit: clients must mask
    } else if (n < 65536) {
        frame += static_cast<char>(0x80 | 126);
        frame += static_cast<char>(n >> 8);
        frame += static_cast<char>(n & 0xff);
    } else {
        frame += static_cast<char>(0x80 | 127);
        for (int i = 7; i >= 0; --i) frame += static_cast<char>((static_cast<uint64_t>(n) >> (8 * i)) & 0xff);
    }
    rng_ = rng_ * 1664525u + 1013904223u;
    uint8_t mask[4] = {static_cast<uint8_t>(rng_ >> 24), static_cast<uint8_t>(rng_ >> 16),
                       static_cast<uint8_t>(rng_ >> 8), static_cast<uint8_t>(rng_)};
    frame.append(reinterpret_cast<char*>(mask), 4);
    for (size_t i = 0; i < n; ++i) frame += static_cast<char>(payload[i] ^ mask[i & 3]);
    if (!send_all(fd_, frame.data(), frame.size())) return fail("send failed");
    return true;
}

bool WebSocketClient::send_text(std::string_view payload) { return send_frame(0x1, payload); }
bool WebSocketClient::send_binary(std::string_view payload) { return send_frame(0x2, payload); }

bool WebSocketClient::fill(size_t n) {
    while (rx_.size() < n) {
        pollfd p{fd_, POLLIN, 0};
        int pr = ::poll(&p, 1, 2000);
        if (pr == 0) return fail("receive timeout mid-frame");
        if (pr < 0) {
            if (errno == EINTR) continue;
            return fail("poll failed");
        }
        char buf[16384];
        ssize_t r = ::recv(fd_, buf, sizeof(buf), 0);
        if (r < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
        if (r <= 0) return fail("connection closed");
        rx_.append(buf, static_cast<size_t>(r));
    }
    return true;
}

WebSocketClient::Poll WebSocketClient::poll(int timeout_ms, std::string& message) {
    if (fd_ < 0) return Poll::Closed;
    if (rx_.empty()) {
        pollfd p{fd_, POLLIN, 0};
        int pr = ::poll(&p, 1, timeout_ms);
        if (pr == 0) return Poll::Idle;
        if (pr < 0) return errno == EINTR ? Poll::Idle : (fail("poll failed"), Poll::Closed);
    }
    while (true) {
        if (!fill(2)) return Poll::Closed;
        auto b0 = static_cast<uint8_t>(rx_[0]), b1 = static_cast<uint8_t>(rx_[1]);
        bool fin = b0 & 0x80;
        uint8_t opcode = b0 & 0x0f;
        bool masked = b1 & 0x80;
        uint64_t len = b1 & 0x7f;
        size_t hdr = 2;
        if (len == 126) {
            if (!fill(4)) return Poll::Closed;
            len = (static_cast<uint8_t>(rx_[2]) << 8) | static_cast<uint8_t>(rx_[3]);
            hdr = 4;
        } else if (len == 127) {
            if (!fill(10)) return Poll::Closed;
            len = 0;
            for (int i = 0; i < 8; ++i) len = (len << 8) | static_cast<uint8_t>(rx_[2 + i]);
            hdr = 10;
        }
        if (len > kMaxMessage) return fail("frame too large"), Poll::Closed;
        size_t mask_off = hdr;
        if (masked) hdr += 4;
        if (!fill(hdr + len)) return Poll::Closed;
        std::string payload = rx_.substr(hdr, len);
        if (masked)
            for (size_t i = 0; i < payload.size(); ++i) payload[i] ^= rx_[mask_off + (i & 3)];
        rx_.erase(0, hdr + len);

        switch (opcode) {
            case 0x9:  // ping
                if (!send_frame(0xA, payload)) return Poll::Closed;
                break;
            case 0xA:  // pong
                break;
            case 0x8:  // close: echo it, then drop the connection
                send_frame(0x8, payload.substr(0, 2));
                error_ = "closed by server";
                close();
                return Poll::Closed;
            case 0x0:  // continuation
            case 0x1:  // text
            case 0x2:  // binary
                if (opcode != 0x0) fragments_.clear();
                fragments_ += payload;
                if (fragments_.size() > kMaxMessage) return fail("message too large"), Poll::Closed;
                if (fin) {
                    message = std::move(fragments_);
                    fragments_.clear();
                    return Poll::Message;
                }
                break;
            default:
                return fail("bad opcode"), Poll::Closed;
        }
        // A control frame or partial message was consumed; continue only if more is buffered.
        if (rx_.empty()) return Poll::Idle;
    }
}

}  // namespace qt
