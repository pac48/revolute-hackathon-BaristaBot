// Minimal blocking RFC 6455 WebSocket client over POSIX sockets (ws:// only, no TLS).
// Used from a single network thread; not thread-safe.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace qt {

struct WsUrl {
    std::string host;
    uint16_t port = 0;
    std::string path = "/";
};

// Accepts "ws://host:port/path", "host:port" or "host" (default port 9090).
bool parse_ws_url(std::string_view text, WsUrl& out);

class WebSocketClient {
public:
    enum class Poll { Idle, Message, Closed };

    ~WebSocketClient();

    // TCP connect + HTTP upgrade. Returns false (and sets error()) on failure.
    bool connect(const WsUrl& url, int timeout_ms);
    void close();
    bool connected() const { return fd_ >= 0; }

    bool send_text(std::string_view payload);
    bool send_binary(std::string_view payload);

    // Waits up to timeout_ms for data, then reads at most one complete message.
    // Ping/pong/close control frames are handled internally.
    Poll poll(int timeout_ms, std::string& message);

    const std::string& error() const { return error_; }

private:
    bool send_frame(uint8_t opcode, std::string_view payload);
    bool fill(size_t n);  // read until rx_ holds at least n bytes (blocking)
    bool fail(std::string what);

    int fd_ = -1;
    std::string rx_;          // received, not yet parsed bytes
    std::string fragments_;   // payload of a fragmented message in progress
    std::string error_;
    uint32_t rng_ = 0x9e3779b9u;
};

}  // namespace qt
