#include "src/teleop/quest_app/core/teleop_link.h"

namespace qt {

TeleopLink::TeleopLink(WsUrl url, std::string device_name)
    : url_(std::move(url)), device_(std::move(device_name)) {}

TeleopLink::~TeleopLink() { stop(); }

void TeleopLink::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread([this] { run(); });
}

void TeleopLink::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
}

void TeleopLink::publish(const protocol::ControllerState& state) {
    std::string msg = protocol::encode_controllers(state);
    std::lock_guard lock(mu_);
    outgoing_ = std::move(msg);
}

void TeleopLink::publish_view(std::string message) {
    std::lock_guard lock(mu_);
    view_ = std::move(message);
}

std::optional<std::pair<protocol::GripperPose, TeleopLink::Clock::time_point>> TeleopLink::gripper_pose()
    const {
    std::lock_guard lock(mu_);
    return gripper_;
}

TeleopLink::Status TeleopLink::status() const {
    std::lock_guard lock(mu_);
    return status_;
}

void TeleopLink::run() {
    WebSocketClient ws;
    std::string incoming;
    while (running_) {
        if (!ws.connected()) {
            {
                std::lock_guard lock(mu_);
                status_.connected = false;
            }
            if (!ws.connect(url_, 1000) || !ws.send_text(protocol::encode_hello(device_))) {
                {
                    std::lock_guard lock(mu_);
                    status_.last_error = ws.error();
                }
                for (int i = 0; i < 10 && running_; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            std::lock_guard lock(mu_);
            status_.connected = true;
            status_.last_error.clear();
            outgoing_.reset();  // don't send a state captured before this connection
            view_.reset();
        }

        // Short poll so a new controller state goes out within ~2 ms of being published.
        auto r = ws.poll(2, incoming);
        if (r == WebSocketClient::Poll::Message) {
            if (auto g = protocol::decode_gripper_pose(incoming)) {
                std::lock_guard lock(mu_);
                gripper_ = std::make_pair(*g, Clock::now());
                ++status_.received;
            }
        } else if (r == WebSocketClient::Poll::Closed) {
            std::lock_guard lock(mu_);
            status_.last_error = ws.error();
            continue;
        }

        std::optional<std::string> out;
        {
            std::lock_guard lock(mu_);
            out.swap(outgoing_);
        }
        if (out) {
            if (ws.send_text(*out)) {
                std::lock_guard lock(mu_);
                ++status_.sent;
            }
        }
        // View frames after controller state, so a frame never delays the controls.
        std::optional<std::string> view;
        {
            std::lock_guard lock(mu_);
            view.swap(view_);
        }
        if (view && ws.connected()) {
            if (ws.send_binary(*view)) {
                std::lock_guard lock(mu_);
                ++status_.views_sent;
            }
        }
    }
    ws.close();
    std::lock_guard lock(mu_);
    status_.connected = false;
}

}  // namespace qt
