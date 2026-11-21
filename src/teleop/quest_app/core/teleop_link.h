// Network thread between the render loop and the host.
//
// The render loop publishes the newest controller state each frame; the thread sends only the
// latest one (stale states are dropped, never queued) and keeps the newest gripper pose
// received. It reconnects every second while the host is unreachable.
#pragma once

#include <atomic>
#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "src/teleop/quest_app/core/protocol.h"
#include "src/teleop/quest_app/core/websocket_client.h"

namespace qt {

class TeleopLink {
public:
    using Clock = std::chrono::steady_clock;

    struct Status {
        bool connected = false;
        uint64_t sent = 0;
        uint64_t received = 0;
        uint64_t views_sent = 0;
        std::string last_error;
    };

    explicit TeleopLink(WsUrl url, std::string device_name = "quest");
    ~TeleopLink();

    void start();
    void stop();

    void publish(const protocol::ControllerState& state);
    // Newest headset view frame (an encoded binary message); replaces one not yet sent.
    void publish_view(std::string message);

    // Newest gripper pose and when it arrived (nullopt until the first one).
    std::optional<std::pair<protocol::GripperPose, Clock::time_point>> gripper_pose() const;

    Status status() const;

private:
    void run();

    WsUrl url_;
    std::string device_;
    std::thread thread_;
    std::atomic<bool> running_{false};

    mutable std::mutex mu_;
    std::optional<std::string> outgoing_;  // newest encoded controller state not yet sent
    std::optional<std::string> view_;      // newest view frame not yet sent
    std::optional<std::pair<protocol::GripperPose, Clock::time_point>> gripper_;
    Status status_;
};

}  // namespace qt
