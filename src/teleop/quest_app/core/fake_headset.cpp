// Stand-in for the headset: streams synthetic controller data at 72 Hz with the app's own
// TeleopLink/WebSocket code and prints the gripper poses the host sends back.
//
//   fake_headset [ws://host:port] [seconds]     (default ws://127.0.0.1:9090, run forever)
//
// The right controller traces a 10 cm circle in front of the robot with the trigger
// ramping; the left one stays still with X held.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "src/teleop/quest_app/core/teleop_link.h"

using namespace qt;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
    WsUrl url;
    if (!parse_ws_url(argc > 1 ? argv[1] : "ws://127.0.0.1:9090", url)) {
        std::fprintf(stderr, "bad url\n");
        return 2;
    }
    double duration = argc > 2 ? std::atof(argv[2]) : 0;

    TeleopLink link(url, "fake_headset");
    link.start();
    auto t0 = Clock::now();
    auto next = t0;
    uint64_t seq = 0;
    bool was_connected = false;
    auto last_print = t0;
    double last_gripper_t = -1;

    while (true) {
        double t = std::chrono::duration<double>(Clock::now() - t0).count();
        if (duration > 0 && t > duration) break;

        protocol::ControllerState s;
        s.seq = ++seq;
        s.t = t;
        s.head_valid = true;
        s.head.p = {-0.3f, 0, 0.6f};
        s.right.active = true;
        s.right.pose.p = {0.35f, -0.1f + 0.1f * static_cast<float>(std::cos(t)),
                          0.25f + 0.1f * static_cast<float>(std::sin(t))};
        s.right.trigger = static_cast<float>(0.5 + 0.5 * std::sin(t));
        s.left.active = true;
        s.left.pose.p = {0.3f, 0.3f, 0.2f};
        s.left.button_lower = true;
        link.publish(s);

        auto st = link.status();
        if (st.connected != was_connected) {
            std::printf(st.connected ? "connected to %s:%u\n" : "disconnected (%s:%u)\n",
                        url.host.c_str(), url.port);
            was_connected = st.connected;
        }
        auto now = Clock::now();
        if (now - last_print > std::chrono::seconds(1)) {
            last_print = now;
            if (auto g = link.gripper_pose(); g && g->first.t != last_gripper_t) {
                last_gripper_t = g->first.t;
                const Pose& p = g->first.pose;
                std::printf("gripper p=[%.3f %.3f %.3f] q=[%.3f %.3f %.3f %.3f]  sent=%llu recv=%llu\n",
                            p.p.x, p.p.y, p.p.z, p.q.x, p.q.y, p.q.z, p.q.w,
                            static_cast<unsigned long long>(st.sent),
                            static_cast<unsigned long long>(st.received));
            } else if (!st.connected) {
                std::printf("waiting for host: %s\n", st.last_error.c_str());
            } else {
                std::printf("sent=%llu, no gripper pose yet\n", static_cast<unsigned long long>(st.sent));
            }
            std::fflush(stdout);
        }
        next += std::chrono::microseconds(13889);  // 72 Hz
        std::this_thread::sleep_until(next);
    }
    auto st = link.status();
    link.stop();
    std::printf("done: sent=%llu received=%llu\n", static_cast<unsigned long long>(st.sent),
                static_cast<unsigned long long>(st.received));
    return st.received > 0 ? 0 : 1;
}
