// Wire protocol between the headset and the host (see PROTOCOL.md).
// All poses are in ROS convention, expressed in the robot base frame.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "src/teleop/quest_app/core/pose_math.h"

namespace qt::protocol {

constexpr int kVersion = 1;

struct Hand {
    bool active = false;  // controller tracked this frame
    Pose pose;            // grip pose
    float trigger = 0;    // 0..1
    float grip = 0;       // 0..1 (squeeze)
    float stick_x = 0, stick_y = 0;
    bool button_lower = false;  // A (right) / X (left)
    bool button_upper = false;  // B (right) / Y (left)
    bool stick_click = false;
    bool menu = false;  // left only
};

struct ControllerState {
    uint64_t seq = 0;
    double t = 0;  // headset steady clock, seconds
    bool head_valid = false;
    Pose head;
    Hand left, right;
};

struct GripperPose {
    double t = 0;
    Pose pose;
};

std::string encode_hello(std::string_view device);
std::string encode_controllers(const ControllerState& s);

// Binary message: the headset view as a JPEG (see PROTOCOL.md):
//   "VIEW" | u16 width | u16 height | f64 t | JPEG bytes      (little-endian)
constexpr size_t kViewHeaderSize = 16;
std::string encode_view(std::string_view jpeg, int width, int height, double t);

// Messages from the host. Unknown types decode to nothing, so the host can add messages
// without breaking older headsets.
std::optional<GripperPose> decode_gripper_pose(std::string_view text);

}  // namespace qt::protocol
