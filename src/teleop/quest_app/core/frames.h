// Conversions between OpenXR and ROS axis conventions.
//
//   OpenXR: x right, y up, z backward (-z is forward)
//   ROS (REP 103): x forward, y left, z up
//
// The change of basis C (ROS <- XR) is a proper rotation:
//   ros.x = -xr.z,  ros.y = -xr.x,  ros.z = xr.y
// A pose expressed in an XR frame converts to the same physical pose in the matching ROS
// frame by p' = C p and R' = C R C^T. For a quaternion, conjugating by a rotation rotates
// its vector part by C and leaves w unchanged, so the same component mapping applies.
//
// Everything on the wire is in ROS convention, so the host never deals with XR axes.
#pragma once

#include "src/teleop/quest_app/core/pose_math.h"

namespace qt {

inline Vec3 xr_to_ros(Vec3 v) { return {-v.z, -v.x, v.y}; }
inline Vec3 ros_to_xr(Vec3 v) { return {-v.y, v.z, -v.x}; }

inline Quat xr_to_ros(Quat q) { return {-q.z, -q.x, q.y, q.w}; }
inline Quat ros_to_xr(Quat q) { return {-q.y, q.z, -q.x, q.w}; }

inline Pose xr_to_ros(const Pose& t) { return {xr_to_ros(t.q), xr_to_ros(t.p)}; }
inline Pose ros_to_xr(const Pose& t) { return {ros_to_xr(t.q), ros_to_xr(t.p)}; }

// Keep only the rotation about the vertical (XR +y) axis, so an anchor placed from a tilted
// controller or head still has a level floor.
inline Quat yaw_only_xr(Quat q) {
    Vec3 fwd = rotate(q, {0, 0, -1});
    float yaw = std::atan2(-fwd.x, -fwd.z);
    return axis_angle({0, 1, 0}, yaw);
}

}  // namespace qt
