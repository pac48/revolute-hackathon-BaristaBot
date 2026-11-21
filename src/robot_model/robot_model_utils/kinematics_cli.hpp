// JSON command-line front end for a Kinematics, so non-C++ code (e.g. the
// Python trajectory_generator) can use FK/IK. Each robot model provides a tiny main:
//
//   int main() { auto m = my_robot::load_mujoco_model();
//                return robot_model::run_kinematics_cli(m.get(), *my_robot::make_kinematics(m.get()),
//                                                       std::cin, std::cout); }
//
// A request on stdin, its answer on stdout (and more of them: below). Joint values are by name; joints
// not given are at the model's qpos0. Poses are of the tip in the root frame:
// {"position": [x, y, z], "quaternion": [x, y, z, w]}.
//
//   {"describe": true}
//     -> {"joint_names": [...], "root_link": "...", "tip_link": "..."}
//   {"fk": {"joint1": 0.1, ...}}            or {"fk": [{...}, {...}]}
//     -> {"pose": POSE}                      or {"poses": [POSE, ...]}
//   {"ik_path": {"start": {"joint1": ...}, "poses": [POSE, ...], "max_joint_step": 0.2,
//                "check_collisions": true,
//                "settings": {"max_attempts": 50, "iterations": 100,
//                             "position_tolerance": 1e-3, "orientation_tolerance": 1e-2}}}
//     -> {"solved": true, "joint_names": [...], "path": [[...], ...], "failed_index": -1,
//         "error": "", "time_ms": ..., "max_position_error": ..., "max_orientation_error": ...}
//
// ik_path solves the poses in order, each from the previous solution (the first
// from start), so a densely sampled Cartesian path gives a continuous joint path.
// It fails at the first pose with no solution, or whose solution moves any joint
// by more than max_joint_step (radians) from the previous one: a jump to another
// IK branch, which would not follow the Cartesian path in between. With
// check_collisions (default), it also fails at the first solution where the
// model's bodies penetrate (MuJoCo collision detection on the full model, e.g.
// self-collision; the error names the bodies). path holds the solutions up to the failure.
//
// Malformed requests answer {"error": "..."} and exit code 2.
//
// Requests are answered in turn until stdin closes, each answer one line, flushed: one
// process can serve many requests (kinematics_client.Kinematics keeps one), without
// loading the model each time. A request that is not JSON ends the stream.

#pragma once

#include "src/robot_model/robot_model_utils/kinematics.hpp"

#include <istream>
#include <ostream>

namespace robot_model
{

int run_kinematics_cli(const mjModel* model, const Kinematics& kinematics, std::istream& in, std::ostream& out);

}  // namespace robot_model
