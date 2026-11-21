// JSON command-line front end for a MotionPlanner, so non-C++ code (e.g. the
// Python trajectory_generator) can plan. Each robot model provides a tiny main:
//
//   int main() { auto m = my_robot::load_mujoco_model();
//                return robot_model::run_planner_cli(m.get(), *my_robot::make_planner(m.get()),
//                                                    std::cin, std::cout); }
//
// Request (stdin), joint values by name; joints not given stay at the model's qpos0:
//   {"start": {"joint1": 0.0, ...}, "goal": {"joint1": 1.2, ...},
//    "obstacles": {"spheres": [{"center": [x, y, z], "radius": r}],
//                  "boxes": [{"center": [x, y, z], "half_extents": [hx, hy, hz], "rpy": [r, p, y]}]},
//    "settings": {"max_iterations": 100000, "range": 1.0, "simplify": true}}
// Response (stdout):
//   {"solved": true, "joint_names": [...], "path": [[...], ...], "planning_time_ms": ...,
//    "simplify_time_ms": ..., "iterations": ..., "error": ""}
// Path waypoints list the planning joints in joint_names order; consecutive
// waypoints are collision-free straight joint-space motions.
// {"describe": true} answers {"joint_names": [...]} (the planning joints).

#pragma once

#include "src/robot_model/robot_model_utils/motion_planner.hpp"

#include <istream>
#include <ostream>

namespace robot_model
{

// Handles one request. Returns 0 if a response was written (solved or not), 2 if
// the request was malformed (the response then has only "error").
int run_planner_cli(const mjModel* model, const MotionPlanner& planner, std::istream& in, std::ostream& out);

}  // namespace robot_model
