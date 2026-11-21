// rebot_dual_arm_gemini_305: functionality for two reBot Arm B601-RS with a Gemini 305 each (C++:
// load_mujoco_model, make_kdl_kinematics, make_planner).
//
// The model files live in //src/robot_model/rebot_dual_arm_gemini_305_data: the single arm's model
// (rebot_arm_gemini_305) attached twice, with the prefixes "left_" (base at y = +0.1725) and "right_"
// (y = -0.1725). Each arm is the single arm: its chain (<prefix>base_link -> <prefix>gripper_end) is the
// single arm's with the prefix. The robot is one robot: one model, one planner over both arms.

#pragma once

#include "src/robot_model/robot_model_utils/kinematics.hpp"
#include "src/robot_model/robot_model_utils/motion_planner.hpp"
#include "src/robot_model/robot_model_utils/mujoco_model.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rebot_dual_arm_gemini_305
{

using MjModelPtr = robot_model::MjModelPtr;

// The arms' name prefixes.
const std::vector<std::string>& arms();

// Absolute path of the composed MJCF (scene.xml), resolved through runfiles.
// Throws std::runtime_error if it can't be found.
std::string mjcf_path();

// Compiles the MJCF (with its meshes) into a MuJoCo model of both arms.
MjModelPtr load_mujoco_model();

// Forward/inverse kinematics <arm>base_link -> <arm>gripper_end (joint1..joint6 of the arm) from
// fast_robot_kinematics (generated code, single precision), for the MuJoCo model `model` from
// load_mujoco_model(). `arm` is "left_" or "right_".
std::unique_ptr<robot_model::Kinematics> make_kinematics(const mjModel* model, const std::string& arm);

// KDL forward/inverse kinematics <arm>base_link -> <arm>gripper_end (joint1..joint6 of the arm), built from
// the MJCF (`model` from load_mujoco_model()). `arm` is "left_" or "right_".
std::unique_ptr<robot_model::Kinematics> make_kdl_kinematics(const mjModel* model, const std::string& arm);

// The motion planner (VAMP) of the whole robot: one 12-joint robot, joint1..joint6 of both arms (the
// grippers held closed), with the cameras in its collision model: the single arm's spheres, twice
// (//src/robot_model/rebot_dual_arm_gemini_305_data:vamp). So it avoids one arm hitting the other as well as
// each arm's self-collisions. Takes mjData of `model` (from load_mujoco_model(); not owned, must outlive the
// planner); the joints are the model's (left_joint1, ..., right_joint6). Obstacles are in the world frame.
std::unique_ptr<robot_model::MotionPlanner> make_planner(const mjModel* model);

}  // namespace rebot_dual_arm_gemini_305
