// rebot_arm_gemini_305: functionality for the reBot Arm B601-RS with an Orbbec Gemini 305 on its wrist.
//
// The model files live in //src/robot_model/rebot_arm_gemini_305_data (the arm's MJCF with the
// camera model attached). The arm is the same as //src/robot_model/rebot_arm's: same joints,
// actuators, chain (base_link -> gripper_end), so its poses are the same. What differs is the
// model: the camera's bodies and MuJoCo cameras (left_color, right_color).

#pragma once

#include "src/robot_model/robot_model_utils/kinematics.hpp"
#include "src/robot_model/robot_model_utils/mujoco_model.hpp"
#include "src/robot_model/robot_model_utils/state_estimator.hpp"
#include "src/robot_model/robot_model_utils/renderer.hpp"

#include <memory>
#include <string>

namespace rebot_arm_gemini_305
{

using MjModelPtr = robot_model::MjModelPtr;

// Absolute path of the composed MJCF (scene.xml), resolved through runfiles.
// Throws std::runtime_error if it can't be found.
std::string mjcf_path();

// Compiles the MJCF (with its meshes) into a MuJoCo model.
// Throws std::runtime_error with MuJoCo's message if loading fails.
MjModelPtr load_mujoco_model();

// Forward/inverse kinematics base_link -> gripper_end (joint1..joint6; fast_robot_kinematics,
// generated from the composed model's URDF). Not thread-safe: one per thread.
std::unique_ptr<robot_model::Kinematics> make_kinematics(const mjModel* model);

// The same chain with KDL (double precision, Levenberg-Marquardt IK), built from the MJCF.
std::unique_ptr<robot_model::Kinematics> make_kdl_kinematics(const mjModel* model);

// Absolute path of the USD scene (rebot_arm_gemini_305_data/scene.usda): the USD files of the arm
// and of the camera layered together with the environment. Its prims correspond to the MJCF:
// /World/<body name> for every body, and a camera /Render/<name> for each of left_color and
// right_color, as well as "overview". Throws std::runtime_error if it can't be found.
std::string usd_path();

// Renderer (ovrtx, needs an NVIDIA RTX GPU) of the USD scene for `model` (from load_mujoco_model();
// not owned, must outlive the renderer). Call update(mjData) to pose the scene, then
// render("left_color") for the wrist camera's image. Throws std::runtime_error if ovrtx fails to start.
std::unique_ptr<robot_model::Renderer> make_renderer(const mjModel* model);

// State estimator for `model` (from load_mujoco_model(); not owned, must outlive the estimator):
// update(mjData, observation) sets the state from observed joint states, e.g. a joint_states
// message (see robot_model_utils/state_estimator.hpp).
std::unique_ptr<robot_model::StateEstimator> make_state_estimator(const mjModel* model);

}  // namespace rebot_arm_gemini_305
