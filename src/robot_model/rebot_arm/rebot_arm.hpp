// rebot_arm: functionality for the reBot Arm B601 robot model.
//
// The model files live in //src/robot_model/rebot_arm_data (fetched from
// Seeed-Projects/reBot-DevArm); this library is the code that works with them.
// It uses the B601-RS variant, whose MJCF has actuators (6 arm joints plus the
// gripper joint7, which drives joint_left/joint_right through equality constraints).

#pragma once

#include "src/robot_model/robot_model_utils/kinematics.hpp"
#include "src/robot_model/robot_model_utils/mujoco_model.hpp"
#include "src/robot_model/robot_model_utils/state_estimator.hpp"
#include "src/robot_model/robot_model_utils/renderer.hpp"

#include <memory>
#include <string>

namespace rebot_arm
{

using MjModelPtr = robot_model::MjModelPtr;

// Absolute path of the RS arm MJCF (RS/mjcf/rs_arm.xml), resolved through runfiles.
// Throws std::runtime_error if it can't be found.
std::string mjcf_path();

// Compiles the MJCF (with its meshes) into a MuJoCo model.
// Throws std::runtime_error with MuJoCo's message if loading fails.
MjModelPtr load_mujoco_model();

// Forward/inverse kinematics of the arm, base_link -> gripper_end (joint1..joint6;
// fast_robot_kinematics, generated from the URDF of the RS MJCF). Poses are of
// gripper_end in base_link's frame (= MJCF world). Takes mjData of `model` (from
// load_mujoco_model(); not owned, must outlive it). Not thread-safe: one per thread.
std::unique_ptr<robot_model::Kinematics> make_kinematics(const mjModel* model);

// The same chain with KDL (double precision, Levenberg-Marquardt IK), built from the MJCF.
std::unique_ptr<robot_model::Kinematics> make_kdl_kinematics(const mjModel* model);

// Absolute path of the USD scene of the arm (usd/scene.usda: the arm's bodies and the environment),
// resolved through runfiles. Its prims correspond to the MJCF: /World/<body name> for every body.
// Throws std::runtime_error if it can't be found.
std::string usd_path();

// Renderer (ovrtx, needs an NVIDIA RTX GPU) of the USD scene for `model` (from load_mujoco_model();
// not owned, must outlive the renderer). Call update(mjData) to pose the scene, then render("overview")
// for an image from the scene's camera. Throws std::runtime_error if ovrtx fails to start.
std::unique_ptr<robot_model::Renderer> make_renderer(const mjModel* model);

// State estimator for `model` (from load_mujoco_model(); not owned, must outlive the estimator):
// update(mjData, observation) sets the state from observed joint states, e.g. a joint_states
// message (see robot_model_utils/state_estimator.hpp).
std::unique_ptr<robot_model::StateEstimator> make_state_estimator(const mjModel* model);

}  // namespace rebot_arm
