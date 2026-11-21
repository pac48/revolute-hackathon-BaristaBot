// two_link_arm: functionality for the two_link_arm robot model.
//
// The model files live in //src/robot_model/two_link_arm_data; this library
// is the code that works with them.

#pragma once

#include "src/robot_model/robot_model_utils/mujoco_model.hpp"
#include "src/robot_model/robot_model_utils/state_estimator.hpp"

#include <memory>
#include <string>

namespace two_link_arm
{

using MjModelPtr = robot_model::MjModelPtr;

// Absolute path of the model's MJCF (scene.xml), resolved through runfiles.
// Throws std::runtime_error if it can't be found.
std::string mjcf_path();

// Compiles the MJCF into a MuJoCo model.
// Throws std::runtime_error with MuJoCo's message if loading fails.
MjModelPtr load_mujoco_model();

// State estimator for `model` (from load_mujoco_model(); not owned, must outlive the estimator):
// update(mjData, observation) sets the state from observed joint states, e.g. a joint_states
// message (see robot_model_utils/state_estimator.hpp).
std::unique_ptr<robot_model::StateEstimator> make_state_estimator(const mjModel* model);

}  // namespace two_link_arm
