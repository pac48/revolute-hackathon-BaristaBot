// KDL backend of Kinematics, built straight from the MuJoCo model.
//
// The chain runs from `root_body` to `tip_body` through the body tree; every hinge or slide joint
// on the way is a chain joint (one joint per body at most). FK is KDL's recursive solver; IK is
// KDL's Levenberg-Marquardt solver (ChainIkSolverPos_LMA), which ignores joint limits, so the
// base class wraps revolute joints into their ranges, projects the rest onto the limits and
// checks the result against the tolerances.

#pragma once

#include "src/robot_model/robot_model_utils/kinematics.hpp"

#include <memory>
#include <string>

namespace robot_model
{

// Throws std::invalid_argument if a body is missing, tip isn't below root, or a body on the
// way has more than one joint or a joint that isn't a hinge or slide.
std::unique_ptr<Kinematics> make_kdl_kinematics(const mjModel* model, const std::string& root_body,
                                                const std::string& tip_body);

}  // namespace robot_model
