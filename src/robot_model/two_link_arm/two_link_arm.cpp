#include "src/robot_model/two_link_arm/two_link_arm.hpp"

#include <memory>
#include <string>

namespace two_link_arm
{

std::string mjcf_path()
{
  return robot_model::runfile_path(TWO_LINK_ARM_MJCF);
}

MjModelPtr load_mujoco_model()
{
  return robot_model::load_mujoco_model(mjcf_path());
}

std::unique_ptr<robot_model::StateEstimator> make_state_estimator(const mjModel* model)
{
  return std::make_unique<robot_model::JointStateEstimator>(model);
}

}  // namespace two_link_arm
