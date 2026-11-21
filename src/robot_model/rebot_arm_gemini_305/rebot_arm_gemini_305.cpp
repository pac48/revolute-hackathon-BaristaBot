#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305.hpp"

#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305_kinematics.hpp"
#include "src/robot_model/robot_model_utils/kdl_kinematics.hpp"
#include "src/robot_model/robot_model_utils/vamp_planner.hpp"

#include <vamp/robots/rebot_arm_gemini_305.hh>

#include <memory>
#include <string>

namespace rebot_arm_gemini_305
{

std::string mjcf_path()
{
  return robot_model::runfile_path(REBOT_ARM_GEMINI_305_MJCF);
}

MjModelPtr load_mujoco_model()
{
  return robot_model::load_mujoco_model(mjcf_path());
}

std::unique_ptr<robot_model::MotionPlanner> make_planner(const mjModel* model)
{
  return std::make_unique<robot_model::VampPlanner<vamp::robots::RebotArmGemini305>>(model);
}

std::unique_ptr<robot_model::Kinematics> make_kinematics(const mjModel* model)
{
  return rebot_arm_gemini_305_kinematics::make_kinematics(model);
}

std::unique_ptr<robot_model::Kinematics> make_kdl_kinematics(const mjModel* model)
{
  return robot_model::make_kdl_kinematics(model, "base_link", "gripper_end");
}

std::unique_ptr<robot_model::StateEstimator> make_state_estimator(const mjModel* model)
{
  return std::make_unique<robot_model::JointStateEstimator>(model);
}

}  // namespace rebot_arm_gemini_305
