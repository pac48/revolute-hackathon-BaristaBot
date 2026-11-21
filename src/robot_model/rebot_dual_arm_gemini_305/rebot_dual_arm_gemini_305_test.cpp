#include "src/robot_model/rebot_dual_arm_gemini_305/rebot_dual_arm_gemini_305.hpp"

#include <gtest/gtest.h>

namespace
{
namespace dual = rebot_dual_arm_gemini_305;

TEST(RebotDualArmGemini305, EachArmHasTheSingleArmsChain)
{
  const auto model = dual::load_mujoco_model();
  for (const std::string& arm : dual::arms())
  {
    const auto kinematics = dual::make_kdl_kinematics(model.get(), arm);
    EXPECT_EQ(kinematics->root_link(), arm + "base_link");
    EXPECT_EQ(kinematics->tip_link(), arm + "gripper_end");
    ASSERT_EQ(kinematics->dof(), 6U);
    for (int i = 0; i < 6; ++i)
      EXPECT_EQ(kinematics->joint_names()[static_cast<std::size_t>(i)], arm + "joint" + std::to_string(i + 1));
  }
}

TEST(RebotDualArmGemini305, ArmsHaveTheSameToolPoseInTheirOwnBaseFrame)
{
  const auto model = dual::load_mujoco_model();
  const auto data = robot_model::make_data(model.get());
  mj_forward(model.get(), data.get());
  const auto left = dual::make_kdl_kinematics(model.get(), "left_")->forward_kinematics(*data);
  const auto right = dual::make_kdl_kinematics(model.get(), "right_")->forward_kinematics(*data);
  EXPECT_LT((left.translation() - right.translation()).norm(), 1e-9);
  EXPECT_LT((left.linear() - right.linear()).norm(), 1e-9);
}

}  // namespace
