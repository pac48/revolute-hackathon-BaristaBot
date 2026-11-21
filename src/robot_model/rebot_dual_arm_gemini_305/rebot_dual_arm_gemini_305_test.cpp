#include "src/robot_model/rebot_dual_arm_gemini_305/rebot_dual_arm_gemini_305.hpp"

#include <gtest/gtest.h>

#include <algorithm>

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

// The planner is one robot over both arms: its joints are both arms' joint1..joint6, and it plans a path that
// moves either (or both).
TEST(RebotDualArmGemini305, PlannerIsOneRobotOverBothArms)
{
  const auto model = dual::load_mujoco_model();
  const auto planner = dual::make_planner(model.get());
  const auto& joints = planner->joint_names();
  ASSERT_EQ(joints.size(), 12U);
  for (const std::string& arm : dual::arms())
    for (int i = 1; i <= 6; ++i)
      EXPECT_NE(std::find(joints.begin(), joints.end(), arm + "joint" + std::to_string(i)), joints.end()) << arm << i;

  auto start = robot_model::make_data(model.get());
  auto goal = robot_model::make_data(model.get());
  // The single arm's planner-test poses, each arm swung away from the other (left to +y, right to -y).
  const double kLeft[6] = {-1.2, 1.2, 1.0, 0.3, -0.3, 0.5}, kRight[6] = {1.2, 1.2, 1.0, 0.3, -0.3, -0.5};
  for (int i = 0; i < 6; ++i)
  {
    robot_model::set_joint_position(model.get(), goal.get(), "left_joint" + std::to_string(i + 1), kLeft[i]);
    robot_model::set_joint_position(model.get(), goal.get(), "right_joint" + std::to_string(i + 1), kRight[i]);
  }
  mj_kinematics(model.get(), start.get());
  mj_kinematics(model.get(), goal.get());
  EXPECT_TRUE(planner->is_valid(*start, robot_model::Environment{}));
  const auto plan = planner->plan(*start, *goal, robot_model::Environment{});
  EXPECT_TRUE(plan.solved) << plan.error;
}

// One arm reaching into the other is a collision to the planner, as it is to MuJoCo.
TEST(RebotDualArmGemini305, PlannerAvoidsOneArmHittingTheOther)
{
  const auto model = dual::load_mujoco_model();
  const auto planner = dual::make_planner(model.get());
  auto data = robot_model::make_data(model.get());
  // Turn the arms toward each other (joint1 about -z: positive turns clockwise from above, so the left arm,
  // at +y, turns toward -y with a positive angle, the right toward +y with a negative one) and stretch them out.
  int found = 0;
  for (const double turn : {0.6, 0.9, 1.2, 1.5, 1.8})
  {
    for (auto* d : {data.get()})
    {
      for (const char* joint : {"left_joint1", "right_joint1"}) robot_model::set_joint_position(model.get(), d, joint, 0.0);
    }
    robot_model::set_joint_position(model.get(), data.get(), "left_joint1", turn);
    robot_model::set_joint_position(model.get(), data.get(), "right_joint1", -turn);
    for (const char* joint : {"left_joint2", "right_joint2", "left_joint3", "right_joint3"})
      robot_model::set_joint_position(model.get(), data.get(), joint, 1.2);
    mj_kinematics(model.get(), data.get());
    const bool mujoco_between_arms = [&] {
      for (const std::string& pair : robot_model::penetrating_contacts(model.get(), data.get()))
      {
        const bool left_first = pair.rfind("left_", 0) == 0;
        const std::size_t dash = pair.find(" - ");
        const bool left_second = pair.compare(dash + 3, 5, "left_") == 0;
        if (left_first != left_second && pair.find("world") == std::string::npos) return true;
      }
      return false;
    }();
    if (mujoco_between_arms)
    {
      ++found;
      EXPECT_FALSE(planner->is_valid(*data, robot_model::Environment{})) << "turn " << turn;
    }
  }
  EXPECT_GT(found, 0) << "no sampled configuration had the arms touching";
}

}  // namespace
