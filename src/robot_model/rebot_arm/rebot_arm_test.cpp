#include "src/robot_model/rebot_arm/rebot_arm.hpp"

#include <gtest/gtest.h>

namespace
{

TEST(RebotArm, LoadsMujocoModel)
{
  const rebot_arm::MjModelPtr model = rebot_arm::load_mujoco_model();
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->njnt, 9);  // joint1-7 + joint_left/joint_right
  EXPECT_EQ(model->nu, 7);    // joint1-7 motors
  for (const char* joint : {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"})
  {
    EXPECT_GE(mj_name2id(model.get(), mjOBJ_JOINT, joint), 0) << joint;
  }
}

TEST(RebotArm, MjcfPathIsAbsolute)
{
  EXPECT_EQ(rebot_arm::mjcf_path().front(), '/');
}


// The gripper's fingers follow joint7 (equality constraints), so observing joint7 alone is enough.
TEST(RebotArm, StateEstimatorMovesTheGripperFingersWithJoint7)
{
  const auto model = rebot_arm::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  const auto estimator = rebot_arm::make_state_estimator(model.get());

  estimator->update_joint_positions(
    data.get(), {{"joint1", 0.4}, {"joint2", 0.2}, {"joint3", -0.5}, {"joint7", 0.03}});
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_left"), 0.03);
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_right"), 0.03);

  // An observed finger is not overridden.
  estimator->update_joint_positions(data.get(), {{"joint7", 0.01}, {"joint_left", 0.02}});
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_left"), 0.02);
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_right"), 0.01);

  // The end effector is where forward kinematics of the observed joints puts it.
  const auto kinematics = rebot_arm::make_kinematics(model.get());
  const int gripper = mj_name2id(model.get(), mjOBJ_BODY, "gripper_end");
  const Eigen::Vector3d at(data->xpos + 3 * gripper);
  EXPECT_NEAR((kinematics->forward_kinematics(*data).translation() - at).norm(), 0.0, 1e-6);
}

}  // namespace
