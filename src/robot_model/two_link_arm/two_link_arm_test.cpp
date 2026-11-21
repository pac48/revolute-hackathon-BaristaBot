#include "src/robot_model/two_link_arm/two_link_arm.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace
{

TEST(TwoLinkArm, LoadsMujocoModel)
{
  const two_link_arm::MjModelPtr model = two_link_arm::load_mujoco_model();
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->njnt, 2);
  EXPECT_EQ(model->nu, 2);
  EXPECT_GE(mj_name2id(model.get(), mjOBJ_JOINT, "joint1"), 0);
  EXPECT_GE(mj_name2id(model.get(), mjOBJ_JOINT, "joint2"), 0);
}

TEST(TwoLinkArm, MjcfPathIsAbsolute)
{
  EXPECT_EQ(two_link_arm::mjcf_path().front(), '/');
}

TEST(TwoLinkArm, StateEstimatorSetsJointStatesAndKinematics)
{
  const auto model = two_link_arm::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  const auto estimator = two_link_arm::make_state_estimator(model.get());

  const int tip = model->nbody - 1;
  const double before = data->xpos[3 * tip + 2];
  robot_model::Observation observation;
  observation.joint_positions = {{"joint1", 0.5}, {"joint2", -0.3}};
  observation.joint_velocities = {{"joint2", 1.5}};
  estimator->update(data.get(), observation);
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint1"), 0.5);
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint2"), -0.3);
  EXPECT_DOUBLE_EQ(data->qvel[model->jnt_dofadr[mj_name2id(model.get(), mjOBJ_JOINT, "joint2")]], 1.5);
  EXPECT_NE(data->xpos[3 * tip + 2], before);  // kinematics are current

  // A joint that is not observed keeps its value.
  estimator->update_joint_positions(data.get(), {{"joint1", 0.1}});
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint1"), 0.1);
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint2"), -0.3);
}

TEST(TwoLinkArm, StateEstimatorRejectsBadObservationsWithoutChangingTheState)
{
  const auto model = two_link_arm::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  const auto estimator = two_link_arm::make_state_estimator(model.get());
  EXPECT_THROW(estimator->update_joint_positions(data.get(), {{"joint1", 0.5}, {"no_such_joint", 1.0}}),
               std::invalid_argument);
  EXPECT_THROW(estimator->update_joint_positions(data.get(), {{"joint1", std::nan("")}}), std::invalid_argument);
  EXPECT_EQ(robot_model::joint_position(model.get(), data.get(), "joint1"), 0.0);
}

}  // namespace
