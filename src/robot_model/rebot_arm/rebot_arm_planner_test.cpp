// rebot_arm::make_planner(): the VAMP planner for the arm, on mjData states.

#include "src/robot_model/rebot_arm/rebot_arm.hpp"
#include "src/robot_model/two_link_arm/two_link_arm.hpp"

#include <gtest/gtest.h>

#include <array>
#include <optional>
#include <stdexcept>

namespace
{

using robot_model::BoxObstacle;
using robot_model::Environment;
using robot_model::MjDataPtr;

using ArmJoints = std::array<double, 6>;
const ArmJoints kHome{0, 0, 0, 0, 0, 0};
// Arm raised and swung to one side, then to the other.
const ArmJoints kLeft{-1.2, 1.2, 1.0, 0.3, -0.3, 0.5};
const ArmJoints kRight{1.2, 1.2, 1.0, 0.3, -0.3, -0.5};

class RebotArmPlannerTest : public ::testing::Test
{
protected:
  // A state of the arm with joint1..joint6 at q (everything else at qpos0).
  MjDataPtr state(const ArmJoints& q) const
  {
    auto d = robot_model::make_data(model_.get());
    for (std::size_t i = 0; i < q.size(); ++i)
    {
      robot_model::set_joint_position(model_.get(), d.get(), "joint" + std::to_string(i + 1), q[i]);
    }
    return d;
  }

  void expect_path_valid(const robot_model::PlanResult& result, const ArmJoints& start, const ArmJoints& goal,
                         const Environment& env) const
  {
    ASSERT_TRUE(result.solved) << result.error;
    ASSERT_GE(result.path.size(), 2U);
    for (std::size_t i = 0; i < 6; ++i)
    {
      const std::string joint = "joint" + std::to_string(i + 1);
      EXPECT_NEAR(robot_model::joint_position(model_.get(), result.path.front().get(), joint), start[i], 1e-5);
      EXPECT_NEAR(robot_model::joint_position(model_.get(), result.path.back().get(), joint), goal[i], 1e-5);
    }
    for (std::size_t i = 0; i + 1 < result.path.size(); ++i)
    {
      EXPECT_TRUE(planner_->is_motion_valid(*result.path[i], *result.path[i + 1], env)) << "segment " << i;
    }
  }

  robot_model::MjModelPtr model_ = rebot_arm::load_mujoco_model();
  std::unique_ptr<robot_model::MotionPlanner> planner_ = rebot_arm::make_planner(model_.get());
};

TEST_F(RebotArmPlannerTest, PlansTheArmJoints)
{
  EXPECT_EQ(planner_->joint_names(),
            (std::vector<std::string>{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}));
  EXPECT_EQ(planner_->model(), model_.get());
}

TEST_F(RebotArmPlannerTest, Validity)
{
  EXPECT_TRUE(planner_->is_valid(*state(kHome), {}));
  ArmJoints out_of_limits = kHome;
  out_of_limits[0] = 3.0;  // joint1 range is +-2.8
  EXPECT_FALSE(planner_->is_valid(*state(out_of_limits), {}));
  EXPECT_EQ(planner_->plan(*state(kHome), *state(out_of_limits), {}).error,
            "goal is outside the joint limits (joint1)");
  const Environment inside_box{{}, {BoxObstacle{{0, 0, 0}, {2, 2, 2}}}};
  EXPECT_FALSE(planner_->is_valid(*state(kLeft), inside_box));
  EXPECT_EQ(planner_->plan(*state(kHome), *state(kLeft), inside_box).error, "start is in collision");
}

TEST_F(RebotArmPlannerTest, PlansInFreeSpace)
{
  expect_path_valid(planner_->plan(*state(kHome), *state(kLeft), {}), kHome, kLeft, {});
}

TEST_F(RebotArmPlannerTest, WaypointsKeepTheStartsOtherJoints)
{
  auto start = state(kHome);
  robot_model::set_joint_position(model_.get(), start.get(), "joint7", 0.02);  // gripper opening: not planned
  const auto result = planner_->plan(*start, *state(kLeft), {});
  ASSERT_TRUE(result.solved) << result.error;
  for (const auto& waypoint : result.path)
  {
    EXPECT_EQ(robot_model::joint_position(model_.get(), waypoint.get(), "joint7"), 0.02);
  }
}

// A pole that blocks the direct motion left -> right but not the endpoints,
// found by search so the test doesn't depend on hand-placed geometry.
std::optional<Environment> blocking_pole(const robot_model::MotionPlanner& planner, const mjData& left,
                                         const mjData& right)
{
  for (double x = -0.6; x <= 0.6; x += 0.05)
  {
    for (double y = -0.6; y <= 0.6; y += 0.05)
    {
      const Environment env{{}, {BoxObstacle{{x, y, 0.4}, {0.03, 0.03, 0.4}}}};
      if (planner.is_valid(left, env) && planner.is_valid(right, env) && !planner.is_motion_valid(left, right, env))
      {
        return env;
      }
    }
  }
  return std::nullopt;
}

TEST_F(RebotArmPlannerTest, PlansAroundAnObstacle)
{
  const auto left = state(kLeft);
  const auto right = state(kRight);
  ASSERT_TRUE(planner_->is_motion_valid(*left, *right, {}));
  const auto env = blocking_pole(*planner_, *left, *right);
  ASSERT_TRUE(env.has_value()) << "no pole position blocks the direct motion";
  const auto result = planner_->plan(*left, *right, *env);
  expect_path_valid(result, kLeft, kRight, *env);
  EXPECT_GT(result.path.size(), 2U);  // had to leave the straight line
}

TEST_F(RebotArmPlannerTest, RejectsAModelWithoutTheArmJoints)
{
  const auto other = two_link_arm::load_mujoco_model();  // has joint1/joint2 only
  EXPECT_THROW(rebot_arm::make_planner(other.get()), std::invalid_argument);
  EXPECT_THROW(rebot_arm::make_planner(nullptr), std::invalid_argument);
}

}  // namespace
