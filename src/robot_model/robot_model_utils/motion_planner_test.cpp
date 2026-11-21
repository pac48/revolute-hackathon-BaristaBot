// MotionPlanner's mjData handling, with a fake backend on the two_link_arm model.

#include "src/robot_model/robot_model_utils/motion_planner.hpp"
#include "src/robot_model/two_link_arm/two_link_arm.hpp"

#include <gtest/gtest.h>

#include <stdexcept>

namespace
{

using robot_model::Environment;
using robot_model::joint_position;
using robot_model::make_data;
using robot_model::set_joint_position;

// Plans the straight line start -> goal and records the joint values it sees.
class StraightLinePlanner : public robot_model::MotionPlanner
{
public:
  StraightLinePlanner(const mjModel* model, std::vector<std::string> joints)
    : MotionPlanner(model, std::move(joints))
  {
  }
  mutable std::vector<Joints> seen;

protected:
  bool joints_valid(const Joints& q, const Environment&) const override
  {
    seen.push_back(q);
    return true;
  }
  bool joint_motion_valid(const Joints& a, const Joints& b, const Environment&) const override
  {
    seen.push_back(a);
    seen.push_back(b);
    return true;
  }
  JointPath plan_joints(const Joints& start, const Joints& goal, const Environment&,
                        const robot_model::PlannerSettings&) const override
  {
    JointPath out;
    out.solved = true;
    Joints middle;
    for (std::size_t i = 0; i < start.size(); ++i)
    {
      middle.push_back((start[i] + goal[i]) / 2);
    }
    out.path = {start, middle, goal};
    return out;
  }
};

class MotionPlannerTest : public ::testing::Test
{
protected:
  two_link_arm::MjModelPtr model_ = two_link_arm::load_mujoco_model();
};

TEST_F(MotionPlannerTest, ChecksJointsAtConstruction)
{
  EXPECT_THROW(StraightLinePlanner(model_.get(), {"joint1", "nope"}), std::invalid_argument);
  EXPECT_THROW(StraightLinePlanner(nullptr, {"joint1"}), std::invalid_argument);
  EXPECT_EQ(StraightLinePlanner(model_.get(), {"joint1", "joint2"}).dof(), 2U);
}

TEST_F(MotionPlannerTest, ReadsPlanningJointsByNameInPlannerOrder)
{
  StraightLinePlanner planner(model_.get(), {"joint2", "joint1"});  // reverse of the model's order
  auto state = make_data(model_.get());
  set_joint_position(model_.get(), state.get(), "joint1", -0.5);
  set_joint_position(model_.get(), state.get(), "joint2", 0.25);
  planner.is_valid(*state, {});
  ASSERT_EQ(planner.seen.size(), 1U);
  EXPECT_EQ(planner.seen[0], (std::vector<double>{0.25, -0.5}));
}

TEST_F(MotionPlannerTest, PathIsMjDataCopiesOfTheStartWithPlanningJointsSet)
{
  StraightLinePlanner planner(model_.get(), {"joint1"});  // joint2 is not planned
  auto start = make_data(model_.get());
  auto goal = make_data(model_.get());
  set_joint_position(model_.get(), start.get(), "joint1", 0.2);
  set_joint_position(model_.get(), start.get(), "joint2", 0.7);
  start->qvel[0] = 1.0;
  set_joint_position(model_.get(), goal.get(), "joint1", 0.6);
  set_joint_position(model_.get(), goal.get(), "joint2", -1.0);

  const auto result = planner.plan(*start, *goal, {});
  ASSERT_TRUE(result.solved);
  ASSERT_EQ(result.path.size(), 3U);
  const double expected[] = {0.2, 0.4, 0.6};
  for (std::size_t i = 0; i < 3; ++i)
  {
    const mjData& waypoint = *result.path[i];
    EXPECT_DOUBLE_EQ(joint_position(model_.get(), &waypoint, "joint1"), expected[i]);
    EXPECT_DOUBLE_EQ(joint_position(model_.get(), &waypoint, "joint2"), 0.7);  // from start, not goal
    EXPECT_EQ(waypoint.qvel[0], 0.0);  // planning joint velocity zeroed
  }
}

TEST_F(MotionPlannerTest, JointAccessorsRejectUnknownJoints)
{
  auto d = make_data(model_.get());
  EXPECT_THROW(joint_position(model_.get(), d.get(), "nope"), std::invalid_argument);
  EXPECT_THROW(set_joint_position(model_.get(), d.get(), "nope", 1), std::invalid_argument);
}

}  // namespace
