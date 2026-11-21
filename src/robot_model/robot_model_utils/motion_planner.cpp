#include "src/robot_model/robot_model_utils/motion_planner.hpp"

#include <stdexcept>
#include <utility>

namespace robot_model
{

MotionPlanner::MotionPlanner(const mjModel* model, std::vector<std::string> joint_names)
  : model_(model), joint_names_(std::move(joint_names))
{
  if (model_ == nullptr)
  {
    throw std::invalid_argument("MotionPlanner needs a MuJoCo model");
  }
  for (const auto& name : joint_names_)
  {
    qpos_addresses_.push_back(joint_qpos_address(model_, name));  // throws if missing / wrong type
    dof_addresses_.push_back(model_->jnt_dofadr[mj_name2id(model_, mjOBJ_JOINT, name.c_str())]);
  }
}

MotionPlanner::Joints MotionPlanner::joints(const mjData& state) const
{
  Joints q;
  q.reserve(qpos_addresses_.size());
  for (const int adr : qpos_addresses_)
  {
    q.push_back(state.qpos[adr]);
  }
  return q;
}

bool MotionPlanner::is_valid(const mjData& state, const Environment& environment) const
{
  return joints_valid(joints(state), environment);
}

bool MotionPlanner::is_motion_valid(const mjData& a, const mjData& b, const Environment& environment) const
{
  return joint_motion_valid(joints(a), joints(b), environment);
}

PlanResult MotionPlanner::plan(const mjData& start, const mjData& goal, const Environment& environment,
                               const PlannerSettings& settings) const
{
  JointPath planned = plan_joints(joints(start), joints(goal), environment, settings);
  PlanResult result;
  result.solved = planned.solved;
  result.planning_time_ms = planned.planning_time_ms;
  result.simplify_time_ms = planned.simplify_time_ms;
  result.iterations = planned.iterations;
  result.error = std::move(planned.error);
  for (const auto& q : planned.path)
  {
    if (q.size() != qpos_addresses_.size())
    {
      return PlanResult{false, {}, result.planning_time_ms, result.simplify_time_ms, result.iterations,
                        "planner backend returned a waypoint with " + std::to_string(q.size()) + " values for " +
                            std::to_string(qpos_addresses_.size()) + " joints"};
    }
    MjDataPtr waypoint = make_data(model_);
    mj_copyData(waypoint.get(), model_, &start);
    for (std::size_t i = 0; i < q.size(); ++i)
    {
      waypoint->qpos[qpos_addresses_[i]] = q[i];
      waypoint->qvel[dof_addresses_[i]] = 0;
    }
    mj_kinematics(model_, waypoint.get());
    result.path.push_back(std::move(waypoint));
  }
  return result;
}

}  // namespace robot_model
