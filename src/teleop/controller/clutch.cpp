#include "src/teleop/controller/clutch.hpp"

#include <algorithm>
#include <cmath>

namespace teleop
{

bool Hysteresis::update(double value)
{
  if (state && value < off)
  {
    state = false;
  }
  else if (!state && value > on)
  {
    state = true;
  }
  return state;
}

std::optional<Eigen::Isometry3d> Clutch::update(bool engaged, const std::optional<Eigen::Isometry3d>& controller,
                                                const Eigen::Isometry3d& tool)
{
  if (!engaged || !controller)
  {
    engaged_ = false;
    return std::nullopt;
  }
  const Eigen::Vector3d cp = controller->translation();
  const Eigen::Quaterniond cq(controller->linear());
  if (!engaged_)
  {
    engaged_ = true;
    controller_start_p_ = cp;
    controller_start_q_ = cq;
    tool_start_p_ = tool.translation();
    tool_start_q_ = Eigen::Quaterniond(tool.linear());
  }
  Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
  target.translation() = tool_start_p_ + scale_ * (cp - controller_start_p_);
  if (rotation_)
  {
    // Base-frame delta: R_tool = (R_c * R_c0^-1) * R_tool0.
    const Eigen::Quaterniond delta = cq * controller_start_q_.conjugate();
    target.linear() = (delta * tool_start_q_).normalized().toRotationMatrix();
  }
  else
  {
    target.linear() = tool_start_q_.toRotationMatrix();
  }
  return target;
}

void ClutchedTarget::reset(const Eigen::Isometry3d& pose)
{
  target_ = pose;
  initialized_ = true;
  clutch_.release();  // an engaged clutch re-anchors on the next update
}

const Eigen::Isometry3d& ClutchedTarget::update(bool engaged, const std::optional<Eigen::Isometry3d>& controller)
{
  // The reference is read only when the clutch engages: the target as it is at that moment.
  if (const auto t = clutch_.update(engaged && initialized_, controller, target_))
  {
    target_ = *t;
  }
  return target_;
}

Eigen::Isometry3d limit_step(const Eigen::Isometry3d& current, const Eigen::Isometry3d& target,
                             double max_distance, double max_angle)
{
  const Eigen::Quaterniond q0(current.linear()), q1(target.linear());
  const double distance = (target.translation() - current.translation()).norm();
  const double angle = q0.angularDistance(q1);
  double s = 1.0;
  if (distance > max_distance)
  {
    s = std::min(s, max_distance / distance);
  }
  if (angle > max_angle)
  {
    s = std::min(s, max_angle / angle);
  }
  if (s >= 1.0)
  {
    return target;
  }
  Eigen::Isometry3d out = Eigen::Isometry3d::Identity();
  out.translation() = current.translation() + s * (target.translation() - current.translation());
  out.linear() = q0.slerp(s, q1).normalized().toRotationMatrix();
  return out;
}

double point_time(const std::vector<double>& from, const std::vector<double>& to,
                  const std::vector<double>& max_velocity, double min_time, int* limiting_joint)
{
  double t = min_time;
  int limiting = -1;
  for (std::size_t i = 0; i < from.size() && i < to.size() && i < max_velocity.size(); ++i)
  {
    const double ti = std::abs(to[i] - from[i]) / max_velocity[i];
    if (ti > t)
    {
      t = ti;
      limiting = static_cast<int>(i);
    }
  }
  if (limiting_joint)
  {
    *limiting_joint = limiting;
  }
  return t;
}

double gripper_position(double squeeze, double open_position, double closed_position)
{
  const double s = std::clamp(squeeze, 0.0, 1.0);
  return open_position + s * (closed_position - open_position);
}

}  // namespace teleop
