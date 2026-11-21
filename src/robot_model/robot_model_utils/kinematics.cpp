#include "src/robot_model/robot_model_utils/kinematics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace robot_model
{

Kinematics::Kinematics(const mjModel* model, std::vector<std::string> joint_names, std::string root_link,
                       std::string tip_link)
  : model_(model)
  , joint_names_(std::move(joint_names))
  , root_link_(std::move(root_link))
  , tip_link_(std::move(tip_link))
{
  if (model_ == nullptr)
  {
    throw std::invalid_argument("Kinematics needs a MuJoCo model");
  }
  constexpr double kPi = 3.14159265358979323846;
  for (const auto& name : joint_names_)
  {
    qpos_addresses_.push_back(joint_qpos_address(model_, name));  // throws if missing / wrong type
    const int id = mj_name2id(model_, mjOBJ_JOINT, name.c_str());
    dof_addresses_.push_back(model_->jnt_dofadr[id]);
    const bool hinge = model_->jnt_type[id] == mjJNT_HINGE;
    const bool limited = model_->jnt_limited[id];
    lower_.push_back(limited ? model_->jnt_range[2 * id] : (hinge ? -kPi : -1.0));
    upper_.push_back(limited ? model_->jnt_range[2 * id + 1] : (hinge ? kPi : 1.0));
    wraps_.push_back(hinge);
    limited_.push_back(limited);
  }
}

Kinematics::Joints Kinematics::lower_limits() const
{
  Joints out;
  for (std::size_t i = 0; i < dof(); ++i)
  {
    out.push_back(limited_[i] ? lower_[i] : -std::numeric_limits<double>::infinity());
  }
  return out;
}

Kinematics::Joints Kinematics::upper_limits() const
{
  Joints out;
  for (std::size_t i = 0; i < dof(); ++i)
  {
    out.push_back(limited_[i] ? upper_[i] : std::numeric_limits<double>::infinity());
  }
  return out;
}

Eigen::Isometry3d Kinematics::forward_kinematics(const mjData& state) const
{
  Joints q;
  for (const int adr : qpos_addresses_)
  {
    q.push_back(state.qpos[adr]);
  }
  return fk(q);
}

int Kinematics::within_limits(Joints& q, double& excess) const
{
  constexpr double kTwoPi = 2 * 3.14159265358979323846;
  // Solutions on a limit (box-constrained solves in float) can be ~1e-7 past the
  // double limit; clamp those onto it.
  constexpr double kSlack = 1e-5;
  int worst = -1;
  excess = 0;
  for (std::size_t i = 0; i < q.size(); ++i)
  {
    if (!std::isfinite(q[i]))
    {
      return -2;
    }
    if (wraps_[i])
    {
      // The representative of q[i] (mod 2*pi) closest to the middle of the range.
      const double mid = 0.5 * (lower_[i] + upper_[i]);
      q[i] -= kTwoPi * std::round((q[i] - mid) / kTwoPi);
    }
    const double past = std::max(lower_[i] - q[i], q[i] - upper_[i]);
    if (past > kSlack && past > excess)
    {
      excess = past;
      worst = static_cast<int>(i);
    }
  }
  if (worst >= 0)
  {
    return worst;
  }
  for (std::size_t i = 0; i < q.size(); ++i)
  {
    q[i] = std::clamp(q[i], lower_[i], upper_[i]);
  }
  return -1;
}

IkResult Kinematics::inverse_kinematics(const Eigen::Isometry3d& target, mjData& state,
                                        const IkSettings& settings) const
{
  const auto start = std::chrono::steady_clock::now();
  std::mt19937 rng(settings.random_seed);
  IkResult result;
  Joints best;
  double best_score = std::numeric_limits<double>::infinity();

  for (int attempt = 0; attempt < settings.max_attempts; ++attempt)
  {
    Joints q(dof());
    for (std::size_t i = 0; i < dof(); ++i)
    {
      q[i] = attempt == 0 ? state.qpos[qpos_addresses_[i]]
                          : std::uniform_real_distribution<double>(lower_[i], upper_[i])(rng);
    }
    const Joints seed = q;
    solve(target, q, settings.iterations);
    ++result.attempts;
    // A non-finite joint falls back to its seed value; then joints past their limits are
    // projected onto them. Either way the attempt is still scored against the tolerances.
    bool non_finite = false;
    for (std::size_t i = 0; i < q.size(); ++i)
    {
      if (!std::isfinite(q[i]))
      {
        q[i] = std::isfinite(seed[i]) ? seed[i] : 0.5 * (lower_[i] + upper_[i]);
        non_finite = true;
      }
    }
    result.non_finite += non_finite;
    double excess = 0;
    const int violation = within_limits(q, excess);
    if (violation >= 0)
    {
      ++result.out_of_limits;
      if (excess > result.worst_excess)
      {
        result.worst_excess = excess;
        result.worst_joint = violation;
      }
    }
    for (std::size_t i = 0; i < q.size(); ++i)
    {
      q[i] = std::clamp(q[i], lower_[i], upper_[i]);
    }
    const Eigen::Isometry3d pose = fk(q);
    const double position_error = (pose.translation() - target.translation()).norm();
    const double orientation_error = Eigen::AngleAxisd(target.linear().transpose() * pose.linear()).angle();
    // Rank attempts by error relative to the tolerances.
    const double score = std::max(position_error / settings.position_tolerance,
                                  orientation_error / settings.orientation_tolerance);
    if (score < best_score)
    {
      best_score = score;
      best = q;
      result.position_error = position_error;
      result.orientation_error = orientation_error;
    }
    if (score <= 1.0)
    {
      result.solved = true;
      break;
    }
  }

  if (result.solved)
  {
    for (std::size_t i = 0; i < best.size(); ++i)
    {
      state.qpos[qpos_addresses_[i]] = best[i];
      state.qvel[dof_addresses_[i]] = 0;
    }
    mj_kinematics(model_, &state);
  }
  result.time_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  return result;
}

}  // namespace robot_model
