// Kinematics backend on fast_robot_kinematics' generated code.
//
// Only for the glue file fast_kinematics() (fast_kinematics.bzl) generates: it
// includes the robot's generated chain header first, which sets
// FAST_FK_NUMBER_OF_JOINTS and renames the fast_fk namespace per robot. The class
// is in an anonymous namespace because its layout depends on that macro.

#pragma once

#include "src/robot_model/robot_model_utils/kinematics.hpp"

#include <fast_forward_kinematics/fast_kinematics.hpp>

#include <array>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#ifndef FAST_FK_NUMBER_OF_JOINTS
#error "include the generated <name>_chain.hpp first"
#endif

namespace
{

class FastKinematics final : public robot_model::Kinematics
{
public:
  static constexpr int N = FAST_FK_NUMBER_OF_JOINTS;

  FastKinematics(const mjModel* model, const std::array<const char*, N>& joint_names, const char* root_link,
                 const char* tip_link, const std::array<double, 16>& tip_transform)
    : Kinematics(model, std::vector<std::string>(joint_names.begin(), joint_names.end()), root_link, tip_link)
  {
    tip_.matrix() = Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(tip_transform.data());
    tip_inverse_ = tip_.inverse();
    // The model's joint limits as box constraints of the IK solve
    // (third_party/fast_robot_kinematics/joint_limit_box_constraints.patch).
    const Joints lower = lower_limits(), upper = upper_limits();
    data_.set_joint_limits(Eigen::Map<const Eigen::VectorXd>(lower.data(), N).cast<float>(),
                           Eigen::Map<const Eigen::VectorXd>(upper.data(), N).cast<float>());
  }

protected:
  Eigen::Isometry3d fk(const Joints& q) const override
  {
    Eigen::Vector<float, N> values;
    for (int i = 0; i < N; ++i)
    {
      values[i] = static_cast<float>(q[i]);
    }
    data_.set_joints(values);
    data_.forward_kinematics();
    Eigen::Matrix<float, 4, 4> frame;
    data_.get_frame(N - 1, frame);
    Eigen::Isometry3d last;
    last.matrix() = frame.cast<double>();
    return last * tip_;
  }

  void solve(const Eigen::Isometry3d& target, Joints& q, int iterations) const override
  {
    // The generated IK drives the last joint's frame; aim it at target * tip^-1.
    Eigen::Matrix<float, 4, 4> last = (target * tip_inverse_).matrix().cast<float>();
    Eigen::VectorX<float> guess(N);
    for (int i = 0; i < N; ++i)
    {
      guess[i] = static_cast<float>(q[i]);
    }
    // The solver holds a reference to param. Upstream stops at a gradient norm of
    // 1e-3, around millimeters of error for this cost; converge further and let
    // the base class judge the result against its tolerances.
    data_.fun->param.max_iterations = iterations;
    data_.fun->param.epsilon = 1e-6f;
    data_.fun->param.epsilon_rel = 1e-6f;
    try
    {
      data_.inverse_kinematics(last, guess);
    }
    catch (const std::exception&)
    {
      // L-BFGS-B can throw std::logic_error (e.g. "the moving direction does not
      // decrease the objective function value"), which upstream doesn't catch; the
      // attempt just failed, keep the iterate it reached.
    }
    for (int i = 0; i < N; ++i)
    {
      q[i] = guess[i];
    }
  }

private:
  Eigen::Isometry3d tip_;
  Eigen::Isometry3d tip_inverse_;
  // Scratch for the generated code (not thread-safe: one Kinematics per thread).
  mutable fast_fk::JointData data_;
};

}  // namespace
