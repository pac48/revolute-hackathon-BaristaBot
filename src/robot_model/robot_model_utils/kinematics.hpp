// Forward and inverse kinematics interface shared by all robot models.
//
// Each robot model package provides a factory (e.g. rebot_arm::make_kinematics()
// in rebot_arm.hpp) returning a Kinematics for one chain (root link -> tip link)
// of its MuJoCo model. Like MotionPlanner, states are mjData of that model: the
// chain's joints (joint_names()) are read and written in qpos by name.
// Poses are of the tip link in the root link's frame.
//
// Implementations: fast_kinematics() (fast_kinematics.bzl), generated with
// fast_robot_kinematics.

#pragma once

#include "src/robot_model/robot_model_utils/mujoco_model.hpp"

#include <Eigen/Geometry>
#include <mujoco/mujoco.h>

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace robot_model
{

struct IkSettings
{
  // Local solves: the first from the seed state, the rest from random joint
  // positions within the limits, until one is within tolerance.
  int max_attempts = 50;
  int iterations = 100;                // solver iterations per attempt
  double position_tolerance = 1e-3;    // meters
  double orientation_tolerance = 1e-2; // radians
  unsigned random_seed = 0;
};

struct IkResult
{
  bool solved = false;
  // Of the best attempt (the solution if solved).
  double position_error = std::numeric_limits<double>::infinity();     // meters
  double orientation_error = std::numeric_limits<double>::infinity();  // radians
  int attempts = 0;
  double time_ms = 0;
  // Attempts repaired before scoring: a non-finite result (those joints reset to the seed), or
  // a joint past its limit (projected onto it; the worst such joint, as an index into
  // joint_names(), and how far past, in rad or m).
  int non_finite = 0;
  int out_of_limits = 0;
  int worst_joint = -1;
  double worst_excess = 0;
};

class Kinematics
{
public:
  virtual ~Kinematics() = default;
  Kinematics(const Kinematics&) = delete;
  Kinematics& operator=(const Kinematics&) = delete;

  // The model whose mjData it takes (not owned; must outlive this).
  const mjModel* model() const { return model_; }
  // Chain joints from root to tip (hinge joints of the model).
  const std::vector<std::string>& joint_names() const { return joint_names_; }
  std::size_t dof() const { return joint_names_.size(); }
  const std::string& root_link() const { return root_link_; }
  const std::string& tip_link() const { return tip_link_; }

  // Tip pose in the root frame for the chain's joint positions in `state`.
  Eigen::Isometry3d forward_kinematics(const mjData& state) const;
  // Joint positions within the model's joint limits (box constraints of the solve
  // where the backend supports them) that put the tip at `target`
  // (in the root frame), starting from `state`'s. If solved, they're written into
  // `state` (the chain's velocities zeroed, kinematics recomputed; other joints
  // untouched); otherwise `state` is unchanged. In place, as making an mjData
  // costs several times the solve; copy first (mj_copyData) to keep the seed.
  IkResult inverse_kinematics(const Eigen::Isometry3d& target, mjData& state,
                              const IkSettings& settings = {}) const;

protected:
  // Throws std::invalid_argument if a joint isn't a hinge/slide joint of the model.
  Kinematics(const mjModel* model, std::vector<std::string> joint_names, std::string root_link,
             std::string tip_link);

  // Backend, on the chain's joint positions in joint_names() order.
  using Joints = std::vector<double>;
  virtual Eigen::Isometry3d fk(const Joints& q) const = 0;
  // One local IK solve from q (updated in place); the base class checks the result.
  virtual void solve(const Eigen::Isometry3d& target, Joints& q, int iterations) const = 0;

  // The model's joint limits in joint_names() order; +-infinity for unlimited joints.
  Joints lower_limits() const;
  Joints upper_limits() const;

private:
  // Wraps revolute joints by 2*pi into their limits; false if a joint can't be.
  // -1 if within, -2 if non-finite, else the index of the joint furthest past its limit
  // (`excess` = how far).
  int within_limits(Joints& q, double& excess) const;

  const mjModel* model_;
  std::vector<std::string> joint_names_;
  std::string root_link_;
  std::string tip_link_;
  std::vector<int> qpos_addresses_;
  std::vector<int> dof_addresses_;
  std::vector<double> lower_;
  std::vector<double> upper_;
  std::vector<bool> wraps_;    // hinge joints (2*pi periodic)
  std::vector<bool> limited_;
};

}  // namespace robot_model
