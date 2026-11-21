// Rigid-body dynamics terms of a URDF's joints (KDL recursive Newton-Euler over the whole tree).
//
// For selected joints it gives the three parts of tau = M(q) qdd + C(q, qd) qd + g(q) separately.
// The tree (not just a chain to a tip) is used so that every link's mass counts, including links
// on branches off the controlled joints, e.g. the fingers hanging off the gripper. Joints of the
// URDF that are not selected are held at zero position, velocity and acceleration.
//
// The terms come from three inverse dynamics passes, and nothing allocates in compute(), so it
// can run in a real-time loop:
//   g           = rne(q, 0,  0)
//   C qd        = rne(q, qd, 0)  - g
//   M qdd       = rne(q, 0,  qdd) - g
#pragma once

#include <array>
#include <map>
#include <string>
#include <vector>

#include "kdl/jntarray.hpp"
#include "kdl/tree.hpp"
#include "kdl/treeidsolver_recursive_newton_euler.hpp"

namespace joint_impedance_controller
{

class TreeDynamics
{
public:
  /// Throws std::runtime_error if the URDF does not parse or a joint is not a movable joint of it.
  TreeDynamics(
    const std::string & urdf_xml, const std::vector<std::string> & joints,
    const std::array<double, 3> & gravity);

  TreeDynamics(const TreeDynamics &) = delete;
  TreeDynamics & operator=(const TreeDynamics &) = delete;

  size_t size() const { return joint_index_.size(); }

  /// Torques (N*m or N) on the joints, all vectors of size(): inertia = M(q) qdd,
  /// coriolis = C(q, qd) qd, gravity = g(q).
  void compute(
    const std::vector<double> & q, const std::vector<double> & qd, const std::vector<double> & qdd,
    std::vector<double> & inertia, std::vector<double> & coriolis, std::vector<double> & gravity);

private:
  void solve(const KDL::JntArray & q, const KDL::JntArray & qd, const KDL::JntArray & qdd);

  KDL::Tree tree_;
  KDL::TreeIdSolver_RNE solver_;
  KDL::WrenchMap no_external_wrenches_;
  std::vector<unsigned int> joint_index_;  // selected joint -> column in the tree's joint arrays
  KDL::JntArray q_, qd_, qdd_, zero_, tau_;
};

}  // namespace joint_impedance_controller
