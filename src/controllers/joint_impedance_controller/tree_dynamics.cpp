#include "tree_dynamics.hpp"

#include <stdexcept>

#include "kdl_parser/kdl_parser.hpp"

namespace joint_impedance_controller
{
namespace
{
KDL::Tree parse(const std::string & urdf_xml)
{
  KDL::Tree tree;
  if (!kdl_parser::treeFromString(urdf_xml, tree))
  {
    throw std::runtime_error("failed to build a KDL tree from the robot description");
  }
  return tree;
}
}  // namespace

TreeDynamics::TreeDynamics(
  const std::string & urdf_xml, const std::vector<std::string> & joints,
  const std::array<double, 3> & gravity)
: tree_(parse(urdf_xml)),
  solver_(tree_, KDL::Vector(gravity[0], gravity[1], gravity[2])),
  q_(tree_.getNrOfJoints()),
  qd_(tree_.getNrOfJoints()),
  qdd_(tree_.getNrOfJoints()),
  zero_(tree_.getNrOfJoints()),
  tau_(tree_.getNrOfJoints())
{
  SetToZero(q_);
  SetToZero(qd_);
  SetToZero(qdd_);
  SetToZero(zero_);
  SetToZero(tau_);

  std::map<std::string, unsigned int> columns;
  for (const auto & [name, segment] : tree_.getSegments())
  {
    if (segment.segment.getJoint().getType() != KDL::Joint::None)
    {
      columns[segment.segment.getJoint().getName()] = segment.q_nr;
    }
  }
  for (const auto & joint : joints)
  {
    const auto it = columns.find(joint);
    if (it == columns.end())
    {
      throw std::runtime_error("joint '" + joint + "' is not a movable joint of the robot description");
    }
    joint_index_.push_back(it->second);
  }
}

void TreeDynamics::solve(const KDL::JntArray & q, const KDL::JntArray & qd, const KDL::JntArray & qdd)
{
  if (solver_.CartToJnt(q, qd, qdd, no_external_wrenches_, tau_) < 0)
  {
    throw std::runtime_error("KDL inverse dynamics failed");
  }
}

void TreeDynamics::compute(
  const std::vector<double> & q, const std::vector<double> & qd, const std::vector<double> & qdd,
  std::vector<double> & inertia, std::vector<double> & coriolis, std::vector<double> & gravity)
{
  for (size_t i = 0; i < joint_index_.size(); ++i)
  {
    q_(joint_index_[i]) = q[i];
    qd_(joint_index_[i]) = qd[i];
    qdd_(joint_index_[i]) = qdd[i];
  }

  solve(q_, zero_, zero_);
  for (size_t i = 0; i < joint_index_.size(); ++i) gravity[i] = tau_(joint_index_[i]);

  solve(q_, qd_, zero_);
  for (size_t i = 0; i < joint_index_.size(); ++i) coriolis[i] = tau_(joint_index_[i]) - gravity[i];

  solve(q_, zero_, qdd_);
  for (size_t i = 0; i < joint_index_.size(); ++i) inertia[i] = tau_(joint_index_[i]) - gravity[i];
}

}  // namespace joint_impedance_controller
