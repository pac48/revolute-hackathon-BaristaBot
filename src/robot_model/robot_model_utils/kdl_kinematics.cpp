#include "src/robot_model/robot_model_utils/kdl_kinematics.hpp"

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_lma.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace robot_model
{
namespace
{

int body_id(const mjModel* m, const std::string& name)
{
  const int id = mj_name2id(m, mjOBJ_BODY, name.c_str());
  if (id < 0) throw std::invalid_argument("no body '" + name + "' in the model");
  return id;
}

// Bodies strictly below root down to tip, root side first.
std::vector<int> path(const mjModel* m, int root, int tip)
{
  std::vector<int> bodies;
  for (int b = tip; b != root; b = m->body_parentid[b])
  {
    if (b == 0) throw std::invalid_argument("the tip body isn't below the root body");
    bodies.push_back(b);
  }
  std::reverse(bodies.begin(), bodies.end());
  return bodies;
}

std::vector<std::string> chain_joints(const mjModel* m, const std::vector<int>& bodies)
{
  std::vector<std::string> names;
  for (const int b : bodies)
  {
    if (m->body_jntnum[b] > 1) throw std::invalid_argument("a chain body has more than one joint");
    if (m->body_jntnum[b] == 1)
    {
      const int j = m->body_jntadr[b];
      if (m->jnt_type[j] != mjJNT_HINGE && m->jnt_type[j] != mjJNT_SLIDE)
        throw std::invalid_argument("chain joints must be hinge or slide joints");
      const char* name = mj_id2name(m, mjOBJ_JOINT, j);
      if (!name) throw std::invalid_argument("chain joints need names");
      names.emplace_back(name);
    }
  }
  return names;
}

class KdlKinematics : public Kinematics
{
public:
  KdlKinematics(const mjModel* model, const std::string& root, const std::string& tip, std::vector<int> bodies)
    : Kinematics(model, chain_joints(model, bodies), root, tip)
  {
    for (const int b : bodies)
    {
      // The body's frame in its parent's at zero joint positions.
      const mjtNum* p = model->body_pos + 3 * b;
      const mjtNum* q = model->body_quat + 4 * b;
      const KDL::Frame parent_body(KDL::Rotation::Quaternion(q[1], q[2], q[3], q[0]), KDL::Vector(p[0], p[1], p[2]));
      if (model->body_jntnum[b] == 0)
      {
        chain_.addSegment(KDL::Segment(KDL::Joint(KDL::Joint::None), parent_body));
        continue;
      }
      // A joint in the body frame turns about its axis through its anchor; KDL wants both in
      // the segment's base (parent) frame, followed by the body frame as the segment tip.
      const int j = model->body_jntadr[b];
      const mjtNum* a = model->jnt_axis + 3 * j;
      const mjtNum* o = model->jnt_pos + 3 * j;
      const KDL::Vector origin = parent_body * KDL::Vector(o[0], o[1], o[2]);
      const KDL::Vector axis = parent_body.M * KDL::Vector(a[0], a[1], a[2]);
      const auto type = model->jnt_type[j] == mjJNT_HINGE ? KDL::Joint::RotAxis : KDL::Joint::TransAxis;
      chain_.addSegment(KDL::Segment(KDL::Joint(mj_id2name(model, mjOBJ_JOINT, j), origin, axis, type), parent_body));
    }
    fk_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
    in_.resize(chain_.getNrOfJoints());
    out_.resize(chain_.getNrOfJoints());
  }

protected:
  Eigen::Isometry3d fk(const Joints& q) const override
  {
    for (std::size_t i = 0; i < q.size(); ++i) in_(static_cast<unsigned>(i)) = q[i];
    KDL::Frame f;
    fk_->JntToCart(in_, f);
    return to_eigen(f);
  }

  void solve(const Eigen::Isometry3d& target, Joints& q, int iterations) const override
  {
    if (!ik_ || iterations != ik_iterations_)
    {
      // Weights: positions in m, rotations in rad scaled down so 1 rad counts like 10 cm.
      Eigen::Matrix<double, 6, 1> weights;
      weights << 1, 1, 1, 0.1, 0.1, 0.1;
      ik_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(chain_, weights, 1e-6, iterations, 1e-12);
      ik_iterations_ = iterations;
    }
    for (std::size_t i = 0; i < q.size(); ++i) in_(static_cast<unsigned>(i)) = q[i];
    KDL::Frame goal;
    const Eigen::Matrix3d r = target.linear();
    goal.M = KDL::Rotation(r(0, 0), r(0, 1), r(0, 2), r(1, 0), r(1, 1), r(1, 2), r(2, 0), r(2, 1), r(2, 2));
    goal.p = KDL::Vector(target.translation().x(), target.translation().y(), target.translation().z());
    ik_->CartToJnt(in_, goal, out_);  // the base class judges whatever it reached
    for (std::size_t i = 0; i < q.size(); ++i) q[i] = out_(static_cast<unsigned>(i));
  }

private:
  static Eigen::Isometry3d to_eigen(const KDL::Frame& f)
  {
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    for (int r = 0; r < 3; ++r)
    {
      for (int c = 0; c < 3; ++c) t.linear()(r, c) = f.M(r, c);
      t.translation()[r] = f.p[r];
    }
    return t;
  }

  KDL::Chain chain_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_;
  // Scratch (not thread-safe: one Kinematics per thread).
  mutable std::unique_ptr<KDL::ChainIkSolverPos_LMA> ik_;
  mutable int ik_iterations_ = -1;
  mutable KDL::JntArray in_, out_;
};

}  // namespace

std::unique_ptr<Kinematics> make_kdl_kinematics(const mjModel* model, const std::string& root_body,
                                                const std::string& tip_body)
{
  if (model == nullptr) throw std::invalid_argument("make_kdl_kinematics needs a MuJoCo model");
  std::vector<int> bodies = path(model, body_id(model, root_body), body_id(model, tip_body));
  return std::make_unique<KdlKinematics>(model, root_body, tip_body, std::move(bodies));
}

}  // namespace robot_model
