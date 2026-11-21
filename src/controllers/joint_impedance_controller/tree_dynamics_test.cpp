// TreeDynamics (KDL over the robot description) against MuJoCo's inverse dynamics of the MJCF the
// URDF was generated from: argv[1] is the URDF, argv[2] the MJCF (rlocationpaths).
//
// mj_inverse's qfrc_inverse is M qacc + qfrc_bias when passive forces and constraints are
// disabled, with the bias (Coriolis + gravity) from MuJoCo's own tree. The two models must agree
// on every part, so a wrong inertia, link frame or gravity direction in the URDF shows here.

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include <rules_cc/cc/runfiles/runfiles.h>

#include <array>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tree_dynamics.hpp"

namespace
{

using joint_impedance_controller::TreeDynamics;

const std::vector<std::string> kArmJoints = {"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};

class TreeDynamicsTest : public testing::Test
{
protected:
  void SetUp() override
  {
    std::vector<std::string> args = testing::internal::GetArgvs();
    ASSERT_GE(args.size(), 3u) << "pass the URDF's and the MJCF's rlocation";
    std::string error;
    std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
        rules_cc::cc::runfiles::Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
    ASSERT_NE(runfiles, nullptr) << error;

    std::ifstream urdf_file(runfiles->Rlocation(args[1]));
    ASSERT_TRUE(urdf_file.good());
    std::stringstream urdf;
    urdf << urdf_file.rdbuf();

    char load_error[1024] = {0};
    model_ = mj_loadXML(runfiles->Rlocation(args[2]).c_str(), nullptr, load_error, sizeof(load_error));
    ASSERT_NE(model_, nullptr) << load_error;
    data_ = mj_makeData(model_);
    model_->opt.disableflags |= mjDSBL_SPRING | mjDSBL_DAMPER | mjDSBL_CONSTRAINT | mjDSBL_ACTUATION;

    dynamics_ = std::make_unique<TreeDynamics>(
        urdf.str(), kArmJoints, std::array<double, 3>{model_->opt.gravity[0], model_->opt.gravity[1], model_->opt.gravity[2]});
    for (const auto& name : kArmJoints)
    {
      const int id = mj_name2id(model_, mjOBJ_JOINT, name.c_str());
      ASSERT_GE(id, 0) << name;
      qpos_.push_back(model_->jnt_qposadr[id]);
      dof_.push_back(model_->jnt_dofadr[id]);
      lower_.push_back(model_->jnt_range[2 * id]);
      upper_.push_back(model_->jnt_range[2 * id + 1]);
    }
  }

  void TearDown() override
  {
    mj_deleteData(data_);
    mj_deleteModel(model_);
  }

  // MuJoCo's net joint forces for the arm joints.
  std::vector<double> mujoco_forces(const std::vector<double>& q, const std::vector<double>& qd, const std::vector<double>& qdd)
  {
    mj_resetData(model_, data_);
    for (size_t i = 0; i < q.size(); ++i)
    {
      data_->qpos[qpos_[i]] = q[i];
      data_->qvel[dof_[i]] = qd[i];
      data_->qacc[dof_[i]] = qdd[i];
    }
    mj_inverse(model_, data_);
    std::vector<double> out;
    for (int dof : dof_) out.push_back(data_->qfrc_inverse[dof]);
    return out;
  }

  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
  std::unique_ptr<TreeDynamics> dynamics_;
  std::vector<int> qpos_, dof_;
  std::vector<double> lower_, upper_;
};

TEST_F(TreeDynamicsTest, MatchesMujocoInverseDynamics)
{
  const size_t n = kArmJoints.size();
  std::mt19937 rng(3);
  const std::vector<double> zero(n, 0.0);
  std::vector<double> inertia(n), coriolis(n), gravity(n);
  double largest = 0.0;
  for (int sample = 0; sample < 50; ++sample)
  {
    std::vector<double> q(n), qd(n), qdd(n);
    for (size_t i = 0; i < n; ++i)
    {
      q[i] = std::uniform_real_distribution<double>(lower_[i], upper_[i])(rng);
      qd[i] = std::uniform_real_distribution<double>(-2.0, 2.0)(rng);
      qdd[i] = std::uniform_real_distribution<double>(-5.0, 5.0)(rng);
    }
    dynamics_->compute(q, qd, qdd, inertia, coriolis, gravity);

    const auto expected_gravity = mujoco_forces(q, zero, zero);
    const auto expected_coriolis = mujoco_forces(q, qd, zero);
    const auto expected_inertia = mujoco_forces(q, zero, qdd);
    for (size_t i = 0; i < n; ++i)
    {
      EXPECT_NEAR(gravity[i], expected_gravity[i], 1e-6) << "gravity, joint " << i;
      EXPECT_NEAR(coriolis[i], expected_coriolis[i] - expected_gravity[i], 1e-6) << "coriolis, joint " << i;
      EXPECT_NEAR(inertia[i], expected_inertia[i] - expected_gravity[i], 1e-6) << "inertia, joint " << i;
      largest = std::max({largest, std::abs(gravity[i]), std::abs(coriolis[i]), std::abs(inertia[i])});
    }
  }
  EXPECT_GT(largest, 0.5) << "the terms should not all be ~0";
}

TEST_F(TreeDynamicsTest, UnknownJointThrows)
{
  EXPECT_THROW(TreeDynamics("<robot name='r'/>", {"nope"}, {0, 0, -9.81}), std::runtime_error);
}

}  // namespace
