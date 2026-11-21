// rebot_arm::make_kinematics: FK against MuJoCo, IK round trips, timing.

#include "src/robot_model/rebot_arm/rebot_arm.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <string>

namespace
{

using robot_model::Kinematics;

// gripper_end in base_link's frame, from MuJoCo's own kinematics.
Eigen::Isometry3d mujoco_pose(const mjModel* m, const mjData* d, const char* body = "gripper_end")
{
  auto pose = [&](const char* name) {
    const int id = mj_name2id(m, mjOBJ_BODY, name);
    Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
    t.translation() = Eigen::Vector3d(d->xpos + 3 * id);
    t.linear() = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(d->xmat + 9 * id);
    return t;
  };
  return pose("base_link").inverse() * pose(body);
}

double angle(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b)
{
  return Eigen::AngleAxisd(a.linear().transpose() * b.linear()).angle();
}

class RebotArmKinematicsTest : public ::testing::TestWithParam<std::string>
{
protected:
  void SetUp() override
  {
    model_ = rebot_arm::load_mujoco_model();
    kinematics_ = GetParam() == "kdl" ? rebot_arm::make_kdl_kinematics(model_.get())
                                      : rebot_arm::make_kinematics(model_.get());
  }

  // Random positions within the limits for the chain joints.
  robot_model::MjDataPtr random_state(std::mt19937& rng)
  {
    auto d = robot_model::make_data(model_.get());
    for (const auto& name : kinematics_->joint_names())
    {
      const int id = mj_name2id(model_.get(), mjOBJ_JOINT, name.c_str());
      const double lo = model_->jnt_range[2 * id], hi = model_->jnt_range[2 * id + 1];
      robot_model::set_joint_position(model_.get(), d.get(), name, std::uniform_real_distribution<>(lo, hi)(rng));
    }
    mj_kinematics(model_.get(), d.get());
    return d;
  }

  robot_model::MjModelPtr model_;
  std::unique_ptr<Kinematics> kinematics_;
};

TEST_P(RebotArmKinematicsTest, Chain)
{
  EXPECT_EQ(kinematics_->joint_names(),
            (std::vector<std::string>{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"}));
  EXPECT_EQ(kinematics_->root_link(), "base_link");
  EXPECT_EQ(kinematics_->tip_link(), "gripper_end");
}

TEST_P(RebotArmKinematicsTest, ForwardKinematicsMatchesMujoco)
{
  std::mt19937 rng(1);
  double worst_position = 0, worst_angle = 0;
  for (int i = 0; i < 1000; ++i)
  {
    const auto d = random_state(rng);
    const Eigen::Isometry3d fk = kinematics_->forward_kinematics(*d);
    const Eigen::Isometry3d mj = mujoco_pose(model_.get(), d.get());
    worst_position = std::max(worst_position, (fk.translation() - mj.translation()).norm());
    worst_angle = std::max(worst_angle, angle(fk, mj));
  }
  std::printf("FK vs MuJoCo over 1000 states: max position error %.2e m, max angle error %.2e rad\n",
              worst_position, worst_angle);
  // The generated code is single precision; the MJCF's quaternions have 7 digits.
  EXPECT_LT(worst_position, 2e-5);
  EXPECT_LT(worst_angle, 2e-5);
}

TEST_P(RebotArmKinematicsTest, InverseKinematicsRoundTrip)
{
  std::mt19937 rng(2);
  int solved = 0, attempts = 0;
  double time_ms = 0;
  constexpr int kTargets = 200;
  for (int i = 0; i < kTargets; ++i)
  {
    const Eigen::Isometry3d target = kinematics_->forward_kinematics(*random_state(rng));
    auto state = random_state(rng);  // the seed
    robot_model::set_joint_position(model_.get(), state.get(), "joint7", 0.01);  // not in the chain
    const robot_model::IkResult result = kinematics_->inverse_kinematics(target, *state);
    attempts += result.attempts;
    time_ms += result.time_ms;
    if (!result.solved)
    {
      continue;
    }
    ++solved;
    // Checked with MuJoCo, not the solver's own FK; within limits; other joints untouched.
    const Eigen::Isometry3d reached = mujoco_pose(model_.get(), state.get());
    EXPECT_LT((reached.translation() - target.translation()).norm(), 1.1e-3);
    EXPECT_LT(angle(reached, target), 1.1e-2);
    for (const auto& name : kinematics_->joint_names())
    {
      const int id = mj_name2id(model_.get(), mjOBJ_JOINT, name.c_str());
      const double q = robot_model::joint_position(model_.get(), state.get(), name);
      EXPECT_GE(q, model_->jnt_range[2 * id] - 1e-9) << name;
      EXPECT_LE(q, model_->jnt_range[2 * id + 1] + 1e-9) << name;
    }
    EXPECT_DOUBLE_EQ(robot_model::joint_position(model_.get(), state.get(), "joint7"), 0.01);
  }
  std::printf("IK: %d/%d solved, %.1f attempts and %.3f ms per target on average\n", solved, kTargets,
              static_cast<double>(attempts) / kTargets, time_ms / kTargets);
  EXPECT_GE(solved, kTargets * 98 / 100);
}

TEST_P(RebotArmKinematicsTest, UnreachableTargetLeavesStateUnchanged)
{
  std::mt19937 rng(4);
  auto state = random_state(rng);
  auto before = robot_model::make_data(model_.get());
  mj_copyData(before.get(), model_.get(), state.get());
  Eigen::Isometry3d far = Eigen::Isometry3d::Identity();
  far.translation() = Eigen::Vector3d(3, 0, 0);  // out of reach
  const robot_model::IkResult result = kinematics_->inverse_kinematics(far, *state, {.max_attempts = 3});
  EXPECT_FALSE(result.solved);
  EXPECT_EQ(result.attempts, 3);
  EXPECT_GT(result.position_error, 1.0);
  for (int i = 0; i < model_->nq; ++i)
  {
    EXPECT_EQ(state->qpos[i], before->qpos[i]);
  }
}

TEST_P(RebotArmKinematicsTest, ForwardKinematicsTiming)
{
  std::mt19937 rng(3);
  const auto d = random_state(rng);
  constexpr int kCalls = 100000;
  double sink = 0;
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kCalls; ++i)
  {
    d->qpos[0] += 1e-7;
    sink += kinematics_->forward_kinematics(*d).translation().x();
  }
  const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
  std::printf("FK: %.0f ns per call (%g)\n", ns / kCalls, sink);
}

}  // namespace

namespace
{

// A pose and seed teleop logged as unsolvable (every seed non-finite) while the CLI solves it.
// One long-lived instance must keep solving it, also after many solves from random seeds and
// seeds on the joint limits.
TEST_P(RebotArmKinematicsTest, LongLivedInstanceStaysFinite)
{
  Eigen::Isometry3d target = Eigen::Isometry3d::Identity();
  target.translation() = Eigen::Vector3d(0.00153, -0.20054, 0.29415);
  target.linear() = Eigen::Quaterniond(0.754147, 0.156887, 0.165756, -0.615771).normalized().toRotationMatrix();
  const std::vector<double> seed{2.67280, 0.65372, 1.13961, -1.57000, 1.05671, 0.99107};
  robot_model::IkSettings settings;
  settings.max_attempts = 1;
  settings.position_tolerance = 0.005;
  settings.orientation_tolerance = 0.05;
  auto d = robot_model::make_data(model_.get());
  auto solve_case = [&] {
    for (std::size_t i = 0; i < seed.size(); ++i)
      robot_model::set_joint_position(model_.get(), d.get(), kinematics_->joint_names()[i], seed[i]);
    mj_kinematics(model_.get(), d.get());
    return kinematics_->inverse_kinematics(target, *d, settings);
  };
  {
    Eigen::Isometry3d t2 = Eigen::Isometry3d::Identity();
    t2.translation() = Eigen::Vector3d(0.32451, -0.04750, 0.40045);
    t2.linear() = Eigen::Quaterniond(0.923798, 0.217714, 0.306598, 0.072079).normalized().toRotationMatrix();
    const std::vector<double> s2{0.55917, 1.42152, 2.04465, -1.39504, 0.71203, 0.00837};
    for (std::size_t i = 0; i < s2.size(); ++i)
      robot_model::set_joint_position(model_.get(), d.get(), kinematics_->joint_names()[i], s2[i]);
    mj_kinematics(model_.get(), d.get());
    const auto r2 = kinematics_->inverse_kinematics(t2, *d, settings);
    std::printf("case2: solved %d non_finite %d oob %d pos %g\n", r2.solved, r2.non_finite, r2.out_of_limits, r2.position_error);
  }
  const auto first = solve_case();
  std::printf("first: solved %d non_finite %d pos %g\n", first.solved, first.non_finite, first.position_error);
  std::mt19937 rng(3);
  int non_finite = 0, step_failed = 0;
  for (int k = 0; k < 20000; ++k)
  {
    auto s = random_state(rng);
    const Eigen::Isometry3d t = kinematics_->forward_kinematics(*random_state(rng));
    const auto r = kinematics_->inverse_kinematics(t, *s, settings);
    non_finite += r.non_finite;
    const auto c = solve_case();
    if (!c.solved && step_failed++ == 0)
      std::printf("case failed after %d random solves (non_finite %d)\n", k, c.non_finite);
  }
  std::printf("random non-finite %d, case failures %d\n", non_finite, step_failed);
  // Targets just past joint4's lower limit (-1.57): every solve runs into that bound.
  int past_nf = 0, past_solved = 0;
  for (int k = 0; k < 200; ++k)
  {
    std::vector<double> q = seed;
    q[3] = -1.57 - 0.001 * (k + 1);
    for (std::size_t i = 0; i < q.size(); ++i)
      robot_model::set_joint_position(model_.get(), d.get(), kinematics_->joint_names()[i], q[i]);
    mj_kinematics(model_.get(), d.get());
    const Eigen::Isometry3d t = kinematics_->forward_kinematics(*d);
    for (std::size_t i = 0; i < q.size(); ++i)
      robot_model::set_joint_position(model_.get(), d.get(), kinematics_->joint_names()[i], seed[i]);
    mj_kinematics(model_.get(), d.get());
    const auto r = kinematics_->inverse_kinematics(t, *d, settings);
    past_nf += r.non_finite;
    past_solved += r.solved;
  }
  std::printf("past the limit: %d non-finite, %d solved of 200\n", past_nf, past_solved);
  EXPECT_EQ(past_nf, 0);
  EXPECT_TRUE(first.solved);
  EXPECT_EQ(step_failed, 0);
}

}  // namespace

#include <cfenv>
#include <xmmintrin.h>

namespace
{
TEST_P(RebotArmKinematicsTest, FloatingPointModes)
{
  Eigen::Matrix4d m;
  m << 0.80389315498322134, 0.0013808679192650608, 0.59477213163943043, 0.32527832831158326, 0.2644532668650289,
      0.89488289120424924, -0.3595122816738498, -0.047204668432337955, -0.53274784374555295, 0.4462988956223321,
      0.71902505571833197, 0.40105004485720774, 0, 0, 0, 1;
  Eigen::Isometry3d t(m);
  const std::vector<double> seed{0.55918765657421865, 1.4241249474619933, 2.0402333831787347, -1.3949776699150187,
                                 0.71634058240728049, 0.0090075927669985532};
  robot_model::IkSettings settings;
  settings.max_attempts = 1;
  settings.position_tolerance = 0.005;
  settings.orientation_tolerance = 0.05;
  auto d = robot_model::make_data(model_.get());
  auto run = [&](const char* name) {
    for (std::size_t i = 0; i < seed.size(); ++i)
      robot_model::set_joint_position(model_.get(), d.get(), kinematics_->joint_names()[i], seed[i]);
    mj_kinematics(model_.get(), d.get());
    const auto r = kinematics_->inverse_kinematics(t, *d, settings);
    std::printf("%-12s solved %d non_finite %d\n", name, r.solved, r.non_finite);
  };
  const unsigned csr = _mm_getcsr();
  run("default");
  _mm_setcsr(csr | 0x8040);
  run("ftz+daz");
  _mm_setcsr(csr);
  for (auto [mode, name] : {std::pair{FE_UPWARD, "upward"}, {FE_DOWNWARD, "downward"}, {FE_TOWARDZERO, "towardzero"}})
  {
    std::fesetround(mode);
    run(name);
  }
  std::fesetround(FE_TONEAREST);
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(Backends, RebotArmKinematicsTest, ::testing::Values("fast", "kdl"),
                         [](const auto& info) { return info.param; });
