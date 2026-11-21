#include "src/teleop/controller/clutch.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace
{

using Eigen::AngleAxisd;
using Eigen::Isometry3d;
using Eigen::Vector3d;

Isometry3d pose(Vector3d p, const AngleAxisd& r = AngleAxisd(0, Vector3d::UnitZ()))
{
  Isometry3d t = Isometry3d::Identity();
  t.translation() = p;
  t.linear() = r.toRotationMatrix();
  return t;
}

double angle(const Isometry3d& a, const Isometry3d& b)
{
  return AngleAxisd(a.linear().transpose() * b.linear()).angle();
}

TEST(Hysteresis, SwitchesWithDeadband)
{
  teleop::Hysteresis h;
  EXPECT_FALSE(h.update(0.5));
  EXPECT_TRUE(h.update(0.7));
  EXPECT_TRUE(h.update(0.5));  // between thresholds: stays on
  EXPECT_FALSE(h.update(0.3));
  EXPECT_FALSE(h.update(0.5));
}

TEST(Clutch, ReleasedGivesNothing)
{
  teleop::Clutch clutch;
  EXPECT_FALSE(clutch.update(false, pose({1, 2, 3}), pose({0.3, 0, 0.2})));
  EXPECT_FALSE(clutch.engaged());
  EXPECT_FALSE(clutch.update(true, std::nullopt, pose({0.3, 0, 0.2})));  // untracked
}

TEST(Clutch, TracksPositionDeltaFromEngage)
{
  teleop::Clutch clutch;
  const Isometry3d tool = pose({0.3, 0.0, 0.2});
  // Engaging anywhere doesn't move the tool.
  auto t = clutch.update(true, pose({1.0, 2.0, 3.0}), tool);
  ASSERT_TRUE(t);
  EXPECT_TRUE(t->isApprox(tool));
  // The controller moves 5 cm in x and -2 cm in z; so does the tool. The tool argument
  // is ignored after engaging.
  t = clutch.update(true, pose({1.05, 2.0, 2.98}), pose({9, 9, 9}));
  ASSERT_TRUE(t);
  EXPECT_TRUE(t->translation().isApprox(Vector3d(0.35, 0.0, 0.18), 1e-12));
  EXPECT_LT(angle(*t, tool), 1e-12);
}

TEST(Clutch, RatchetsAfterRelease)
{
  teleop::Clutch clutch;
  Isometry3d tool = pose({0.3, 0.0, 0.2});
  clutch.update(true, pose({0, 0, 0}), tool);
  tool = *clutch.update(true, pose({0.1, 0, 0}), tool);  // +10 cm
  EXPECT_FALSE(clutch.update(false, pose({0.1, 0, 0}), tool));
  // Hand back to the start while released, engage again, move +10 cm again.
  clutch.update(true, pose({0, 0, 0}), tool);
  const auto t = clutch.update(true, pose({0.1, 0, 0}), tool);
  EXPECT_TRUE(t->translation().isApprox(Vector3d(0.5, 0.0, 0.2), 1e-12));
}

TEST(Clutch, AppliesRotationAboutBaseAxes)
{
  teleop::Clutch clutch;
  // Tool pitched down 90 deg; controller held at an arbitrary orientation.
  const Isometry3d tool = pose({0.3, 0, 0.2}, AngleAxisd(M_PI / 2, Vector3d::UnitY()));
  const AngleAxisd held(0.7, Vector3d(1, 2, 3).normalized());
  clutch.update(true, pose({0, 0, 0}, held), tool);
  // Turning the controller 30 deg about base z turns the tool 30 deg about base z.
  const Eigen::Matrix3d yaw = AngleAxisd(M_PI / 6, Vector3d::UnitZ()).toRotationMatrix();
  Isometry3d c = pose({0, 0, 0});
  c.linear() = yaw * held.toRotationMatrix();
  const auto t = clutch.update(true, c, tool);
  EXPECT_TRUE(t->linear().isApprox(yaw * tool.linear(), 1e-9));
  EXPECT_TRUE(t->translation().isApprox(tool.translation(), 1e-12));
}

TEST(Clutch, ScaleAndPositionOnly)
{
  teleop::Clutch clutch(0.5, /*rotation=*/false);
  const Isometry3d tool = pose({0.3, 0, 0.2});
  clutch.update(true, pose({0, 0, 0}), tool);
  const auto t = clutch.update(true, pose({0.1, -0.2, 0}, AngleAxisd(1.0, Vector3d::UnitX())), tool);
  EXPECT_TRUE(t->translation().isApprox(Vector3d(0.35, -0.1, 0.2), 1e-12));
  EXPECT_LT(angle(*t, tool), 1e-12);
}

TEST(ClutchedTarget, IsStatefulAcrossEngagements)
{
  teleop::ClutchedTarget target;
  target.reset(pose({0.3, 0.0, 0.2}));
  // Moving the controller while released does nothing.
  EXPECT_TRUE(target.update(false, pose({0.5, 0.5, 0.5})).isApprox(pose({0.3, 0.0, 0.2})));
  // Engage anywhere; the target moves by the controller's displacement.
  target.update(true, pose({1.0, 0.0, 0.0}));
  EXPECT_TRUE(target.update(true, pose({1.1, 0.0, 0.0})).translation().isApprox(Vector3d(0.4, 0.0, 0.2), 1e-12));
  // Released: the target stays put, wherever the robot got to.
  EXPECT_TRUE(target.update(false, pose({0.0, 0.0, 0.0})).translation().isApprox(Vector3d(0.4, 0.0, 0.2), 1e-12));
  EXPECT_FALSE(target.engaged());
  // Engaging again continues from the target, not from where engaging happens.
  target.update(true, pose({5.0, 5.0, 5.0}));
  const Isometry3d t = target.update(true, pose({5.0, 5.1, 5.0}));
  EXPECT_TRUE(t.translation().isApprox(Vector3d(0.4, 0.1, 0.2), 1e-12));
}

TEST(ClutchedTarget, ResetSnapsAndReanchorsWhileEngaged)
{
  teleop::ClutchedTarget target;
  EXPECT_FALSE(target.initialized());
  target.reset(pose({0.3, 0.0, 0.2}));
  target.update(true, pose({0, 0, 0}));
  target.update(true, pose({0.2, 0, 0}));  // target at 0.5
  // Reset to the end effector (say it lags at 0.35) while still engaged.
  target.reset(pose({0.35, 0.0, 0.2}));
  EXPECT_TRUE(target.target().translation().isApprox(Vector3d(0.35, 0.0, 0.2), 1e-12));
  // Holding the controller still: no jump; moving it: motion from the reset pose.
  EXPECT_TRUE(target.update(true, pose({0.2, 0, 0})).translation().isApprox(Vector3d(0.35, 0.0, 0.2), 1e-12));
  EXPECT_TRUE(target.update(true, pose({0.2, 0, 0.05})).translation().isApprox(Vector3d(0.35, 0.0, 0.25), 1e-12));
}

TEST(ClutchedTarget, UntrackedControllerReleases)
{
  teleop::ClutchedTarget target;
  target.reset(pose({0.3, 0.0, 0.2}));
  target.update(true, pose({0, 0, 0}));
  target.update(true, std::nullopt);
  EXPECT_FALSE(target.engaged());
  // Tracking comes back with the trigger still held: re-engages there, no jump.
  EXPECT_TRUE(target.update(true, pose({1, 1, 1})).isApprox(pose({0.3, 0.0, 0.2})));
}

TEST(LimitStep, ClampsDistanceAndAngle)
{
  const Isometry3d a = pose({0, 0, 0});
  EXPECT_TRUE(teleop::limit_step(a, pose({0.001, 0, 0}), 0.01, 0.1).isApprox(pose({0.001, 0, 0})));
  const Isometry3d far = pose({0.1, 0, 0}, AngleAxisd(0.2, Vector3d::UnitZ()));
  const Isometry3d s = teleop::limit_step(a, far, 0.01, 1.0);
  EXPECT_NEAR(s.translation().norm(), 0.01, 1e-12);
  EXPECT_NEAR(angle(a, s), 0.02, 1e-9);  // same fraction of the way
  const Isometry3d r = teleop::limit_step(a, far, 1.0, 0.05);  // angle binds
  EXPECT_NEAR(angle(a, r), 0.05, 1e-9);
  EXPECT_NEAR(r.translation().x(), 0.025, 1e-9);
}

TEST(PointTime, SlowestJointSetsTheTime)
{
  const std::vector<double> vmax{1.0, 2.0, 0.5};
  int joint = 0;
  // Small changes: the minimum time.
  EXPECT_DOUBLE_EQ(teleop::point_time({0, 0, 0}, {0.001, -0.001, 0.001}, vmax, 0.02, &joint), 0.02);
  EXPECT_EQ(joint, -1);
  // A large change on joint 1 at 2 rad/s: 1.5 s; joint 2's 0.4 rad at 0.5 rad/s is 0.8 s.
  EXPECT_DOUBLE_EQ(teleop::point_time({0, 0, 0}, {0.1, -3.0, 0.4}, vmax, 0.02, &joint), 1.5);
  EXPECT_EQ(joint, 1);
  EXPECT_DOUBLE_EQ(teleop::point_time({0, 0, 0}, {0.1, -1.0, 0.75}, vmax, 0.02, &joint), 1.5);
  EXPECT_EQ(joint, 2);
}

TEST(Gripper, MapsSqueeze)
{
  EXPECT_DOUBLE_EQ(teleop::gripper_position(0.0, 0.05, 0.0), 0.05);
  EXPECT_DOUBLE_EQ(teleop::gripper_position(1.0, 0.05, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(teleop::gripper_position(0.5, 0.05, 0.0), 0.025);
  EXPECT_DOUBLE_EQ(teleop::gripper_position(2.0, 0.05, 0.0), 0.0);
}

}  // namespace
