// Clutched end-effector teleoperation: the ROS-free math of //src/teleop/controller:teleop.
//
//   teleop::ClutchedTarget target(scale);
//   target.reset(end_effector_pose);                       // once, and on the reset button
//   const Eigen::Isometry3d& goal = target.update(engaged, controller_pose);
//
// The target is persistent state. While the clutch is engaged, the controller's motion since
// the moment of engaging is applied to the target as it was then: its position moves by the
// controller's displacement (times scale), and it turns by the controller's rotation, both
// about base-frame axes. Releasing leaves the target where it is; the controller can then be
// moved freely and engaged again anywhere ("ratcheting"), so the operator's arm never has to
// match the robot's. The robot chases the target; if it falls behind (speed limit, an
// unreachable target), the target doesn't jump back to it. reset() snaps the target to a pose
// (e.g. the end effector's) and re-anchors an engaged clutch there.
//
// Clutch is the stateless part: the target for one engagement, from a reference pose.
//
// All poses are in one frame (the robot base). Allocation-free, for the control loop.

#pragma once

#include <Eigen/Geometry>

#include <optional>
#include <vector>

namespace teleop
{

// A button from an analog value (e.g. a trigger): on above `on`, off below `off`.
struct Hysteresis
{
  double on = 0.6;
  double off = 0.4;
  bool state = false;

  bool update(double value);
};

class Clutch
{
public:
  explicit Clutch(double scale = 1.0, bool rotation = true) : scale_(scale), rotation_(rotation) {}

  // The tool target for this cycle, or nullopt while released. `tool` (the tool's current,
  // last commanded pose) is read only on the cycle that engages: the reference the
  // controller's motion is applied to. A missing `controller` (not tracked) releases.
  std::optional<Eigen::Isometry3d> update(bool engaged, const std::optional<Eigen::Isometry3d>& controller,
                                          const Eigen::Isometry3d& tool);

  bool engaged() const { return engaged_; }
  void release() { engaged_ = false; }

private:
  double scale_;
  bool rotation_;  // false: keep the tool's orientation from engaging
  bool engaged_ = false;
  Eigen::Vector3d controller_start_p_;
  Eigen::Quaterniond controller_start_q_;
  Eigen::Vector3d tool_start_p_;
  Eigen::Quaterniond tool_start_q_;
};

class ClutchedTarget
{
public:
  explicit ClutchedTarget(double scale = 1.0, bool rotation = true) : clutch_(scale, rotation) {}

  // Snaps the target to `pose`. If engaged, the clutch re-anchors: motion from the next
  // update on is applied to `pose`.
  void reset(const Eigen::Isometry3d& pose);

  // Applies the controller's motion while `engaged` (and `controller` is tracked); otherwise
  // leaves the target unchanged. Returns the target. Call reset() first.
  const Eigen::Isometry3d& update(bool engaged, const std::optional<Eigen::Isometry3d>& controller);

  bool initialized() const { return initialized_; }
  bool engaged() const { return clutch_.engaged(); }
  const Eigen::Isometry3d& target() const { return target_; }

private:
  Clutch clutch_;
  bool initialized_ = false;
  Eigen::Isometry3d target_ = Eigen::Isometry3d::Identity();
};

// `target`, or the pose on the way there from `current` at most `max_distance` (m) and
// `max_angle` (rad) away: a speed limit when applied once per cycle.
Eigen::Isometry3d limit_step(const Eigen::Isometry3d& current, const Eigen::Isometry3d& target,
                             double max_distance, double max_angle);

// Time for a one-point trajectory from `from` to `to` (joint positions) so that no joint
// exceeds its velocity limit: max over joints of |to - from| / max_velocity, and at least
// `min_time`. `limiting_joint` (optional) gets the joint that sets the time, or -1 if min_time
// does. Large joint changes are fine; they just take longer.
double point_time(const std::vector<double>& from, const std::vector<double>& to,
                  const std::vector<double>& max_velocity, double min_time, int* limiting_joint = nullptr);

// Gripper joint position for a 0..1 squeeze (0 = open).
double gripper_position(double squeeze, double open_position, double closed_position);

}  // namespace teleop
