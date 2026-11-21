// Feedforward joint impedance controller driven by joint trajectories.
//
// Trajectories come as trajectory_msgs/JointTrajectory on ~/joint_trajectory or as
// control_msgs/FollowJointTrajectory goals on ~/follow_joint_trajectory (succeeds when the
// trajectory ends; no goal tolerances). Their points' positions (velocities and accelerations are
// ignored) go into a //src/controllers/trajectory_buffer TrajectoryBuffer, timestamped at
// header.stamp (or, if zero, when the control loop receives them) + time_from_start. Each joint's
// points replace its points from the message's first one on, so a new trajectory preempts the rest
// of the old one; a partial message (allow_partial_joints_goal) leaves the other joints' points
// alone. trajectory.latency delays every point, so streamed messages overlap. The buffer smooths the
// points into a C2 piecewise quintic (trajectory.*_weight) within the robot description's joint
// position limits (trajectory.position_limits: a QP with the limits as inequalities) that starts
// at the current desired state, so preemption and streaming are continuous, and reaches the last
// point with the message's velocity there (if it gives velocities; a free one otherwise, as for
// streamed positions-only points), then holds it. It gives a desired q_d, qd_d, qdd_d each cycle. The effort
// written to each joint is
//
//   tau = M(q) qdd_d + C(q, qd) qd + g(q)  +  Kp (q_d - q) + Kd (qd_d - qd)
//
// The model terms (each switchable, see feedforward.*) are evaluated at the measured state, from
// the robot description's masses and inertias, with the trajectory supplying the acceleration.
// Kp (gains.<joint>.stiffness) and Kd (gains.<joint>.damping) are the feedback on the position and
// velocity targets. ~/controller_state (control_msgs/JointTrajectoryControllerState) reports the
// reference (only positions: the buffer's targets, linearly interpolated and before the fit; see
// TrajectoryBuffer::query_target), the measured state, the error (q_d - q, qd_d - qd: the fitted
// curve less the measured state) and the output (q_d, qd_d, qdd_d from the fitted curve, and the
// effort tau above). reference - output.positions is the fit's deviation from the targets. With no trajectory,
// the joints hold their position from activation.
//
// Joints in mit_joints are driven like the reBot Arm RS's motors, which run the feedback themselves
// (tau = kp (position - q) + kd (velocity - dq) + t_ff): the controller writes q_d, qd_d, Kp, Kd and
// the model terms as t_ff to their position, velocity, kp, kd and t_ff command interfaces, and an
// effort to the others. Joints in position_joints (the real gripper) only take a position (and, with
// gains.<joint>.velocity_limit, a speed limit): their motor does the rest, and no model terms or gains apply.
#pragma once

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "control_msgs/msg/joint_trajectory_controller_state.hpp"
#include "controller_interface/controller_interface.hpp"
#include "joint_impedance_controller/joint_impedance_controller_parameters.hpp"
#include "rclcpp/subscription.hpp"
#include "rclcpp/timer.hpp"
#include "rclcpp_action/server.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "src/controllers/trajectory_buffer/trajectory_buffer.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "tree_dynamics.hpp"

namespace joint_impedance_controller
{

class JointImpedanceController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State &) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State &) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  using TrajectoryMsg = trajectory_msgs::msg::JointTrajectory;
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  using GoalHandle = rclcpp_action::ServerGoalHandle<FollowJointTrajectory>;
  using StateMsg = control_msgs::msg::JointTrajectoryControllerState;

  /// A validated trajectory's points, laid out for TrajectoryBuffer::set_commands (allocated off the
  /// control loop), numbered so that a goal can tell when its own trajectory has finished.
  struct Command
  {
    int64_t stamp_ns = 0;                 ///< header.stamp; 0: starts when the control loop gets it
    std::vector<double> time_from_start;  ///< per point (s)
    std::vector<double> positions;        ///< row-major, a row of dof per point; NaN: joint left out
    std::vector<double> final_velocities; ///< per joint: the last point's velocity; NaN: none given
    std::vector<double> times;            ///< the control loop's: the points' buffer times
    bool stop = false;                    ///< come to rest at the desired position (one point)
    uint64_t id = 0;
  };

  /// A message as a Command, in the controller's joint order; nullptr (and a log) if invalid.
  std::shared_ptr<Command> validated(const TrajectoryMsg & msg) const;
  /// Come to rest (in trajectory.stop_duration) where the joints are desired to be.
  std::shared_ptr<Command> stop_command() const;
  /// The trajectory.*_weight parameters.
  trajectory_buffer::Weights weights() const;
  /// Hand a command to the control loop; returns its number.
  uint64_t submit(std::shared_ptr<Command> command);
  /// Abort the action goal in progress, if any (its trajectory was replaced).
  void abort_active_goal(const std::string & reason);
  void check_active_goal();
  bool read_state();
  bool is_mit(const std::string & joint) const;
  bool is_position_only(const std::string & joint) const;
  double velocity_limit(const std::string & joint) const;
  void write_command();
  void write_zero();
  void publish_state(const rclcpp::Time & time, double now);

  std::shared_ptr<ParamListener> param_listener_;
  Params params_;
  size_t dof_ = 0;

  std::unique_ptr<TreeDynamics> dynamics_;
  rclcpp::Subscription<TrajectoryMsg>::SharedPtr subscription_;
  rclcpp_action::Server<FollowJointTrajectory>::SharedPtr action_server_;
  rclcpp::TimerBase::SharedPtr goal_timer_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<Command>> new_command_;
  std::atomic<uint64_t> next_id_{1};
  std::atomic<uint64_t> finished_id_{0};  // the control loop's: the last trajectory that reached its end

  std::mutex goal_mutex_;  // guards active_goal_ and active_goal_id_ (non-real-time threads)
  std::shared_ptr<GoalHandle> active_goal_;
  uint64_t active_goal_id_ = 0;

  // The control loop's.
  std::unique_ptr<trajectory_buffer::TrajectoryBuffer> buffer_;
  bool started_ = false;     // buffer_ reset at the first update after activation
  int64_t epoch_ns_ = 0;     // the buffer's time 0: that update's time
  std::shared_ptr<Command> installed_;
  double installed_end_ = 0; // buffer time when installed_'s last point is reached
  rclcpp::Time last_state_publish_{0, 0, RCL_ROS_TIME};
  trajectory_msgs::msg::JointTrajectoryPoint state_, state_desired_;

  using StatePublisher = realtime_tools::RealtimePublisher<StateMsg>;
  rclcpp::Publisher<StateMsg>::SharedPtr state_pub_;
  std::unique_ptr<StatePublisher> state_publisher_;
  StateMsg state_msg_;

  std::vector<double> kp_, kd_, inertia_, coriolis_, gravity_, effort_, feedforward_, target_, zero_;
};

}  // namespace joint_impedance_controller
