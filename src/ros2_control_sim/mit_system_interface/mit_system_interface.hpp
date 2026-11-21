// MuJoCo simulation of a robot whose joints are driven like the real reBot Arm RS (RobStride motors,
// rebot_robstride_protocol's RobStrideProtocol): five command interfaces per joint,
//   position (q_des), velocity (dq_des), kp, kd, t_ff,
// and the state interfaces position, velocity and effort. The motor's MIT law
//   tau = kp (q_des - q) + kd (dq_des - dq) + t_ff
// is evaluated in write(), once per controller_manager cycle, and applied through the effort
// command of the MuJoCo torque actuator; the MJCF's ctrlrange limits it, like the motor's torque
// limit. Joints without a kp command interface are left to mujoco_ros2_control (e.g. the gripper).
//
// mujoco_ros2_control only knows position, velocity and effort commands. This subclass hides the MIT
// joints' interfaces from it (so it sees an effort-only torque joint), exports the MIT ones itself, and
// feeds its effort handle.
//
// The whole MuJoCo state (every joint's, e.g. a free object's too):
//   /mujoco_state      mujoco_state_msgs/MujocoState, published on every read() (qpos, qvel, sim time)
//   /set_mujoco_state  mujoco_state_msgs/SetMujocoState, sets qpos and qvel (the sim time is kept)
// (the base class's node; third_party/mujoco_ros2_control/subclass_access.patch gives access to it).

#pragma once

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "hardware_interface/handle.hpp"
#include "hardware_interface/hardware_info.hpp"
#include "hardware_interface/types/hardware_component_interface_params.hpp"
#include "mujoco_ros2_control/mujoco_system_interface.hpp"
#include "mujoco_state_msgs/msg/mujoco_state.hpp"
#include "mujoco_state_msgs/srv/set_mujoco_state.hpp"
#include "realtime_tools/realtime_publisher.hpp"

namespace mit_system_interface
{

class MitMujocoSystemInterface : public mujoco_ros2_control::MujocoSystemInterface
{
public:
  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;
  hardware_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous) override;
  hardware_interface::return_type perform_command_mode_switch(
    const std::vector<std::string> & start_interfaces,
    const std::vector<std::string> & stop_interfaces) override;
  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  ~MitMujocoSystemInterface() override;

private:
  struct MitJoint
  {
    std::string name;
    // The MIT commands, exported as the joint's command interfaces. Defaults are the interfaces'
    // initial_value params (0 if none); the position is the measured one when the system activates.
    double position_command{std::numeric_limits<double>::quiet_NaN()};
    double velocity_command{0.0};
    double kp{0.0};
    double kd{0.0};
    double t_ff{0.0};
    // mujoco_ros2_control's handles: the torque it applies, and the measured state.
    std::optional<hardware_interface::CommandInterface> effort;
    std::optional<hardware_interface::StateInterface> position, velocity;
  };

  MitJoint * find(const std::string & joint);
  void set_state(
    const mujoco_state_msgs::srv::SetMujocoState::Request & request,
    mujoco_state_msgs::srv::SetMujocoState::Response & response);

  std::vector<MitJoint> mit_joints_;

  // The whole state: published on every read(), set by the service.
  mjModel * model_{nullptr};  // a copy, for nq, nv and mj_forward
  std::shared_ptr<rclcpp::Publisher<mujoco_state_msgs::msg::MujocoState>> state_publisher_;
  std::unique_ptr<realtime_tools::RealtimePublisher<mujoco_state_msgs::msg::MujocoState>> state_realtime_publisher_;
  mujoco_state_msgs::msg::MujocoState state_msg_;
  rclcpp::Service<mujoco_state_msgs::srv::SetMujocoState>::SharedPtr set_state_service_;
};

}  // namespace mit_system_interface
