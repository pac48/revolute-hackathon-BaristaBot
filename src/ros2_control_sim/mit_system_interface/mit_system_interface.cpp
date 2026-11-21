#include "mit_system_interface.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>

#include "pluginlib/class_list_macros.hpp"
#include "realtime_tools/realtime_publisher.hpp"

namespace mit_system_interface
{

namespace
{
constexpr const char * kMitInterfaces[] = {"position", "velocity", "kp", "kd", "t_ff"};

bool is_mit_joint(const hardware_interface::ComponentInfo & joint)
{
  for (const auto & interface : joint.command_interfaces)
    if (interface.name == "kp") return true;
  return false;
}

double initial_value(const hardware_interface::ComponentInfo & joint, const std::string & name)
{
  for (const auto & interface : joint.command_interfaces)
    if (interface.name == name && !interface.initial_value.empty())
      return std::stod(interface.initial_value);
  return name == "position" ? std::numeric_limits<double>::quiet_NaN() : 0.0;
}
}  // namespace

MitMujocoSystemInterface::MitJoint * MitMujocoSystemInterface::find(const std::string & joint)
{
  for (auto & j : mit_joints_)
    if (j.name == joint) return &j;
  return nullptr;
}

hardware_interface::CallbackReturn MitMujocoSystemInterface::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  // mujoco_ros2_control sees the MIT joints as effort-only torque joints.
  auto inner = params;
  mit_joints_.clear();
  for (auto & joint : inner.hardware_info.joints)
  {
    if (!is_mit_joint(joint)) continue;
    for (const auto & interface : joint.command_interfaces)
    {
      if (std::find(std::begin(kMitInterfaces), std::end(kMitInterfaces), interface.name) ==
          std::end(kMitInterfaces))
      {
        RCLCPP_ERROR(
          get_logger(), "Joint '%s': '%s' is not an MIT command interface (%s).", joint.name.c_str(),
          interface.name.c_str(), "position, velocity, kp, kd, t_ff");
        return hardware_interface::CallbackReturn::ERROR;
      }
    }
    MitJoint mit;
    mit.name = joint.name;
    mit.position_command = initial_value(joint, "position");
    mit.velocity_command = initial_value(joint, "velocity");
    mit.kp = initial_value(joint, "kp");
    mit.kd = initial_value(joint, "kd");
    mit.t_ff = initial_value(joint, "t_ff");
    mit_joints_.push_back(std::move(mit));
    hardware_interface::InterfaceInfo effort;
    effort.name = "effort";
    effort.data_type = "double";
    joint.command_interfaces = {effort};
  }

  const auto result = MujocoSystemInterface::on_init(inner);
  info_ = params.hardware_info;  // what the resource manager exports against
  if (result != hardware_interface::CallbackReturn::SUCCESS) return result;

  get_model(model_);
  state_msg_.qpos.assign(model_->nq, 0.0);
  state_msg_.qvel.assign(model_->nv, 0.0);
  const auto node = ros_node();
  state_publisher_ = node->create_publisher<mujoco_state_msgs::msg::MujocoState>("/mujoco_state", 10);
  state_realtime_publisher_ =
    std::make_unique<realtime_tools::RealtimePublisher<mujoco_state_msgs::msg::MujocoState>>(state_publisher_);
  set_state_service_ = node->create_service<mujoco_state_msgs::srv::SetMujocoState>(
    "/set_mujoco_state",
    [this](
      const std::shared_ptr<mujoco_state_msgs::srv::SetMujocoState::Request> request,
      std::shared_ptr<mujoco_state_msgs::srv::SetMujocoState::Response> response)
    { set_state(*request, *response); });
  RCLCPP_INFO(
    get_logger(), "Publishing the MuJoCo state (nq %d, nv %d) on /mujoco_state; /set_mujoco_state sets it.",
    model_->nq, model_->nv);
  return result;
}

MitMujocoSystemInterface::~MitMujocoSystemInterface()
{
  set_state_service_.reset();
  state_realtime_publisher_.reset();
  state_publisher_.reset();
  if (model_) mj_deleteModel(model_);
}

hardware_interface::return_type MitMujocoSystemInterface::read(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  const auto result = MujocoSystemInterface::read(time, period);
  if (state_realtime_publisher_)
  {
    // The snapshot the base read() just took: the same step as the joint states.
    const auto & state = latest_control_state();
    state_msg_.header.stamp = time;
    state_msg_.sim_time = state.time;
    std::copy(state.qpos.begin(), state.qpos.end(), state_msg_.qpos.begin());
    std::copy(state.qvel.begin(), state.qvel.end(), state_msg_.qvel.begin());
    state_realtime_publisher_->try_publish(state_msg_);
  }
  return result;
}

void MitMujocoSystemInterface::set_state(
  const mujoco_state_msgs::srv::SetMujocoState::Request & request,
  mujoco_state_msgs::srv::SetMujocoState::Response & response)
{
  const auto nq = static_cast<std::size_t>(model_->nq), nv = static_cast<std::size_t>(model_->nv);
  if (request.qpos.size() != nq || (!request.qvel.empty() && request.qvel.size() != nv))
  {
    response.success = false;
    response.message = "qpos needs " + std::to_string(nq) + " values (got " + std::to_string(request.qpos.size()) +
                       "), qvel " + std::to_string(nv) + " or none (got " + std::to_string(request.qvel.size()) + ")";
    return;
  }
  mjData * data = nullptr;
  get_data(data);  // the rest of the state (time, ctrl, warm start) as it is
  std::copy(request.qpos.begin(), request.qpos.end(), data->qpos);
  if (request.qvel.empty())
    std::fill(data->qvel, data->qvel + nv, 0.0);
  else
    std::copy(request.qvel.begin(), request.qvel.end(), data->qvel);
  mj_normalizeQuat(model_, data->qpos);
  mj_forward(model_, data);
  set_data(data);
  mj_deleteData(data);
  response.success = true;
  response.message = "set";
}

std::vector<hardware_interface::StateInterface> MitMujocoSystemInterface::export_state_interfaces()
{
  auto interfaces = MujocoSystemInterface::export_state_interfaces();
  for (auto & interface : interfaces)
  {
    auto * joint = find(interface.get_prefix_name());
    if (!joint) continue;
    if (interface.get_interface_name() == "position") joint->position.emplace(interface);
    if (interface.get_interface_name() == "velocity") joint->velocity.emplace(interface);
  }
  return interfaces;
}

std::vector<hardware_interface::CommandInterface>
MitMujocoSystemInterface::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> out;
  for (auto & interface : MujocoSystemInterface::export_command_interfaces())
  {
    if (auto * joint = find(interface.get_prefix_name()))
      joint->effort.emplace(std::move(interface));  // fed by write(); not exported
    else
      out.push_back(std::move(interface));
  }
  for (auto & joint : mit_joints_)
  {
    // The pointer constructor is what mujoco_ros2_control itself uses.
    out.emplace_back(joint.name, "position", &joint.position_command);
    out.emplace_back(joint.name, "velocity", &joint.velocity_command);
    out.emplace_back(joint.name, "kp", &joint.kp);
    out.emplace_back(joint.name, "kd", &joint.kd);
    out.emplace_back(joint.name, "t_ff", &joint.t_ff);
  }
  return out;
}

hardware_interface::CallbackReturn MitMujocoSystemInterface::on_activate(
  const rclcpp_lifecycle::State & previous)
{
  // Hold where the joints are until a controller commands them.
  for (auto & joint : mit_joints_)
  {
    if (joint.position && std::isnan(joint.position_command))
    {
      if (const auto q = joint.position->get_optional<double>()) joint.position_command = q.value();
    }
  }
  return MujocoSystemInterface::on_activate(previous);
}

hardware_interface::return_type MitMujocoSystemInterface::perform_command_mode_switch(
  const std::vector<std::string> & start_interfaces, const std::vector<std::string> & stop_interfaces)
{
  // The MIT joints always run on their effort (torque) actuators; pass on the other joints'.
  const auto others = [this](const std::vector<std::string> & names)
  {
    std::vector<std::string> out;
    for (const auto & name : names)
    {
      if (!find(name.substr(0, name.rfind('/')))) out.push_back(name);
    }
    return out;
  };
  return MujocoSystemInterface::perform_command_mode_switch(
    others(start_interfaces), others(stop_interfaces));
}

hardware_interface::return_type MitMujocoSystemInterface::write(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  for (auto & joint : mit_joints_)
  {
    if (!joint.effort || !joint.position || !joint.velocity) continue;
    const auto q = joint.position->get_optional<double>();
    const auto dq = joint.velocity->get_optional<double>();
    if (!q || !dq) continue;
    // An unset (NaN) target is "here": no feedback from it.
    const double q_des = std::isfinite(joint.position_command) ? joint.position_command : q.value();
    const double dq_des = std::isfinite(joint.velocity_command) ? joint.velocity_command : dq.value();
    const double t_ff = std::isfinite(joint.t_ff) ? joint.t_ff : 0.0;
    (void)joint.effort->set_value(
      joint.kp * (q_des - q.value()) + joint.kd * (dq_des - dq.value()) + t_ff);
  }
  return MujocoSystemInterface::write(time, period);
}

}  // namespace mit_system_interface

PLUGINLIB_EXPORT_CLASS(
  mit_system_interface::MitMujocoSystemInterface, hardware_interface::SystemInterface)
