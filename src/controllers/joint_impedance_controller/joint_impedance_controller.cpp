#include "joint_impedance_controller.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>

#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/logging.hpp"
#include "rclcpp_action/create_server.hpp"
#include "urdf/model.hpp"

namespace joint_impedance_controller
{

namespace
{
constexpr const char * kMitInterfaces[] = {"position", "velocity", "kp", "kd", "t_ff"};
}  // namespace

controller_interface::CallbackReturn JointImpedanceController::on_init()
{
  try
  {
    param_listener_ = std::make_shared<ParamListener>(get_node());
    params_ = param_listener_->get_params();
  }
  catch (const std::exception & e)
  {
    fprintf(stderr, "Exception thrown during init stage with message: %s \n", e.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : params_.joints)
  {
    if (is_mit(joint))
    {
      for (const char * name : kMitInterfaces) conf.names.push_back(joint + "/" + name);
    }
    else if (is_position_only(joint))
    {
      conf.names.push_back(joint + "/position");
      if (velocity_limit(joint) > 0.0) conf.names.push_back(joint + "/velocity");
    }
    else
    {
      conf.names.push_back(joint + "/effort");
    }
  }
  return conf;
}

controller_interface::InterfaceConfiguration
JointImpedanceController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration conf;
  conf.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : params_.joints)
  {
    conf.names.push_back(joint + "/position");
    conf.names.push_back(joint + "/velocity");
  }
  return conf;
}

controller_interface::CallbackReturn JointImpedanceController::on_configure(
  const rclcpp_lifecycle::State &)
{
  const auto logger = get_node()->get_logger();
  param_listener_->refresh_dynamic_parameters();
  params_ = param_listener_->get_params();
  dof_ = params_.joints.size();
  for (const auto * list : {&params_.mit_joints, &params_.position_joints})
  {
    for (const auto & joint : *list)
    {
      if (std::find(params_.joints.begin(), params_.joints.end(), joint) == params_.joints.end())
      {
        RCLCPP_ERROR(logger, "'%s' of mit_joints or position_joints is not in joints.", joint.c_str());
        return CallbackReturn::ERROR;
      }
    }
  }
  for (const auto & joint : params_.position_joints)
  {
    if (is_mit(joint))
    {
      RCLCPP_ERROR(logger, "'%s' is in both mit_joints and position_joints.", joint.c_str());
      return CallbackReturn::ERROR;
    }
  }

  const std::string & urdf = get_robot_description();
  if (urdf.empty())
  {
    RCLCPP_ERROR(logger, "The robot description is empty; the model terms need its inertias.");
    return CallbackReturn::ERROR;
  }
  try
  {
    dynamics_ = std::make_unique<TreeDynamics>(
      urdf, params_.joints,
      std::array<double, 3>{params_.gravity[0], params_.gravity[1], params_.gravity[2]});
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(logger, "%s", e.what());
    return CallbackReturn::ERROR;
  }

  try
  {
    trajectory_buffer::Options options;
    options.num_joints = dof_;
    options.capacity = static_cast<size_t>(params_.trajectory.capacity);
    options.weights = weights();
    options.latency = params_.trajectory.latency;
    if (params_.trajectory.position_limits)
    {
      urdf::Model model;
      if (!model.initString(urdf)) throw std::runtime_error("Could not parse the robot description for joint limits.");
      const double margin = params_.trajectory.position_limit_margin;
      for (const auto & name : params_.joints)
      {
        const auto joint = model.getJoint(name);
        double lower = -std::numeric_limits<double>::infinity(), upper = -lower;
        if (joint && joint->limits &&
            (joint->type == urdf::Joint::REVOLUTE || joint->type == urdf::Joint::PRISMATIC))
        {
          lower = joint->limits->lower + margin;
          upper = joint->limits->upper - margin;
          if (lower > upper) throw std::runtime_error("position_limit_margin leaves no range for '" + name + "'.");
        }
        options.lower.push_back(lower);
        options.upper.push_back(upper);
        RCLCPP_INFO(logger, "Trajectory limits of '%s': [%g, %g]", name.c_str(), lower, upper);
      }
    }
    buffer_ = std::make_unique<trajectory_buffer::TrajectoryBuffer>(options);
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(logger, "%s", e.what());
    return CallbackReturn::ERROR;
  }

  for (auto * point : {&state_, &state_desired_})
  {
    point->positions.assign(dof_, 0.0);
    point->velocities.assign(dof_, 0.0);
    point->accelerations.assign(dof_, 0.0);
  }
  for (auto * v : {&kp_, &kd_, &inertia_, &coriolis_, &gravity_, &effort_, &feedforward_, &target_, &zero_}) v->assign(dof_, 0.0);

  subscription_ = get_node()->create_subscription<TrajectoryMsg>(
    "~/joint_trajectory", rclcpp::SystemDefaultsQoS(),
    [this](const std::shared_ptr<const TrajectoryMsg> msg)
    {
      if (auto checked = validated(*msg))
      {
        abort_active_goal("replaced by a trajectory message");
        submit(checked);
      }
    });

  using namespace std::placeholders;
  action_server_ = rclcpp_action::create_server<FollowJointTrajectory>(
    get_node()->get_node_base_interface(), get_node()->get_node_clock_interface(),
    get_node()->get_node_logging_interface(), get_node()->get_node_waitables_interface(),
    std::string(get_node()->get_name()) + "/follow_joint_trajectory",
    [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const FollowJointTrajectory::Goal> goal)
    {
      return validated(goal->trajectory) ? rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE
                                         : rclcpp_action::GoalResponse::REJECT;
    },
    [this](const std::shared_ptr<GoalHandle> goal)
    {
      // Cancelling stops following the trajectory: hold the desired position it reached.
      std::lock_guard lock(goal_mutex_);
      if (goal == active_goal_)
      {
        active_goal_.reset();
        submit(stop_command());
      }
      return rclcpp_action::CancelResponse::ACCEPT;
    },
    [this](const std::shared_ptr<GoalHandle> goal)
    {
      auto checked = validated(goal->get_goal()->trajectory);
      if (!checked)
      {
        goal->abort(std::make_shared<FollowJointTrajectory::Result>());
        return;
      }
      abort_active_goal("replaced by a new goal");
      std::lock_guard lock(goal_mutex_);
      active_goal_ = goal;
      active_goal_id_ = submit(checked);
    });
  // Goals finish when the control loop reports their trajectory's end.
  goal_timer_ = get_node()->create_wall_timer(std::chrono::milliseconds(20), [this] { check_active_goal(); });

  state_pub_ = get_node()->create_publisher<StateMsg>("~/controller_state", rclcpp::SystemDefaultsQoS());
  state_publisher_ = std::make_unique<StatePublisher>(state_pub_);
  state_msg_.joint_names = params_.joints;
  for (auto * point : {&state_msg_.reference, &state_msg_.feedback, &state_msg_.error, &state_msg_.output})
  {
    point->positions.assign(dof_, 0.0);
    point->velocities.assign(dof_, 0.0);
    point->accelerations.assign(dof_, 0.0);
    point->effort.assign(dof_, 0.0);
  }
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointImpedanceController::on_activate(
  const rclcpp_lifecycle::State &)
{
  param_listener_->refresh_dynamic_parameters();
  params_ = param_listener_->get_params();
  if (!read_state())
  {
    RCLCPP_ERROR(get_node()->get_logger(), "Could not read the joint state.");
    return CallbackReturn::ERROR;
  }
  try
  {
    buffer_->set_weights(weights());
    buffer_->set_latency(params_.trajectory.latency);
  }
  catch (const std::exception & e)
  {
    RCLCPP_ERROR(get_node()->get_logger(), "%s", e.what());
    return CallbackReturn::ERROR;
  }
  // Hold where the joints are (from the first update) until a trajectory arrives; drop any
  // command from before.
  new_command_.writeFromNonRT(nullptr);
  installed_.reset();
  started_ = false;
  finished_id_ = 0;
  return CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn JointImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  abort_active_goal("the controller was deactivated");
  std::fill(effort_.begin(), effort_.end(), 0.0);
  write_zero();
  return CallbackReturn::SUCCESS;
}

uint64_t JointImpedanceController::submit(std::shared_ptr<Command> command)
{
  command->id = next_id_++;
  new_command_.writeFromNonRT(command);
  return command->id;
}

void JointImpedanceController::abort_active_goal(const std::string & reason)
{
  std::lock_guard lock(goal_mutex_);
  if (!active_goal_) return;
  auto result = std::make_shared<FollowJointTrajectory::Result>();
  result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
  result->error_string = reason;
  active_goal_->abort(result);
  active_goal_.reset();
}

void JointImpedanceController::check_active_goal()
{
  std::lock_guard lock(goal_mutex_);
  if (active_goal_ && finished_id_.load() == active_goal_id_)
  {
    auto result = std::make_shared<FollowJointTrajectory::Result>();
    result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
    active_goal_->succeed(result);
    active_goal_.reset();
  }
}

trajectory_buffer::Weights JointImpedanceController::weights() const
{
  return {params_.trajectory.position_weight, params_.trajectory.velocity_weight,
          params_.trajectory.acceleration_weight};
}

std::shared_ptr<JointImpedanceController::Command> JointImpedanceController::stop_command() const
{
  auto command = std::make_shared<Command>();
  command->stop = true;
  command->time_from_start.assign(1, params_.trajectory.stop_duration);
  command->times.assign(1, 0.0);
  command->positions.assign(dof_, 0.0);  // the control loop's desired positions when it gets it
  command->final_velocities.assign(dof_, 0.0);  // at rest
  return command;
}

std::shared_ptr<JointImpedanceController::Command> JointImpedanceController::validated(
  const TrajectoryMsg & msg) const
{
  const auto logger = get_node()->get_logger();
  if (msg.points.empty())
  {
    RCLCPP_ERROR(logger, "Ignoring a trajectory without points.");
    return nullptr;
  }
  // The controlled joints, each once, in any order; a partial message only if allowed.
  const size_t unset = std::numeric_limits<size_t>::max();
  std::vector<size_t> order(dof_, unset);  // controller joint -> index in the message
  for (size_t m = 0; m < msg.joint_names.size(); ++m)
  {
    const auto it = std::find(params_.joints.begin(), params_.joints.end(), msg.joint_names[m]);
    if (it == params_.joints.end())
    {
      RCLCPP_ERROR(logger, "Ignoring a trajectory with unknown joint '%s'.", msg.joint_names[m].c_str());
      return nullptr;
    }
    size_t & slot = order[static_cast<size_t>(it - params_.joints.begin())];
    if (slot != unset)
    {
      RCLCPP_ERROR(logger, "Ignoring a trajectory with joint '%s' twice.", msg.joint_names[m].c_str());
      return nullptr;
    }
    slot = m;
  }
  if (msg.joint_names.size() != dof_ && !params_.allow_partial_joints_goal)
  {
    RCLCPP_ERROR(
      logger, "Ignoring a trajectory of %zu joints; need all %zu (allow_partial_joints_goal is off).",
      msg.joint_names.size(), dof_);
    return nullptr;
  }
  const size_t given = msg.joint_names.size();

  auto out = std::make_shared<Command>();
  out->stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
  out->time_from_start.resize(msg.points.size());
  out->times.resize(msg.points.size());
  out->positions.assign(msg.points.size() * dof_, std::numeric_limits<double>::quiet_NaN());
  out->final_velocities.assign(dof_, std::numeric_limits<double>::quiet_NaN());
  rclcpp::Duration previous(0, 0);
  for (size_t p = 0; p < msg.points.size(); ++p)
  {
    const auto & in = msg.points[p];
    const auto sizes_ok = [&](const std::vector<double> & v) { return v.empty() || v.size() == given; };
    if (in.positions.size() != given || !sizes_ok(in.velocities) || !sizes_ok(in.accelerations))
    {
      RCLCPP_ERROR(logger, "Ignoring a trajectory: point %zu does not have a value per joint.", p);
      return nullptr;
    }
    const rclcpp::Duration time_from_start(in.time_from_start);
    if (p > 0 && time_from_start <= previous)
    {
      RCLCPP_ERROR(logger, "Ignoring a trajectory: time_from_start must increase (point %zu).", p);
      return nullptr;
    }
    previous = time_from_start;
    out->time_from_start[p] = time_from_start.seconds();
    // Left-out joints stay NaN: the buffer leaves their points alone.
    for (size_t i = 0; i < dof_; ++i)
    {
      if (order[i] == unset) continue;
      const double position = in.positions[order[i]];
      if (!std::isfinite(position))
      {
        RCLCPP_ERROR(logger, "Ignoring a trajectory with a non-finite position (point %zu).", p);
        return nullptr;
      }
      out->positions[p * dof_ + i] = position;
      if (p + 1 == msg.points.size() && !in.velocities.empty())
      {
        if (!std::isfinite(in.velocities[order[i]]))
        {
          RCLCPP_ERROR(logger, "Ignoring a trajectory with a non-finite final velocity.");
          return nullptr;
        }
        out->final_velocities[i] = in.velocities[order[i]];
      }
    }
  }
  return out;
}

bool JointImpedanceController::read_state()
{
  for (size_t i = 0; i < dof_; ++i)
  {
    const auto position = state_interfaces_[2 * i].get_optional();
    const auto velocity = state_interfaces_[2 * i + 1].get_optional();
    if (!position.has_value() || !velocity.has_value()) return false;
    state_.positions[i] = position.value();
    state_.velocities[i] = velocity.value();
  }
  return true;
}

bool JointImpedanceController::is_mit(const std::string & joint) const
{
  return std::find(params_.mit_joints.begin(), params_.mit_joints.end(), joint) !=
         params_.mit_joints.end();
}

bool JointImpedanceController::is_position_only(const std::string & joint) const
{
  return std::find(params_.position_joints.begin(), params_.position_joints.end(), joint) !=
         params_.position_joints.end();
}

double JointImpedanceController::velocity_limit(const std::string & joint) const
{
  const auto gains = params_.gains.joints_map.find(joint);
  return gains == params_.gains.joints_map.end() ? 0.0 : gains->second.velocity_limit;
}

// command_interfaces_ follows command_interface_configuration(): an effort per joint; for an MIT
// joint its position, velocity, kp, kd and t_ff; for a position-only joint its position and, with a
// velocity_limit, its velocity.
void JointImpedanceController::write_command()
{
  size_t next = 0;
  const auto & q_d = state_desired_.positions;
  const auto & qd_d = state_desired_.velocities;
  for (size_t i = 0; i < dof_; ++i)
  {
    if (is_mit(params_.joints[i]))
    {
      (void)command_interfaces_[next++].set_value(q_d[i]);
      (void)command_interfaces_[next++].set_value(qd_d[i]);
      (void)command_interfaces_[next++].set_value(kp_[i]);
      (void)command_interfaces_[next++].set_value(kd_[i]);
      (void)command_interfaces_[next++].set_value(feedforward_[i]);
    }
    else if (is_position_only(params_.joints[i]))
    {
      (void)command_interfaces_[next++].set_value(q_d[i]);
      if (velocity_limit(params_.joints[i]) > 0.0)
        (void)command_interfaces_[next++].set_value(velocity_limit(params_.joints[i]));
    }
    else
    {
      (void)command_interfaces_[next++].set_value(effort_[i]);
    }
  }
}

// No torque: an MIT joint with zero gains and no feedforward.
void JointImpedanceController::write_zero()
{
  size_t next = 0;
  for (size_t i = 0; i < dof_; ++i)
  {
    if (is_mit(params_.joints[i]))
    {
      // Keep the position target where the joint is, so the (zero) gains have nothing to add.
      (void)command_interfaces_[next++].set_value(state_.positions[i]);
      for (int k = 0; k < 4; ++k) (void)command_interfaces_[next++].set_value(0.0);
    }
    else if (is_position_only(params_.joints[i]))
    {
      // Hold where it is; the velocity limit (if any) stays.
      (void)command_interfaces_[next++].set_value(state_.positions[i]);
      if (velocity_limit(params_.joints[i]) > 0.0)
        (void)command_interfaces_[next++].set_value(velocity_limit(params_.joints[i]));
    }
    else
    {
      (void)command_interfaces_[next++].set_value(0.0);
    }
  }
}

controller_interface::return_type JointImpedanceController::update(
  const rclcpp::Time & time, const rclcpp::Duration & /*period*/)
{
  if (param_listener_->is_old(params_))
  {
    params_ = param_listener_->get_params();
    try
    {
      buffer_->set_weights(weights());
      buffer_->set_latency(params_.trajectory.latency);
    }
    catch (const std::exception & e)
    {
      RCLCPP_ERROR(get_node()->get_logger(), "%s; keeping the previous trajectory weights.", e.what());
    }
  }
  for (size_t i = 0; i < dof_; ++i)
  {
    const auto & gains = params_.gains.joints_map.at(params_.joints[i]);
    kp_[i] = gains.stiffness;
    kd_[i] = gains.damping;
  }

  if (!read_state())
  {
    RCLCPP_ERROR_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 1000, "Could not read the joint state.");
    return controller_interface::return_type::ERROR;
  }

  // Buffer time: seconds since the first update after activation.
  if (!started_)
  {
    epoch_ns_ = time.nanoseconds();
    buffer_->reset(0.0, state_.positions);
    started_ = true;
  }
  const double now = static_cast<double>(time.nanoseconds() - epoch_ns_) * 1e-9;

  // A new trajectory's points go into the buffer, replacing each of its joints' later ones.
  const auto command = *new_command_.readFromRT();
  if (command && command != installed_)
  {
    installed_ = command;
    if (command->stop)
    {
      buffer_->query_command(now, command->positions, {}, {});
      // Not delayed by the latency.
      command->times[0] = now - buffer_->latency() + command->time_from_start[0];
    }
    else
    {
      const double start =
        command->stamp_ns == 0 ? now : static_cast<double>(command->stamp_ns - epoch_ns_) * 1e-9;
      for (size_t k = 0; k < command->times.size(); ++k)
        command->times[k] = start + command->time_from_start[k];
    }
    const auto status =
      buffer_->set_commands(now, command->times, command->positions, command->final_velocities);
    if (status == trajectory_buffer::Status::kInvalid)
    {
      RCLCPP_ERROR(get_node()->get_logger(), "The trajectory buffer rejected a trajectory.");
    }
    else if (status == trajectory_buffer::Status::kLimitsViolated)
    {
      RCLCPP_WARN_THROTTLE(
        get_node()->get_logger(), *get_node()->get_clock(), 1000,
        "The trajectory could not be kept within the joint limits (e.g. moving out too fast to stop).");
    }
    else if (status == trajectory_buffer::Status::kTruncated)
    {
      RCLCPP_ERROR(
        get_node()->get_logger(), "A trajectory has more points than trajectory.capacity (%ld); it was cut.",
        params_.trajectory.capacity);
    }
    installed_end_ = std::max(now, command->times.back() + buffer_->latency());
  }

  buffer_->query_command(
    now, state_desired_.positions, state_desired_.velocities, state_desired_.accelerations);
  // The trajectory has ended once its last point's time has passed.
  if (installed_ && now >= installed_end_) finished_id_ = installed_->id;

  const auto & q_d = state_desired_.positions;
  const auto & qd_d = state_desired_.velocities;
  const auto & qdd_d = state_desired_.accelerations;
  // Model terms at the measured state; only the acceleration comes from the trajectory. An MIT joint's
  // motor adds the feedback to this t_ff itself.
  dynamics_->compute(state_.positions, state_.velocities, qdd_d, inertia_, coriolis_, gravity_);
  const auto & ff = params_.feedforward;
  for (size_t i = 0; i < dof_; ++i)
  {
    feedforward_[i] = 0.0;
    if (ff.inertia) feedforward_[i] += inertia_[i];
    if (ff.coriolis) feedforward_[i] += coriolis_[i];
    if (ff.gravity) feedforward_[i] += gravity_[i];
    effort_[i] = is_position_only(params_.joints[i]) ?
                   0.0 :
                   kp_[i] * (q_d[i] - state_.positions[i]) + kd_[i] * (qd_d[i] - state_.velocities[i]) +
                     feedforward_[i];
  }
  write_command();
  publish_state(time, now);
  return controller_interface::return_type::OK;
}

void JointImpedanceController::publish_state(const rclcpp::Time & time, double now)
{
  if (!state_publisher_ || params_.state_publish_rate <= 0.0) return;
  if (last_state_publish_.nanoseconds() != 0 && (time - last_state_publish_).seconds() < 1.0 / params_.state_publish_rate)
    return;
  last_state_publish_ = time;
  auto & msg = state_msg_;
  msg.header.stamp = time;
  // The reference is the commanded targets, before the fit (positions only; the rest stays zero).
  buffer_->query_target(now, target_);
  msg.reference.positions = target_;
  msg.feedback.positions = state_.positions;
  msg.feedback.velocities = state_.velocities;
  // The output is what the control law makes of it: the fitted q_d, qd_d, qdd_d and the effort.
  msg.output.positions = state_desired_.positions;
  msg.output.velocities = state_desired_.velocities;
  msg.output.accelerations = state_desired_.accelerations;
  for (size_t i = 0; i < dof_; ++i)
  {
    // The error is the control law's: the fitted curve less the measured state.
    msg.error.positions[i] = state_desired_.positions[i] - state_.positions[i];
    msg.error.velocities[i] = state_desired_.velocities[i] - state_.velocities[i];
    msg.output.effort[i] = effort_[i];
  }
  state_publisher_->try_publish(state_msg_);
}

}  // namespace joint_impedance_controller

PLUGINLIB_EXPORT_CLASS(
  joint_impedance_controller::JointImpedanceController, controller_interface::ControllerInterface)
