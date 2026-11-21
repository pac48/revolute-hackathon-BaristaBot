// Teleoperate the robot's tool with a VR controller (Quest app + //src/teleop/quest_bridge).
//
//   bazel run //src/teleop/controller:teleop -- [--hand right] [--scale 1] [--no-rotation]
//   bazel run //src/teleop/controller:teleop -- --model rebot_dual_arm_gemini_305   # both hands, both arms
//
// A robot with several arms (rebot_dual_arm_gemini_305) is one robot: one node, one joint state, one
// trajectory controller (every joint in one trajectory), one collision
// model. The left hand drives the left arm and the right hand the right, each with its own trigger, grip and
// reset button; each arm's commands and every plan avoid the other arm as well as themselves. Every robot has a
// world frame ("world"): the controller poses, the targets and the TF frames below are in it, and each arm's IK
// works in its own base frame (at its base's pose in the world).
//
// The tool target is persistent (clutch.hpp: ClutchedTarget). It starts at the end effector.
// Hold the TRIGGER to engage the clutch: the controller's motion since then (position delta x
// --scale, and its rotation) is applied to the target, and the arm chases the target. Release
// to leave the target (and the arm) where they are; move your hand back and engage again to
// keep going from the same target. Press A (X on the left hand) to snap the target to the end
// effector's pose. Squeeze the GRIP to close the gripper.
//
// While engaged, every control cycle (--rate, default 200 Hz) the step toward the target is
// speed-limited and solved with the model's IK in-process from --ik-seeds seeds: the robot's
// measured joint angles (/joint_states), then random configurations within the joint limits.
// Of the converged solutions, the one closest to the measured angles (largest single-joint
// difference) that passes the checks below is streamed to joint_trajectory_controller as a
// one-point trajectory. Its time_from_start comes from the joint velocity limits
// (--max-joint-velocity, --joint-velocity-limits): the slowest joint's distance from the
// controller's setpoint over its limit, and at least --lookahead. So a large joint change (a
// different IK branch, a wrist flip) is a slower move, not a rejection. The arm moves only while
// the trigger is held; if it lags the target, it keeps closing the gap on the next engagement.
//
// Safety: IK solves within the joint ranges (box constraints), without collision checks. The
// command then advances along the straight joint-space line from the last command (collision-free
// by construction) to the IK solution, in samples of --collision-resolution, and stops at the
// last sample before a self-collision: the whole segment if the line is clear, nothing (the
// robot holds) if it collides at once. Collisions within the first --collision-escape (5 deg) of
// joint motion along the line are tolerated when the line gets clear within it, so an arm at or
// in a contact can always back out of it, but not push further in. There is no motion planner: a
// blocked line is not routed around; move the hand so the straight line clears the obstacle.
//
// On Ctrl-C (or SIGTERM) the robot goes home before the node exits: a straight joint-space line (all arms
// together) from the controller's setpoint to all joints at zero, checked for collisions in MuJoCo like the
// commands (contacts already there at the start or at home are ignored), played at --home-velocity; if the line
// collides the robot stays where it is. So the arms are not left where they were (and don't drop when the driver is then
// stopped). Ctrl-C again, or --no-home, exits at once. The grippers are left as they are.
// --no-collision-check skips all of this (and goes home unchecked).
// If no seed converges, the robot holds. The clutch
// releases when the controller isn't tracked, its pose is older than --stale, or /quest/active
// is false.
//
// Threads: the ROS executor (subscriptions) on the main thread; the control loop on its
// own thread, optionally SCHED_FIFO (--rt-priority). They share only the latest inputs.
//
// Subscribes /quest/<hand>/pose (in the world frame), /quest/<hand>/joy, /quest/active and
// <controller>/controller_state; publishes <controller>/joint_trajectory,
// /teleop/target_pose (the persistent target, on change and at 5 Hz), /teleop/engaged, and
// TF frames teleop_target (the target) and teleop_command (the tool pose being commanded,
// where the arm's setpoint is headed) under world, at 50 Hz. With several arms the same under
// /teleop/<hand>/ and as teleop_target_<hand>, teleop_command_<hand>.

#include "src/robot_model/active_model/active_model.hpp"
#include "src/robot_model/rebot_arm/rebot_arm.hpp"
#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305.hpp"
#include "src/robot_model/rebot_dual_arm_gemini_305/rebot_dual_arm_gemini_305.hpp"
#include "src/robot_model/robot_model_utils/kinematics.hpp"
#include "src/robot_model/robot_model_utils/mujoco_model.hpp"
#include "src/teleop/controller/clutch.hpp"

#include <control_msgs/msg/joint_trajectory_controller_state.hpp>
#include <cxxopts.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <csignal>
#include <chrono>
#include <cfenv>
#include <cmath>
#include <xmmintrin.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;
using control_msgs::msg::JointTrajectoryControllerState;
using geometry_msgs::msg::PoseStamped;
using sensor_msgs::msg::JointState;
using sensor_msgs::msg::Joy;
using trajectory_msgs::msg::JointTrajectory;
using trajectory_msgs::msg::JointTrajectoryPoint;

constexpr int kTrigger = 0, kGrip = 1;  // Joy axes from quest_bridge: [trigger, grip, stick_x, stick_y]
constexpr int kReset = 0;               // Joy buttons: right [a, b, stick], left [x, y, stick, menu]

struct Options
{
  std::string model = robot_model::kActiveModel;  // active_model.txt
  std::string hand = "right";
  std::string controller = "joint_trajectory_controller";
  double rate = 200.0;
  double scale = 1.0;
  bool rotation = true;
  double max_speed = 1.0;          // m/s
  double max_angular_speed = 4.0;  // rad/s
  double max_joint_velocity = 3.0;     // rad/s (m/s for slide joints), every joint
  std::string joint_velocity_limits;  // "joint1=1.0,joint6=3.0": per-joint overrides
  std::vector<std::pair<std::string, double>> joint_velocity_overrides;  // parsed from the above
  double lookahead = 0.02;         // s
  double stale = 0.1;              // s
  bool collision_check = true;
  double busy_time = 0.1;  // s: trajectories longer than this play out before the next command
  double collision_resolution = 0.01;  // rad of the largest joint change between path samples
  double collision_escape = 5 * M_PI / 180;  // rad: collisions tolerated this close to the start if the path gets clear
  int collision_max_samples = 100;     // per cycle; long moves are sampled more coarsely
  int ik_seeds = 8;
  double ik_position_tolerance = 0.005;     // m
  double ik_orientation_tolerance = 0.05;  // rad
  int ik_iterations = 100;
  std::string ik_solver = "kdl";
  std::string gripper_joint = "joint7";
  double gripper_open = 0.05;
  double gripper_closed = 0.0;
  int rt_priority = 0;
  bool home_on_exit = true;
  bool swap_hands = false;  // several arms: each arm follows the other hand (the left controller drives the right arm)
  double home_velocity = 0.5;  // rad/s (m/s for slide joints): the speed of the trip home on exit
};

// The robot models with kinematics (add an entry per model). A robot with several arms is one robot: one
// model of all its arms; each arm has its own chain.
struct Model
{
  std::vector<std::string> arms;  // the arms' joint-name prefixes ({""}: one arm, the whole model)
  robot_model::MjModelPtr (*load)();
  // For one arm (`arm` is its prefix).
  std::function<std::unique_ptr<robot_model::Kinematics>(const mjModel*, const std::string& arm)> kinematics;  // fast_robot_kinematics
  std::function<std::unique_ptr<robot_model::Kinematics>(const mjModel*, const std::string& arm)> kdl_kinematics;  // KDL
};

template <typename Make>
auto single_arm(Make make)
{
  return [make](const mjModel* model, const std::string&) { return make(model); };
}

const std::unordered_map<std::string, Model> kModels = {
    {"rebot_arm",
     {{""},
      &rebot_arm::load_mujoco_model,
      single_arm(&rebot_arm::make_kinematics),
      single_arm(&rebot_arm::make_kdl_kinematics)}},
    // The same arm with the Gemini 305 on its wrist.
    {"rebot_arm_gemini_305",
     {{""},
      &rebot_arm_gemini_305::load_mujoco_model,
      single_arm(&rebot_arm_gemini_305::make_kinematics),
      single_arm(&rebot_arm_gemini_305::make_kdl_kinematics)}},
    // Two of them, 62 cm apart in y (left_ and right_), as one robot.
    {"rebot_dual_arm_gemini_305",
     {rebot_dual_arm_gemini_305::arms(),
      &rebot_dual_arm_gemini_305::load_mujoco_model,
      &rebot_dual_arm_gemini_305::make_kinematics,
      &rebot_dual_arm_gemini_305::make_kdl_kinematics}},
};

Eigen::Isometry3d to_eigen(const geometry_msgs::msg::Pose& p)
{
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() = Eigen::Vector3d(p.position.x, p.position.y, p.position.z);
  t.linear() = Eigen::Quaterniond(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)
                   .normalized()
                   .toRotationMatrix();
  return t;
}

void to_msg(const Eigen::Isometry3d& t, geometry_msgs::msg::Pose& p)
{
  const Eigen::Quaterniond q(t.linear());
  p.position.x = t.translation().x();
  p.position.y = t.translation().y();
  p.position.z = t.translation().z();
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
}

// Why IK solutions were rejected over a reporting window, with what explains each reason.
struct RejectStats
{
  enum Reason { kNoSolution, kCollision, kCount };
  static constexpr const char* kNames[kCount] = {"no IK solution", "self-collision"};

  std::array<int, kCount> count{};
  std::array<std::map<std::string, int>, kCount> detail;  // joint, joint + side, or body pair -> count
  // No solution: the best attempt's errors (infinite if no attempt stayed within the limits).
  int unsolved_outside_limits = 0;
  // Why seeds' attempts were discarded before scoring (IkResult): non-finite, or a joint past
  // its limit (joint -> count, and the largest excess).
  int seeds_non_finite = 0;
  std::map<std::string, int> seeds_past_limit;
  std::map<std::string, double> seeds_past_limit_max;
  // Accepted solutions: from seed 0 (measured angles) or a random seed; converged seeds.
  int solves = 0, won_measured = 0, won_random = 0;
  int truncated = 0;  // commands cut short before a collision
  int toward_goal = 0;  // steps out of reach, solved for the target itself instead
  int inexact = 0;    // commands outside the IK tolerances that still got closer to the target
  int kept = 0;       // of those, dropped: the last command ends closer to the target
  std::map<std::string, int> truncated_by;  // body pair -> count
  long converged = 0;
  double unsolved_pos_sum = 0, unsolved_ori_sum = 0;
  double unsolved_pos_min = std::numeric_limits<double>::infinity(), unsolved_ori_min = 0;

  int total() const
  {
    int n = 0;
    for (int c : count) n += c;
    return n;
  }

  void clear() { *this = RejectStats{}; }

  // e.g. "6.2 of 8 seeds converge per solve; accepted 240 from the measured angles, 3 from random seeds"
  std::string seed_summary(int seeds_per_solve) const
  {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out << std::setprecision(1) << (solves ? double(converged) / solves : 0.0) << " of " << seeds_per_solve
        << " seeds converge per solve; accepted " << won_measured << " from the measured angles, " << won_random
        << " from random seeds";
    if (truncated > 0)
    {
      out << "; " << truncated << " cut short before a collision [";
      bool first = true;
      for (const auto& [pair, n] : truncated_by)
      {
        out << (first ? "" : ", ") << pair << " " << n;
        first = false;
      }
      out << "]";
      if (kept > 0) out << ", " << kept << " of them dropped for the last command (closer to the target)";
    }
    if (inexact > 0) out << "; " << inexact << " outside the IK tolerances but closer to the target";
    if (toward_goal > 0) out << "; " << toward_goal << " steps out of reach, solved for the target instead";
    return out.str();
  }

  // e.g. "no IK solution 3 [best attempt: ...]; self-collision 5 [link3 - link5 5]"
  std::string summary() const
  {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    bool first_reason = true;
    for (int r = 0; r < kCount; ++r)
    {
      if (count[r] == 0) continue;
      out << (first_reason ? "" : "; ") << kNames[r] << " " << count[r] << " [";
      first_reason = false;
      bool first = true;
      for (const auto& [key, n] : detail[r])
      {
        out << (first ? "" : ", ") << key << " " << n;
        first = false;
      }
      if (r == kNoSolution)
      {
        const int finite = count[r] - unsolved_outside_limits;
        if (finite > 0)
        {
          out << std::setprecision(1) << "best attempt: min " << unsolved_pos_min * 1e3 << " mm / "
              << unsolved_ori_min * 180 / M_PI << " deg, mean " << unsolved_pos_sum / finite * 1e3 << " mm / "
              << unsolved_ori_sum / finite * 180 / M_PI << " deg";
        }
        if (unsolved_outside_limits > 0)
        {
          out << (finite > 0 ? "; " : "") << unsolved_outside_limits
              << " where every seed's solve ended outside the joint limits or non-finite (seeds: "
              << seeds_non_finite << " non-finite";
          for (const auto& [joint, n] : seeds_past_limit)
          {
            out << ", " << joint << " past its limit " << n << " (max " << std::setprecision(3)
                << seeds_past_limit_max.at(joint) << ")";
          }
          out << ")";
        }
      }
      out << "]";
    }
    return out.str();
  }
};

// One teleoperated arm of the robot (the chain from its base to its tool): its IK, its clutched target
// and its hand. A robot with several arms has one Arm per arm, all in one Teleop.
struct Candidate
{
  int seed;
  double distance;  // largest joint difference from the measured angles
  std::vector<double> q;
  double position_error, orientation_error;  // from the solve's target
  bool converged;                            // within the IK tolerances
};

struct Arm
{
  Arm(const Options& options) : target_(options.scale, options.rotation) {}

  std::string prefix;         // its joints' name prefix ("" for a robot with one arm)
  std::string name;           // "left" or "right" for a robot with several arms; "" for one
  std::string hand;           // the controller that drives it: "left" or "right"
  std::string gripper_joint;  // empty: no gripper
  std::size_t first = 0;      // its chain joints are the robot's joints [first, first + n)
  std::unique_ptr<robot_model::Kinematics> kinematics_;
  std::function<std::unique_ptr<robot_model::Kinematics>(const mjModel*)> make_kinematics_;
  std::vector<int> qpos_;                               // the chain joints' addresses in the model's qpos
  std::vector<double> q_, q_prev_;                      // its chain joints: the last command, the one before
  std::vector<double> seed_, seed_lower_, seed_upper_;  // IK seeds, and where random ones are drawn (the joint ranges)
  std::vector<double> path_q_, path_result_;            // advance_until_collision() scratch
  std::vector<double> measured, setpoint;               // this cycle's slices of the robot's
  std::vector<Candidate> candidates_;
  bool truncated_ = false;  // the last solve's command was cut short
  bool inexact_ = false;    // the last solve's command is outside the IK tolerances
  Eigen::Isometry3d goal_ = Eigen::Isometry3d::Identity();         // the clutch target, for solve()
  Eigen::Isometry3d tool_ = Eigen::Isometry3d::Identity();         // FK of q_: the commanded end effector
  Eigen::Isometry3d base_offset_ = Eigen::Isometry3d::Identity();  // its base in the world frame
  teleop::ClutchedTarget target_;  // persistent tool target, in the arm's base frame
  teleop::Hysteresis trigger_;
  bool reset_was_down_ = false;
  std::optional<double> gripper_sent_;
  Clock::time_point next_recheck_{};
  std::string blocked_pair_;  // the last solve stopped before this collision
  bool moved = false;         // this cycle's command moves the arm
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_sub;
  rclcpp::Subscription<Joy>::SharedPtr joy_sub;
  rclcpp::Publisher<PoseStamped>::SharedPtr target_pub;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr engaged_pub;
};

// All poses (the controllers', the targets', TF) are in the world frame: every robot has one. Each arm's
// IK works in its own base frame, at base_offset_ in the world.
constexpr const char* kWorld = "world";

// Teleoperates a robot with one or several arms: each arm follows its own hand, and the robot is one: one
// joint state, one trajectory controller, one collision model. So each arm's commands avoid the other arms
// as well as themselves.
class Teleop : public rclcpp::Node
{
public:
  Teleop(const Options& options, robot_model::MjModelPtr model, const Model& entry,
         std::function<std::unique_ptr<robot_model::Kinematics>(const mjModel*, const std::string&)> make_kinematics)
      : Node("teleop"),
        options_(options),
        model_(std::move(model)),
        state_(robot_model::make_data(model_.get()))
  {
    const bool several = entry.arms.size() > 1;
    arms_.reserve(entry.arms.size());
    for (const std::string& prefix : entry.arms)
    {
      Arm& a = arms_.emplace_back(options);
      a.prefix = prefix;
      a.name = several ? prefix.substr(0, prefix.size() - 1) : "";
      // With --swap-hands each arm follows the other hand: facing the robot, the left controller drives the
      // arm on your right.
      a.hand = !several ? options.hand : (options.swap_hands == (a.name == "left") ? "right" : "left");
      if (!options.gripper_joint.empty()) a.gripper_joint = prefix + options.gripper_joint;
      a.make_kinematics_ = [make_kinematics, prefix](const mjModel* m) { return make_kinematics(m, prefix); };
      a.kinematics_ = a.make_kinematics_(model_.get());
      const auto& joints = a.kinematics_->joint_names();
      a.first = joints_.size();
      for (const auto& j : joints)
      {
        joint_index_[j] = joints_.size();
        joints_.push_back(j);
        a.qpos_.push_back(robot_model::joint_qpos_address(model_.get(), j));
        all_qpos_.push_back(a.qpos_.back());
        // The model's joint range (what the IK solves within).
        const int id = mj_name2id(model_.get(), mjOBJ_JOINT, j.c_str());
        const bool limited = id >= 0 && model_->jnt_limited[id];
        a.seed_lower_.push_back(limited ? model_->jnt_range[2 * id] : -M_PI);
        a.seed_upper_.push_back(limited ? model_->jnt_range[2 * id + 1] : M_PI);
      }
      const std::size_t n = joints.size();
      for (auto* v : {&a.q_, &a.q_prev_, &a.seed_, &a.path_q_, &a.path_result_, &a.measured, &a.setpoint}) v->assign(n, 0.0);
      // Where the arm's base is in the world (its chain's root body, at the model's default pose).
      mj_kinematics(model_.get(), state_.get());
      const int base = mj_name2id(model_.get(), mjOBJ_BODY, a.kinematics_->root_link().c_str());
      if (base >= 0)
      {
        a.base_offset_.translation() = Eigen::Map<const Eigen::Vector3d>(state_->xpos + 3 * base);
        a.base_offset_.linear() = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(state_->xmat + 9 * base);
      }
    }
    for (const Arm& a : arms_)
    {
      if (several) RCLCPP_INFO(get_logger(), "the %s controller drives the %s arm", a.hand.c_str(), a.name.c_str());
    }
    vmax_.assign(joints_.size(), options.max_joint_velocity);
    for (const auto& [joint, v] : options.joint_velocity_overrides)
    {
      const auto it = std::find(joints_.begin(), joints_.end(), joint);
      if (it == joints_.end())
      {
        throw std::invalid_argument("--joint-velocity-limits: " + joint + " is not a chain joint");
      }
      vmax_[static_cast<std::size_t>(it - joints_.begin())] = v;
    }
    inputs_.hands.resize(arms_.size());
    inputs_.measured.assign(joints_.size(), 0.0);
    inputs_.setpoint.assign(joints_.size(), 0.0);
    candidate_reserve_ = static_cast<std::size_t>(std::max(1, options.ik_seeds));
    for (auto& a : arms_) a.candidates_.reserve(candidate_reserve_);
    settings_.max_attempts = 1;  // one local solve per seed; the seeds are ours (solve())
    settings_.iterations = options.ik_iterations;
    // Every attempt that stays finite comes back; solve() judges them (within the tolerances, or
    // at least closer to the target than the last command).
    settings_.position_tolerance = std::numeric_limits<double>::infinity();
    settings_.orientation_tolerance = std::numeric_limits<double>::infinity();

    for (std::size_t i = 0; i < arms_.size(); ++i)
    {
      Arm& a = arms_[i];
      a.pose_sub = create_subscription<PoseStamped>("/quest/" + a.hand + "/pose", rclcpp::SensorDataQoS().keep_last(1),
                                                    [this, i](const PoseStamped& msg) { on_pose(i, msg); });
      a.joy_sub = create_subscription<Joy>("/quest/" + a.hand + "/joy", rclcpp::SensorDataQoS().keep_last(1),
                                           [this, i](const Joy& msg) { on_joy(i, msg); });
      // One arm: /teleop/target_pose, /teleop/engaged, TF teleop_target and teleop_command. Several: the same
      // under /teleop/<hand>/ and with the hand as a suffix (teleop_target_left).
      const std::string topics = several ? "/teleop/" + a.name : "/teleop";
      a.target_pub = create_publisher<PoseStamped>(topics + "/target_pose", 10);
      a.engaged_pub = create_publisher<std_msgs::msg::Bool>(topics + "/engaged", rclcpp::QoS(1).transient_local());
      publish_engaged(a, false);
    }
    active_sub_ = create_subscription<std_msgs::msg::Bool>("/quest/active", 10, [this](const std_msgs::msg::Bool& msg) {
      std::lock_guard lock(mutex_);
      inputs_.active = msg.data;
    });
    joint_state_sub_ = create_subscription<JointState>("/joint_states", rclcpp::SensorDataQoS().keep_last(1),
                                                       [this](const JointState& msg) { on_joint_states(msg); });
    state_sub_ = create_subscription<JointTrajectoryControllerState>(
        "/" + options.controller + "/controller_state", 10,
        [this](const JointTrajectoryControllerState& msg) { on_controller_state(msg); });
    command_pub_ = create_publisher<JointTrajectory>("/" + options.controller + "/joint_trajectory", 10);
    tf_pub_ = create_publisher<tf2_msgs::msg::TFMessage>("/tf", rclcpp::QoS(100));
    tf_msg_.transforms.resize(2 * arms_.size());
    for (std::size_t i = 0; i < arms_.size(); ++i)
    {
      const std::string suffix = several ? "_" + arms_[i].name : "";
      tf_msg_.transforms[2 * i].header.frame_id = kWorld;
      tf_msg_.transforms[2 * i].child_frame_id = "teleop_target" + suffix;
      tf_msg_.transforms[2 * i + 1].header.frame_id = kWorld;
      tf_msg_.transforms[2 * i + 1].child_frame_id = "teleop_command" + suffix;
    }
    command_.points.resize(1);
    full_.assign(joints_.size(), 0.0);
    from_.assign(joints_.size(), 0.0);
  }

  void run(const std::atomic<bool>& running);
  // Moves the whole robot on a straight joint-space line from the controller's setpoint to all joints at zero
  // (if it is collision-free, see line_clear) at --home-velocity, waiting for it to end. `abort` ends the wait early. Returns whether it was sent.
  bool go_home(const std::function<bool()>& abort);

private:
  struct HandInputs
  {
    std::optional<Eigen::Isometry3d> pose;  // in the arm's base frame
    Clock::time_point pose_time{};
    std::array<double, 4> axes{};
    std::array<int, 4> buttons{};
  };
  struct Inputs
  {
    std::vector<HandInputs> hands;                       // one per arm
    bool active = false;
    std::unordered_map<std::string, double> references;  // controller setpoint by joint
    std::vector<double> measured;                        // the robot's joints (joints_) from /joint_states
    Clock::time_point measured_time{};
    std::vector<double> setpoint;  // likewise the controller's reference (where a new command starts)
    Clock::time_point setpoint_time{};
  };

  void on_pose(std::size_t arm, const PoseStamped& msg)
  {
    if (msg.header.frame_id != kWorld)
    {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                            "controller pose is in '%s' but poses are in the '%s' frame: run quest_bridge with "
                            "--world-frame %s",
                            msg.header.frame_id.c_str(), kWorld, kWorld);
      return;
    }
    // In the arm's base frame, where its IK works.
    const Eigen::Isometry3d pose = arms_[arm].base_offset_.inverse() * to_eigen(msg.pose);
    std::lock_guard lock(mutex_);
    inputs_.hands[arm].pose = pose;
    inputs_.hands[arm].pose_time = Clock::now();
  }

  void on_joy(std::size_t arm, const Joy& msg)
  {
    std::lock_guard lock(mutex_);
    auto& hand = inputs_.hands[arm];
    for (std::size_t i = 0; i < hand.axes.size() && i < msg.axes.size(); ++i)
    {
      hand.axes[i] = msg.axes[i];
    }
    for (std::size_t i = 0; i < hand.buttons.size() && i < msg.buttons.size(); ++i)
    {
      hand.buttons[i] = msg.buttons[i];
    }
  }

  void on_joint_states(const JointState& msg)
  {
    std::lock_guard lock(mutex_);
    std::size_t found = 0;
    for (std::size_t i = 0; i < msg.name.size() && i < msg.position.size(); ++i)
    {
      if (const auto it = joint_index_.find(msg.name[i]); it != joint_index_.end())
      {
        inputs_.measured[it->second] = msg.position[i];
        ++found;
      }
    }
    if (found == joint_index_.size())
    {
      inputs_.measured_time = Clock::now();
    }
  }

  void on_controller_state(const JointTrajectoryControllerState& msg)
  {
    std::lock_guard lock(mutex_);
    std::size_t found = 0;
    for (std::size_t i = 0; i < msg.joint_names.size() && i < msg.reference.positions.size(); ++i)
    {
      inputs_.references[msg.joint_names[i]] = msg.reference.positions[i];
      if (const auto it = joint_index_.find(msg.joint_names[i]); it != joint_index_.end())
      {
        inputs_.setpoint[it->second] = msg.reference.positions[i];
        ++found;
      }
    }
    if (found == joint_index_.size())
    {
      inputs_.setpoint_time = Clock::now();
    }
  }

  // The controller's setpoint for the arm's chain into state_ / a.q_ (and its FK into a.tool_): where
  // the controller holds the end effector. False if not known yet.
  bool load_setpoint(Arm& a, const Inputs& in)
  {
    const auto& joints = a.kinematics_->joint_names();
    for (std::size_t i = 0; i < joints.size(); ++i)
    {
      const auto it = in.references.find(joints[i]);
      if (it == in.references.end())
      {
        return false;
      }
      a.q_[i] = it->second;
    }
    write_q(a, a.q_);
    a.tool_ = a.kinematics_->forward_kinematics(*state_);
    return true;
  }

  void read_q(Arm& a, std::vector<double>& q) const
  {
    for (std::size_t i = 0; i < a.qpos_.size(); ++i)
    {
      q[i] = state_->qpos[a.qpos_[i]];
    }
  }

  // Sets the arm's chain joints in state_ (the other arms' stay where their last command put them).
  void write_q(Arm& a, const std::vector<double>& q)
  {
    for (std::size_t i = 0; i < a.qpos_.size(); ++i)
    {
      state_->qpos[a.qpos_[i]] = q[i];
    }
    mj_kinematics(model_.get(), state_.get());
  }

  // The penetrating contacts of state_ that moving arm `a` can change: those with one of its bodies. A
  // contact within another arm (it rests in one at home, say) is not this arm's to avoid; it would
  // otherwise block every motion of this arm for as long as it lasts. With one arm (empty prefix) that
  // is every contact.
  std::vector<std::string> arm_contacts(const Arm& a)
  {
    std::vector<std::string> pairs = robot_model::penetrating_contacts(model_.get(), state_.get());
    if (a.prefix.empty()) return pairs;
    pairs.erase(std::remove_if(pairs.begin(), pairs.end(),
                               [&](const std::string& pair) {
                                 const std::size_t dash = pair.find(" - ");
                                 return pair.rfind(a.prefix, 0) != 0 && pair.compare(dash + 3, a.prefix.size(), a.prefix) != 0;
                               }),
                pairs.end());
    return pairs;
  }

  // Why `q` can't be commanded, or kCount if it can. IK keeps it within the joint ranges, and
  // joint changes of any size are fine: the point's time comes from the velocity limits.
  // Advances along the straight joint-space line from `from` (the last command) toward `to`
  // and returns the last sample before a self-collision in `q` (= `to` if the line is clear),
  // and the fraction of the line covered (0: blocked); `pair` names the collision that stopped it.
  //
  // Escape window: collisions within the first --collision-escape rad (largest joint change) of
  // the line, extended past `to` for short steps, are tolerated if the line reaches a
  // collision-free sample within that window. So an arm in or at a contact can move out of it,
  // but not further in.
  double advance_until_collision(Arm& a, const std::vector<double>& from, const std::vector<double>& to,
                                 std::vector<double>& q, std::string& pair)
  {
    q = to;
    double span = 0;
    for (std::size_t i = 0; i < to.size(); ++i) span = std::max(span, std::abs(to[i] - from[i]));
    if (span <= 0) return 1.0;
    auto colliding = [&](double s) {
      for (std::size_t i = 0; i < to.size(); ++i) a.path_q_[i] = from[i] + s * (to[i] - from[i]);
      write_q(a, a.path_q_);
      const std::vector<std::string> c = arm_contacts(a);
      if (c.empty()) return false;
      pair = c.front();
      return true;
    };
    auto stop_at = [&](double done) {
      for (std::size_t i = 0; i < to.size(); ++i) q[i] = from[i] + done * (to[i] - from[i]);
      return done;
    };
    // The first collision-free sample of the window.
    const double h = options_.collision_resolution / span;
    const double window = options_.collision_escape / span;
    double clear = -1;
    for (int k = 1; k * h <= window + 1e-12 || k == 1; ++k)
    {
      if (!colliding(k * h))
      {
        clear = k * h;
        break;
      }
    }
    if (clear < 0) return stop_at(0.0);  // no way out in this direction
    if (clear >= 1.0)
    {
      return 1.0;  // the whole step is inside the window and the line gets clear past it
    }
    const int samples = std::clamp(static_cast<int>(std::ceil((1.0 - clear) * span / options_.collision_resolution)),
                                   1, options_.collision_max_samples);
    double last = clear;
    for (int k = 1; k <= samples; ++k)
    {
      const double s = clear + (1.0 - clear) * k / samples;
      if (colliding(s)) return stop_at(last);
      last = s;
    }
    return 1.0;
  }

  // Whether the straight joint-space line from `start` to `goal` (the robot's joints, joints_ order) is free of
  // collisions, sampled every --collision-resolution of the largest joint change. Contacts already there at
  // `start` or at `goal` (the arm rests in one at home: link2 - link5) are ignored. `pair` names the first other
  // collision. state_ is restored.
  bool line_clear(const std::vector<double>& start, const std::vector<double>& goal, std::string& pair)
  {
    std::vector<double> saved(all_qpos_.size());
    for (std::size_t i = 0; i < all_qpos_.size(); ++i) saved[i] = state_->qpos[all_qpos_[i]];
    auto contacts_at = [&](double s) {
      for (std::size_t i = 0; i < all_qpos_.size(); ++i) state_->qpos[all_qpos_[i]] = start[i] + s * (goal[i] - start[i]);
      return robot_model::penetrating_contacts(model_.get(), state_.get());
    };
    std::vector<std::string> already = contacts_at(0.0);
    for (const std::string& c : contacts_at(1.0)) already.push_back(c);
    double span = 0;
    for (std::size_t i = 0; i < start.size(); ++i) span = std::max(span, std::abs(goal[i] - start[i]));
    const int samples = std::max(1, static_cast<int>(std::ceil(span / options_.collision_resolution)));
    bool clear = true;
    for (int k = 1; k <= samples && clear; ++k)
    {
      for (const std::string& c : contacts_at(static_cast<double>(k) / samples))
      {
        if (std::find(already.begin(), already.end(), c) != already.end()) continue;
        pair = c;
        clear = false;
        break;
      }
    }
    for (std::size_t i = 0; i < all_qpos_.size(); ++i) state_->qpos[all_qpos_[i]] = saved[i];
    mj_kinematics(model_.get(), state_.get());
    return clear;
  }

  // IK for `target` from --ik-seeds seeds: `measured` (the robot's joint angles) first, then
  // random configurations within the joint limits. The converged solution closest to
  // `measured` (largest single-joint difference) is the goal; the command is how far toward it the
  // joint-space line from the last command gets before a self-collision (advance_until_collision).
  // On progress a.q_ becomes that (returns true; a.truncated_ says whether it was cut short).
  // Otherwise a.q_ stays the last command and the reason goes into `stats`.
  // IK for `target` from every seed; converged solutions go into a.candidates_, closest to
  // `measured` first. Failures are tallied in `stats`, the closest in `best_failed`.
  void ik_candidates(Arm& a, const Eigen::Isometry3d& target, const std::vector<double>& measured, RejectStats& stats,
                     robot_model::IkResult& best_failed)
  {
    const std::size_t n = a.q_.size();
    a.candidates_.clear();
    // The seeds are tried in order and the loop stops at the first that converged: seed 0 (the measured angles)
    // wins ties, so the random seeds, the slow ones (they start far from a solution), only matter when it and
    // the last command (seed 1; needed for continuity with a running plan, so tried then even if seed 0
    // converged) don't. With two arms this runs twice per cycle.
    const bool plan_running = !plan_times_.empty() && plan_times_.back() > Clock::now();
    for (int k = 0; k < options_.ik_seeds; ++k)
    {
      if (!a.candidates_.empty() && a.candidates_.back().converged && (k >= 2 || (k == 1 && !plan_running))) break;
      for (std::size_t i = 0; i < n; ++i)
      {
        // Seed 0: measured; seed 1: the last command (continuity with a running plan); then random.
        a.seed_[i] = k == 0   ? measured[i]
                   : k == 1 ? a.q_[i]
                            : std::uniform_real_distribution<double>(a.seed_lower_[i], a.seed_upper_[i])(rng_);
      }
      write_q(a, a.seed_);
      const robot_model::IkResult r = a.kinematics_->inverse_kinematics(target, *state_, settings_);
      if (!r.solved)
      {
        if (r.position_error < best_failed.position_error) best_failed = r;
        stats.seeds_non_finite += r.non_finite;
        if (r.out_of_limits > 0 && r.worst_joint >= 0)
        {
          const std::string& j = a.kinematics_->joint_names()[static_cast<std::size_t>(r.worst_joint)];
          stats.seeds_past_limit[j] += r.out_of_limits;
          double& mx = stats.seeds_past_limit_max[j];
          mx = std::max(mx, r.worst_excess);
        }
        continue;
      }
      Candidate c;
      c.seed = k;
      c.q.resize(n);
      read_q(a, c.q);
      const Eigen::Isometry3d reached = a.kinematics_->forward_kinematics(*state_);
      c.position_error = (reached.translation() - target.translation()).norm();
      c.orientation_error = Eigen::AngleAxisd(target.linear().transpose() * reached.linear()).angle();
      c.converged = c.position_error <= options_.ik_position_tolerance &&
                    c.orientation_error <= options_.ik_orientation_tolerance;
      if (!c.converged && c.position_error < best_failed.position_error)
      {
        best_failed.position_error = c.position_error;
        best_failed.orientation_error = c.orientation_error;
      }
      c.distance = 0;
      for (std::size_t i = 0; i < n; ++i) c.distance = std::max(c.distance, std::abs(c.q[i] - measured[i]));
      a.candidates_.push_back(std::move(c));
    }
    stats.converged += std::count_if(a.candidates_.begin(), a.candidates_.end(), [](const Candidate& c) { return c.converged; });
    // Seeds in the same IK branch converge to the same angles up to solver noise; seed 0 (the
    // measured angles) wins such ties, so a random seed only wins with a genuinely closer
    // solution.
    constexpr double kTie = 1e-3;  // rad
    // Solutions within the tolerances first, closest to the measured angles; then the rest,
    // closest to the target.
    const double pt = options_.ik_position_tolerance, ot = options_.ik_orientation_tolerance;
    std::sort(a.candidates_.begin(), a.candidates_.end(), [pt, ot](const Candidate& a, const Candidate& b) {
      if (a.converged != b.converged) return a.converged;
      if (!a.converged)
        return std::hypot(a.position_error / pt, a.orientation_error / ot) <
               std::hypot(b.position_error / pt, b.orientation_error / ot);
      return a.distance - (a.seed == 0 ? kTie : 0.0) < b.distance - (b.seed == 0 ? kTie : 0.0);
    });
  }

  bool solve(Arm& a, const Eigen::Isometry3d& target, const std::vector<double>& measured, RejectStats& stats)
  {
    a.q_prev_ = a.q_;
    ++stats.solves;
    const std::size_t n = a.q_.size();
    robot_model::IkResult best_failed;
    ik_candidates(a, target, measured, stats, best_failed);

    a.truncated_ = false;
    a.inexact_ = false;
    new_plan_ = false;
    a.blocked_pair_.clear();
    // Outside the tolerances, the best attempt still goes if it gets the tool closer to the
    // target (a.goal_) than the last command does. If the step can't be reached that way (a joint
    // limit or the workspace edge in its direction), solve for the target itself: the closest
    // the arm can get to it, timed by the velocity limits like any other move.
    auto improves = [&](const Candidate& c) {
      write_q(a, c.q);
      return pose_error(a.kinematics_->forward_kinematics(*state_), a.goal_) < pose_error(a.tool_, a.goal_);
    };
    if (!a.candidates_.empty() && !a.candidates_.front().converged && !improves(a.candidates_.front()))
    {
      RejectStats ignored;
      robot_model::IkResult toward_goal;
      ik_candidates(a, a.goal_, measured, ignored, toward_goal);
      if (!a.candidates_.empty() && improves(a.candidates_.front()))
      {
        ++stats.toward_goal;
        a.inexact_ = true;  // the command isn't the step: the tool goes where FK says
      }
      else
      {
        a.candidates_.clear();
      }
    }
    if (!a.candidates_.empty() && !a.candidates_.front().converged)
    {
      a.inexact_ = true;
      ++stats.inexact;
    }
    // (a.inexact_ also means a.tool_ comes from FK of the command rather than the step.)
    if (!a.candidates_.empty())
    {
      const Candidate& best = a.candidates_.front();
      if (!options_.collision_check)
      {
        a.q_ = best.q;
      }
      else
      {
        std::string pair;
        const double done = advance_until_collision(a, a.q_prev_, best.q, a.path_result_, pair);
        a.blocked_pair_ = done < 1 ? pair : std::string();
        if (done <= 0)
        {
          ++stats.count[RejectStats::kCollision];
          ++stats.detail[RejectStats::kCollision][pair];
          a.q_ = a.q_prev_;
          write_q(a, a.q_);
          return false;
        }
        a.q_ = a.path_result_;
        if (done < 1)
        {
          a.truncated_ = true;
          ++stats.truncated;
          ++stats.truncated_by[pair];
        }
      }
      write_q(a, a.q_);
      ++(best.seed == 0 ? stats.won_measured : stats.won_random);
      return true;
    }

    if (a.candidates_.empty())
    {
      ++stats.count[RejectStats::kNoSolution];
      write_q(a, a.q_prev_);
      if (std::isfinite(best_failed.position_error))
      {
        stats.unsolved_pos_sum += best_failed.position_error;
        stats.unsolved_ori_sum += best_failed.orientation_error;
        if (best_failed.position_error < stats.unsolved_pos_min)
        {
          stats.unsolved_pos_min = best_failed.position_error;
          stats.unsolved_ori_min = best_failed.orientation_error;
        }
      }
      else
      {
        ++stats.unsolved_outside_limits;
        // A reproducible case: replay with the kinematics CLI (ik_path from these joints).
        const Eigen::Quaterniond tq(target.linear());
        std::ostringstream joints;
        joints.setf(std::ios::fixed);
        joints.precision(5);
        for (std::size_t i = 0; i < n; ++i)
        {
          joints << (i ? ", " : "") << "\"" << a.kinematics_->joint_names()[i] << "\": " << measured[i];
        }
        recheck_on_fresh_kinematics(a, target, measured, stats);
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "no seed ended inside the joint limits. Case: target position [%.5f, %.5f, %.5f] "
                             "quaternion [%.6f, %.6f, %.6f, %.6f], seed 0 {%s}",
                             target.translation().x(), target.translation().y(), target.translation().z(), tq.x(),
                             tq.y(), tq.z(), tq.w(), joints.str().c_str());
      }
    }
    a.q_ = a.q_prev_;
    write_q(a, a.q_);
    return false;
  }

  // Every seed came back non-finite. Re-solve the same case (seed 0) on a newly made
  // Kinematics: if that solves it, the long-lived instance has gone bad; replace it.
  void recheck_on_fresh_kinematics(Arm& a, const Eigen::Isometry3d& target, const std::vector<double>& measured,
                                   RejectStats& stats)
  {
    if (!a.make_kinematics_ || stats.seeds_non_finite < options_.ik_seeds) return;
    const auto now = Clock::now();
    if (now < a.next_recheck_) return;
    a.next_recheck_ = now + std::chrono::seconds(1);
    const bool target_finite = target.matrix().allFinite();
    std::unique_ptr<robot_model::Kinematics> fresh = a.make_kinematics_(model_.get());
    write_q(a, measured);
    const robot_model::IkResult again = a.kinematics_->inverse_kinematics(target, *state_, settings_);
    write_q(a, measured);
    const robot_model::IkResult r = fresh->inverse_kinematics(target, *state_, settings_);
    RCLCPP_ERROR(get_logger(),
                 "all seeds non-finite: target finite %d; same instance again: solved %d non-finite %d; "
                 "fresh instance: solved %d non-finite %d (pos %.4f m)%s",
                 target_finite, again.solved, again.non_finite, r.solved, r.non_finite, r.position_error,
                 r.solved && !again.solved ? "; replacing the IK instance" : "");
    // The case as logged (rounded): solves in a fresh process. Does it here?
    Eigen::Isometry3d rounded = Eigen::Isometry3d::Identity();
    auto round_to = [](double v, double d) { return std::round(v * d) / d; };
    const Eigen::Quaterniond tq(target.linear());
    for (int i = 0; i < 3; ++i) rounded.translation()[i] = round_to(target.translation()[i], 1e5);
    rounded.linear() = Eigen::Quaterniond(round_to(tq.w(), 1e6), round_to(tq.x(), 1e6), round_to(tq.y(), 1e6),
                                          round_to(tq.z(), 1e6)).normalized().toRotationMatrix();
    std::vector<double> rounded_seed = measured;
    for (double& v : rounded_seed) v = round_to(v, 1e5);
    write_q(a, rounded_seed);
    const robot_model::IkResult rr = fresh->inverse_kinematics(rounded, *state_, settings_);
    std::ostringstream m;
    m.precision(17);
    m << "target matrix [" << target.matrix().format(Eigen::IOFormat(17, 0, ", ", "; ")) << "], seed [";
    for (std::size_t i = 0; i < measured.size(); ++i) m << (i ? ", " : "") << measured[i];
    m << "], orthonormality error " << (target.linear().transpose() * target.linear() - Eigen::Matrix3d::Identity()).norm();
    RCLCPP_ERROR(get_logger(), "rounded case on the fresh instance: solved %d non-finite %d; %s", rr.solved,
                 rr.non_finite, m.str().c_str());
    // Is the shared model intact (joint ranges as at startup), and does a new mjData help?
    std::ostringstream ranges;
    ranges.precision(9);
    for (std::size_t i = 0; i < a.seed_lower_.size(); ++i)
    {
      const int id = mj_name2id(model_.get(), mjOBJ_JOINT, a.kinematics_->joint_names()[i].c_str());
      const double lo = model_->jnt_range[2 * id], hi = model_->jnt_range[2 * id + 1];
      if (!(lo == a.seed_lower_[i] && hi == a.seed_upper_[i]))
        ranges << " " << a.kinematics_->joint_names()[i] << " [" << lo << ", " << hi << "] was [" << a.seed_lower_[i]
               << ", " << a.seed_upper_[i] << "]";
    }
    robot_model::MjDataPtr scratch = robot_model::make_data(model_.get());
    for (std::size_t i = 0; i < a.qpos_.size(); ++i) scratch->qpos[a.qpos_[i]] = measured[i];
    mj_kinematics(model_.get(), scratch.get());
    const robot_model::IkResult nd = fresh->inverse_kinematics(target, *scratch, settings_);
    // Per-thread floating-point state: MXCSR here, and the same solve on a new thread.
    robot_model::IkResult other;
    std::thread([&] {
      robot_model::MjDataPtr d2 = robot_model::make_data(model_.get());
      for (std::size_t i = 0; i < a.qpos_.size(); ++i) d2->qpos[a.qpos_[i]] = measured[i];
      mj_kinematics(model_.get(), d2.get());
      other = a.make_kinematics_(model_.get())->inverse_kinematics(target, *d2, settings_);
    }).join();
    RCLCPP_ERROR(get_logger(), "MXCSR 0x%x, rounding %d; on a new thread: solved %d non-finite %d", _mm_getcsr(),
                 std::fegetround(), other.solved, other.non_finite);
    int bad_qpos = 0;
    for (int i = 0; i < model_->nq; ++i) bad_qpos += !std::isfinite(state_->qpos[i]);
    RCLCPP_ERROR(get_logger(),
                 "new mjData: solved %d non-finite %d; non-finite qpos in state_: %d of %d; joint ranges changed:%s; "
                 "iterations %d, tolerances %g m %g rad",
                 nd.solved, nd.non_finite, bad_qpos, model_->nq, ranges.str().empty() ? " none" : ranges.str().c_str(),
                 settings_.iterations, settings_.position_tolerance, settings_.orientation_tolerance);
    if (r.solved && !again.solved) a.kinematics_ = std::move(fresh);
    write_q(a, a.q_prev_);
  }


  // Distance of `a` from `b` in units of the IK tolerances (position and orientation).
  double pose_error(const Eigen::Isometry3d& a, const Eigen::Isometry3d& b) const
  {
    const double p = (a.translation() - b.translation()).norm() / options_.ik_position_tolerance;
    const double r = Eigen::AngleAxisd(a.linear().transpose() * b.linear()).angle() / options_.ik_orientation_tolerance;
    return std::hypot(p, r);
  }

  // Sends `points` (the robot's joint positions) as one trajectory starting now from `from` (the controller's
  // setpoint), each segment timed by point_time(); the first takes at least --lookahead. Plan
  // waypoints (the first `plan_count` points) get their scheduled end times in plan_times_.
  // `grippers` (one per arm; set ones only) are added to every point. Returns the trajectory's duration.
  // `send_arm` (one per arm, or empty for all) says which arms' joints go in the message: an arm that is not being
  // moved is left out, and the controller holds it where it is desired to be. (Its joints would otherwise
  // carry the controller_state setpoint, which is up to 25 ms old: a moving arm that doesn't step this cycle
  // would be pulled back to where it was.) The timing still sees every joint.
  double send_trajectory(const std::vector<double>& from, const std::vector<std::vector<double>>& points,
                         std::size_t plan_count, const std::vector<std::optional<double>>& grippers,
                         Clock::time_point now, int* limiting_joint, const std::vector<bool>& send_arm = {})
  {
    command_.joint_names.clear();
    std::vector<std::size_t> sent;  // indices into joints_
    for (std::size_t ai = 0; ai < arms_.size(); ++ai)
    {
      if (!send_arm.empty() && !send_arm[ai]) continue;
      for (std::size_t i = 0; i < arms_[ai].q_.size(); ++i) sent.push_back(arms_[ai].first + i);
    }
    for (std::size_t j : sent) command_.joint_names.push_back(joints_[j]);
    std::vector<double> grip;
    for (std::size_t i = 0; i < arms_.size(); ++i)
    {
      if (!grippers[i]) continue;
      command_.joint_names.push_back(arms_[i].gripper_joint);
      grip.push_back(*grippers[i]);
      arms_[i].gripper_sent_ = *grippers[i];
    }
    command_.points.resize(points.size());
    double t = 0;
    const std::vector<double>* prev = &from;
    if (plan_count > 0) plan_times_.assign(plan_count, now);
    for (std::size_t k = 0; k < points.size(); ++k)
    {
      int joint = -1;
      t += teleop::point_time(*prev, points[k], vmax_, k == 0 ? options_.lookahead : 1e-3, &joint);
      if (joint >= 0 && limiting_joint) *limiting_joint = joint;
      auto& p = command_.points[k];
      p.positions.clear();
      for (std::size_t j : sent) p.positions.push_back(points[k][j]);
      p.positions.insert(p.positions.end(), grip.begin(), grip.end());
      p.time_from_start = rclcpp::Duration::from_seconds(t);
      if (k < plan_count)
      {
        plan_times_[k] = now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(t));
      }
      prev = &points[k];
    }
    command_pub_->publish(command_);  // header.stamp 0: starts now, replacing the last command
    // A long trajectory (a plan, or a big velocity-limited move) must play out: replacing it
    // every cycle restarts the controller's spline each time, and the arm barely moves.
    busy_until_ = t > options_.busy_time
                      ? now + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(t))
                      : Clock::time_point{};
    return t;
  }

  // The grippers alone (the set ones), over --lookahead.
  void send_gripper(const std::vector<std::optional<double>>& grippers)
  {
    command_.joint_names.clear();
    std::vector<double> positions;
    for (std::size_t i = 0; i < arms_.size(); ++i)
    {
      if (!grippers[i]) continue;
      command_.joint_names.push_back(arms_[i].gripper_joint);
      positions.push_back(*grippers[i]);
      arms_[i].gripper_sent_ = *grippers[i];
    }
    command_.points.resize(1);
    command_.points[0].positions = positions;
    command_.points[0].time_from_start = rclcpp::Duration::from_seconds(options_.lookahead);
    command_pub_->publish(command_);
  }

  // The arms' targets and the tool poses being commanded, in the world frame.
  void publish_tf()
  {
    const auto stamp = get_clock()->now();
    for (std::size_t a = 0; a < arms_.size(); ++a)
    {
      const Eigen::Isometry3d poses[2] = {arms_[a].base_offset_ * arms_[a].target_.target(),
                                          arms_[a].base_offset_ * arms_[a].tool_};
      for (int i = 0; i < 2; ++i)
      {
        auto& tf = tf_msg_.transforms[2 * a + static_cast<std::size_t>(i)];
        tf.header.stamp = stamp;
        const Eigen::Quaterniond q(poses[i].linear());
        tf.transform.translation.x = poses[i].translation().x();
        tf.transform.translation.y = poses[i].translation().y();
        tf.transform.translation.z = poses[i].translation().z();
        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf.transform.rotation.w = q.w();
      }
    }
    tf_pub_->publish(tf_msg_);
  }

  void publish_engaged(Arm& a, bool engaged)
  {
    std_msgs::msg::Bool msg;
    msg.data = engaged;
    a.engaged_pub->publish(msg);
  }

  Options options_;
  robot_model::MjModelPtr model_;
  robot_model::MjDataPtr state_;  // the robot at the last commanded solutions (each arm's), and the grippers' joints
  std::vector<Arm> arms_;
  std::vector<std::string> joints_;                          // every arm's chain joints, arm after arm
  std::unordered_map<std::string, std::size_t> joint_index_;  // joint -> index in joints_
  std::vector<int> all_qpos_;                                // their qpos addresses
  std::vector<double> full_, from_;                          // a command and its start, over joints_ (per cycle)
  std::size_t candidate_reserve_ = 1;
  Clock::time_point busy_until_{};  // a long trajectory is playing out until then
  bool new_plan_ = false;                          // the last solve planned (plan_points_)
  std::vector<std::vector<double>> plan_points_;   // the executing plan's waypoints (after its start), over joints_
  std::vector<Clock::time_point> plan_times_;      // when each is scheduled to be reached
  std::vector<std::vector<double>> traj_points_;   // send_trajectory() input scratch
  robot_model::IkSettings settings_;
  std::vector<double> vmax_;           // joint velocity limits (point_time()), over joints_
  std::mt19937 rng_{12345};
  JointTrajectory command_;

  std::mutex mutex_;
  Inputs inputs_;

  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr active_sub_;
  rclcpp::Subscription<JointTrajectoryControllerState>::SharedPtr state_sub_;
  rclcpp::Subscription<JointState>::SharedPtr joint_state_sub_;
  rclcpp::Publisher<JointTrajectory>::SharedPtr command_pub_;
  rclcpp::Publisher<tf2_msgs::msg::TFMessage>::SharedPtr tf_pub_;
  tf2_msgs::msg::TFMessage tf_msg_;
};


void Teleop::run(const std::atomic<bool>& running)
{
  const auto period = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / options_.rate));
  const double dt = 1.0 / options_.rate;
  const auto stale = std::chrono::duration<double>(options_.stale);
  auto next = Clock::now();
  Inputs in;
  in.hands.resize(arms_.size());
  std::vector<PoseStamped> target_msgs(arms_.size());
  for (auto& msg : target_msgs) msg.header.frame_id = kWorld;
  std::vector<std::optional<double>> grippers(arms_.size());
  std::vector<std::optional<double>> gripper_goal(arms_.size());

  // Stats over each reporting window (while engaged).
  int cycles = 0, solves = 0, overruns = 0;
  double solve_us_sum = 0, solve_us_max = 0;
  RejectStats rejected;
  int points = 0, velocity_limited = 0;  // arm commands; those timed by a joint velocity limit
  double longest_point = 0;
  std::string longest_joint;
  auto report = Clock::now() + std::chrono::seconds(5);
  auto next_target_publish = Clock::now();
  auto next_tf_publish = Clock::now();

  while (running.load(std::memory_order_relaxed))
  {
    next += period;
    std::this_thread::sleep_until(next);
    const auto now = Clock::now();
    if (now - next > period)
    {
      ++overruns;
      next = now;  // don't try to catch up with a burst
    }
    const bool any_released = std::any_of(arms_.begin(), arms_.end(), [](const Arm& a) { return !a.target_.engaged(); });
    {
      std::lock_guard lock(mutex_);
      in.hands = inputs_.hands;
      in.active = inputs_.active;
      in.measured = inputs_.measured;
      in.measured_time = inputs_.measured_time;
      in.setpoint = inputs_.setpoint;
      in.setpoint_time = inputs_.setpoint_time;
      if (any_released)
      {
        in.references = inputs_.references;  // only needed while released (init, engage, reset)
      }
    }
    const bool measured_fresh = now - in.measured_time < stale;
    const bool setpoint_fresh = now - in.setpoint_time < stale;
    const bool busy = now < busy_until_;
    for (Arm& a : arms_)
    {
      for (std::size_t i = 0; i < a.q_.size(); ++i)
      {
        a.measured[i] = in.measured[a.first + i];
        a.setpoint[i] = in.setpoint[a.first + i];
      }
      a.moved = false;
    }
    new_plan_ = false;
    bool target_changed_any = false;

    for (std::size_t ai = 0; ai < arms_.size(); ++ai)
    {
      Arm& a = arms_[ai];
      const HandInputs& hand = in.hands[ai];
      const bool was_engaged = a.target_.engaged();
      const bool fresh = in.active && hand.pose && now - hand.pose_time < stale;
      bool target_changed = false;

      // The target starts at the end effector, once the controller's setpoint is known.
      if (!a.target_.initialized() && load_setpoint(a, in))
      {
        a.target_.reset(a.tool_);
        target_changed = true;
      }
      // An arm that isn't being driven is where the controller holds it (something else may move it), and
      // the other arms' collision checks see it there.
      if (!was_engaged && a.target_.initialized() && !busy) load_setpoint(a, in);

      // Reset button (rising edge): the target snaps to the end effector, so the arm stays put.
      const bool reset_down = in.active && hand.buttons[kReset] != 0;
      if (reset_down && !a.reset_was_down_ && a.target_.initialized())
      {
        if (was_engaged || load_setpoint(a, in))  // engaged: a.tool_ is the latest commanded pose
        {
          a.target_.reset(a.tool_);
          target_changed = true;
          RCLCPP_INFO(get_logger(), "%starget reset to the end effector", a.prefix.empty() ? "" : (a.name + " arm: ").c_str());
        }
      }
      a.reset_was_down_ = reset_down;

      const bool pressed = a.trigger_.update(hand.axes[kTrigger]) && fresh && a.target_.initialized();
      bool engaged = pressed;
      if (pressed && !was_engaged)
      {
        // Engaging: continue from where the controller holds the arm now (something else may
        // have moved it while released). The target is unchanged. (A reset while engaged
        // re-anchors inside ClutchedTarget; a.tool_ is current then.)
        if (!load_setpoint(a, in))
        {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no %s/controller_state yet; can't engage",
                               options_.controller.c_str());
          engaged = false;
        }
        else
        {
          RCLCPP_INFO(get_logger(), "%sclutch engaged", a.prefix.empty() ? "" : (a.name + " arm: ").c_str());
          publish_engaged(a, true);
        }
      }
      else if (was_engaged && !pressed)
      {
        RCLCPP_INFO(get_logger(), "%sclutch released%s", a.prefix.empty() ? "" : (a.name + " arm: ").c_str(),
                    fresh ? "" : " (controller lost or headset inactive)");
        publish_engaged(a, false);
      }

      bool arm_moved = false;
      const Eigen::Isometry3d before = a.target_.target();
      const Eigen::Isometry3d& goal = a.target_.update(engaged, fresh ? hand.pose : std::nullopt);
      target_changed = target_changed || !goal.isApprox(before, 1e-12);
      if (engaged && !busy && !goal.isApprox(a.tool_, 1e-9))
      {
        // Chase the target, speed-limited.
        const Eigen::Isometry3d step =
            teleop::limit_step(a.tool_, goal, options_.max_speed * dt, options_.max_angular_speed * dt);
        const auto t0 = Clock::now();
        // Seed 0: the robot's measured angles, or the last command if /joint_states is stale.
        a.goal_ = goal;
        const bool solved = solve(a, step, measured_fresh ? a.measured : a.q_, rejected);
        const double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
        ++solves;
        solve_us_sum += us;
        solve_us_max = std::max(solve_us_max, us);
        if (options_.collision_check && !a.blocked_pair_.empty())
        {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "stopped before collision %s",
                               a.blocked_pair_.c_str());
        }
        if (solved)
        {
          // Cut short before a collision: the tool is where the command actually goes. If that
          // gets less close to the target than where the last command (or a running plan) ends,
          // keep the last command rather than replace it.
          const Eigen::Isometry3d reached =
              a.truncated_ || a.inexact_ ? a.kinematics_->forward_kinematics(*state_) : step;
          if ((a.truncated_ || a.inexact_) && pose_error(reached, goal) >= pose_error(a.tool_, goal))
          {
            ++rejected.kept;
            a.q_ = a.q_prev_;
            write_q(a, a.q_);
          }
          else
          {
            a.tool_ = reached;
            arm_moved = true;
            ++points;
          }
        }
      }
      a.moved = arm_moved;

      // The target, on every change and at 5 Hz otherwise (RViz shows it while released too).
      target_changed_any = target_changed_any || target_changed;
      if (a.target_.initialized() && (target_changed || now >= next_target_publish))
      {
        target_msgs[ai].header.stamp = get_clock()->now();
        to_msg(a.base_offset_ * goal, target_msgs[ai].pose);
        a.target_pub->publish(target_msgs[ai]);
      }

      grippers[ai].reset();
      gripper_goal[ai].reset();
      if (!a.gripper_joint.empty() && in.active)
      {
        const double g = teleop::gripper_position(hand.axes[kGrip], options_.gripper_open, options_.gripper_closed);
        gripper_goal[ai] = g;
        if (!a.gripper_sent_ || std::abs(g - *a.gripper_sent_) > 1e-4)
        {
          grippers[ai] = g;
        }
      }
    }
    if (now >= next_target_publish) next_target_publish = now + std::chrono::milliseconds(200);

    // teleop_target / teleop_command TF frames at 50 Hz (a.tool_ is the commanded end effector).
    if (std::all_of(arms_.begin(), arms_.end(), [](const Arm& a) { return a.target_.initialized(); }) && now >= next_tf_publish)
    {
      publish_tf();
      next_tf_publish = now + std::chrono::milliseconds(20);
    }

    const bool any_moved = std::any_of(arms_.begin(), arms_.end(), [](const Arm& a) { return a.moved; });
    if (any_moved)
    {
      // A trajectory that leaves a gripper out replaces its travel with a hold at where it is now, so a
      // gripper still closing while the clutch moves the arm would stall after a few ms. Keep sending its goal.
      for (std::size_t ai = 0; ai < arms_.size(); ++ai) grippers[ai] = gripper_goal[ai];
    }
    const bool any_gripper = std::any_of(grippers.begin(), grippers.end(), [](const auto& g) { return g.has_value(); });
    if (any_moved)
    {
      // One trajectory for the robot: the moved arms to their commands, the others where the controller
      // holds them (a partial goal is fine too, but this keeps every arm in one spline set), from where the
      // controller's setpoint is now (where the controller starts each trajectory), or the last command if controller_state
      // is stale. A plan still executing keeps its remaining waypoints ahead of the new goal.
      for (const Arm& b : arms_)
      {
        for (std::size_t i = 0; i < b.q_.size(); ++i)
        {
          const std::size_t j = b.first + i;
          // An engaged arm that didn't step keeps its last command: a trajectory that left it out would stop it
          // dead (the controller holds left-out joints at their desired position, velocity 0).
          full_[j] = b.moved || b.target_.engaged() || !setpoint_fresh ? b.q_[i] : in.setpoint[j];
          from_[j] = setpoint_fresh ? in.setpoint[j] : (b.moved ? b.q_prev_[i] : b.q_[i]);
        }
      }
      traj_points_.clear();
      std::size_t plan_count = 0;
      if (new_plan_)
      {
        traj_points_ = plan_points_;
        plan_count = plan_points_.size();
      }
      else
      {
        for (std::size_t k = 0; k < plan_points_.size() && k < plan_times_.size(); ++k)
        {
          if (plan_times_[k] > now) traj_points_.push_back(plan_points_[k]);
        }
        if (traj_points_.empty())
        {
          plan_points_.clear();
          plan_times_.clear();
        }
        else
        {
          plan_points_ = traj_points_;
          plan_count = traj_points_.size();
        }
        traj_points_.push_back(full_);
      }
      int joint = -1;
      std::vector<bool> send_arm;  // a plan moves the whole robot; otherwise the arms that are being driven
      if (plan_count == 0)
      {
        for (const Arm& b : arms_) send_arm.push_back(b.moved || b.target_.engaged());
      }
      const double time = send_trajectory(from_, traj_points_, plan_count, grippers, now, &joint, send_arm);
      if (joint >= 0)
      {
        ++velocity_limited;
        if (time > longest_point)
        {
          longest_point = time;
          longest_joint = joints_[static_cast<std::size_t>(joint)];
        }
      }
    }
    else if (any_gripper && !busy)  // a gripper-only command would cut the arm's trajectory short
    {
      send_gripper(grippers);
    }

    ++cycles;
    if (now >= report)
    {
      if (solves > 0 || overruns > 0)
      {
        RCLCPP_INFO(get_logger(), "%d cycles, %d IK solves (mean %.0f us, max %.0f us), %d rejected, %d overruns",
                    cycles, solves, solves ? solve_us_sum / solves : 0.0, solve_us_max, rejected.total(), overruns);
        if (rejected.solves > 0)
        {
          RCLCPP_INFO(get_logger(), "IK %s", rejected.seed_summary(options_.ik_seeds).c_str());
        }
        if (velocity_limited > 0)
        {
          RCLCPP_INFO(get_logger(), "velocity-limited %d of %d points (longest %.2f s, %s)", velocity_limited,
                      points, longest_point, longest_joint.c_str());
        }
        if (rejected.total() > 0)
        {
          RCLCPP_WARN(get_logger(), "rejected %d of %d: %s", rejected.total(), solves, rejected.summary().c_str());
        }
      }
      rejected.clear();
      points = velocity_limited = 0;
      longest_point = 0;
      cycles = solves = overruns = 0;
      solve_us_sum = solve_us_max = 0;
      report = now + std::chrono::seconds(5);
    }
  }
}

bool Teleop::go_home(const std::function<bool()>& abort)
{
  const std::size_t n = joints_.size();
  Inputs in;
  {
    std::lock_guard lock(mutex_);
    in = inputs_;
  }
  const auto now = Clock::now();
  const auto recent = [&](Clock::time_point t) { return now - t < std::chrono::seconds(2); };
  std::vector<double> start;
  if (in.setpoint.size() == n && recent(in.setpoint_time))
  {
    start = in.setpoint;  // where the controller's trajectory is: the new one continues from it
  }
  else if (in.measured.size() == n && recent(in.measured_time))
  {
    start = in.measured;
  }
  else
  {
    RCLCPP_ERROR(get_logger(), "going home: no recent %s/controller_state or /joint_states; not moving", options_.controller.c_str());
    return false;
  }
  if (std::all_of(start.begin(), start.end(), [](double q) { return std::abs(q) < 1e-3; }))
  {
    RCLCPP_INFO(get_logger(), "going home: already there");
    return true;
  }
  const std::vector<double> home(n, 0.0);
  // The whole robot's line, so the arms don't meet on the way.
  std::string pair;
  if (options_.collision_check && !line_clear(start, home, pair))
  {
    RCLCPP_ERROR(get_logger(), "going home: the straight line home collides (%s); not moving", pair.c_str());
    return false;
  }
  vmax_.assign(n, options_.home_velocity);
  traj_points_ = {home};
  const double seconds = send_trajectory(start, traj_points_, 0, std::vector<std::optional<double>>(arms_.size()), now, nullptr);
  RCLCPP_INFO(get_logger(), "going home: %zu waypoints, %.1f s (Ctrl-C again to stop)", traj_points_.size(), seconds);
  const auto end = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds + 0.5));
  while (Clock::now() < end && !abort())
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return true;
}

// Ctrl-C and SIGTERM are counted, not handled by rclcpp's shutdown: the first one stops the node's
// control loop and sends the arm home (which publishes, so the context must stay valid), the second
// ends the wait.
std::atomic<int> g_signals{0};
extern "C" void on_signal(int)
{
  g_signals.fetch_add(1);
}

}  // namespace

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv, rclcpp::InitOptions(), rclcpp::SignalHandlerOptions::None);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  std::vector<char*> argv2;
  for (const auto& a : args)
  {
    argv2.push_back(const_cast<char*>(a.c_str()));
  }

  Options o;
  cxxopts::Options cli("teleop", "Clutched end-effector teleoperation from a VR controller (hold the trigger to move).");
  // clang-format off
  cli.add_options()
    ("model", "robot model (by default the one in active_model.txt)", cxxopts::value(o.model)->default_value(o.model))
    ("hand", "left or right controller (one arm)", cxxopts::value(o.hand)->default_value(o.hand))
    ("swap-hands", "several arms: each arm follows the other hand (the left controller drives the right arm), "
                   "which is the natural way round when you face the robot")
    ("controller", "joint_trajectory_controller name", cxxopts::value(o.controller)->default_value(o.controller))
    ("rate", "control rate, Hz", cxxopts::value(o.rate)->default_value("200"))
    ("scale", "tool motion per controller motion", cxxopts::value(o.scale)->default_value("1"))
    ("no-rotation", "track position only, keep the tool's orientation")
    ("max-speed", "tool speed limit, m/s", cxxopts::value(o.max_speed)->default_value("1"))
    ("max-angular-speed", "tool turn rate limit, rad/s", cxxopts::value(o.max_angular_speed)->default_value("4"))
    ("max-joint-velocity", "joint velocity limit for timing commands, rad/s (every joint)",
     cxxopts::value(o.max_joint_velocity)->default_value("3"))
    ("joint-velocity-limits", "per-joint overrides, e.g. joint1=1.0,joint6=3.0 (rad/s)",
     cxxopts::value(o.joint_velocity_limits)->default_value(""))
    ("lookahead", "time the controller gets to reach each setpoint, s", cxxopts::value(o.lookahead)->default_value("0.02"))
    ("stale", "release if the controller pose is older, s", cxxopts::value(o.stale)->default_value("0.1"))
    ("no-collision-check", "don't stop commands before self-collisions")
    ("collision-resolution", "joint-space path sampling for collision checks, rad",
     cxxopts::value(o.collision_resolution)->default_value("0.01"))
    ("busy-time", "trajectories longer than this (s) play out before teleop sends the next command",
     cxxopts::value(o.busy_time)->default_value("0.1"))
    ("collision-escape", "largest joint change at the start of a path within which a collision is tolerated "
     "if the path gets collision-free, rad",
     cxxopts::value(o.collision_escape)->default_value("0.0872665"))
    ("collision-max-samples", "most path samples per cycle (long moves are sampled more coarsely)",
     cxxopts::value(o.collision_max_samples)->default_value("100"))
    ("ik-seeds", "IK seeds per cycle: the measured joint angles, then random ones; the closest valid solution wins",
     cxxopts::value(o.ik_seeds)->default_value("8"))
    ("ik-position-tolerance", "accept IK solutions within this distance of the target, m",
     cxxopts::value(o.ik_position_tolerance)->default_value("0.005"))
    ("ik-orientation-tolerance", "accept IK solutions within this angle of the target, rad",
     cxxopts::value(o.ik_orientation_tolerance)->default_value("0.05"))
    ("ik-solver", "kdl (KDL, Levenberg-Marquardt) or fast (fast_robot_kinematics, single precision)",
     cxxopts::value(o.ik_solver)->default_value("kdl"))
    ("ik-iterations", "IK solver iterations per attempt", cxxopts::value(o.ik_iterations)->default_value("100"))
    ("gripper-joint", "gripper joint driven by the grip squeeze (empty: none)",
     cxxopts::value(o.gripper_joint)->default_value(o.gripper_joint))
    ("no-gripper", "don't drive the gripper")
    ("gripper-open", "gripper position at no squeeze", cxxopts::value(o.gripper_open)->default_value("0.05"))
    ("gripper-closed", "gripper position at full squeeze", cxxopts::value(o.gripper_closed)->default_value("0"))
    ("no-home", "on Ctrl-C, exit at once instead of sending the arm home (all joints at zero)")
    ("home-velocity", "speed of the trip home on Ctrl-C, rad/s", cxxopts::value(o.home_velocity)->default_value("0.5"))
    ("rt-priority", "SCHED_FIFO priority of the control thread (0: normal scheduling)",
     cxxopts::value(o.rt_priority)->default_value("0"))
    ("h,help", "show this help");
  // clang-format on
  try
  {
    const auto parsed = cli.parse(static_cast<int>(argv2.size()), argv2.data());
    if (parsed.count("help"))
    {
      std::cout << cli.help() << "\n";
      rclcpp::shutdown();
      return 0;
    }
    o.rotation = parsed.count("no-rotation") == 0;
    o.collision_check = parsed.count("no-collision-check") == 0;
    o.home_on_exit = parsed.count("no-home") == 0;
    o.swap_hands = parsed.count("swap-hands") > 0;
    if (parsed.count("no-gripper"))
    {
      o.gripper_joint.clear();
    }
    if (o.hand != "left" && o.hand != "right")
    {
      throw std::invalid_argument("--hand must be left or right");
    }
    if (o.home_velocity <= 0) throw std::invalid_argument("--home-velocity must be positive");
    if (o.rate <= 0 || o.max_speed <= 0 || o.max_angular_speed <= 0 || o.ik_seeds < 1 || o.max_joint_velocity <= 0)
    {
      throw std::invalid_argument(
          "--rate, --max-speed, --max-angular-speed, --ik-seeds and --max-joint-velocity must be positive");
    }
    std::stringstream limits(o.joint_velocity_limits);
    for (std::string item; std::getline(limits, item, ',');)
    {
      const auto eq = item.find('=');
      if (item.empty()) continue;
      if (eq == std::string::npos) throw std::invalid_argument("--joint-velocity-limits: expected joint=value, got " + item);
      const double v = std::stod(item.substr(eq + 1));
      if (v <= 0) throw std::invalid_argument("--joint-velocity-limits: " + item + " must be positive");
      o.joint_velocity_overrides.emplace_back(item.substr(0, eq), v);
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << "teleop: " << e.what() << "\n" << cli.help() << "\n";
    rclcpp::shutdown();
    return 2;
  }
  const auto model_entry = kModels.find(o.model);
  if (model_entry == kModels.end())
  {
    std::cerr << "teleop: no kinematics for model '" << o.model << "'"
              << (o.model == robot_model::kActiveModel ? " (the active model, from active_model.txt: pass --model)" : "") << "\n";
    rclcpp::shutdown();
    return 2;
  }

  const Model& entry = model_entry->second;
  if (o.ik_solver != "kdl" && o.ik_solver != "fast")
  {
    std::fprintf(stderr, "--ik-solver: kdl or fast, not '%s'\n", o.ik_solver.c_str());
    return 1;
  }
  const auto& make_kinematics = o.ik_solver == "kdl" ? entry.kdl_kinematics : entry.kinematics;
  if (!make_kinematics)
  {
    std::fprintf(stderr, "--ik-solver %s is not available for %s\n", o.ik_solver.c_str(), o.model.c_str());
    return 1;
  }

  // One node for the robot: with several arms, each follows its own hand (left/right), and everything else (the
  // joint states, the trajectory controller, the collision checks) is shared, so the arms avoid
  // each other. The controller poses are in the world frame (quest_bridge, default frame world).
  std::shared_ptr<Teleop> node;
  try
  {
    node = std::make_shared<Teleop>(o, entry.load(), entry, make_kinematics);
  }
  catch (const std::invalid_argument& e)
  {
    std::cerr << "teleop: " << e.what() << "\n";
    rclcpp::shutdown();
    return 2;
  }
  {
    std::string arms;
    for (const std::string& prefix : entry.arms) arms += (arms.empty() ? "" : ", ") + (prefix.empty() ? "arm" : prefix.substr(0, prefix.size() - 1) + " arm");
    RCLCPP_INFO(node->get_logger(), "%s (%s), %.0f Hz; hold a trigger to move its arm, the reset button (A / X) resets its target%s",
                o.model.c_str(), arms.c_str(), o.rate,
                o.gripper_joint.empty() ? "" : ", squeeze the grip to close its gripper");
  }

  std::atomic<bool> running{true};
  std::thread control([&] {
    if (o.rt_priority > 0)
    {
      sched_param param{};
      param.sched_priority = o.rt_priority;
      if (const int err = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param); err != 0)
      {
        RCLCPP_WARN(node->get_logger(), "SCHED_FIFO priority %d not allowed (%s); running unprioritized",
                    o.rt_priority, std::strerror(err));
      }
    }
    node->run(running);
  });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  while (g_signals.load() == 0 && rclcpp::ok())
  {
    executor.spin_once(std::chrono::milliseconds(20));
  }
  running = false;
  control.join();
  if (g_signals.load() > 0 && o.home_on_exit)
  {
    RCLCPP_INFO(node->get_logger(), "stopping; sending the robot home");
    node->go_home([] { return g_signals.load() > 1; });
  }
  rclcpp::shutdown();
  return 0;
}
