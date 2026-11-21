// Motion planning interface shared by all robot models.
//
// Each robot model package provides a factory (e.g. rebot_arm::make_planner()
// in rebot_arm.hpp) returning a MotionPlanner bound to its MuJoCo model. States
// are mjData of that model: the planner reads and writes its planning joints
// (joint_names()) in qpos by name, so joint order always follows the model.
// Joint positions can be set with robot_model::set_joint_position
// (mujoco_model.hpp).
//
// Implementations: VampPlanner<Robot> (vamp_planner.hpp) for robots generated
// with vamp_robot (//third_party/vamp:defs.bzl).

#pragma once

#include "src/robot_model/robot_model_utils/mujoco_model.hpp"

#include <mujoco/mujoco.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace robot_model
{

struct SphereObstacle
{
  std::array<double, 3> center;
  double radius;
};

struct BoxObstacle
{
  std::array<double, 3> center;
  std::array<double, 3> half_extents;
  std::array<double, 3> rpy{0, 0, 0};  // rotation about x, then y, then z (fixed axes)
};

// Obstacles in the planning model's base frame (meters).
struct Environment
{
  std::vector<SphereObstacle> spheres;
  std::vector<BoxObstacle> boxes;
};

struct PlannerSettings
{
  std::size_t max_iterations = 100000;
  double range = 1.0;    // maximum step of the tree search (radians/meters)
  bool simplify = true;  // shorten and smooth the path
};

struct PlanResult
{
  bool solved = false;
  // start ... goal. Each waypoint is a copy of the start state with the
  // planning joints' positions set (their velocities zero) and kinematics
  // computed; consecutive waypoints are valid straight joint-space motions.
  std::vector<MjDataPtr> path;
  double planning_time_ms = 0;
  double simplify_time_ms = 0;
  std::size_t iterations = 0;
  std::string error;  // why it failed, if it did
};

class MotionPlanner
{
public:
  virtual ~MotionPlanner() = default;
  MotionPlanner(const MotionPlanner&) = delete;
  MotionPlanner& operator=(const MotionPlanner&) = delete;

  // The model whose mjData the planner takes (not owned; must outlive the planner).
  const mjModel* model() const { return model_; }
  // Planning joints (hinge/slide joints of the model); others are left as they are.
  const std::vector<std::string>& joint_names() const { return joint_names_; }
  std::size_t dof() const { return joint_names_.size(); }

  // The planning joints are within limits and the robot is free of obstacles
  // and of self-collision.
  bool is_valid(const mjData& state, const Environment& environment) const;
  // The straight joint-space motion of the planning joints from a to b is valid.
  bool is_motion_valid(const mjData& a, const mjData& b, const Environment& environment) const;
  // Collision-free path for the planning joints from start to goal.
  PlanResult plan(const mjData& start, const mjData& goal, const Environment& environment,
                  const PlannerSettings& settings = {}) const;

protected:
  // Throws std::invalid_argument if a joint isn't a hinge/slide joint of the model.
  MotionPlanner(const mjModel* model, std::vector<std::string> joint_names);

  // Backend interface, on the planning joints' positions in joint_names() order.
  using Joints = std::vector<double>;
  struct JointPath
  {
    bool solved = false;
    std::vector<Joints> path;
    double planning_time_ms = 0;
    double simplify_time_ms = 0;
    std::size_t iterations = 0;
    std::string error;
  };
  virtual bool joints_valid(const Joints& q, const Environment& environment) const = 0;
  virtual bool joint_motion_valid(const Joints& a, const Joints& b, const Environment& environment) const = 0;
  virtual JointPath plan_joints(const Joints& start, const Joints& goal, const Environment& environment,
                                const PlannerSettings& settings) const = 0;

private:
  Joints joints(const mjData& state) const;

  const mjModel* model_;
  std::vector<std::string> joint_names_;
  std::vector<int> qpos_addresses_;
  std::vector<int> dof_addresses_;
};

}  // namespace robot_model
