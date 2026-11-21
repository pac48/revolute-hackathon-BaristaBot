// State estimation interface shared by all robot models.
//
// A StateEstimator updates a model's state (mjData) given observations: what sensors report,
// keyed by name. Each robot model package provides a factory (e.g. rebot_arm::make_state_estimator()
// in rebot_arm.hpp). The estimator of every model is a JointStateEstimator for now: it takes joint
// states (from a joint_states topic, say) as they are. Observation is where other kinds of
// measurements will be added, for estimators that fuse them with the state in mjData.
//
// Like Kinematics, states are mjData of the model the estimator was made for.

#pragma once

#include <mujoco/mujoco.h>

#include <map>
#include <string>

namespace robot_model
{

// Joint name -> value: positions in rad (hinge) or m (slide), velocities in rad/s or m/s.
using JointValues = std::map<std::string, double>;

// What was observed. Joints and quantities that are not in it are not observed.
struct Observation
{
  JointValues joint_positions;
  JointValues joint_velocities;
};

class StateEstimator
{
public:
  virtual ~StateEstimator() = default;
  StateEstimator(const StateEstimator&) = delete;
  StateEstimator& operator=(const StateEstimator&) = delete;

  // Updates `data` (of the model) given `observation`. Afterwards `data`'s kinematics are
  // current (xpos, xmat, ...: mj_kinematics), so it can be rendered or used for kinematics.
  // Throws std::invalid_argument, leaving `data` unchanged, for a joint that is not a hinge or slide
  // joint of the model, or a value that is not finite.
  virtual void update(mjData* data, const Observation& observation) = 0;

  // update() with joint positions only, e.g. a joint_states message.
  void update_joint_positions(mjData* data, const JointValues& positions)
  {
    update(data, Observation{positions, {}});
  }

protected:
  StateEstimator() = default;
};

// Takes the observed joint positions and velocities as the state: qpos and qvel of each observed
// joint are set to them, the others are left as they are. A joint that the model's equality
// constraints tie to an observed one (the reBot's gripper fingers follow joint7) and that is not
// itself observed follows it, so a model driven by one joint value stays consistent.
class JointStateEstimator : public StateEstimator
{
public:
  // `model` is not owned and must outlive the estimator.
  explicit JointStateEstimator(const mjModel* model);

  void update(mjData* data, const Observation& observation) override;

private:
  const mjModel* model_;
};

}  // namespace robot_model
