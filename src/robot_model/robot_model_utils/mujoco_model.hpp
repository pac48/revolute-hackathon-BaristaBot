// Shared helpers for the robot model libraries (two_link_arm, rebot_arm, ...):
// find a model file in runfiles, compile it into a MuJoCo model, and access
// joint positions in mjData by joint name.

#pragma once

#include <mujoco/mujoco.h>

#include <memory>
#include <string>
#include <vector>

namespace robot_model
{

struct MjModelDeleter
{
  void operator()(mjModel* model) const { mj_deleteModel(model); }
};
using MjModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;

struct MjDataDeleter
{
  void operator()(mjData* data) const { mj_deleteData(data); }
};
using MjDataPtr = std::unique_ptr<mjData, MjDataDeleter>;

// Absolute path of a runfile given its rlocation path (`$(rlocationpath ...)`).
// Works under `bazel run`/`bazel test` and when a binary is started directly.
// Throws std::runtime_error if the file can't be found.
std::string runfile_path(const std::string& rlocation);

// Compiles the MJCF at `path` (meshes etc. resolved relative to it).
// Throws std::runtime_error with MuJoCo's message if loading fails.
MjModelPtr load_mujoco_model(const std::string& path);

// New mjData for `model`, at its default configuration (qpos0).
MjDataPtr make_data(const mjModel* model);

// Address in qpos of a hinge or slide joint, by name.
// Throws std::invalid_argument if there is no such joint, or it has another type.
int joint_qpos_address(const mjModel* model, const std::string& joint);

// Position of a hinge or slide joint in d->qpos (radians or meters).
double joint_position(const mjModel* model, const mjData* data, const std::string& joint);
void set_joint_position(const mjModel* model, mjData* data, const std::string& joint, double value);

// Bodies in contact (penetrating) at `data`'s configuration, as "body_a - body_b"
// (MuJoCo's own collision filtering: e.g. parent/child bodies don't collide).
// Runs kinematics and collision detection on `data`.
std::vector<std::string> penetrating_contacts(const mjModel* model, mjData* data);

}  // namespace robot_model
