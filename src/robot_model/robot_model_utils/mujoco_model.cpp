#include "src/robot_model/robot_model_utils/mujoco_model.hpp"

#include <algorithm>

#include <rules_cc/cc/runfiles/runfiles.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace robot_model
{
namespace
{

using rules_cc::cc::runfiles::Runfiles;

// Runfiles of the running binary. Works under `bazel run`/`bazel test` (env
// vars) and when a binary is started directly (<exe>.runfiles or
// <exe>.runfiles_manifest next to it), without needing argv[0].
const Runfiles& runfiles()
{
  static const std::unique_ptr<Runfiles> instance = []
  {
    std::error_code ec;
    const std::string exe = std::filesystem::read_symlink("/proc/self/exe", ec).string();
    std::string error;
    std::unique_ptr<Runfiles> r(Runfiles::Create(exe, BAZEL_CURRENT_REPOSITORY, &error));
    if (!r)
    {
      throw std::runtime_error("robot_model: cannot locate runfiles: " + error);
    }
    return r;
  }();
  return *instance;
}

}  // namespace

std::string runfile_path(const std::string& rlocation)
{
  const std::string path = runfiles().Rlocation(rlocation);
  if (path.empty() || !std::filesystem::exists(path))
  {
    throw std::runtime_error("robot_model: " + rlocation + " not found in runfiles");
  }
  return path;
}

MjModelPtr load_mujoco_model(const std::string& path)
{
  char error[1024] = {0};
  MjModelPtr model(mj_loadXML(path.c_str(), nullptr, error, sizeof(error)));
  if (!model)
  {
    throw std::runtime_error("robot_model: failed to load " + path + ": " + error);
  }
  return model;
}

MjDataPtr make_data(const mjModel* model)
{
  return MjDataPtr(mj_makeData(model));
}

int joint_qpos_address(const mjModel* model, const std::string& joint)
{
  const int id = mj_name2id(model, mjOBJ_JOINT, joint.c_str());
  if (id < 0)
  {
    throw std::invalid_argument("joint '" + joint + "' is not in the MuJoCo model");
  }
  if (model->jnt_type[id] != mjJNT_HINGE && model->jnt_type[id] != mjJNT_SLIDE)
  {
    throw std::invalid_argument("joint '" + joint + "' is not a hinge or slide joint");
  }
  return model->jnt_qposadr[id];
}

double joint_position(const mjModel* model, const mjData* data, const std::string& joint)
{
  return data->qpos[joint_qpos_address(model, joint)];
}

void set_joint_position(const mjModel* model, mjData* data, const std::string& joint, double value)
{
  data->qpos[joint_qpos_address(model, joint)] = value;
}

std::vector<std::string> penetrating_contacts(const mjModel* model, mjData* data)
{
  mj_kinematics(model, data);
  mj_collision(model, data);
  std::vector<std::string> out;
  for (int i = 0; i < data->ncon; ++i)
  {
    const mjContact& c = data->contact[i];
    if (c.dist >= 0)
    {
      continue;  // within the margin, not touching
    }
    auto body = [&](int geom) {
      const char* name = mj_id2name(model, mjOBJ_BODY, model->geom_bodyid[geom]);
      return std::string(name ? name : "?");
    };
    std::string pair = body(c.geom[0]) + " - " + body(c.geom[1]);
    if (std::find(out.begin(), out.end(), pair) == out.end())
    {
      out.push_back(pair);
    }
  }
  return out;
}

}  // namespace robot_model
