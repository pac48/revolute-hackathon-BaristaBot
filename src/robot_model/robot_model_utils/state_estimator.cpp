#include "src/robot_model/robot_model_utils/state_estimator.hpp"

#include "src/robot_model/robot_model_utils/mujoco_model.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace robot_model
{
namespace
{

struct Assignment
{
  int address;   // qpos or qvel address
  double value;
};

// Address of a joint's value in qvel (hinge and slide joints: one dof).
int joint_dof_address(const mjModel* model, const std::string& joint)
{
  joint_qpos_address(model, joint);  // validates the name and type
  return model->jnt_dofadr[mj_name2id(model, mjOBJ_JOINT, joint.c_str())];
}

void check_finite(const std::string& joint, double value)
{
  if (!std::isfinite(value))
  {
    throw std::invalid_argument("observation of joint '" + joint + "' is not finite");
  }
}

}  // namespace

JointStateEstimator::JointStateEstimator(const mjModel* model) : model_(model) {}

void JointStateEstimator::update(mjData* data, const Observation& observation)
{
  std::vector<Assignment> qpos, qvel;  // validated first, so a bad observation changes nothing
  for (const auto& [joint, value] : observation.joint_positions)
  {
    check_finite(joint, value);
    qpos.push_back({joint_qpos_address(model_, joint), value});
  }
  for (const auto& [joint, value] : observation.joint_velocities)
  {
    check_finite(joint, value);
    qvel.push_back({joint_dof_address(model_, joint), value});
  }

  // Joints tied to an observed one by a joint equality: joint1 = poly(joint2), polycoef in eq_data.
  auto observed = [&](const std::map<std::string, double>& values, int joint_id) {
    return values.count(mj_id2name(model_, mjOBJ_JOINT, joint_id)) != 0;
  };
  for (int e = 0; e < model_->neq; ++e)
  {
    if (model_->eq_type[e] != mjEQ_JOINT || !model_->eq_active0[e] || model_->eq_obj2id[e] < 0)
    {
      continue;
    }
    const int y = model_->eq_obj1id[e], x = model_->eq_obj2id[e];
    const mjtNum* c = model_->eq_data + mjNEQDATA * e;
    const bool linear = c[2] == 0 && c[3] == 0 && c[4] == 0 && c[1] != 0;
    const auto follow = [&](const JointValues& values, std::vector<Assignment>& out, bool velocity) {
      const bool has_y = observed(values, y), has_x = observed(values, x);
      if (has_y == has_x)
      {
        return;  // neither is observed, or both are: nothing to derive
      }
      const auto address = [&](int joint) {
        return velocity ? model_->jnt_dofadr[joint] : model_->jnt_qposadr[joint];
      };
      const std::string x_name = mj_id2name(model_, mjOBJ_JOINT, x);
      const std::string y_name = mj_id2name(model_, mjOBJ_JOINT, y);
      if (has_x)
      {
        const double xv = values.at(x_name);
        if (velocity)
        {
          if (linear) out.push_back({address(y), c[1] * xv});  // dy/dt = c1 dx/dt
        }
        else
        {
          double yv = 0, power = 1;
          for (int k = 0; k < 5; ++k, power *= xv) yv += c[k] * power;
          out.push_back({address(y), yv});
        }
      }
      else if (linear)
      {
        const double yv = values.at(y_name);
        out.push_back({address(x), velocity ? yv / c[1] : (yv - c[0]) / c[1]});
      }
    };
    follow(observation.joint_positions, qpos, false);
    follow(observation.joint_velocities, qvel, true);
  }

  for (const Assignment& a : qpos) data->qpos[a.address] = a.value;
  for (const Assignment& a : qvel) data->qvel[a.address] = a.value;
  mj_kinematics(model_, data);
}

}  // namespace robot_model
