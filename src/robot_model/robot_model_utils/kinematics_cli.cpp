#include "src/robot_model/robot_model_utils/kinematics_cli.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <string>
#include <vector>

namespace robot_model
{
namespace
{

using nlohmann::json;

MjDataPtr state(const mjModel* model, const json& joints)
{
  auto d = make_data(model);
  for (const auto& [name, value] : joints.items())
  {
    set_joint_position(model, d.get(), name, value.get<double>());
  }
  mj_kinematics(model, d.get());
  return d;
}

json to_json(const Eigen::Isometry3d& pose)
{
  const Eigen::Quaterniond q(pose.linear());
  const Eigen::Vector3d p = pose.translation();
  return {{"position", {p.x(), p.y(), p.z()}}, {"quaternion", {q.x(), q.y(), q.z(), q.w()}}};
}

Eigen::Isometry3d pose_from_json(const json& j)
{
  const auto p = j.at("position").get<std::vector<double>>();
  const auto q = j.at("quaternion").get<std::vector<double>>();
  if (p.size() != 3 || q.size() != 4)
  {
    throw std::invalid_argument("a pose needs position [x, y, z] and quaternion [x, y, z, w]");
  }
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(p[0], p[1], p[2]);
  pose.linear() = Eigen::Quaterniond(q[3], q[0], q[1], q[2]).normalized().toRotationMatrix();
  return pose;
}

std::vector<double> chain_positions(const mjModel* model, const mjData& d, const Kinematics& kinematics)
{
  std::vector<double> q;
  for (const auto& name : kinematics.joint_names())
  {
    q.push_back(joint_position(model, &d, name));
  }
  return q;
}

json ik_path(const mjModel* model, const Kinematics& kinematics, const json& request)
{
  IkSettings settings;
  const json s = request.value("settings", json::object());
  settings.max_attempts = s.value("max_attempts", settings.max_attempts);
  settings.iterations = s.value("iterations", settings.iterations);
  settings.position_tolerance = s.value("position_tolerance", settings.position_tolerance);
  settings.orientation_tolerance = s.value("orientation_tolerance", settings.orientation_tolerance);
  const double max_step = request.value("max_joint_step", 0.2);
  const bool check_collisions = request.value("check_collisions", true);
  IkSettings seeded = settings;
  seeded.max_attempts = 1;  // from the previous solution only

  const auto start_time = std::chrono::steady_clock::now();
  auto current = state(model, request.value("start", json::object()));
  std::vector<double> previous = chain_positions(model, *current, kinematics);
  json path = json::array();
  int failed_index = -1;
  std::string error;
  double max_position_error = 0, max_orientation_error = 0;
  const auto poses = request.at("poses");
  for (std::size_t i = 0; i < poses.size(); ++i)
  {
    const Eigen::Isometry3d target = pose_from_json(poses[i]);
    IkResult result = kinematics.inverse_kinematics(target, *current, seeded);
    if (!result.solved)
    {
      result = kinematics.inverse_kinematics(target, *current, settings);  // random restarts
    }
    if (!result.solved)
    {
      failed_index = static_cast<int>(i);
      error = "no IK solution for pose " + std::to_string(i) +
              (std::isfinite(result.position_error)
                   ? " (best: " + std::to_string(result.position_error) + " m, " +
                         std::to_string(result.orientation_error) + " rad)"
                   : " (no attempt within the joint limits)");
      break;
    }
    const std::vector<double> q = chain_positions(model, *current, kinematics);
    double step = 0;
    std::string joint;
    for (std::size_t j = 0; j < q.size(); ++j)
    {
      if (std::abs(q[j] - previous[j]) > step)
      {
        step = std::abs(q[j] - previous[j]);
        joint = kinematics.joint_names()[j];
      }
    }
    if (step > max_step)
    {
      failed_index = static_cast<int>(i);
      error = "IK jumped at pose " + std::to_string(i) + ": " + joint + " moved " + std::to_string(step) +
              " rad (max_joint_step " + std::to_string(max_step) + "); sample the path more densely or "
              "the path passes near a singularity or a joint limit";
      break;
    }
    if (check_collisions)
    {
      const std::vector<std::string> contacts = penetrating_contacts(model, current.get());
      if (!contacts.empty())
      {
        failed_index = static_cast<int>(i);
        error = "collision at pose " + std::to_string(i) + ": " + contacts.front() +
                (contacts.size() > 1 ? " (+" + std::to_string(contacts.size() - 1) + " more)" : "");
        break;
      }
    }
    max_position_error = std::max(max_position_error, result.position_error);
    max_orientation_error = std::max(max_orientation_error, result.orientation_error);
    path.push_back(q);
    previous = q;
  }
  return {{"solved", failed_index < 0},
          {"joint_names", kinematics.joint_names()},
          {"path", path},
          {"failed_index", failed_index},
          {"error", error},
          {"time_ms",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count()},
          {"max_position_error", max_position_error},
          {"max_orientation_error", max_orientation_error}};
}

// The answer to one request (throws on a malformed one).
json answer(const mjModel* model, const Kinematics& kinematics, const json& request)
{
  if (request.value("describe", false))
  {
    return json{{"joint_names", kinematics.joint_names()},
                {"root_link", kinematics.root_link()},
                {"tip_link", kinematics.tip_link()}};
  }
  if (request.contains("fk"))
  {
    const json& fk = request.at("fk");
    if (fk.is_array())
    {
      json poses = json::array();
      for (const auto& joints : fk)
      {
        poses.push_back(to_json(kinematics.forward_kinematics(*state(model, joints))));
      }
      return json{{"poses", poses}};
    }
    return json{{"pose", to_json(kinematics.forward_kinematics(*state(model, fk)))}};
  }
  if (request.contains("ik_path"))
  {
    return ik_path(model, kinematics, request.at("ik_path"));
  }
  throw std::invalid_argument("expected one of describe, fk, ik_path");
}

}  // namespace

int run_kinematics_cli(const mjModel* model, const Kinematics& kinematics, std::istream& in, std::ostream& out)
{
  int code = 0;
  int requests = 0;
  while ((in >> std::ws) && in.peek() != std::char_traits<char>::eof())
  {
    json request;
    try
    {
      in >> request;  // one JSON value: the rest of the stream stays for the next request
    }
    catch (const std::exception& e)
    {
      out << json{{"error", e.what()}} << std::endl;
      return 2;  // not JSON: where the next request starts is unknown
    }
    ++requests;
    try
    {
      out << answer(model, kinematics, request) << std::endl;  // flushed: the caller may be waiting for it
    }
    catch (const std::exception& e)
    {
      out << json{{"error", e.what()}} << std::endl;
      code = 2;
    }
  }
  if (requests == 0)
  {
    out << json{{"error", "no request"}} << std::endl;
    return 2;
  }
  return code;
}

}  // namespace robot_model
