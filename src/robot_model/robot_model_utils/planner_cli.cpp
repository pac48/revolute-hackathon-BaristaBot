#include "src/robot_model/robot_model_utils/planner_cli.hpp"

#include <nlohmann/json.hpp>

#include <exception>
#include <string>

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

std::array<double, 3> vec3(const json& j, const std::array<double, 3>& fallback = {0, 0, 0})
{
  return j.is_null() ? fallback : j.get<std::array<double, 3>>();
}

Environment environment(const json& obstacles)
{
  Environment env;
  for (const auto& s : obstacles.value("spheres", json::array()))
  {
    env.spheres.push_back({vec3(s.at("center")), s.at("radius").get<double>()});
  }
  for (const auto& b : obstacles.value("boxes", json::array()))
  {
    env.boxes.push_back({vec3(b.at("center")), vec3(b.at("half_extents")), vec3(b.value("rpy", json()))});
  }
  return env;
}

}  // namespace

int run_planner_cli(const mjModel* model, const MotionPlanner& planner, std::istream& in, std::ostream& out)
{
  try
  {
    const json request = json::parse(in);
    if (request.value("describe", false))
    {
      out << json{{"joint_names", planner.joint_names()}} << "\n";
      return 0;
    }
    const json settings_json = request.value("settings", json::object());
    PlannerSettings settings;
    settings.max_iterations = settings_json.value("max_iterations", settings.max_iterations);
    settings.range = settings_json.value("range", settings.range);
    settings.simplify = settings_json.value("simplify", settings.simplify);

    const auto start = state(model, request.value("start", json::object()));
    const auto goal = state(model, request.at("goal"));
    const auto result =
        planner.plan(*start, *goal, environment(request.value("obstacles", json::object())), settings);

    json path = json::array();
    for (const auto& waypoint : result.path)
    {
      json q = json::array();
      for (const auto& name : planner.joint_names())
      {
        q.push_back(joint_position(model, waypoint.get(), name));
      }
      path.push_back(q);
    }
    out << json{{"solved", result.solved},
                {"joint_names", planner.joint_names()},
                {"path", path},
                {"planning_time_ms", result.planning_time_ms},
                {"simplify_time_ms", result.simplify_time_ms},
                {"iterations", result.iterations},
                {"error", result.error}}
        << "\n";
    return 0;
  }
  catch (const std::exception& e)
  {
    out << json{{"error", e.what()}} << "\n";
    return 2;
  }
}

}  // namespace robot_model
