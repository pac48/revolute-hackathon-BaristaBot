// rebot_arm planner CLI: one JSON planning request on stdin, the path on stdout
// (see //src/robot_model/robot_model_utils:planner_cli.hpp).

#include "src/robot_model/rebot_arm/rebot_arm.hpp"
#include "src/robot_model/robot_model_utils/planner_cli.hpp"

#include <iostream>

int main()
{
  const auto model = rebot_arm::load_mujoco_model();
  const auto planner = rebot_arm::make_planner(model.get());
  return robot_model::run_planner_cli(model.get(), *planner, std::cin, std::cout);
}
