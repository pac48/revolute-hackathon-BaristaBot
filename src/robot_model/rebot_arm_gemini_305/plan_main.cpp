// rebot_arm_gemini_305 planner CLI: one JSON planning request on stdin, the path on stdout
// (see //src/robot_model/robot_model_utils:planner_cli.hpp).

#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305.hpp"
#include "src/robot_model/robot_model_utils/planner_cli.hpp"

#include <iostream>

int main()
{
  const auto model = rebot_arm_gemini_305::load_mujoco_model();
  const auto planner = rebot_arm_gemini_305::make_planner(model.get());
  return robot_model::run_planner_cli(model.get(), *planner, std::cin, std::cout);
}
