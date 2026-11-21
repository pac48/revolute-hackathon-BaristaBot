// rebot_arm kinematics CLI: one JSON FK/IK request on stdin, the answer on stdout
// (see //src/robot_model/robot_model_utils:kinematics_cli.hpp).

#include "src/robot_model/rebot_arm/rebot_arm.hpp"
#include "src/robot_model/robot_model_utils/kinematics_cli.hpp"

#include <iostream>

int main()
{
  const auto model = rebot_arm::load_mujoco_model();
  const auto kinematics = rebot_arm::make_kinematics(model.get());
  return robot_model::run_kinematics_cli(model.get(), *kinematics, std::cin, std::cout);
}
