// rebot_arm_gemini_305::usd_path and make_renderer, apart from the rest of rebot_arm_gemini_305.cpp so that only the
// programs that render link ovrtx (a 1.4 GB runtime); see //src/robot_model/rebot_arm_gemini_305:rebot_arm_gemini_305_renderer.

#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305.hpp"

#include <memory>
#include <string>

namespace rebot_arm_gemini_305
{

std::string usd_path()
{
  return robot_model::runfile_path(REBOT_ARM_GEMINI_305_USD);
}

std::unique_ptr<robot_model::Renderer> make_renderer(const mjModel* model)
{
  return std::make_unique<robot_model::Renderer>(model, usd_path());
}

}  // namespace rebot_arm_gemini_305
