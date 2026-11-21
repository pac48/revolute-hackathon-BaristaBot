// The USD of the arm with the camera corresponds to the composed MJCF: an Xform per body (the arm's
// and the camera's) at its default pose, and the camera's cameras and RenderProducts.
// args: rlocation paths of scene.usda, then of the layers (rebot_arm.usda, gemini_305.usda, environment.usda).

#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305.hpp"
#include "src/robot_model/robot_model_utils/usd_correspondence.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

namespace
{
std::string g_scene;
std::vector<std::string> g_usd;

TEST(RebotArmGemini305Usd, CorrespondsToTheMjcf)
{
  const auto model = rebot_arm_gemini_305::load_mujoco_model();
  const auto problems = robot_model::usd_mismatches(model.get(), g_usd);
  for (const std::string& p : problems) ADD_FAILURE() << p;
}

// The composition is the base models' USD files, layered.
TEST(RebotArmGemini305Usd, SceneLayersTheBaseModels)
{
  std::ifstream in(g_scene);
  std::stringstream text;
  text << in.rdbuf();
  EXPECT_NE(text.str().find("rebot_arm.usda"), std::string::npos);
  EXPECT_NE(text.str().find("gemini_305.usda"), std::string::npos);
}
}  // namespace

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  g_scene = robot_model::runfile_path(argv[1]);
  for (int i = 2; i < argc; ++i) g_usd.push_back(robot_model::runfile_path(argv[i]));
  return RUN_ALL_TESTS();
}
