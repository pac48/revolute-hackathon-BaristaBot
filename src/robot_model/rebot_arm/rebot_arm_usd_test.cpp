// The USD of the arm corresponds to the MJCF: an Xform per body at its default pose.
// args: rlocation paths of scene.usda, then of its layers (rebot_arm.usda, environment.usda).

#include "src/robot_model/rebot_arm/rebot_arm.hpp"
#include "src/robot_model/robot_model_utils/usd_correspondence.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>

namespace
{
std::string g_scene;
std::vector<std::string> g_usd;

TEST(RebotArmUsd, CorrespondsToTheMjcf)
{
  const auto model = rebot_arm::load_mujoco_model();
  const auto problems = robot_model::usd_mismatches(model.get(), g_usd);
  for (const std::string& p : problems) ADD_FAILURE() << p;
}

// The scene is the arm's layer and the environment, layered.
TEST(RebotArmUsd, SceneLayersTheArmAndTheEnvironment)
{
  std::ifstream in(g_scene);
  std::stringstream text;
  text << in.rdbuf();
  EXPECT_NE(text.str().find("rebot_arm.usda"), std::string::npos);
  EXPECT_NE(text.str().find("environment.usda"), std::string::npos);
}
}  // namespace

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  g_scene = robot_model::runfile_path(argv[1]);
  for (int i = 2; i < argc; ++i) g_usd.push_back(robot_model::runfile_path(argv[i]));
  return RUN_ALL_TESTS();
}
