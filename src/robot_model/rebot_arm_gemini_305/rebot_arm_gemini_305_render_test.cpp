// Renders the arm with the Gemini 305 with ovrtx. Needs an NVIDIA RTX GPU; see the BUILD target.

#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305.hpp"

#include <gtest/gtest.h>

namespace
{

long long difference(const robot_model::Image& a, const robot_model::Image& b)
{
  long long sum = 0;
  for (std::size_t i = 0; i < a.rgb.size(); ++i) sum += std::abs(int(a.rgb[i]) - int(b.rgb[i]));
  return sum;
}

// The wrist camera is in the model: its image is at the camera's resolution and follows the arm.
TEST(RebotArmGemini305Render, WristCameraFollowsTheArm)
{
  const auto model = rebot_arm_gemini_305::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  const auto renderer = rebot_arm_gemini_305::make_renderer(model.get());

  mj_forward(model.get(), data.get());
  renderer->update(data.get());
  const robot_model::Image home = renderer->render("left_color");
  EXPECT_EQ(home.width, 1280);
  EXPECT_EQ(home.height, 720);
  // Rendering is not bit-exact (denoising), so repeating a pose is the baseline for "the arm moved".
  const long long noise = difference(home, renderer->render("left_color"));

  robot_model::set_joint_position(model.get(), data.get(), "joint1", 0.7);
  robot_model::set_joint_position(model.get(), data.get(), "joint3", 0.5);
  mj_forward(model.get(), data.get());
  renderer->update(data.get());
  EXPECT_GT(difference(home, renderer->render("left_color")), 3 * noise + 10000);

  const robot_model::Image overview = renderer->render("overview");
  EXPECT_EQ(overview.width, 640);
}

}  // namespace
