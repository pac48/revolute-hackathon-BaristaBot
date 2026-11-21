// Renders the arm with ovrtx. Needs an NVIDIA RTX GPU; see the BUILD target.

#include "src/robot_model/rebot_arm/rebot_arm.hpp"

#include <gtest/gtest.h>

namespace
{

long long difference(const robot_model::Image& a, const robot_model::Image& b)
{
  long long sum = 0;
  for (std::size_t i = 0; i < a.rgb.size(); ++i) sum += std::abs(int(a.rgb[i]) - int(b.rgb[i]));
  return sum;
}

TEST(RebotArmRender, ImageFollowsMjData)
{
  const auto model = rebot_arm::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  const auto renderer = rebot_arm::make_renderer(model.get());

  mj_forward(model.get(), data.get());
  renderer->update(data.get());
  const robot_model::Image home = renderer->render("overview");
  EXPECT_EQ(home.width, 640);
  EXPECT_EQ(home.height, 480);
  ASSERT_EQ(home.rgb.size(), 640u * 480u * 3u);
  // Rendering is not bit-exact (denoising), so repeating a pose is the baseline for "the arm moved".
  const long long noise = difference(home, renderer->render("overview"));

  robot_model::set_joint_position(model.get(), data.get(), "joint1", 1.0);
  robot_model::set_joint_position(model.get(), data.get(), "joint2", 0.8);
  mj_forward(model.get(), data.get());
  renderer->update(data.get());
  EXPECT_GT(difference(home, renderer->render("overview")), 3 * noise + 10000);  // the arm moved
}

TEST(RebotArmRender, UnknownCameraThrows)
{
  const auto model = rebot_arm::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  const auto renderer = rebot_arm::make_renderer(model.get());
  mj_forward(model.get(), data.get());
  renderer->update(data.get());
  EXPECT_THROW(renderer->render("no_such_camera"), std::runtime_error);
}

}  // namespace
