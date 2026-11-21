// The reBot with the Gemini 305: the model loads with its cameras and the arm is the reBot's.

#include "src/robot_model/rebot_arm/rebot_arm.hpp"
#include "src/robot_model/rebot_arm_gemini_305/rebot_arm_gemini_305.hpp"

#include <gtest/gtest.h>

#include <string>

namespace
{

TEST(RebotArmGemini305, LoadsMujocoModelWithCameras)
{
  const auto model = rebot_arm_gemini_305::load_mujoco_model();
  ASSERT_NE(model, nullptr);
  EXPECT_EQ(model->njnt, 10);  // the reBot's 9 joints and the scene cube's free joint: the cameras add none
  EXPECT_EQ(model->nu, 7);
  EXPECT_EQ(model->ncam, 3);  // the Gemini's two and the scene camera
  for (const char* camera : {"left_color", "right_color", "scene_color"})
  {
    EXPECT_GE(mj_name2id(model.get(), mjOBJ_CAMERA, camera), 0) << camera;
  }
  // Frame names are the driver's (orbbec_camera, camera_name "camera").
  for (const char* body : {"camera_bottom_screw", "camera_link", "camera_left_color_optical_frame"})
  {
    EXPECT_GE(mj_name2id(model.get(), mjOBJ_BODY, body), 0) << body;
  }
}

// The Gemini's left color optical frame is where the reBot-DevArm's D405 color camera was
// (wrist_camera_optical_frame): the camera is mounted in that place.
TEST(RebotArmGemini305, CameraIsAtTheD405sPlace)
{
  const auto model = rebot_arm_gemini_305::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  mj_kinematics(model.get(), data.get());
  const int base = mj_name2id(model.get(), mjOBJ_BODY, "gripper_end");
  const int optical = mj_name2id(model.get(), mjOBJ_BODY, "camera_left_color_optical_frame");
  Eigen::Isometry3d gripper = Eigen::Isometry3d::Identity(), cam = Eigen::Isometry3d::Identity();
  gripper.translation() = Eigen::Vector3d(data->xpos + 3 * base);
  gripper.linear() = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(data->xmat + 9 * base);
  cam.translation() = Eigen::Vector3d(data->xpos + 3 * optical);
  cam.linear() = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(data->xmat + 9 * optical);
  const Eigen::Vector3d p = (gripper.inverse() * cam).translation();
  EXPECT_NEAR(p.x(), -0.108727455312, 1e-6);
  EXPECT_NEAR(p.y(), 0.00907532251669, 1e-6);
  EXPECT_NEAR(p.z(), 0.107614316844, 1e-6);
}

// Composing must not change the arm: every geom of the bare reBot is where it was, with the same
// mesh (the camera's MJCF once leaked a default quat into the gripper's meshes).
TEST(RebotArmGemini305, ArmGeomsAreTheRebots)
{
  const auto plain = rebot_arm::load_mujoco_model();
  const auto composed = rebot_arm_gemini_305::load_mujoco_model();
  auto d_plain = robot_model::make_data(plain.get());
  auto d_composed = robot_model::make_data(composed.get());
  mj_forward(plain.get(), d_plain.get());
  mj_forward(composed.get(), d_composed.get());
  for (int i = 0; i < plain->ngeom; ++i)
  {
    const int body = plain->geom_bodyid[i];
    const char* body_name = mj_id2name(plain.get(), mjOBJ_BODY, body);
    const int composed_body = mj_name2id(composed.get(), mjOBJ_BODY, body_name);
    ASSERT_GE(composed_body, 0) << body_name;
    int k = 0;  // the same body's geoms are in the same order
    for (int t = 0; t < i; ++t) k += plain->geom_bodyid[t] == body;
    int j = -1;
    for (int t = 0, seen = 0; t < composed->ngeom && j < 0; ++t)
    {
      if (composed->geom_bodyid[t] == composed_body && seen++ == k) j = t;
    }
    ASSERT_GE(j, 0) << "geom " << i << " of " << body_name;
    const std::string what = std::string(body_name) + " geom " + std::to_string(k);
    for (int q = 0; q < 3; ++q)
    {
      EXPECT_NEAR(d_plain->geom_xpos[3 * i + q], d_composed->geom_xpos[3 * j + q], 1e-9) << what;
    }
    for (int q = 0; q < 9; ++q)
    {
      EXPECT_NEAR(d_plain->geom_xmat[9 * i + q], d_composed->geom_xmat[9 * j + q], 1e-9) << what;
    }
    const int mesh_plain = plain->geom_dataid[i], mesh_composed = composed->geom_dataid[j];
    EXPECT_EQ(mesh_plain < 0, mesh_composed < 0) << what;
    if (mesh_plain >= 0 && mesh_composed >= 0)
    {
      EXPECT_EQ(plain->mesh_vertnum[mesh_plain], composed->mesh_vertnum[mesh_composed]) << what;
    }
  }
}

TEST(RebotArmGemini305, GripperEndPoseIsTheRebotsWithoutCamera)
{
  const auto plain = rebot_arm::load_mujoco_model();
  const auto composed = rebot_arm_gemini_305::load_mujoco_model();
  const auto k_plain = rebot_arm::make_kinematics(plain.get());
  const auto k_composed = rebot_arm_gemini_305::make_kinematics(composed.get());
  EXPECT_EQ(k_plain->joint_names(), k_composed->joint_names());
  auto d_plain = robot_model::make_data(plain.get());
  auto d_composed = robot_model::make_data(composed.get());
  const std::vector<double> q{0.3, 0.9, 1.1, -0.4, 0.5, -0.7};
  for (std::size_t i = 0; i < q.size(); ++i)
  {
    const std::string joint = "joint" + std::to_string(i + 1);
    robot_model::set_joint_position(plain.get(), d_plain.get(), joint, q[i]);
    robot_model::set_joint_position(composed.get(), d_composed.get(), joint, q[i]);
  }
  const Eigen::Isometry3d a = k_plain->forward_kinematics(*d_plain), b = k_composed->forward_kinematics(*d_composed);
  EXPECT_LT((a.translation() - b.translation()).norm(), 1e-9);
  EXPECT_LT(Eigen::AngleAxisd(a.linear().transpose() * b.linear()).angle(), 1e-9);
}

// The gripper's fingers follow joint7 (equality constraints), so observing joint7 alone is enough.
TEST(RebotArmGemini305, StateEstimatorMovesTheGripperFingersWithJoint7)
{
  const auto model = rebot_arm_gemini_305::load_mujoco_model();
  auto data = robot_model::make_data(model.get());
  const auto estimator = rebot_arm_gemini_305::make_state_estimator(model.get());

  estimator->update_joint_positions(
    data.get(), {{"joint1", 0.4}, {"joint2", 0.2}, {"joint3", -0.5}, {"joint7", 0.03}});
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_left"), 0.03);
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_right"), 0.03);

  // An observed finger is not overridden.
  estimator->update_joint_positions(data.get(), {{"joint7", 0.01}, {"joint_left", 0.02}});
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_left"), 0.02);
  EXPECT_DOUBLE_EQ(robot_model::joint_position(model.get(), data.get(), "joint_right"), 0.01);

  // The end effector is where forward kinematics of the observed joints puts it.
  const auto kinematics = rebot_arm_gemini_305::make_kinematics(model.get());
  const int gripper = mj_name2id(model.get(), mjOBJ_BODY, "gripper_end");
  const Eigen::Vector3d at(data->xpos + 3 * gripper);
  EXPECT_NEAR((kinematics->forward_kinematics(*data).translation() - at).norm(), 0.0, 1e-6);
}

}  // namespace
