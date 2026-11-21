// The dual-arm MJCF loads in MuJoCo (meshes resolved from runfiles) and has two identical arms,
// prefixed left_ and right_, side by side: the same x and z, 61 cm apart in y.

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include <rules_cc/cc/runfiles/runfiles.h>

#include <memory>
#include <string>

namespace
{

class DualArmMjcf : public testing::Test
{
protected:
  void SetUp() override
  {
    std::string error;
    std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
        rules_cc::cc::runfiles::Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
    ASSERT_NE(runfiles, nullptr) << error;
    const std::string path = runfiles->Rlocation(testing::internal::GetArgvs().at(1));
    char load_error[1024] = {0};
    model_ = mj_loadXML(path.c_str(), nullptr, load_error, sizeof(load_error));
    ASSERT_NE(model_, nullptr) << load_error;
    data_ = mj_makeData(model_);
    mj_forward(model_, data_);
  }
  void TearDown() override
  {
    mj_deleteData(data_);
    mj_deleteModel(model_);
  }
  int id(mjtObj type, const std::string& name) const { return mj_name2id(model_, type, name.c_str()); }

  mjModel* model_ = nullptr;
  mjData* data_ = nullptr;
};

TEST_F(DualArmMjcf, EachArmHasSevenActuatedJointsAndTheGripperFingers)
{
  for (const char* arm : {"left_", "right_"})
  {
    for (int i = 1; i <= 7; ++i)
    {
      EXPECT_GE(id(mjOBJ_JOINT, arm + std::string("joint") + std::to_string(i)), 0) << arm << i;
      EXPECT_GE(id(mjOBJ_ACTUATOR, arm + std::string("joint") + std::to_string(i) + "_motor"), 0) << arm << i;
    }
    EXPECT_GE(id(mjOBJ_JOINT, arm + std::string("joint_left")), 0);
    EXPECT_GE(id(mjOBJ_JOINT, arm + std::string("joint_right")), 0);
    EXPECT_GE(id(mjOBJ_BODY, arm + std::string("camera_link")), 0);
  }
  EXPECT_EQ(model_->nu, 14);
  EXPECT_EQ(model_->neq, 4);  // the fingers follow joint7 on each arm
}

TEST_F(DualArmMjcf, BasesAreAlignedInXAndSeparatedBy34_5CmInY)
{
  const double* left = data_->xpos + 3 * id(mjOBJ_BODY, "left_base_link");
  const double* right = data_->xpos + 3 * id(mjOBJ_BODY, "right_base_link");
  EXPECT_NEAR(left[0], right[0], 1e-9);
  EXPECT_NEAR(left[2], right[2], 1e-9);
  EXPECT_NEAR(left[1] - right[1], 0.61, 1e-9);
  // The same pose of the arms: the tools are 61 cm apart at the zero configuration, too.
  const double* left_tool = data_->xpos + 3 * id(mjOBJ_BODY, "left_gripper_end");
  const double* right_tool = data_->xpos + 3 * id(mjOBJ_BODY, "right_gripper_end");
  EXPECT_NEAR(left_tool[0], right_tool[0], 1e-6);
  EXPECT_NEAR(left_tool[1] - right_tool[1], 0.61, 1e-6);
  EXPECT_NEAR(left_tool[2], right_tool[2], 1e-6);
}

TEST_F(DualArmMjcf, ArmsDoNotTouchEachOtherAtTheZeroConfiguration)
{
  for (int i = 0; i < data_->ncon; ++i)
  {
    const bool left1 = std::string(mj_id2name(model_, mjOBJ_BODY, model_->geom_bodyid[data_->contact[i].geom1])).rfind("left_", 0) == 0;
    const bool left2 = std::string(mj_id2name(model_, mjOBJ_BODY, model_->geom_bodyid[data_->contact[i].geom2])).rfind("left_", 0) == 0;
    const int b1 = model_->geom_bodyid[data_->contact[i].geom1], b2 = model_->geom_bodyid[data_->contact[i].geom2];
    if (b1 == 0 || b2 == 0) continue;  // the floor
    EXPECT_EQ(left1, left2) << "contact between the arms";
  }
}

}  // namespace
