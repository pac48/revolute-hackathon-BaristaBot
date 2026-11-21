// Loads every MJCF passed as an argument (runfiles paths) with MuJoCo, so a
// missing mesh or broken include in the fetched model fails the build's tests.

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include <rules_cc/cc/runfiles/runfiles.h>

#include <memory>
#include <string>
#include <vector>

namespace
{

TEST(RebotArmData, MjcfsLoad)
{
  // Test arguments after argv[0] (set by `args` in the BUILD file).
  std::vector<std::string> mjcfs = testing::internal::GetArgvs();
  mjcfs.erase(mjcfs.begin());
  ASSERT_FALSE(mjcfs.empty()) << "no MJCF paths passed";
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
  ASSERT_NE(runfiles, nullptr) << error;

  for (const std::string& rlocation : mjcfs)
  {
    SCOPED_TRACE(rlocation);
    const std::string path = runfiles->Rlocation(rlocation);
    char load_error[1024] = {0};
    mjModel* model = mj_loadXML(path.c_str(), nullptr, load_error, sizeof(load_error));
    ASSERT_NE(model, nullptr) << load_error;
    EXPECT_GT(model->njnt, 0);
    mj_deleteModel(model);
  }
}

}  // namespace
