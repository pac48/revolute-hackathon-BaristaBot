// Self-contacts of the RS arm MJCF (argv[1]) over random joint configurations.
//
// Adjacent bodies (parent and child) share a joint, and their convex collision meshes
// overlap around it, so MuJoCo must never report contacts between them: they would stall the
// joint in the sim and make IK collision checks reject valid poses. The test samples
// configurations uniformly within the joint ranges, runs MuJoCo's collision detection and
// fails on any contact between adjacent bodies. It prints every contacting pair it sees.

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include <rules_cc/cc/runfiles/runfiles.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace
{

constexpr int kSamples = 5000;

std::string body_name(const mjModel* m, int body)
{
  const char* name = mj_id2name(m, mjOBJ_BODY, body);
  return name ? name : "body" + std::to_string(body);
}

TEST(RebotArmData, NoContactsBetweenAdjacentBodies)
{
  std::vector<std::string> args = testing::internal::GetArgvs();
  ASSERT_GE(args.size(), 2u) << "pass the RS MJCF's rlocation";
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
  ASSERT_NE(runfiles, nullptr) << error;
  const std::string path = runfiles->Rlocation(args[1]);
  char load_error[1024] = {0};
  mjModel* m = mj_loadXML(path.c_str(), nullptr, load_error, sizeof(load_error));
  ASSERT_NE(m, nullptr) << load_error;
  mjData* d = mj_makeData(m);

  std::mt19937 rng(7);
  std::map<std::pair<std::string, std::string>, int> pairs;  // penetrating contacts per body pair
  std::map<std::pair<std::string, std::string>, int> adjacent;
  for (int s = 0; s < kSamples; ++s)
  {
    mj_resetData(m, d);
    for (int j = 0; j < m->njnt; ++j)
    {
      if (m->jnt_type[j] != mjJNT_HINGE && m->jnt_type[j] != mjJNT_SLIDE)
      {
        continue;
      }
      double lo = -3.14159, hi = 3.14159;
      if (m->jnt_limited[j])
      {
        lo = m->jnt_range[2 * j];
        hi = m->jnt_range[2 * j + 1];
      }
      d->qpos[m->jnt_qposadr[j]] = std::uniform_real_distribution<double>(lo, hi)(rng);
    }
    mj_kinematics(m, d);
    mj_collision(m, d);
    for (int c = 0; c < d->ncon; ++c)
    {
      const mjContact& con = d->contact[c];
      if (con.dist >= 0)
      {
        continue;
      }
      int b1 = m->geom_bodyid[con.geom[0]], b2 = m->geom_bodyid[con.geom[1]];
      std::pair<std::string, std::string> key = std::minmax(body_name(m, b1), body_name(m, b2));
      ++pairs[key];
      if (m->body_parentid[b1] == b2 || m->body_parentid[b2] == b1)
      {
        ++adjacent[key];
      }
    }
  }

  std::printf("penetrating contacts over %d random configurations:\n", kSamples);
  for (const auto& [key, n] : pairs)
  {
    std::printf("  %-16s - %-16s %5d%s\n", key.first.c_str(), key.second.c_str(), n,
                adjacent.count(key) ? "  (adjacent)" : "");
  }
  for (const auto& [key, n] : adjacent)
  {
    ADD_FAILURE() << "adjacent bodies collide: " << key.first << " - " << key.second << " in " << n << " of "
                  << kSamples << " configurations; add <exclude body1=... body2=...> to the MJCF";
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}

}  // namespace
