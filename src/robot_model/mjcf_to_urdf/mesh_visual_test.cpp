// mjcf_to_urdf --mesh-prefix places each visual mesh where MuJoCo has it.
//
// Runs the converter on the MJCF (argv[2]) and, for every visual mesh in the URDF, maps
// vertices of the original mesh file through the URDF visual origin into the link frame and
// checks each lands on a vertex of MuJoCo's compiled (re-centered) mesh placed by the geom's
// pose. This checks the inverse of MuJoCo's mesh re-centering independently of how the
// converter computes it.

#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include <rules_cc/cc/runfiles/runfiles.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using Vec3 = std::array<double, 3>;
constexpr char kPrefix[] = "MESH:";

std::vector<Vec3> read_binary_stl(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<Vec3> out;
  if (data.size() < 84)
  {
    return out;
  }
  uint32_t n;
  std::memcpy(&n, data.data() + 80, 4);
  if (data.size() != 84 + 50 * static_cast<size_t>(n))
  {
    return out;  // not a binary STL
  }
  for (uint32_t t = 0; t < n; ++t)
  {
    for (int v = 0; v < 3; ++v)
    {
      float xyz[3];
      std::memcpy(xyz, data.data() + 84 + 50 * t + 12 + 12 * v, 12);
      out.push_back({xyz[0], xyz[1], xyz[2]});
    }
  }
  return out;
}

// URDF rpy (fixed axes x, y, z) -> rotation matrix, row-major.
std::array<double, 9> rpy_matrix(double r, double p, double y)
{
  const double cr = std::cos(r), sr = std::sin(r), cp = std::cos(p), sp = std::sin(p), cy = std::cos(y),
               sy = std::sin(y);
  return {cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
          sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
          -sp,     cp * sr,                cp * cr};
}

Vec3 parse3(const std::string& s)
{
  Vec3 v{};
  std::istringstream(s) >> v[0] >> v[1] >> v[2];
  return v;
}

TEST(MjcfToUrdf, VisualMeshesLineUpWithMujoco)
{
  std::vector<std::string> args = testing::internal::GetArgvs();
  ASSERT_EQ(args.size(), 3u) << "args: converter, MJCF (rlocations)";
  std::string error;
  std::unique_ptr<rules_cc::cc::runfiles::Runfiles> runfiles(
      rules_cc::cc::runfiles::Runfiles::CreateForTest(BAZEL_CURRENT_REPOSITORY, &error));
  ASSERT_NE(runfiles, nullptr) << error;
  const std::string converter = runfiles->Rlocation(args[1]);
  const std::string mjcf = runfiles->Rlocation(args[2]);
  const std::string meshdir = mjcf.substr(0, mjcf.rfind('/')) + "/meshes/";
  const std::string urdf_path = std::string(std::getenv("TEST_TMPDIR")) + "/visual.urdf";
  ASSERT_EQ(std::system((converter + " --mesh-prefix " + kPrefix + " " + mjcf + " " + urdf_path).c_str()), 0);

  char load_error[1024] = {0};
  mjModel* m = mj_loadXML(mjcf.c_str(), nullptr, load_error, sizeof(load_error));
  ASSERT_NE(m, nullptr) << load_error;

  std::ifstream f(urdf_path);
  const std::string urdf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const std::regex visual_re(
      "<visual>\\s*<origin xyz=\"([^\"]*)\" rpy=\"([^\"]*)\"/>\\s*<geometry><mesh filename=\"MESH:([^\"]*)\" "
      "scale=\"([^\"]*)\"/></geometry>\\s*<material name=\"([^\"]*)\"");
  int checked = 0;
  for (auto it = std::sregex_iterator(urdf.begin(), urdf.end(), visual_re); it != std::sregex_iterator(); ++it)
  {
    const std::smatch& v = *it;
    SCOPED_TRACE(v[3].str());
    const Vec3 xyz = parse3(v[1]), rpy = parse3(v[2]), scale = parse3(v[4]);
    const auto R = rpy_matrix(rpy[0], rpy[1], rpy[2]);
    // The material is named after the geom ("geom<id>" when unnamed).
    std::string geom_name = v[5];
    int g = mj_name2id(m, mjOBJ_GEOM, geom_name.c_str());
    if (g < 0 && geom_name.rfind("geom", 0) == 0)
    {
      g = std::atoi(geom_name.c_str() + 4);
    }
    ASSERT_GE(g, 0) << geom_name;
    const int mesh = m->geom_dataid[g];
    const int body = m->geom_bodyid[g];
    // Link frame = body frame shifted to its joint (see mjcf_to_urdf.cpp).
    Vec3 offset{0, 0, 0};
    if (m->body_jntnum[body] == 1)
    {
      const int j = m->body_jntadr[body];
      offset = {m->jnt_pos[3 * j], m->jnt_pos[3 * j + 1], m->jnt_pos[3 * j + 2]};
    }
    // MuJoCo's compiled vertices, placed by the geom pose, in the link frame.
    std::vector<Vec3> mujoco;
    for (int k = 0; k < m->mesh_vertnum[mesh]; ++k)
    {
      const float* p = m->mesh_vert + 3 * (m->mesh_vertadr[mesh] + k);
      const mjtNum local[3] = {p[0], p[1], p[2]};
      mjtNum world[3];
      mju_rotVecQuat(world, local, m->geom_quat + 4 * g);
      mujoco.push_back({world[0] + m->geom_pos[3 * g] - offset[0], world[1] + m->geom_pos[3 * g + 1] - offset[1],
                        world[2] + m->geom_pos[3 * g + 2] - offset[2]});
    }
    const std::vector<Vec3> file = read_binary_stl(meshdir + v[3].str());
    ASSERT_FALSE(file.empty()) << "can't read " << meshdir + v[3].str();
    // A spread of file vertices must each coincide with some compiled vertex.
    for (size_t k = 0; k < file.size(); k += std::max<size_t>(1, file.size() / 8))
    {
      const Vec3 s{file[k][0] * scale[0], file[k][1] * scale[1], file[k][2] * scale[2]};
      const Vec3 q{R[0] * s[0] + R[1] * s[1] + R[2] * s[2] + xyz[0],
                   R[3] * s[0] + R[4] * s[1] + R[5] * s[2] + xyz[1],
                   R[6] * s[0] + R[7] * s[1] + R[8] * s[2] + xyz[2]};
      double best = std::numeric_limits<double>::infinity();
      for (const Vec3& c : mujoco)
      {
        best = std::min(best, std::hypot(q[0] - c[0], q[1] - c[1], q[2] - c[2]));
      }
      EXPECT_LT(best, 1e-5) << "file vertex " << k << " is " << best << " m from MuJoCo's mesh";
    }
    ++checked;
  }
  EXPECT_GT(checked, 20) << "visual meshes checked";
  std::printf("checked %d visual meshes\n", checked);
  mj_deleteModel(m);
}

}  // namespace
