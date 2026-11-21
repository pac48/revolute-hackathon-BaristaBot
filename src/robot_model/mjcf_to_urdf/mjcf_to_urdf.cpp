// mjcf_to_urdf: write a URDF for the robot described by an MJCF file.
//
//   mjcf_to_urdf [--mesh-prefix PREFIX] <model.xml> <out.urdf>
//
// The MJCF is compiled with MuJoCo, so the URDF carries what MuJoCo actually
// simulates: body poses, joint axes and ranges, masses and inertias (including
// ones MuJoCo infers from geoms), damping/friction, and box/sphere/cylinder/
// capsule geometry. The MJCF stays the single source of truth for the model.
//
// Mapping:
//   world body                 -> link "world" (its geoms, e.g. the floor, are scene, not robot)
//   body with no joint         -> fixed joint "<parent>_to_<body>"
//   body with one hinge        -> revolute (continuous if the hinge has no range)
//   body with one slide        -> prismatic (must have a range)
//   body with a free joint     -> skipped, with its subtree (a loose scene object, e.g. a
//                                 cube to grasp: scene, not robot, like the world's geoms)
//   joint range                -> <limit lower upper>
//   joint actuatorfrcrange     -> <limit effort>  (URDF 1.2: omitted means unlimited)
//   damping / frictionloss     -> <dynamics>
// MJCF has no joint velocity limit, so <limit velocity> is always omitted
// (unlimited under URDF 1.2). Ball joints, several joints in one body, and
// unlimited slides are rejected; ellipsoid/plane/hfield geoms are skipped.
//
// Mesh geoms are skipped too, unless --mesh-prefix is given: then visual-only mesh geoms
// (contype = conaffinity = 0) become <visual> meshes, filename = PREFIX + the mesh's file
// path as written in the MJCF (relative to its meshdir), with the geom's material color.
// Collision meshes are never written, so tools that read URDF collision geometry see the
// same model with or without the flag. MuJoCo re-centers each mesh on its inertial frame
// (mesh_pos/mesh_quat); the visual origin undoes that so the original file lines up.
//
// A MuJoCo joint can sit anywhere in its body, while a URDF joint is at its
// child link's origin. Each link frame is therefore the body frame shifted to
// the joint position (same orientation), and the body's inertia and geoms are
// shifted back by the same offset.

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using Vec3 = std::array<double, 3>;

std::string num(double v)
{
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.9g", std::abs(v) < 1e-12 ? 0.0 : v);
  return buf;
}

std::string vec(const Vec3& v)
{
  return num(v[0]) + " " + num(v[1]) + " " + num(v[2]);
}

// URDF rpy (fixed-axis roll, pitch, yaw) of a MuJoCo quaternion (w, x, y, z).
Vec3 rpy(const mjtNum* q)
{
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double sinp = std::clamp(2 * (w * y - z * x), -1.0, 1.0);
  return {std::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), std::asin(sinp),
          std::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))};
}

std::string origin(const Vec3& xyz, const mjtNum* quat)
{
  return "<origin xyz=\"" + vec(xyz) + "\" rpy=\"" + vec(rpy(quat)) + "\"/>";
}

Vec3 sub(const mjtNum* a, const Vec3& b)
{
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

std::string name(const mjModel* m, mjtObj type, int id, const char* fallback)
{
  const char* n = mj_id2name(m, type, id);
  return n && *n ? n : std::string(fallback) + std::to_string(id);
}

[[noreturn]] void die(const std::string& msg)
{
  std::cerr << "mjcf_to_urdf: " << msg << "\n";
  std::exit(1);
}

// Geoms skipped because URDF has no equivalent (reported once at the end).
int g_skipped_geoms = 0;
// Bodies skipped as loose scene objects (free joint, or under one).
int g_skipped_bodies = 0;
// --mesh-prefix; empty: mesh geoms are skipped.
std::string g_mesh_prefix;

// <visual> for a visual-only mesh geom (see --mesh-prefix).
void mesh_visual(std::ostream& out, const mjModel* m, int g, const Vec3& offset)
{
  const int mesh = m->geom_dataid[g];
  if (mesh < 0 || m->mesh_pathadr[mesh] < 0)
  {
    ++g_skipped_geoms;
    return;
  }
  // Pose of the mesh file's frame in the link: geom pose * inverse(mesh_pos, mesh_quat).
  mjtNum inv_q[4], inv_p[3], q[4], p[3];
  mju_negQuat(inv_q, m->mesh_quat + 4 * mesh);
  mju_rotVecQuat(inv_p, m->mesh_pos + 3 * mesh, inv_q);
  mju_scl3(inv_p, inv_p, -1);
  const mjtNum* geom_q = m->geom_quat + 4 * g;
  mju_mulQuat(q, geom_q, inv_q);
  mju_rotVecQuat(p, inv_p, geom_q);
  const Vec3 pos = sub(m->geom_pos + 3 * g, offset);
  const Vec3 xyz = {pos[0] + p[0], pos[1] + p[1], pos[2] + p[2]};

  const float* rgba = m->geom_matid[g] >= 0 ? m->mat_rgba + 4 * m->geom_matid[g] : m->geom_rgba + 4 * g;
  const mjtNum* scale = m->mesh_scale + 3 * mesh;
  out << "    <visual>\n"
      << "      " << origin(xyz, q) << "\n"
      << "      <geometry><mesh filename=\"" << g_mesh_prefix << (m->paths + m->mesh_pathadr[mesh]) << "\" scale=\""
      << vec({scale[0], scale[1], scale[2]}) << "\"/></geometry>\n"
      << "      <material name=\"" << name(m, mjOBJ_GEOM, g, "geom") << "\"><color rgba=\"" << num(rgba[0]) << " "
      << num(rgba[1]) << " " << num(rgba[2]) << " " << num(rgba[3]) << "\"/></material>\n"
      << "    </visual>\n";
}

// <visual>/<collision> elements for one geom; capsules become a cylinder plus two spheres.
void geometry(std::ostream& out, const mjModel* m, int g, const Vec3& offset)
{
  const mjtNum* size = m->geom_size + 3 * g;
  const mjtNum* quat = m->geom_quat + 4 * g;
  const Vec3 pos = sub(m->geom_pos + 3 * g, offset);

  std::vector<std::pair<Vec3, std::string>> shapes;  // (position, <geometry> body)
  switch (m->geom_type[g])
  {
    case mjGEOM_BOX:
      shapes.push_back({pos, "<box size=\"" + vec({2 * size[0], 2 * size[1], 2 * size[2]}) + "\"/>"});
      break;
    case mjGEOM_SPHERE:
      shapes.push_back({pos, "<sphere radius=\"" + num(size[0]) + "\"/>"});
      break;
    case mjGEOM_CYLINDER:
      shapes.push_back({pos, "<cylinder radius=\"" + num(size[0]) + "\" length=\"" + num(2 * size[1]) + "\"/>"});
      break;
    case mjGEOM_CAPSULE:
    {
      shapes.push_back({pos, "<cylinder radius=\"" + num(size[0]) + "\" length=\"" + num(2 * size[1]) + "\"/>"});
      for (double sign : {-1.0, 1.0})
      {
        // Cap centers are at +-half-length along the geom's z axis.
        const mjtNum local[3] = {0, 0, sign * size[1]};
        mjtNum world[3];
        mju_rotVecQuat(world, local, quat);
        shapes.push_back({{pos[0] + world[0], pos[1] + world[1], pos[2] + world[2]},
                          "<sphere radius=\"" + num(size[0]) + "\"/>"});
      }
      break;
    }
    case mjGEOM_MESH:
      if (!g_mesh_prefix.empty() && !m->geom_contype[g] && !m->geom_conaffinity[g])
      {
        mesh_visual(out, m, g, offset);
        return;
      }
      ++g_skipped_geoms;
      return;
    default:
      ++g_skipped_geoms;
      return;
  }

  const bool collides = m->geom_contype[g] || m->geom_conaffinity[g];
  for (const auto& [xyz, shape] : shapes)
  {
    const std::string body = "      " + origin(xyz, quat) + "\n      <geometry>" + shape + "</geometry>\n";
    out << "    <visual>\n" << body << "    </visual>\n";
    if (collides)
    {
      out << "    <collision>\n" << body << "    </collision>\n";
    }
  }
}

std::string convert(const mjModel* m)
{
  std::ostringstream out;
  out << "<?xml version=\"1.0\"?>\n"
      << "<!-- Generated from MJCF by mjcf_to_urdf. Do not edit; edit the MJCF. -->\n"
      << "<robot name=\"" << m->names << "\" version=\"1.2\">\n"
      << "  <link name=\"world\"/>\n";

  // Link frame of each body = body frame shifted to its joint's position.
  std::vector<Vec3> offset(m->nbody, Vec3{0, 0, 0});
  // Free bodies and everything under them: scene objects, left out of the robot.
  std::vector<bool> loose(m->nbody, false);

  for (int b = 1; b < m->nbody; ++b)  // bodies are topologically sorted: parents first
  {
    const std::string link = name(m, mjOBJ_BODY, b, "body");
    const int parent = m->body_parentid[b];
    const std::string parent_link = parent == 0 ? "world" : name(m, mjOBJ_BODY, parent, "body");

    loose[b] = loose[parent];
    for (int k = 0; k < m->body_jntnum[b]; ++k)
    {
      loose[b] = loose[b] || m->jnt_type[m->body_jntadr[b] + k] == mjJNT_FREE;
    }
    if (loose[b])
    {
      ++g_skipped_bodies;
      continue;
    }

    if (m->body_jntnum[b] > 1)
    {
      die("body '" + link + "' has " + std::to_string(m->body_jntnum[b]) + " joints; URDF allows one");
    }
    const int j = m->body_jntnum[b] == 1 ? m->body_jntadr[b] : -1;
    if (j >= 0 && m->jnt_type[j] != mjJNT_FREE)
    {
      offset[b] = {m->jnt_pos[3 * j], m->jnt_pos[3 * j + 1], m->jnt_pos[3 * j + 2]};
    }

    // Link: inertia (principal axes) and geometry, relative to the shifted frame.
    out << "  <link name=\"" << link << "\">\n";
    if (m->body_mass[b] > 0)
    {
      const mjtNum* inertia = m->body_inertia + 3 * b;
      out << "    <inertial>\n"
          << "      " << origin(sub(m->body_ipos + 3 * b, offset[b]), m->body_iquat + 4 * b) << "\n"
          << "      <mass value=\"" << num(m->body_mass[b]) << "\"/>\n"
          << "      <inertia ixx=\"" << num(inertia[0]) << "\" iyy=\"" << num(inertia[1]) << "\" izz=\""
          << num(inertia[2]) << "\" ixy=\"0\" ixz=\"0\" iyz=\"0\"/>\n"
          << "    </inertial>\n";
    }
    for (int g = m->body_geomadr[b]; g >= 0 && g < m->body_geomadr[b] + m->body_geomnum[b]; ++g)
    {
      geometry(out, m, g, offset[b]);
    }
    out << "  </link>\n";

    // Joint: child link origin expressed in the parent's (shifted) link frame.
    const mjtNum* quat = m->body_quat + 4 * b;
    mjtNum shifted[3];
    const mjtNum local[3] = {offset[b][0], offset[b][1], offset[b][2]};
    mju_rotVecQuat(shifted, local, quat);
    const Vec3 xyz = {m->body_pos[3 * b] + shifted[0] - offset[parent][0],
                      m->body_pos[3 * b + 1] + shifted[1] - offset[parent][1],
                      m->body_pos[3 * b + 2] + shifted[2] - offset[parent][2]};

    std::string joint_name = parent_link + "_to_" + link;
    std::string type = "fixed";
    if (j >= 0)
    {
      joint_name = name(m, mjOBJ_JOINT, j, "joint");
      switch (m->jnt_type[j])
      {
        case mjJNT_HINGE:
          type = m->jnt_limited[j] ? "revolute" : "continuous";
          break;
        case mjJNT_SLIDE:
          if (!m->jnt_limited[j])
          {
            die("slide joint '" + joint_name + "' has no range; URDF prismatic joints need limits");
          }
          type = "prismatic";
          break;
        default:
          die("joint '" + joint_name + "' is a ball joint, which URDF cannot represent");
      }
    }

    out << "  <joint name=\"" << joint_name << "\" type=\"" << type << "\">\n"
        << "    <parent link=\"" << parent_link << "\"/>\n"
        << "    <child link=\"" << link << "\"/>\n"
        << "    " << origin(xyz, quat) << "\n";
    if (type == "revolute" || type == "continuous" || type == "prismatic")
    {
      const mjtNum* axis = m->jnt_axis + 3 * j;
      out << "    <axis xyz=\"" << vec({axis[0], axis[1], axis[2]}) << "\"/>\n";

      std::string limit;
      if (m->jnt_limited[j])
      {
        limit += " lower=\"" + num(m->jnt_range[2 * j]) + "\" upper=\"" + num(m->jnt_range[2 * j + 1]) + "\"";
      }
      if (m->jnt_actfrclimited[j])
      {
        const double effort =
            std::max(std::abs(m->jnt_actfrcrange[2 * j]), std::abs(m->jnt_actfrcrange[2 * j + 1]));
        limit += " effort=\"" + num(effort) + "\"";
      }
      if (!limit.empty())
      {
        out << "    <limit" << limit << "/>\n";
      }

      const int dof = m->jnt_dofadr[j];
      if (m->dof_damping[dof] > 0 || m->dof_frictionloss[dof] > 0)
      {
        out << "    <dynamics damping=\"" << num(m->dof_damping[dof]) << "\" friction=\""
            << num(m->dof_frictionloss[dof]) << "\"/>\n";
      }
    }
    out << "  </joint>\n";
  }

  out << "</robot>\n";
  return out.str();
}

}  // namespace

int main(int argc, char** argv)
{
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() >= 2 && args[0] == "--mesh-prefix")
  {
    g_mesh_prefix = args[1];
    args.erase(args.begin(), args.begin() + 2);
  }
  if (args.size() != 2)
  {
    std::cerr << "usage: mjcf_to_urdf [--mesh-prefix PREFIX] <model.xml> <out.urdf>\n";
    return 2;
  }
  char error[1024] = {0};
  mjModel* m = mj_loadXML(args[0].c_str(), nullptr, error, sizeof(error));
  if (!m)
  {
    die("failed to load " + args[0] + ": " + error);
  }
  const std::string urdf = convert(m);
  mj_deleteModel(m);
  if (g_skipped_geoms > 0)
  {
    std::cerr << "mjcf_to_urdf: skipped " << g_skipped_geoms
              << " mesh/ellipsoid/plane/hfield geoms (no URDF equivalent"
              << (g_mesh_prefix.empty() ? "; --mesh-prefix writes visual meshes" : ", or collision meshes") << ")\n";
  }

  if (g_skipped_bodies > 0)
  {
    std::cerr << "mjcf_to_urdf: skipped " << g_skipped_bodies
              << " free-joint bodies (with their subtrees) (scene objects, not robot)\n";
  }

  std::ofstream(args[1]) << urdf;
  return 0;
}
