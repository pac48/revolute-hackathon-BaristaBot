// Renderer: images of a MuJoCo model's USD scene, rendered with NVIDIA ovrtx (RTX).
//
// The scene is a USD stage whose prims correspond to the MJCF: /World/<body name> is an Xform for
// every MuJoCo body, and every MuJoCo camera <name> has a USD Camera and a RenderProduct
// /Render/<name> (a stage may have more cameras than the MJCF, e.g. "overview"). update() sets
// the bodies' world poses from mjData; render() renders a camera.
//
// Needs an NVIDIA RTX GPU and driver (and, in a sandbox, access to them). The first renderer of
// a machine compiles shaders, which can take minutes; the renderer itself is then created in
// seconds. Not thread-safe: one renderer is one ovrtx renderer and one ovstage instance.

#pragma once

#include <mujoco/mujoco.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace robot_model
{

// An 8-bit RGB image, row-major from the top-left, 3 bytes per pixel.
struct Image
{
  int width = 0;
  int height = 0;
  std::vector<std::uint8_t> rgb;
};

class Renderer
{
public:
  // Opens the USD file at `usd_path` (absolute; its sublayers and references are resolved relative
  // to it) for the bodies of `model`. Every named MuJoCo body needs a /World/<name> prim.
  // Throws std::runtime_error with ovrtx's message on failure.
  Renderer(const mjModel* model, const std::string& usd_path);
  ~Renderer();
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;

  // Sets the pose of every body from `data` (xpos/xquat as of its last forward kinematics:
  // mj_forward, mj_kinematics or mj_step). `data` is of the model given to the constructor.
  void update(const mjData* data);

  // Renders the camera `camera` (a MuJoCo camera name, or a USD-only camera such as "overview")
  // at the poses of the last update(), at the resolution of its RenderProduct.
  // Throws std::runtime_error if there is no such RenderProduct or rendering fails.
  Image render(const std::string& camera);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace robot_model
