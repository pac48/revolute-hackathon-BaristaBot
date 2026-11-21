// Test helper: checks that USD files (.usda text) correspond to an MJCF model.
//
// What the renderer (renderer.hpp) relies on: /World/<body name> is an Xform for every named
// MuJoCo body, whose authored world pose matches the body's at the model's default configuration;
// every MuJoCo camera <name> is a Camera prim named <name> and has a RenderProduct /Render/<name>.
// The files are the layers of one stage; prims are found by name (the files here are flat).

#pragma once

#include <mujoco/mujoco.h>

#include <string>
#include <vector>

namespace robot_model
{

// Problems found, one message each; empty if the USD corresponds to the MJCF.
std::vector<std::string> usd_mismatches(const mjModel* model, const std::vector<std::string>& usda_paths);

}  // namespace robot_model
