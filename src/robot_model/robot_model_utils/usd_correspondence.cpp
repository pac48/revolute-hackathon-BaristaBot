#include "src/robot_model/robot_model_utils/usd_correspondence.hpp"

#include "src/robot_model/robot_model_utils/mujoco_model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>

namespace robot_model
{

std::vector<std::string> usd_mismatches(const mjModel* model, const std::vector<std::string>& usda_paths)
{
  std::string text;
  for (const std::string& path : usda_paths) {
    std::ifstream in(path);
    if (!in) {
      return {"cannot read " + path};
    }
    std::stringstream ss;
    ss << in.rdbuf();
    text += ss.str() + "\n";
  }

  // `def Xform "name"` followed by its block's authored matrix4d (16 numbers, row by row).
  std::map<std::string, std::vector<double>> body_matrix;
  std::set<std::string> cameras, products;
  const std::regex def(R"re(def (\w+) "([^"]+)")re");
  const std::regex matrix(R"re(matrix4d xformOp:transform = \(\s*(.*?)\s*\)\s*\n)re");
  for (auto it = std::sregex_iterator(text.begin(), text.end(), def); it != std::sregex_iterator(); ++it) {
    const std::string type = (*it)[1], name = (*it)[2];
    if (type == "Camera") cameras.insert(name);
    if (type == "RenderProduct") products.insert(name);
    if (type != "Xform") continue;
    // The first matrix after this def, if it comes before the next def.
    const std::size_t begin = it->position() + it->length();
    const std::size_t next = text.find("def ", begin);
    std::smatch m;
    const std::string block = text.substr(begin, next == std::string::npos ? std::string::npos : next - begin);
    if (!std::regex_search(block, m, matrix)) {
      body_matrix[name];  // present, without a pose
      continue;
    }
    std::string numbers = m[1];
    for (char& c : numbers) {
      if (c == '(' || c == ')' || c == ',') c = ' ';
    }
    std::istringstream in(numbers);
    std::vector<double> values;
    for (double v; in >> v;) values.push_back(v);
    body_matrix[name] = values;
  }

  MjDataPtr data = make_data(model);
  mj_forward(model, data.get());

  std::vector<std::string> problems;
  for (int b = 1; b < model->nbody; ++b) {
    const char* raw = mj_id2name(model, mjOBJ_BODY, b);
    if (!raw || !*raw) continue;
    // Loose scene objects (a free body, e.g. a cube to grasp, and its subtree) are not in the USD.
    const int root = model->body_rootid[b];
    if (model->body_jntnum[root] > 0 && model->jnt_type[model->body_jntadr[root]] == mjJNT_FREE) continue;
    const std::string name = raw;
    const auto found = body_matrix.find(name);
    if (found == body_matrix.end()) {
      problems.push_back("MJCF body '" + name + "' has no Xform in the USD");
      continue;
    }
    if (found->second.size() != 16) {
      problems.push_back("USD Xform '" + name + "' has no authored matrix4d xformOp:transform");
      continue;
    }
    // Row-vector convention: rows are the rotation's transpose, the last row is the translation.
    const mjtNum* rot = data->xmat + 9 * b;
    const mjtNum* pos = data->xpos + 3 * b;
    double worst = 0;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) worst = std::max(worst, std::abs(found->second[4 * r + c] - rot[3 * c + r]));
      worst = std::max(worst, std::abs(found->second[12 + r] - pos[r]));
    }
    if (worst > 1e-4) {
      problems.push_back("USD Xform '" + name + "' is not at the MJCF body's default pose (off by " + std::to_string(worst) + ")");
    }
  }
  for (int c = 0; c < model->ncam; ++c) {
    const std::string name = mj_id2name(model, mjOBJ_CAMERA, c);
    if (!cameras.count(name)) problems.push_back("MJCF camera '" + name + "' has no Camera in the USD");
    if (!products.count(name)) problems.push_back("MJCF camera '" + name + "' has no RenderProduct in the USD");
  }
  return problems;
}

}  // namespace robot_model
