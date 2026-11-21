#include "src/robot_model/robot_model_utils/renderer.hpp"

#include <ovrtx/ovrtx.h>
#include <ovrtx/ovrtx_config.h>
#include <ovrtx/ovrtx_types.h>
#include <ovstage/ovstage.h>

#include "src/robot_model/robot_model_utils/mujoco_model.hpp"

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace robot_model
{
namespace
{

constexpr char kRenderVar[] = "/Render/Vars/LdrColor";
constexpr double kStepSeconds = 1.0 / 60.0;

ovx_string_t ovx(const std::string& s) { return {s.data(), s.size()}; }

[[noreturn]] void fail(const std::string& what, ovx_string_t detail)
{
  std::string message = "ovrtx: " + what;
  if (detail.ptr && detail.length > 0) {
    message += ": " + std::string(detail.ptr, detail.length);
  }
  throw std::runtime_error(message);
}

void check(const ovrtx_result_t& result, const char* what)
{
  if (result.status == OVRTX_API_ERROR) {
    fail(what, ovrtx_get_last_error());
  }
}

void check_enqueue(const ovrtx_enqueue_result_t& result, const char* what)
{
  if (result.status == OVRTX_API_ERROR) {
    fail(what, ovrtx_get_last_error());
  }
}

void check_stage(ovstage_api_status_t status, ovstage_instance_t* stage, const char* what)
{
  if (status != OVSTAGE_OK) {
    ovx_string_t detail = ovstage_population_get_last_error();
    if ((!detail.ptr || detail.length == 0) && stage) {
      detail = ovstage_get_last_error();
    }
    fail(std::string(what) + " (ovstage status " + std::to_string(static_cast<int>(status)) + ")", detail);
  }
}

// Directory `bin/` of an SDK package: the parent of its marker file in runfiles.
std::string package_bin_dir(const std::string& marker_rlocation)
{
  return std::filesystem::path(runfile_path(marker_rlocation)).parent_path().string();
}

}  // namespace

struct Renderer::Impl
{
  ovrtx_renderer_t* renderer = nullptr;
  ovstage_instance_t* stage = nullptr;
  bool stage_initialized = false;
  bool attached = false;

  ovstage_ordinal_t ordinal = 0;  // last ordinal committed with the write floor

  std::vector<int> bodies;                 // MuJoCo body ids with a prim, in prim order
  ovstage_query_handle_t query = OVSTAGE_INVALID_QUERY_HANDLE;
  ovx_token_t xform_token{};
  std::vector<double> matrices;            // 16 per body, USD row-vector convention
  const mjModel* model = nullptr;

  ~Impl()
  {
    // Teardown never throws; failures here have nowhere to go.
    if (query != OVSTAGE_INVALID_QUERY_HANDLE && stage) {
      ovstage_release_query(stage, query);
    }
    if (attached) {
      ovrtx_detach_ovstage(renderer);
    }
    if (renderer) {
      ovrtx_destroy_renderer(renderer);
    }
    if (stage) {
      ovstage_destroy_instance(stage);
    }
    if (stage_initialized) {
      ovstage_shutdown();
    }
  }

  void wait_stage_op(const ovstage_enqueue_result_t& enqueue, const char* what)
  {
    check_stage(enqueue.status, stage, what);
    ovstage_op_wait_result_t wait{};
    const ovstage_api_status_t status = ovstage_wait_op(stage, enqueue.op_index, OVSTAGE_TIMEOUT_INFINITE, &wait);
    ovstage_release_op(stage, enqueue.op_index);
    check_stage(status, stage, what);
    if (wait.error_op_id_count != 0) {
      fail(what, ovstage_get_last_op_error(stage, wait.error_op_ids[0]));
    }
  }

  void wait_population_op(const ovstage_population_enqueue_result_t& enqueue, const char* what)
  {
    check_stage(enqueue.status, stage, what);
    ovstage_population_op_wait_result_t wait{};
    check_stage(ovstage_population_wait_op(stage, enqueue.op_index, OVSTAGE_TIMEOUT_INFINITE, &wait), stage, what);
    if (wait.error_op_id_count != 0) {
      fail(what, ovstage_population_get_last_op_error(wait.error_op_ids[0]));
    }
  }

  void commit(ovstage_ordinal_t at)
  {
    ovstage_write_floor_desc_t floor{};
    floor.ordinal = at;
    floor.scope = OVSTAGE_SCOPE_ALL;
    wait_stage_op(ovstage_advance_write_floor(stage, &floor), "advance_write_floor");
    ordinal = at;
  }
};

Renderer::Renderer(const mjModel* model, const std::string& usd_path) : impl_(std::make_unique<Impl>())
{
  Impl& s = *impl_;
  s.model = model;

  // The static loaders open the runtime from the package's bin/ directory.
  const std::string ovrtx_bin = package_bin_dir(OVRTX_RUNTIME_MARKER);
  const std::string ovstage_bin = package_bin_dir(OVSTAGE_RUNTIME_MARKER);

  ovrtx_config_entry_t entries[] = {ovrtx_config_entry_binary_package_root_path(ovx(ovrtx_bin))};
  ovrtx_config_t config{};
  config.entries = entries;
  config.entry_count = sizeof(entries) / sizeof(entries[0]);
  check(ovrtx_register_schema_paths(&config), "register_schema_paths");
  check(ovrtx_create_renderer(&config, &s.renderer), "create_renderer");

  ovstage_config_entry_t stage_entries[] = {ovstage_config_entry_binary_package_root_path(ovx(ovstage_bin))};
  ovstage_config_t stage_config{};
  stage_config.entries = stage_entries;
  stage_config.entry_count = sizeof(stage_entries) / sizeof(stage_entries[0]);
  check_stage(ovstage_initialize(&stage_config), nullptr, "ovstage_initialize");
  s.stage_initialized = true;

  ovstage_instance_desc_t desc{};
  desc.name = "robot_model_renderer";
  check_stage(ovstage_create_instance(&desc, &s.stage), nullptr, "create_instance");
  check(ovrtx_attach_ovstage(s.renderer, s.stage), "attach_ovstage");
  s.attached = true;

  s.ordinal = 1;
  s.wait_population_op(
    ovstage_population_open_usd_from_file(
      s.stage, ovx(usd_path), s.ordinal, /*time=*/0.0, OVSTAGE_POPULATION_DOMAIN_RENDERING),
    "open_usd");
  s.commit(s.ordinal);

  // One query over the prims of the model's bodies: /World/<body name>, in body order.
  std::vector<std::string> paths;
  for (int b = 1; b < model->nbody; ++b) {
    const char* name = mj_id2name(model, mjOBJ_BODY, b);
    if (name && *name) {
      s.bodies.push_back(b);
      paths.push_back(std::string("/World/") + name);
    }
  }
  std::vector<ovx_string_t> path_strings;
  for (const std::string& p : paths) {
    path_strings.push_back(ovx(p));
  }
  path_dictionary_instance_t* dict = ovstage_get_path_dictionary(s.stage);
  ovx_primpath_list_t list{};
  if (path_dictionary_create_path_list_from_strings(dict, path_strings.data(), path_strings.size(), &list).status !=
      OVX_API_SUCCESS) {
    throw std::runtime_error("ovrtx: could not intern the body prim paths");
  }
  const ovstage_api_status_t query_status = ovstage_query_from_path_list(s.stage, list, &s.query);
  path_dictionary_release_path_list_reference(dict, list);
  check_stage(query_status, s.stage, "query_from_path_list");

  const std::string attribute = "omni:xform";
  ovx_string_t attribute_string = ovx(attribute);
  if (path_dictionary_create_tokens_from_strings(dict, &attribute_string, 1, &s.xform_token).status !=
      OVX_API_SUCCESS) {
    throw std::runtime_error("ovrtx: could not intern omni:xform");
  }
  s.matrices.assign(16 * s.bodies.size(), 0.0);
}

Renderer::~Renderer() = default;

void Renderer::update(const mjData* data)
{
  Impl& s = *impl_;
  for (std::size_t i = 0; i < s.bodies.size(); ++i) {
    const int b = s.bodies[i];
    const mjtNum* rot = data->xmat + 9 * b;  // row-major, rotation of the body in the world
    const mjtNum* pos = data->xpos + 3 * b;
    double* m = &s.matrices[16 * i];
    // USD matrices act on row vectors: the rotation is transposed and the translation is the last row.
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        m[4 * r + c] = rot[3 * c + r];
      }
      m[4 * r + 3] = 0.0;
    }
    m[12] = pos[0];
    m[13] = pos[1];
    m[14] = pos[2];
    m[15] = 1.0;
  }

  int64_t shape[1] = {static_cast<int64_t>(s.bodies.size())};
  DLTensor tensor{};
  tensor.data = s.matrices.data();
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = {kDLFloat, 64, 16};
  tensor.shape = shape;

  ovstage_write_data_t write{};
  write.tensors = &tensor;
  write.tensor_count = 1;
  write.is_array = false;
  write.semantic = OVSTAGE_SEMANTIC_MATRIX;
  ovx_token_or_string_t attribute{};
  attribute.token = s.xform_token;

  const ovstage_ordinal_t at = s.ordinal + 1;
  s.wait_stage_op(
    ovstage_write_attribute(s.stage, s.query, attribute, at, write, OVSTAGE_PRIM_MODE_UPSERT), "write omni:xform");
  s.commit(at);
}

Image Renderer::render(const std::string& camera)
{
  Impl& s = *impl_;
  const std::string product = "/Render/" + camera;
  ovx_string_t product_string = ovx(product);
  ovrtx_render_product_set_t products{};
  products.render_products = &product_string;
  products.num_render_products = 1;

  check_enqueue(ovrtx_update_from_stage(s.renderer, s.ordinal), "update_from_stage");
  ovrtx_step_result_handle_t step{};
  const ovrtx_enqueue_result_t enqueue = ovrtx_step_with_stage(s.renderer, products, kStepSeconds, s.ordinal, &step);
  check_enqueue(enqueue, "step");

  ovrtx_op_wait_result_t wait{};
  check(ovrtx_wait_op(s.renderer, enqueue.op_index, ovrtx_timeout_infinite, &wait), "wait_op");

  // Whatever happens from here, the step's results are released.
  struct Release
  {
    ovrtx_renderer_t* renderer;
    ovrtx_step_result_handle_t step;
    ~Release() { ovrtx_destroy_results(renderer, step); }
  } release{s.renderer, step};

  ovrtx_render_product_set_outputs_t outputs{};
  check(ovrtx_fetch_results(s.renderer, step, ovrtx_timeout_infinite, &outputs), "fetch_results");

  ovrtx_render_var_output_handle_t handle = OVRTX_INVALID_RENDER_VAR_OUTPUT_HANDLE;
  for (std::size_t i = 0; i < outputs.output_count && handle == OVRTX_INVALID_RENDER_VAR_OUTPUT_HANDLE; ++i) {
    const ovrtx_render_product_output_t& out = outputs.outputs[i];
    for (std::size_t f = 0; f < out.output_frame_count; ++f) {
      const ovrtx_render_product_frame_output_t& frame = out.output_frames[f];
      for (std::size_t v = 0; v < frame.render_var_count; ++v) {
        const auto& var = frame.output_render_vars[v];
        if (var.render_var_path.ptr && std::string_view(var.render_var_path.ptr, var.render_var_path.length) == kRenderVar) {
          handle = var.output_handle;
        }
      }
    }
  }
  if (handle == OVRTX_INVALID_RENDER_VAR_OUTPUT_HANDLE) {
    throw std::runtime_error("ovrtx: no LdrColor output for the camera '" + camera + "' (is there a RenderProduct " + product + "?)");
  }

  ovrtx_map_output_description_t map_desc{};
  map_desc.device_type = OVRTX_MAP_DEVICE_TYPE_CPU;
  ovrtx_render_var_output_t mapped{};
  check(ovrtx_map_render_var_output(s.renderer, handle, &map_desc, ovrtx_timeout_infinite, &mapped), "map_render_var_output");
  struct Unmap
  {
    ovrtx_renderer_t* renderer;
    ovrtx_render_var_output_map_handle_t map;
    ~Unmap()
    {
      ovrtx_cuda_sync_t none{};
      ovrtx_unmap_render_var_output(renderer, map, none);
    }
  } unmap{s.renderer, mapped.map_handle};

  if (mapped.num_tensors != 1) {
    throw std::runtime_error("ovrtx: unexpected LdrColor output (" + std::to_string(mapped.num_tensors) + " tensors)");
  }
  const DLTensor& tensor = *mapped.tensors[0].dl;
  if (tensor.ndim != 3 || !tensor.shape || tensor.shape[2] != 4 || tensor.dtype.lanes != 1 || tensor.dtype.bits != 8) {
    throw std::runtime_error("ovrtx: unexpected LdrColor layout (expected uint8 [H, W, 4])");
  }

  Image image;
  image.height = static_cast<int>(tensor.shape[0]);
  image.width = static_cast<int>(tensor.shape[1]);
  image.rgb.resize(static_cast<std::size_t>(image.width) * image.height * 3);
  const auto* rgba = static_cast<const std::uint8_t*>(tensor.data);
  for (std::size_t p = 0, n = static_cast<std::size_t>(image.width) * image.height; p < n; ++p) {
    image.rgb[3 * p + 0] = rgba[4 * p + 0];
    image.rgb[3 * p + 1] = rgba[4 * p + 1];
    image.rgb[3 * p + 2] = rgba[4 * p + 2];
  }
  return image;
}

}  // namespace robot_model
