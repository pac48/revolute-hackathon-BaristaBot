// Quest Teleop: native OpenXR app (NativeActivity, OpenGL ES 3).
//
// Streams both controllers (grip pose, trigger, squeeze, thumbstick, buttons) and the head pose
// to the host over a WebSocket, and draws the gripper pose the host sends back.
//
// Frames: the app keeps a "robot base anchor" in the XR stage space. Every pose on the wire is
// in ROS convention, relative to that anchor (= the robot base frame). The anchor starts 0.5 m
// in front of and 0.5 m below the head, facing the way you face; click the LEFT thumbstick to
// move it to the left controller (position + heading).
//
// Host address: adb shell setprop debug.quest_teleop.host <ip>:<port>  (then restart the app).
// Default 127.0.0.1:9090, which reaches the PC over USB after `adb reverse tcp:9090 tcp:9090`.
//
// View stream: the left eye's image, downscaled and JPEG-encoded, goes to the host as binary
// messages (view_capture.h). debug.quest_teleop.view_width (default 720; 0 = off),
// debug.quest_teleop.view_fps (default 15) and debug.quest_teleop.view_quality (default 70).

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include "sources/android/native_app_glue/android_native_app_glue.h"  // @androidndk//:native_app_glue
#include <jni.h>
#include <sys/system_properties.h>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "src/teleop/quest_app/core/frames.h"
#include "src/teleop/quest_app/app/gl_renderer.h"
#include "src/teleop/quest_app/core/pose_math.h"
#include "src/teleop/quest_app/core/protocol.h"
#include "src/teleop/quest_app/core/teleop_link.h"
#include "src/teleop/quest_app/app/view_capture.h"

#define LOG_TAG "quest_teleop"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#define XR_CHECK(call)                                                        \
    do {                                                                      \
        XrResult r_ = (call);                                                 \
        if (XR_FAILED(r_)) {                                                  \
            LOGE("%s failed: %d (%s:%d)", #call, r_, __FILE__, __LINE__);     \
            return false;                                                     \
        }                                                                     \
    } while (0)

using namespace qt;

namespace {

constexpr float kNear = 0.05f, kFar = 50.0f;
constexpr auto kGripperStale = std::chrono::milliseconds(500);

Pose to_pose(const XrPosef& p) {
    return {{p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w},
            {p.position.x, p.position.y, p.position.z}};
}

// Maps ROS-local coordinates of an object to XR-local coordinates, so geometry can be authored
// in ROS axes (x forward, y left, z up) and drawn under an XR pose.
Mat4 ros_basis() {
    Mat4 m;
    Vec3 cols[3] = {ros_to_xr(Vec3{1, 0, 0}), ros_to_xr(Vec3{0, 1, 0}), ros_to_xr(Vec3{0, 0, 1})};
    for (int c = 0; c < 3; ++c) {
        m.m[c * 4 + 0] = cols[c].x;
        m.m[c * 4 + 1] = cols[c].y;
        m.m[c * 4 + 2] = cols[c].z;
    }
    return m;
}

int int_property(const char* name, int fallback) {
    char value[PROP_VALUE_MAX] = {};
    if (__system_property_get(name, value) <= 0) return fallback;
    char* end = nullptr;
    long v = std::strtol(value, &end, 10);
    return end != value ? static_cast<int>(v) : fallback;
}

std::string host_address() {
    char value[PROP_VALUE_MAX] = {};
    if (__system_property_get("debug.quest_teleop.host", value) > 0) return value;
    return "127.0.0.1:9090";
}

struct EyeSwapchain {
    XrSwapchain handle = XR_NULL_HANDLE;
    int width = 0, height = 0;
    std::vector<XrSwapchainImageOpenGLESKHR> images;
    std::vector<GLuint> framebuffers;
    GLuint depth = 0;
};

struct HandInput {
    XrPath path = XR_NULL_PATH;
    XrSpace grip_space = XR_NULL_HANDLE;
};

class App {
public:
    explicit App(android_app* app) : android_(app) {}
    ~App() { shutdown(); }

    bool init();
    void shutdown();
    void poll_events();
    void frame();

    bool session_running() const { return session_running_; }
    bool exit_requested() const { return exit_requested_; }
    bool resumed = false;

private:
    bool init_egl();
    bool init_instance();
    bool init_session();
    bool init_actions();
    bool init_swapchains();
    void handle_state(XrSessionState state);
    void update_input(XrTime time, protocol::ControllerState& out, Pose stage_hand[2], bool hand_valid[2]);
    void render_eye(int eye, const XrView& view, const Pose stage_hand[2], const bool hand_valid[2]);

    android_app* android_;
    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLConfig egl_config_ = nullptr;
    EGLContext egl_context_ = EGL_NO_CONTEXT;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;

    XrInstance instance_ = XR_NULL_HANDLE;
    XrSystemId system_ = XR_NULL_SYSTEM_ID;
    XrSession session_ = XR_NULL_HANDLE;
    XrSpace stage_ = XR_NULL_HANDLE;
    XrSpace view_space_ = XR_NULL_HANDLE;
    XrSessionState state_ = XR_SESSION_STATE_UNKNOWN;
    bool session_running_ = false;
    bool exit_requested_ = false;

    XrActionSet action_set_ = XR_NULL_HANDLE;
    XrAction grip_pose_ = XR_NULL_HANDLE, trigger_ = XR_NULL_HANDLE, squeeze_ = XR_NULL_HANDLE;
    XrAction stick_ = XR_NULL_HANDLE, stick_click_ = XR_NULL_HANDLE;
    XrAction lower_ = XR_NULL_HANDLE, upper_ = XR_NULL_HANDLE, menu_ = XR_NULL_HANDLE;
    HandInput hands_[2];  // 0 = left, 1 = right

    std::vector<XrViewConfigurationView> config_views_;
    std::vector<EyeSwapchain> eyes_;
    int64_t color_format_ = 0;

    GlRenderer renderer_;
    std::unique_ptr<TeleopLink> link_;
    ViewCapture view_;  // after link_: destroyed first, as its encoder thread sends through link_
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    uint64_t seq_ = 0;

    bool anchor_set_ = false;
    Pose anchor_;  // robot base anchor in stage space (XR axes, yaw only)
    bool prev_left_stick_click_ = false;
};

// ---- Setup -------------------------------------------------------------------------------

bool App::init_egl() {
    egl_display_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (!eglInitialize(egl_display_, nullptr, nullptr)) return false;
    const EGLint attribs[] = {EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8,
                              EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                              EGL_NONE};
    EGLint n = 0;
    if (!eglChooseConfig(egl_display_, attribs, &egl_config_, 1, &n) || n == 0) return false;
    const EGLint ctx_attribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    egl_context_ = eglCreateContext(egl_display_, egl_config_, EGL_NO_CONTEXT, ctx_attribs);
    if (egl_context_ == EGL_NO_CONTEXT) return false;
    // OpenXR renders into its own swapchain images; the surface only makes the context current.
    const EGLint pb[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
    egl_surface_ = eglCreatePbufferSurface(egl_display_, egl_config_, pb);
    return eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_);
}

bool App::init_instance() {
    PFN_xrInitializeLoaderKHR init_loader = nullptr;
    XR_CHECK(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                                   reinterpret_cast<PFN_xrVoidFunction*>(&init_loader)));
    XrLoaderInitInfoAndroidKHR loader_info{XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
    loader_info.applicationVM = android_->activity->vm;
    loader_info.applicationContext = android_->activity->clazz;
    XR_CHECK(init_loader(reinterpret_cast<XrLoaderInitInfoBaseHeaderKHR*>(&loader_info)));

    const char* extensions[] = {XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
                                XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME};
    XrInstanceCreateInfoAndroidKHR android_info{XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
    android_info.applicationVM = android_->activity->vm;
    android_info.applicationActivity = android_->activity->clazz;

    XrInstanceCreateInfo info{XR_TYPE_INSTANCE_CREATE_INFO};
    info.next = &android_info;
    std::strncpy(info.applicationInfo.applicationName, "Quest Teleop", XR_MAX_APPLICATION_NAME_SIZE - 1);
    info.applicationInfo.applicationVersion = 1;
    std::strncpy(info.applicationInfo.engineName, "none", XR_MAX_ENGINE_NAME_SIZE - 1);
    info.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34);  // what Quest 2 firmware supports
    info.enabledExtensionCount = 2;
    info.enabledExtensionNames = extensions;
    XR_CHECK(xrCreateInstance(&info, &instance_));

    XrSystemGetInfo sys{XR_TYPE_SYSTEM_GET_INFO};
    sys.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    XR_CHECK(xrGetSystem(instance_, &sys, &system_));

    XrSystemProperties props{XR_TYPE_SYSTEM_PROPERTIES};
    XR_CHECK(xrGetSystemProperties(instance_, system_, &props));
    LOGI("OpenXR system: %s", props.systemName);
    return true;
}

bool App::init_session() {
    // Required before xrCreateSession with an OpenGL ES binding.
    PFN_xrGetOpenGLESGraphicsRequirementsKHR get_reqs = nullptr;
    XR_CHECK(xrGetInstanceProcAddr(instance_, "xrGetOpenGLESGraphicsRequirementsKHR",
                                   reinterpret_cast<PFN_xrVoidFunction*>(&get_reqs)));
    XrGraphicsRequirementsOpenGLESKHR reqs{XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR};
    XR_CHECK(get_reqs(instance_, system_, &reqs));

    XrGraphicsBindingOpenGLESAndroidKHR binding{XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR};
    binding.display = egl_display_;
    binding.config = egl_config_;
    binding.context = egl_context_;
    XrSessionCreateInfo info{XR_TYPE_SESSION_CREATE_INFO};
    info.next = &binding;
    info.systemId = system_;
    XR_CHECK(xrCreateSession(instance_, &info, &session_));

    // Stage (floor-level, guardian-centered) if available, else local.
    uint32_t n = 0;
    XR_CHECK(xrEnumerateReferenceSpaces(session_, 0, &n, nullptr));
    std::vector<XrReferenceSpaceType> types(n);
    XR_CHECK(xrEnumerateReferenceSpaces(session_, n, &n, types.data()));
    XrReferenceSpaceType base_type = XR_REFERENCE_SPACE_TYPE_LOCAL;
    for (auto t : types)
        if (t == XR_REFERENCE_SPACE_TYPE_STAGE) base_type = t;
    LOGI("reference space: %s", base_type == XR_REFERENCE_SPACE_TYPE_STAGE ? "stage" : "local");

    XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
    space.poseInReferenceSpace.orientation.w = 1;
    space.referenceSpaceType = base_type;
    XR_CHECK(xrCreateReferenceSpace(session_, &space, &stage_));
    space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    XR_CHECK(xrCreateReferenceSpace(session_, &space, &view_space_));
    return true;
}

bool App::init_actions() {
    XrActionSetCreateInfo set_info{XR_TYPE_ACTION_SET_CREATE_INFO};
    std::strcpy(set_info.actionSetName, "teleop");
    std::strcpy(set_info.localizedActionSetName, "Teleop");
    XR_CHECK(xrCreateActionSet(instance_, &set_info, &action_set_));

    XR_CHECK(xrStringToPath(instance_, "/user/hand/left", &hands_[0].path));
    XR_CHECK(xrStringToPath(instance_, "/user/hand/right", &hands_[1].path));
    XrPath both[2] = {hands_[0].path, hands_[1].path};

    auto make = [&](XrAction& action, const char* name, XrActionType type, uint32_t n_sub,
                    const XrPath* sub) -> bool {
        XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO};
        info.actionType = type;
        std::strcpy(info.actionName, name);
        std::strcpy(info.localizedActionName, name);
        info.countSubactionPaths = n_sub;
        info.subactionPaths = sub;
        XR_CHECK(xrCreateAction(action_set_, &info, &action));
        return true;
    };
    if (!make(grip_pose_, "grip_pose", XR_ACTION_TYPE_POSE_INPUT, 2, both) ||
        !make(trigger_, "trigger", XR_ACTION_TYPE_FLOAT_INPUT, 2, both) ||
        !make(squeeze_, "squeeze", XR_ACTION_TYPE_FLOAT_INPUT, 2, both) ||
        !make(stick_, "thumbstick", XR_ACTION_TYPE_VECTOR2F_INPUT, 2, both) ||
        !make(stick_click_, "thumbstick_click", XR_ACTION_TYPE_BOOLEAN_INPUT, 2, both) ||
        !make(lower_, "lower_button", XR_ACTION_TYPE_BOOLEAN_INPUT, 2, both) ||
        !make(upper_, "upper_button", XR_ACTION_TYPE_BOOLEAN_INPUT, 2, both) ||
        !make(menu_, "menu", XR_ACTION_TYPE_BOOLEAN_INPUT, 1, &hands_[0].path))
        return false;

    std::vector<XrActionSuggestedBinding> bindings;
    auto bind = [&](XrAction action, const char* path) -> bool {
        XrPath p;
        XR_CHECK(xrStringToPath(instance_, path, &p));
        bindings.push_back({action, p});
        return true;
    };
    bool ok = bind(grip_pose_, "/user/hand/left/input/grip/pose") &&
              bind(grip_pose_, "/user/hand/right/input/grip/pose") &&
              bind(trigger_, "/user/hand/left/input/trigger/value") &&
              bind(trigger_, "/user/hand/right/input/trigger/value") &&
              bind(squeeze_, "/user/hand/left/input/squeeze/value") &&
              bind(squeeze_, "/user/hand/right/input/squeeze/value") &&
              bind(stick_, "/user/hand/left/input/thumbstick") &&
              bind(stick_, "/user/hand/right/input/thumbstick") &&
              bind(stick_click_, "/user/hand/left/input/thumbstick/click") &&
              bind(stick_click_, "/user/hand/right/input/thumbstick/click") &&
              bind(lower_, "/user/hand/left/input/x/click") &&
              bind(lower_, "/user/hand/right/input/a/click") &&
              bind(upper_, "/user/hand/left/input/y/click") &&
              bind(upper_, "/user/hand/right/input/b/click") &&
              bind(menu_, "/user/hand/left/input/menu/click");
    if (!ok) return false;

    XrPath profile;
    XR_CHECK(xrStringToPath(instance_, "/interaction_profiles/oculus/touch_controller", &profile));
    XrInteractionProfileSuggestedBinding suggested{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
    suggested.interactionProfile = profile;
    suggested.suggestedBindings = bindings.data();
    suggested.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
    XR_CHECK(xrSuggestInteractionProfileBindings(instance_, &suggested));

    for (auto& hand : hands_) {
        XrActionSpaceCreateInfo info{XR_TYPE_ACTION_SPACE_CREATE_INFO};
        info.action = grip_pose_;
        info.subactionPath = hand.path;
        info.poseInActionSpace.orientation.w = 1;
        XR_CHECK(xrCreateActionSpace(session_, &info, &hand.grip_space));
    }

    XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
    attach.countActionSets = 1;
    attach.actionSets = &action_set_;
    XR_CHECK(xrAttachSessionActionSets(session_, &attach));
    return true;
}

bool App::init_swapchains() {
    uint32_t n = 0;
    XR_CHECK(xrEnumerateViewConfigurationViews(instance_, system_, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                               0, &n, nullptr));
    config_views_.assign(n, {XR_TYPE_VIEW_CONFIGURATION_VIEW});
    XR_CHECK(xrEnumerateViewConfigurationViews(instance_, system_, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                               n, &n, config_views_.data()));

    uint32_t nf = 0;
    XR_CHECK(xrEnumerateSwapchainFormats(session_, 0, &nf, nullptr));
    std::vector<int64_t> formats(nf);
    XR_CHECK(xrEnumerateSwapchainFormats(session_, nf, &nf, formats.data()));
    color_format_ = formats.empty() ? GL_RGBA8 : formats[0];
    for (int64_t f : formats)
        if (f == GL_SRGB8_ALPHA8) color_format_ = f;  // linear shading, sRGB encode on write

    eyes_.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        EyeSwapchain& eye = eyes_[i];
        eye.width = static_cast<int>(config_views_[i].recommendedImageRectWidth);
        eye.height = static_cast<int>(config_views_[i].recommendedImageRectHeight);
        XrSwapchainCreateInfo info{XR_TYPE_SWAPCHAIN_CREATE_INFO};
        info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
        info.format = color_format_;
        info.sampleCount = 1;
        info.width = static_cast<uint32_t>(eye.width);
        info.height = static_cast<uint32_t>(eye.height);
        info.faceCount = 1;
        info.arraySize = 1;
        info.mipCount = 1;
        XR_CHECK(xrCreateSwapchain(session_, &info, &eye.handle));

        uint32_t ni = 0;
        XR_CHECK(xrEnumerateSwapchainImages(eye.handle, 0, &ni, nullptr));
        eye.images.assign(ni, {XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR});
        XR_CHECK(xrEnumerateSwapchainImages(eye.handle, ni, &ni,
                                            reinterpret_cast<XrSwapchainImageBaseHeader*>(eye.images.data())));

        glGenRenderbuffers(1, &eye.depth);
        glBindRenderbuffer(GL_RENDERBUFFER, eye.depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, eye.width, eye.height);
        eye.framebuffers.resize(ni);
        glGenFramebuffers(static_cast<GLsizei>(ni), eye.framebuffers.data());
        for (uint32_t k = 0; k < ni; ++k) {
            glBindFramebuffer(GL_FRAMEBUFFER, eye.framebuffers[k]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, eye.images[k].image, 0);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, eye.depth);
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                LOGE("framebuffer incomplete");
                return false;
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        LOGI("eye %u: %dx%d, %u images", i, eye.width, eye.height, ni);
    }
    return true;
}

bool App::init() {
    if (!init_egl()) {
        LOGE("EGL init failed");
        return false;
    }
    if (!init_instance() || !init_session() || !init_actions() || !init_swapchains()) return false;
    if (!renderer_.init()) return false;

    std::string addr = host_address();
    WsUrl url;
    if (!parse_ws_url(addr, url)) {
        LOGE("bad debug.quest_teleop.host '%s', using 127.0.0.1:9090", addr.c_str());
        parse_ws_url("127.0.0.1:9090", url);
    }
    LOGI("host: ws://%s:%u%s", url.host.c_str(), url.port, url.path.c_str());
    link_ = std::make_unique<TeleopLink>(url, "quest");
    link_->start();

    const int view_width = int_property("debug.quest_teleop.view_width", 720);
    if (view_width > 0 && !eyes_.empty()) {
        TeleopLink* link = link_.get();
        view_.init(eyes_[0].width, eyes_[0].height, view_width, int_property("debug.quest_teleop.view_fps", 15),
                   int_property("debug.quest_teleop.view_quality", 70),
                   [link](std::string jpeg, int w, int h, double t) {
                       link->publish_view(protocol::encode_view(jpeg, w, h, t));
                   });
    }
    return true;
}

void App::shutdown() {
    view_.shutdown();  // needs the GL context, and sends through link_
    if (link_) link_->stop();
    link_.reset();
    renderer_.shutdown();
    for (auto& eye : eyes_) {
        if (!eye.framebuffers.empty())
            glDeleteFramebuffers(static_cast<GLsizei>(eye.framebuffers.size()), eye.framebuffers.data());
        if (eye.depth) glDeleteRenderbuffers(1, &eye.depth);
        if (eye.handle) xrDestroySwapchain(eye.handle);
    }
    eyes_.clear();
    for (auto& hand : hands_)
        if (hand.grip_space) xrDestroySpace(hand.grip_space), hand.grip_space = XR_NULL_HANDLE;
    if (action_set_) xrDestroyActionSet(action_set_), action_set_ = XR_NULL_HANDLE;
    if (view_space_) xrDestroySpace(view_space_), view_space_ = XR_NULL_HANDLE;
    if (stage_) xrDestroySpace(stage_), stage_ = XR_NULL_HANDLE;
    if (session_) xrDestroySession(session_), session_ = XR_NULL_HANDLE;
    if (instance_) xrDestroyInstance(instance_), instance_ = XR_NULL_HANDLE;
    if (egl_display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(egl_display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (egl_surface_ != EGL_NO_SURFACE) eglDestroySurface(egl_display_, egl_surface_);
        if (egl_context_ != EGL_NO_CONTEXT) eglDestroyContext(egl_display_, egl_context_);
        eglTerminate(egl_display_);
        egl_display_ = EGL_NO_DISPLAY;
    }
}

// ---- Session lifecycle -------------------------------------------------------------------

void App::handle_state(XrSessionState state) {
    state_ = state;
    LOGI("session state %d", state);
    switch (state) {
        case XR_SESSION_STATE_READY: {
            XrSessionBeginInfo info{XR_TYPE_SESSION_BEGIN_INFO};
            info.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            if (XR_SUCCEEDED(xrBeginSession(session_, &info))) session_running_ = true;
            break;
        }
        case XR_SESSION_STATE_STOPPING:
            xrEndSession(session_);
            session_running_ = false;
            break;
        case XR_SESSION_STATE_EXITING:
        case XR_SESSION_STATE_LOSS_PENDING:
            exit_requested_ = true;
            break;
        default:
            break;
    }
}

void App::poll_events() {
    XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(instance_, &event) == XR_SUCCESS) {
        switch (event.type) {
            case XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED:
                handle_state(reinterpret_cast<XrEventDataSessionStateChanged&>(event).state);
                break;
            case XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING:
                exit_requested_ = true;
                break;
            default:
                break;
        }
        event = {XR_TYPE_EVENT_DATA_BUFFER};
    }
}

// ---- Per frame ---------------------------------------------------------------------------

void App::update_input(XrTime time, protocol::ControllerState& out, Pose stage_hand[2], bool hand_valid[2]) {
    hand_valid[0] = hand_valid[1] = false;
    out.seq = ++seq_;
    out.t = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();

    XrSpaceLocation head{XR_TYPE_SPACE_LOCATION};
    bool head_ok = XR_SUCCEEDED(xrLocateSpace(view_space_, stage_, time, &head)) &&
                   (head.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
                   (head.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT);
    Pose stage_head = head_ok ? to_pose(head.pose) : Pose{};

    if (!anchor_set_ && head_ok) {
        Quat yaw = yaw_only_xr(stage_head.q);
        anchor_.q = yaw;
        anchor_.p = stage_head.p + rotate(yaw, {0, -0.5f, -0.5f});
        anchor_set_ = true;
    }
    Pose anchor_inv = inverse(anchor_);
    if (head_ok && anchor_set_) {
        out.head_valid = true;
        out.head = xr_to_ros(anchor_inv * stage_head);
    }

    // Only the focused app gets input; otherwise report both hands inactive.
    if (state_ != XR_SESSION_STATE_FOCUSED) return;
    XrActiveActionSet active{action_set_, XR_NULL_PATH};
    XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
    sync.countActiveActionSets = 1;
    sync.activeActionSets = &active;
    if (XR_FAILED(xrSyncActions(session_, &sync))) return;

    for (int h = 0; h < 2; ++h) {
        protocol::Hand& hand = h == 0 ? out.left : out.right;
        XrActionStateGetInfo get{XR_TYPE_ACTION_STATE_GET_INFO};
        get.subactionPath = hands_[h].path;

        XrActionStateFloat f{XR_TYPE_ACTION_STATE_FLOAT};
        get.action = trigger_;
        if (XR_SUCCEEDED(xrGetActionStateFloat(session_, &get, &f)) && f.isActive) hand.trigger = f.currentState;
        get.action = squeeze_;
        f = {XR_TYPE_ACTION_STATE_FLOAT};
        if (XR_SUCCEEDED(xrGetActionStateFloat(session_, &get, &f)) && f.isActive) hand.grip = f.currentState;

        XrActionStateVector2f v{XR_TYPE_ACTION_STATE_VECTOR2F};
        get.action = stick_;
        if (XR_SUCCEEDED(xrGetActionStateVector2f(session_, &get, &v)) && v.isActive) {
            hand.stick_x = v.currentState.x;
            hand.stick_y = v.currentState.y;
        }
        auto boolean = [&](XrAction action) {
            XrActionStateBoolean b{XR_TYPE_ACTION_STATE_BOOLEAN};
            get.action = action;
            return XR_SUCCEEDED(xrGetActionStateBoolean(session_, &get, &b)) && b.isActive && b.currentState;
        };
        hand.stick_click = boolean(stick_click_);
        hand.button_lower = boolean(lower_);
        hand.button_upper = boolean(upper_);
        if (h == 0) hand.menu = boolean(menu_);

        XrSpaceLocation loc{XR_TYPE_SPACE_LOCATION};
        if (XR_SUCCEEDED(xrLocateSpace(hands_[h].grip_space, stage_, time, &loc)) &&
            (loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) &&
            (loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT)) {
            stage_hand[h] = to_pose(loc.pose);
            hand_valid[h] = true;
        }
    }

    // Left thumbstick click (rising edge): put the robot base at the left controller.
    if (out.left.stick_click && !prev_left_stick_click_ && hand_valid[0]) {
        anchor_.q = yaw_only_xr(stage_hand[0].q);
        anchor_.p = stage_hand[0].p;
        anchor_inv = inverse(anchor_);
        LOGI("robot base anchor moved to left controller");
    }
    prev_left_stick_click_ = out.left.stick_click;

    for (int h = 0; h < 2; ++h) {
        protocol::Hand& hand = h == 0 ? out.left : out.right;
        hand.active = hand_valid[h];
        if (hand_valid[h]) hand.pose = xr_to_ros(anchor_inv * stage_hand[h]);
    }
    if (out.head_valid) out.head = xr_to_ros(anchor_inv * stage_head);
}

void App::render_eye(int i, const XrView& view, const Pose stage_hand[2], const bool hand_valid[2]) {
    EyeSwapchain& eye = eyes_[i];
    uint32_t image = 0;
    XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
    if (XR_FAILED(xrAcquireSwapchainImage(eye.handle, &acquire, &image))) return;
    XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
    wait.timeout = XR_INFINITE_DURATION;
    xrWaitSwapchainImage(eye.handle, &wait);

    Mat4 proj = projection_from_fov(view.fov.angleLeft, view.fov.angleRight, view.fov.angleUp,
                                    view.fov.angleDown, kNear, kFar);
    Mat4 view_m = to_matrix(inverse(to_pose(view.pose)));
    renderer_.begin_eye(eye.framebuffers[image], eye.width, eye.height, proj * view_m);

    // Floor grid, 4 x 4 m in 0.5 m steps (stage y = 0).
    Mat4 identity;
    for (int k = -4; k <= 4; ++k) {
        float c = k * 0.5f;
        renderer_.box(identity, {c, 0, 0}, {0.005f, 0.002f, 4.0f}, {0.25f, 0.27f, 0.3f});
        renderer_.box(identity, {0, 0, c}, {4.0f, 0.002f, 0.005f}, {0.25f, 0.27f, 0.3f});
    }

    const Mat4 basis = ros_basis();
    auto status = link_->status();

    // Robot base: axes plus a status cube (green = connected to host, red = not).
    Mat4 base_m = to_matrix(anchor_) * basis;
    renderer_.axes(base_m, 0.15f, 0.008f);
    renderer_.box(base_m, {0, 0, 0}, {0.04f, 0.04f, 0.04f},
                  status.connected ? Color{0.1f, 0.8f, 0.2f} : Color{0.85f, 0.1f, 0.1f});

    // Controllers: axes of the pose that is sent (ROS convention, x = pointing direction).
    for (int h = 0; h < 2; ++h) {
        if (!hand_valid[h]) continue;
        Mat4 m = to_matrix(stage_hand[h]) * basis;
        renderer_.axes(m, 0.08f, 0.006f);
    }

    // Gripper from the host, in robot base frame (ROS). Approach axis is +x of the tool frame;
    // drawn orange when fresh, grey when older than kGripperStale.
    if (auto g = link_->gripper_pose()) {
        bool fresh = std::chrono::steady_clock::now() - g->second < kGripperStale;
        Mat4 m = to_matrix(anchor_ * ros_to_xr(g->first.pose)) * basis;
        Color c = fresh ? Color{1.0f, 0.55f, 0.1f} : Color{0.45f, 0.45f, 0.45f};
        renderer_.box(m, {-0.06f, 0, 0}, {0.06f, 0.09f, 0.04f}, c);         // palm
        renderer_.box(m, {-0.015f, 0.035f, 0}, {0.05f, 0.012f, 0.02f}, c);   // finger
        renderer_.box(m, {-0.015f, -0.035f, 0}, {0.05f, 0.012f, 0.02f}, c);  // finger
        renderer_.axes(m, 0.06f, 0.005f);
    }

    if (i == 0) {
        view_.frame(eye.framebuffers[image],
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count());
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
    xrReleaseSwapchainImage(eye.handle, &release);
}

void App::frame() {
    XrFrameState fs{XR_TYPE_FRAME_STATE};
    XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
    if (XR_FAILED(xrWaitFrame(session_, &wait, &fs))) return;
    XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
    if (XR_FAILED(xrBeginFrame(session_, &begin))) return;

    protocol::ControllerState state;
    Pose stage_hand[2];
    bool hand_valid[2] = {false, false};
    update_input(fs.predictedDisplayTime, state, stage_hand, hand_valid);
    if (anchor_set_) link_->publish(state);

    std::vector<XrCompositionLayerProjectionView> proj_views;
    XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
    const XrCompositionLayerBaseHeader* layers[1] = {};
    uint32_t layer_count = 0;

    if (fs.shouldRender) {
        XrViewState vs{XR_TYPE_VIEW_STATE};
        XrViewLocateInfo li{XR_TYPE_VIEW_LOCATE_INFO};
        li.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        li.displayTime = fs.predictedDisplayTime;
        li.space = stage_;
        std::vector<XrView> views(eyes_.size(), {XR_TYPE_VIEW});
        uint32_t nv = 0;
        if (XR_SUCCEEDED(xrLocateViews(session_, &li, &vs, static_cast<uint32_t>(views.size()), &nv, views.data())) &&
            (vs.viewStateFlags & XR_VIEW_STATE_POSITION_VALID_BIT) &&
            (vs.viewStateFlags & XR_VIEW_STATE_ORIENTATION_VALID_BIT)) {
            proj_views.resize(nv, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW});
            for (uint32_t i = 0; i < nv; ++i) {
                render_eye(static_cast<int>(i), views[i], stage_hand, hand_valid);
                proj_views[i].pose = views[i].pose;
                proj_views[i].fov = views[i].fov;
                proj_views[i].subImage.swapchain = eyes_[i].handle;
                proj_views[i].subImage.imageRect = {{0, 0}, {eyes_[i].width, eyes_[i].height}};
            }
            layer.space = stage_;
            layer.viewCount = nv;
            layer.views = proj_views.data();
            layers[0] = reinterpret_cast<XrCompositionLayerBaseHeader*>(&layer);
            layer_count = 1;
        }
    }

    XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
    end.displayTime = fs.predictedDisplayTime;
    end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    end.layerCount = layer_count;
    end.layers = layers;
    xrEndFrame(session_, &end);
}

void on_app_cmd(android_app* app, int32_t cmd) {
    auto* self = static_cast<App*>(app->userData);
    if (!self) return;
    switch (cmd) {
        case APP_CMD_RESUME: self->resumed = true; break;
        case APP_CMD_PAUSE: self->resumed = false; break;
        default: break;
    }
}

}  // namespace

void android_main(android_app* app) {
    JNIEnv* env = nullptr;
    app->activity->vm->AttachCurrentThread(&env, nullptr);

    {
        App xr(app);
        app->userData = &xr;
        app->onAppCmd = on_app_cmd;

        if (!xr.init()) {
            LOGE("initialization failed; exiting");
            ANativeActivity_finish(app->activity);
        } else {
            while (!app->destroyRequested) {
                // Block on Android events only while there is nothing to render.
                while (true) {
                    int timeout = (!xr.resumed && !xr.session_running() && !app->destroyRequested) ? -1 : 0;
                    android_poll_source* source = nullptr;
                    if (ALooper_pollOnce(timeout, nullptr, nullptr, reinterpret_cast<void**>(&source)) < 0) break;
                    if (source) source->process(app, source);
                }
                xr.poll_events();
                if (xr.exit_requested()) {
                    ANativeActivity_finish(app->activity);
                    break;
                }
                if (xr.session_running()) xr.frame();
            }
        }
        // Drain remaining events so the activity can finish cleanly.
        while (!app->destroyRequested) {
            android_poll_source* source = nullptr;
            if (ALooper_pollOnce(-1, nullptr, nullptr, reinterpret_cast<void**>(&source)) >= 0 && source)
                source->process(app, source);
        }
        app->userData = nullptr;
    }
    app->activity->vm->DetachCurrentThread();
}
