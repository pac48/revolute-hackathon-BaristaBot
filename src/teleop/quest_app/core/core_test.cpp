// Unit tests for the headset core. No framework: each CHECK prints the failure and the
// process exits non-zero if any failed.
#include <cmath>
#include <cstdio>
#include <cstring>

#include "src/teleop/quest_app/core/frames.h"
#include "src/teleop/quest_app/core/json.h"
#include "src/teleop/quest_app/core/pose_math.h"
#include "src/teleop/quest_app/core/protocol.h"
#include "src/teleop/quest_app/core/websocket_client.h"

using namespace qt;

static int failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                \
        }                                                              \
    } while (0)

static bool near(float a, float b, float tol = 1e-4f) { return std::fabs(a - b) < tol; }
static bool near(Vec3 a, Vec3 b, float tol = 1e-4f) {
    return near(a.x, b.x, tol) && near(a.y, b.y, tol) && near(a.z, b.z, tol);
}
static bool near_rot(Quat a, Quat b) {  // q and -q are the same rotation
    float d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    return std::fabs(std::fabs(d) - 1) < 1e-4f;
}

static void test_math() {
    Pose a{axis_angle({0, 0, 1}, 0.7f), {1, 2, 3}};
    Pose b{axis_angle({1, 0, 0}, -0.3f), {-0.5f, 0.2f, 0.1f}};
    Pose id = a * inverse(a);
    CHECK(near(id.p, {0, 0, 0}));
    CHECK(near_rot(id.q, {}));
    // Composition matches applying one after the other.
    Vec3 v{0.3f, -0.1f, 0.8f};
    Pose ab = a * b;
    CHECK(near(ab.p + rotate(ab.q, v), a.p + rotate(a.q, b.p + rotate(b.q, v))));
    // Matrix agrees with the pose.
    Mat4 m = to_matrix(a);
    Vec3 r = a.p + rotate(a.q, v);
    Vec3 mv{m.m[0] * v.x + m.m[4] * v.y + m.m[8] * v.z + m.m[12],
            m.m[1] * v.x + m.m[5] * v.y + m.m[9] * v.z + m.m[13],
            m.m[2] * v.x + m.m[6] * v.y + m.m[10] * v.z + m.m[14]};
    CHECK(near(r, mv));
    // Matrix product agrees with pose product.
    Mat4 mab = to_matrix(a) * to_matrix(b), pab = to_matrix(ab);
    bool same = true;
    for (int i = 0; i < 16; ++i) same &= near(mab.m[i], pab.m[i]);
    CHECK(same);
}

static void test_frames() {
    // XR forward (-z) is ROS +x; XR up (+y) is ROS +z; XR right (+x) is ROS -y.
    CHECK(near(xr_to_ros(Vec3{0, 0, -1}), {1, 0, 0}));
    CHECK(near(xr_to_ros(Vec3{0, 1, 0}), {0, 0, 1}));
    CHECK(near(xr_to_ros(Vec3{1, 0, 0}), {0, -1, 0}));
    Vec3 v{0.1f, -0.7f, 2.0f};
    CHECK(near(ros_to_xr(xr_to_ros(v)), v));
    // Rotations: a yaw about XR +y is a yaw about ROS +z by the same angle.
    Quat yaw_xr = axis_angle({0, 1, 0}, 0.4f);
    CHECK(near_rot(xr_to_ros(yaw_xr), axis_angle({0, 0, 1}, 0.4f)));
    // Conversion commutes with rotating a vector.
    Quat q = normalize({0.2f, -0.5f, 0.1f, 0.8f});
    CHECK(near(xr_to_ros(rotate(q, v)), rotate(xr_to_ros(q), xr_to_ros(v))));
    CHECK(near_rot(ros_to_xr(xr_to_ros(q)), q));
    // Pose composition commutes with conversion.
    Pose a{q, {1, 2, 3}}, b{yaw_xr, {-1, 0, 0.5f}};
    Pose lhs = xr_to_ros(a * b), rhs = xr_to_ros(a) * xr_to_ros(b);
    CHECK(near(lhs.p, rhs.p));
    CHECK(near_rot(lhs.q, rhs.q));
    // yaw_only keeps the heading and drops pitch/roll.
    Quat tilted = axis_angle({0, 1, 0}, 1.0f) * axis_angle({1, 0, 0}, 0.5f);
    CHECK(near_rot(yaw_only_xr(tilted), axis_angle({0, 1, 0}, 1.0f)));
}

static void test_json() {
    auto v = json::parse(R"({"a": [1, -2.5e1, true, null], "s": "x\"y\n", "o": {}})");
    CHECK(v.has_value());
    if (v) {
        CHECK(v->get("a")->arr.size() == 4);
        CHECK(near(static_cast<float>(v->get("a")->arr[1].num), -25.0f));
        CHECK(v->get("a")->arr[2].b);
        CHECK(v->get("s")->str == "x\"y\n");
        CHECK(v->get("o")->is_object());
        CHECK(v->get("missing") == nullptr);
    }
    CHECK(!json::parse("{\"a\":1,}"));
    CHECK(!json::parse("[1 2]"));
    CHECK(!json::parse("{} trailing"));
    CHECK(!json::parse(""));
    std::string deep(100, '[');
    CHECK(!json::parse(deep));
}

static void test_protocol() {
    protocol::ControllerState s;
    s.seq = 42;
    s.t = 1.5;
    s.head_valid = true;
    s.head.p = {0.1f, 0.2f, 1.6f};
    s.right.active = true;
    s.right.pose.p = {0.4f, -0.2f, 0.3f};
    s.right.trigger = 0.75f;
    s.right.button_lower = true;
    std::string msg = protocol::encode_controllers(s);
    auto v = json::parse(msg);
    CHECK(v.has_value());
    if (v) {
        CHECK(v->get("type")->str == "controllers");
        CHECK(v->get("seq")->num == 42);
        CHECK(v->get("head")->get("p")->arr.size() == 3);
        CHECK(v->get("left")->get("active")->b == false);
        CHECK(v->get("left")->get("p") == nullptr);  // inactive hands carry no pose
        CHECK(near(static_cast<float>(v->get("right")->get("p")->arr[0].num), 0.4f));
        CHECK(near(static_cast<float>(v->get("right")->get("trigger")->num), 0.75f));
        CHECK(v->get("right")->get("buttons")->get("a")->num == 1);
        CHECK(v->get("left")->get("buttons")->get("menu") != nullptr);
    }
    CHECK(json::parse(protocol::encode_hello("quest\"2")).has_value());

    auto g = protocol::decode_gripper_pose(
        R"({"type":"gripper_pose","t":3.0,"p":[0.3,0.0,0.2],"q":[0,0,0,2]})");
    CHECK(g.has_value());
    if (g) {
        CHECK(near(g->pose.p, {0.3f, 0, 0.2f}));
        CHECK(near(g->pose.q.w, 1.0f));  // normalized
    }
    CHECK(!protocol::decode_gripper_pose(R"({"type":"other","p":[0,0,0],"q":[0,0,0,1]})"));
    CHECK(!protocol::decode_gripper_pose(R"({"type":"gripper_pose","p":[0,0],"q":[0,0,0,1]})"));
    CHECK(!protocol::decode_gripper_pose("not json"));
}

static void test_view() {
    std::string msg = protocol::encode_view("\xff\xd8jpeg", 720, 792, 12.5);
    CHECK(msg.size() == protocol::kViewHeaderSize + 6);
    CHECK(msg.compare(0, 4, "VIEW") == 0);
    uint16_t w, h;
    double t;
    std::memcpy(&w, msg.data() + 4, 2);
    std::memcpy(&h, msg.data() + 6, 2);
    std::memcpy(&t, msg.data() + 8, 8);
    CHECK(w == 720 && h == 792 && t == 12.5);
    CHECK(msg.compare(16, 6, "\xff\xd8jpeg") == 0);
}

static void test_url() {
    WsUrl u;
    CHECK(parse_ws_url("ws://192.168.1.5:8765/quest", u) && u.host == "192.168.1.5" &&
          u.port == 8765 && u.path == "/quest");
    CHECK(parse_ws_url("localhost", u) && u.host == "localhost" && u.port == 9090 && u.path == "/");
    CHECK(parse_ws_url("10.0.0.2:1234", u) && u.port == 1234);
    CHECK(!parse_ws_url("", u));
    CHECK(!parse_ws_url("host:99999", u));
    CHECK(!parse_ws_url("host:abc", u));
}

int main() {
    test_math();
    test_frames();
    test_json();
    test_protocol();
    test_view();
    test_url();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all core tests passed\n");
    return 0;
}
