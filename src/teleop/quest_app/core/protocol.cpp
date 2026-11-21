#include "src/teleop/quest_app/core/protocol.h"

#include <cstdio>
#include <cstring>

#include "src/teleop/quest_app/core/json.h"

namespace qt::protocol {

namespace {

void append(std::string& out, const char* fmt, auto... args) {
    char buf[256];
    int n = std::snprintf(buf, sizeof(buf), fmt, args...);
    if (n > 0) out.append(buf, static_cast<size_t>(n) < sizeof(buf) ? n : sizeof(buf) - 1);
}

void append_pose(std::string& out, const Pose& t) {
    append(out, "\"p\":[%.5f,%.5f,%.5f],\"q\":[%.6f,%.6f,%.6f,%.6f]", t.p.x, t.p.y, t.p.z, t.q.x,
           t.q.y, t.q.z, t.q.w);
}

void append_hand(std::string& out, const char* name, const Hand& h, bool left) {
    append(out, ",\"%s\":{\"active\":%s", name, h.active ? "true" : "false");
    if (h.active) {
        out += ',';
        append_pose(out, h.pose);
    }
    append(out, ",\"trigger\":%.4f,\"grip\":%.4f,\"stick\":[%.4f,%.4f]", h.trigger, h.grip,
           h.stick_x, h.stick_y);
    if (left)
        append(out, ",\"buttons\":{\"x\":%d,\"y\":%d,\"stick\":%d,\"menu\":%d}}", h.button_lower,
               h.button_upper, h.stick_click, h.menu);
    else
        append(out, ",\"buttons\":{\"a\":%d,\"b\":%d,\"stick\":%d}}", h.button_lower,
               h.button_upper, h.stick_click);
}

bool read_vec(const json::Value* v, float* out, size_t n) {
    if (!v || !v->is_array() || v->arr.size() != n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (!v->arr[i].is_number()) return false;
        out[i] = static_cast<float>(v->arr[i].num);
    }
    return true;
}

}  // namespace

std::string encode_hello(std::string_view device) {
    std::string out;
    append(out, "{\"type\":\"hello\",\"version\":%d,\"device\":\"", kVersion);
    for (char c : device)
        if (c != '"' && c != '\\' && c >= 0x20) out += c;
    out += "\"}";
    return out;
}

std::string encode_controllers(const ControllerState& s) {
    std::string out;
    out.reserve(640);
    append(out, "{\"type\":\"controllers\",\"seq\":%llu,\"t\":%.6f",
           static_cast<unsigned long long>(s.seq), s.t);
    if (s.head_valid) {
        out += ",\"head\":{";
        append_pose(out, s.head);
        out += '}';
    }
    append_hand(out, "left", s.left, true);
    append_hand(out, "right", s.right, false);
    out += '}';
    return out;
}

std::string encode_view(std::string_view jpeg, int width, int height, double t) {
    std::string out(kViewHeaderSize, '\0');
    std::memcpy(out.data(), "VIEW", 4);
    const uint16_t w = static_cast<uint16_t>(width), h = static_cast<uint16_t>(height);
    // Quest and x86 are little-endian; the header is defined little-endian.
    std::memcpy(out.data() + 4, &w, 2);
    std::memcpy(out.data() + 6, &h, 2);
    std::memcpy(out.data() + 8, &t, 8);
    out.append(jpeg);
    return out;
}

std::optional<GripperPose> decode_gripper_pose(std::string_view text) {
    auto v = json::parse(text);
    if (!v || !v->is_object()) return std::nullopt;
    const json::Value* type = v->get("type");
    if (!type || !type->is_string() || type->str != "gripper_pose") return std::nullopt;
    float p[3], q[4];
    if (!read_vec(v->get("p"), p, 3) || !read_vec(v->get("q"), q, 4)) return std::nullopt;
    GripperPose g;
    if (const json::Value* t = v->get("t"); t && t->is_number()) g.t = t->num;
    g.pose.p = {p[0], p[1], p[2]};
    g.pose.q = normalize({q[0], q[1], q[2], q[3]});
    return g;
}

}  // namespace qt::protocol
