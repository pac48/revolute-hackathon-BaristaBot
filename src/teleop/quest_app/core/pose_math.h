// Small rigid-transform math: vectors, unit quaternions (x, y, z, w), poses, 4x4 matrices.
// Header-only, no platform dependencies, so it builds for the headset and the host tests.
#pragma once

#include <cmath>

namespace qt {

struct Vec3 {
    float x = 0, y = 0, z = 0;
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(Vec3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline Vec3 cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

struct Quat {
    float x = 0, y = 0, z = 0, w = 1;
};

inline Quat operator*(Quat a, Quat b) {
    return {
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    };
}
inline Quat conjugate(Quat q) { return {-q.x, -q.y, -q.z, q.w}; }
inline Quat normalize(Quat q) {
    float n = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (n < 1e-9f) return {};
    return {q.x / n, q.y / n, q.z / n, q.w / n};
}
inline Vec3 rotate(Quat q, Vec3 v) {
    // v' = v + 2w(u x v) + 2u x (u x v), u = vector part
    Vec3 u{q.x, q.y, q.z};
    Vec3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}
inline Quat axis_angle(Vec3 axis, float angle) {
    float s = std::sin(angle * 0.5f);
    return {axis.x * s, axis.y * s, axis.z * s, std::cos(angle * 0.5f)};
}

struct Pose {
    Quat q;
    Vec3 p;
};

// a * b: apply b, then a (frames compose left to right: T_a_c = T_a_b * T_b_c).
inline Pose operator*(const Pose& a, const Pose& b) { return {a.q * b.q, a.p + rotate(a.q, b.p)}; }
inline Pose inverse(const Pose& t) {
    Quat qi = conjugate(t.q);
    return {qi, rotate(qi, t.p * -1.0f)};
}

// Column-major 4x4, as OpenGL expects.
struct Mat4 {
    float m[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c)
        for (int row = 0; row < 4; ++row) {
            float s = 0;
            for (int k = 0; k < 4; ++k) s += a.m[k * 4 + row] * b.m[c * 4 + k];
            r.m[c * 4 + row] = s;
        }
    return r;
}

inline Mat4 to_matrix(const Pose& t, Vec3 scale = {1, 1, 1}) {
    const Quat& q = t.q;
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    Mat4 r;
    r.m[0] = (1 - 2 * (yy + zz)) * scale.x;
    r.m[1] = (2 * (xy + wz)) * scale.x;
    r.m[2] = (2 * (xz - wy)) * scale.x;
    r.m[3] = 0;
    r.m[4] = (2 * (xy - wz)) * scale.y;
    r.m[5] = (1 - 2 * (xx + zz)) * scale.y;
    r.m[6] = (2 * (yz + wx)) * scale.y;
    r.m[7] = 0;
    r.m[8] = (2 * (xz + wy)) * scale.z;
    r.m[9] = (2 * (yz - wx)) * scale.z;
    r.m[10] = (1 - 2 * (xx + yy)) * scale.z;
    r.m[11] = 0;
    r.m[12] = t.p.x;
    r.m[13] = t.p.y;
    r.m[14] = t.p.z;
    r.m[15] = 1;
    return r;
}

// Asymmetric-frustum projection from OpenXR field-of-view angles (radians), GL clip space.
inline Mat4 projection_from_fov(float left, float right, float up, float down, float near_z, float far_z) {
    float tl = std::tan(left), tr = std::tan(right), tu = std::tan(up), td = std::tan(down);
    float w = tr - tl, h = tu - td;
    Mat4 r;
    for (float& v : r.m) v = 0;
    r.m[0] = 2 / w;
    r.m[5] = 2 / h;
    r.m[8] = (tr + tl) / w;
    r.m[9] = (tu + td) / h;
    r.m[10] = -(far_z + near_z) / (far_z - near_z);
    r.m[11] = -1;
    r.m[14] = -2 * far_z * near_z / (far_z - near_z);
    return r;
}

}  // namespace qt
