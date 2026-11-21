// Flat-shaded boxes in OpenGL ES 3: enough for axes, a floor grid and a simple gripper.
#pragma once

#include <GLES3/gl3.h>

#include "src/teleop/quest_app/core/pose_math.h"

namespace qt {

struct Color {
    float r, g, b;
};

class GlRenderer {
public:
    bool init();
    void shutdown();

    // Per eye: bind the target, clear, set the view-projection used by the draw calls.
    void begin_eye(GLuint framebuffer, int width, int height, const Mat4& view_proj);

    // Box of `size` centered at `center`, in the frame given by `model`.
    void box(const Mat4& model, Vec3 center, Vec3 size, Color color);
    // RGB = xyz axes of `model`, each `length` long.
    void axes(const Mat4& model, float length, float thickness);

private:
    GLuint program_ = 0, vao_ = 0, vbo_ = 0, ibo_ = 0;
    GLint u_mvp_ = -1, u_model_ = -1, u_color_ = -1;
    Mat4 view_proj_;
};

}  // namespace qt
