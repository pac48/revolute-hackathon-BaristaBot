#include "src/teleop/quest_app/app/gl_renderer.h"

#include <android/log.h>

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "quest_teleop", __VA_ARGS__)

namespace qt {

namespace {

const char* kVertex = R"(#version 300 es
layout(location = 0) in vec3 a_pos;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
void main() {
    v_normal = mat3(u_model) * a_normal;
    gl_Position = u_mvp * vec4(a_pos, 1.0);
}
)";

const char* kFragment = R"(#version 300 es
precision mediump float;
in vec3 v_normal;
uniform vec3 u_color;
out vec4 frag;
void main() {
    // Fixed light from above-front in XR world coordinates (y up).
    float l = 0.55 + 0.45 * max(dot(normalize(v_normal), normalize(vec3(0.3, 1.0, 0.5))), 0.0);
    frag = vec4(u_color * l, 1.0);
}
)";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        LOGE("shader compile failed: %s", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

Mat4 translate_scale(Vec3 t, Vec3 s) {
    Mat4 m;
    m.m[0] = s.x;
    m.m[5] = s.y;
    m.m[10] = s.z;
    m.m[12] = t.x;
    m.m[13] = t.y;
    m.m[14] = t.z;
    return m;
}

}  // namespace

bool GlRenderer::init() {
    GLuint vs = compile(GL_VERTEX_SHADER, kVertex), fs = compile(GL_FRAGMENT_SHADER, kFragment);
    if (!vs || !fs) return false;
    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        LOGE("program link failed");
        return false;
    }
    u_mvp_ = glGetUniformLocation(program_, "u_mvp");
    u_model_ = glGetUniformLocation(program_, "u_model");
    u_color_ = glGetUniformLocation(program_, "u_color");

    // Unit cube centered at the origin, 4 vertices per face so each face has its own normal.
    float v[24 * 6];
    unsigned short idx[36];
    const float n[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (int f = 0; f < 6; ++f) {
        int a = (f / 2 + 1) % 3, b = (f / 2 + 2) % 3, c = f / 2;  // tangent axes, normal axis
        float sign = n[f][c];
        const float corners[4][2] = {{-0.5f, -0.5f}, {0.5f, -0.5f}, {0.5f, 0.5f}, {-0.5f, 0.5f}};
        for (int k = 0; k < 4; ++k) {
            float* p = &v[(f * 4 + k) * 6];
            p[c] = 0.5f * sign;
            // Flip winding on negative faces so all faces are counter-clockwise from outside.
            p[a] = corners[k][0] * sign;
            p[b] = corners[k][1];
            p[3] = n[f][0];
            p[4] = n[f][1];
            p[5] = n[f][2];
        }
        unsigned short base = static_cast<unsigned short>(f * 4);
        unsigned short tri[6] = {0, 1, 2, 0, 2, 3};
        for (int k = 0; k < 6; ++k) idx[f * 6 + k] = base + tri[k];
    }
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
    glGenBuffers(1, &ibo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glBindVertexArray(0);
    return true;
}

void GlRenderer::shutdown() {
    if (ibo_) glDeleteBuffers(1, &ibo_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
    if (program_) glDeleteProgram(program_);
    ibo_ = vbo_ = vao_ = program_ = 0;
}

void GlRenderer::begin_eye(GLuint framebuffer, int width, int height, const Mat4& view_proj) {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glViewport(0, 0, width, height);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);  // boxes are tiny; skip winding bugs entirely
    glClearColor(0.05f, 0.06f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glUseProgram(program_);
    glBindVertexArray(vao_);
    view_proj_ = view_proj;
}

void GlRenderer::box(const Mat4& model, Vec3 center, Vec3 size, Color color) {
    Mat4 m = model * translate_scale(center, size);
    Mat4 mvp = view_proj_ * m;
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.m);
    glUniformMatrix4fv(u_model_, 1, GL_FALSE, m.m);
    glUniform3f(u_color_, color.r, color.g, color.b);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
}

void GlRenderer::axes(const Mat4& model, float length, float t) {
    box(model, {length / 2, 0, 0}, {length, t, t}, {0.9f, 0.15f, 0.15f});
    box(model, {0, length / 2, 0}, {t, length, t}, {0.15f, 0.85f, 0.2f});
    box(model, {0, 0, length / 2}, {t, t, length}, {0.2f, 0.35f, 0.95f});
}

}  // namespace qt
