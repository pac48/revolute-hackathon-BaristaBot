// Spectator stream: a downscaled copy of one eye's image, JPEG-encoded off the render thread.
//
// Per captured frame, on the render thread (cheap, never waits on the GPU):
//   blit the eye image into a small framebuffer (scaled, flipped to top-down rows), start an
//   asynchronous glReadPixels into a pixel buffer object, and fence it. A later frame maps the
//   buffer once its fence has signaled and hands the pixels to the encoder thread.
// The encoder thread compresses to JPEG (AndroidBitmap_compress, API 30+) and calls `sink`.
// If the encoder is still busy with the previous frame, the new one replaces it.
#pragma once

#include <GLES3/gl3.h>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace qt {

class ViewCapture {
public:
    // jpeg, width, height, capture time (s, steady clock of the caller)
    using Sink = std::function<void(std::string jpeg, int width, int height, double t)>;

    ~ViewCapture() { shutdown(); }

    // width: output width in pixels (height keeps the eye's aspect); fps: capture rate.
    bool init(int eye_width, int eye_height, int width, double fps, int quality, Sink sink);
    void shutdown();

    // Call after rendering the eye, with its framebuffer. `t` is now, in seconds.
    void frame(GLuint eye_framebuffer, double t);

    int width() const { return width_; }
    int height() const { return height_; }

private:
    struct Slot {
        GLuint pbo = 0;
        GLsync fence = nullptr;
        double t = 0;
    };

    void encode_loop();

    int eye_width_ = 0, eye_height_ = 0, width_ = 0, height_ = 0;
    double period_ = 0, next_capture_ = 0;
    int quality_ = 70;
    GLuint texture_ = 0, framebuffer_ = 0;
    Slot slots_[2];
    int write_slot_ = 0;
    Sink sink_;

    std::thread encoder_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool has_pending_ = false;
    std::vector<uint8_t> pending_;  // RGBA rows, top-down
    double pending_t_ = 0;
};

}  // namespace qt
