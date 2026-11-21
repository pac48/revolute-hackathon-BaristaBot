#include "src/teleop/quest_app/app/view_capture.h"

#include <android/bitmap.h>
#include <android/data_space.h>
#include <android/log.h>

#include <cstring>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "quest_teleop", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "quest_teleop", __VA_ARGS__)

namespace qt {

bool ViewCapture::init(int eye_width, int eye_height, int width, double fps, int quality, Sink sink) {
    if (width <= 0 || fps <= 0 || eye_width <= 0 || eye_height <= 0) return false;
    eye_width_ = eye_width;
    eye_height_ = eye_height;
    width_ = width & ~1;
    height_ = (eye_height * width_ / eye_width) & ~1;
    period_ = 1.0 / fps;
    quality_ = quality;
    sink_ = std::move(sink);

    // sRGB like the eye swapchain, so the blit copies the displayed values unchanged.
    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_SRGB8_ALPHA8, width_, height_);
    glBindTexture(GL_TEXTURE_2D, 0);
    glGenFramebuffers(1, &framebuffer_);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
    bool ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (!ok) {
        LOGE("view capture framebuffer incomplete");
        return false;
    }
    for (Slot& s : slots_) {
        glGenBuffers(1, &s.pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, s.pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, static_cast<GLsizeiptr>(width_) * height_ * 4, nullptr, GL_STREAM_READ);
    }
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

    stop_ = false;
    encoder_ = std::thread([this] { encode_loop(); });
    LOGI("view stream: %dx%d at %.0f fps, JPEG quality %d", width_, height_, fps, quality_);
    return true;
}

void ViewCapture::shutdown() {
    {
        std::lock_guard lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    if (encoder_.joinable()) encoder_.join();
    for (Slot& s : slots_) {
        if (s.fence) glDeleteSync(s.fence), s.fence = nullptr;
        if (s.pbo) glDeleteBuffers(1, &s.pbo), s.pbo = 0;
    }
    if (framebuffer_) glDeleteFramebuffers(1, &framebuffer_), framebuffer_ = 0;
    if (texture_) glDeleteTextures(1, &texture_), texture_ = 0;
}

void ViewCapture::frame(GLuint eye_framebuffer, double t) {
    if (!framebuffer_) return;

    // Collect finished readbacks (oldest first).
    for (int k = 1; k <= 2; ++k) {
        Slot& s = slots_[(write_slot_ + k) % 2];
        if (!s.fence) continue;
        if (glClientWaitSync(s.fence, 0, 0) == GL_TIMEOUT_EXPIRED) continue;
        glDeleteSync(s.fence);
        s.fence = nullptr;
        glBindBuffer(GL_PIXEL_PACK_BUFFER, s.pbo);
        const size_t bytes = static_cast<size_t>(width_) * height_ * 4;
        if (void* p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, static_cast<GLsizeiptr>(bytes), GL_MAP_READ_BIT)) {
            {
                std::lock_guard lock(mu_);
                pending_.resize(bytes);
                std::memcpy(pending_.data(), p, bytes);
                pending_t_ = s.t;
                has_pending_ = true;
            }
            cv_.notify_one();
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    if (t < next_capture_) return;
    Slot& s = slots_[write_slot_];
    if (s.fence) return;  // GPU still busy with this buffer; skip this capture
    next_capture_ = t + period_;

    glBindFramebuffer(GL_READ_FRAMEBUFFER, eye_framebuffer);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, framebuffer_);
    // Destination y reversed: rows come back top-down from glReadPixels.
    glBlitFramebuffer(0, 0, eye_width_, eye_height_, 0, height_, width_, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer_);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, s.pbo);
    glReadPixels(0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    s.fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    s.t = t;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    write_slot_ = (write_slot_ + 1) % 2;
}

void ViewCapture::encode_loop() {
    std::vector<uint8_t> pixels;
    while (true) {
        double t;
        {
            std::unique_lock lock(mu_);
            cv_.wait(lock, [this] { return stop_ || has_pending_; });
            if (stop_) return;
            pixels.swap(pending_);
            t = pending_t_;
            has_pending_ = false;
        }
        AndroidBitmapInfo info{};
        info.width = static_cast<uint32_t>(width_);
        info.height = static_cast<uint32_t>(height_);
        info.stride = static_cast<uint32_t>(width_) * 4;
        info.format = ANDROID_BITMAP_FORMAT_RGBA_8888;
        info.flags = ANDROID_BITMAP_FLAGS_ALPHA_OPAQUE;
        std::string jpeg;
        jpeg.reserve(96 * 1024);
        auto write = [](void* user, const void* data, size_t size) -> bool {
            static_cast<std::string*>(user)->append(static_cast<const char*>(data), size);
            return true;
        };
        int rc = AndroidBitmap_compress(&info, ADATASPACE_SRGB, pixels.data(), ANDROID_BITMAP_COMPRESS_FORMAT_JPEG,
                                        quality_, &jpeg, write);
        if (rc != ANDROID_BITMAP_RESULT_SUCCESS) {
            LOGE("JPEG encode failed: %d", rc);
            continue;
        }
        sink_(std::move(jpeg), width_, height_, t);
    }
}

}  // namespace qt
