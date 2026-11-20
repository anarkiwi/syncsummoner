// One connector's CRTC and primary plane, driven by atomic commits on libdrm.
#pragma once

#include "kmsplay.hpp"

#include <xf86drmMode.h>

namespace kmsplay {

// kms++'s VideoMode::to_string_long, so the status names a mode as before.
std::string mode_name(const drmModeModeInfo &mode);

class KmsDisplay : public Display {
public:
  KmsDisplay(const std::string &device, const std::string &connector,
             uint32_t width, uint32_t height, Rate fps);
  ~KmsDisplay() override;
  KmsDisplay(const KmsDisplay &) = delete;
  KmsDisplay &operator=(const KmsDisplay &) = delete;

  void fill(int buf, const uint8_t *data) override;
  void start(int buf) override;
  void set_active(bool active, int buf) override;
  void flip(int buf, int64_t v) override;
  std::optional<Event> wait(double timeout_s) override;
  // A buffer's bytes, row after row without padding.
  std::vector<uint8_t> frame(int buf) const;

  Rate refresh_hz, period_ns;
  int64_t hold = 1;
  bool modeset = false;
  std::string mode;
  size_t frame_bytes;

private:
  struct Buffer {
    uint32_t handle = 0, pitch = 0, fb = 0;
    uint64_t size = 0;
    uint8_t *map = nullptr;
  };
  struct Flip {
    uint32_t objs[1], counts[1], props[10];
    uint64_t values[10];
    struct drm_mode_atomic req;
  };
  uint32_t prop(uint32_t object, uint32_t type, const char *name) const;
  void commit_sync(int buf, std::optional<bool> active, bool allow_modeset,
                   const char *what);

  int fd_ = -1;
  uint32_t width_, height_, conn_ = 0, crtc_ = 0, plane_ = 0, blob_ = 0;
  drmModeModeInfo info_{};
  Buffer buffers_[BUFFERS];
  Flip flips_[BUFFERS];
  uint32_t conn_crtc_ = 0, crtc_active_ = 0, crtc_mode_ = 0;
};

} // namespace kmsplay
