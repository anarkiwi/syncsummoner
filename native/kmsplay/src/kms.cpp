#include "kms.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drm.h>

namespace kmsplay {

namespace {

std::string sformat(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
std::string sformat(const char *fmt, ...) {
  char buf[256];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  return buf;
}

std::string join(const std::vector<std::string> &parts) {
  std::string out;
  for (const std::string &p : parts) {
    out += (out.empty() ? "" : "|") + p;
  }
  return out;
}

char sync(uint32_t flags, uint32_t pos, uint32_t neg) {
  return (flags & pos) != 0 ? '+' : (flags & neg) != 0 ? '-' : '?';
}

Mode as_mode(const drmModeModeInfo &m) {
  return {m.hdisplay, m.vdisplay, m.clock, m.htotal, m.vtotal, m.flags};
}

} // namespace

std::string mode_name(const drmModeModeInfo &m) {
  bool interlaced = (m.flags & DRM_MODE_FLAG_INTERLACE) != 0;
  double hz = m.clock * 1000.0 / (static_cast<double>(m.htotal) * m.vtotal) *
              (interlaced ? 2 : 1);
  std::string h =
      sformat("%u/%u/%u/%u/%c", m.hdisplay, m.hsync_start - m.hdisplay,
              m.hsync_end - m.hsync_start, m.htotal - m.hsync_end,
              sync(m.flags, DRM_MODE_FLAG_PHSYNC, DRM_MODE_FLAG_NHSYNC));
  std::string v =
      sformat("%u/%u/%u/%u/%c", m.vdisplay, m.vsync_start - m.vdisplay,
              m.vsync_end - m.vsync_start, m.vtotal - m.vsync_end,
              sync(m.flags, DRM_MODE_FLAG_PVSYNC, DRM_MODE_FLAG_NVSYNC));
  std::vector<std::string> types, flags;
  if ((m.type & DRM_MODE_TYPE_PREFERRED) != 0) {
    types.emplace_back("P");
  }
  if ((m.type & DRM_MODE_TYPE_USERDEF) != 0) {
    types.emplace_back("U");
  }
  if ((m.type & DRM_MODE_TYPE_DRIVER) != 0) {
    types.emplace_back("D");
  }
  if ((m.flags & DRM_MODE_FLAG_DBLCLK) != 0) {
    flags.emplace_back("2x");
  }
  static const char *aspects[] = {nullptr, "4:3", "16:9", "64:27", "256:135"};
  uint32_t aspect = (m.flags & DRM_MODE_FLAG_PIC_AR_MASK) >> 19;
  if (aspect > 0 && aspect < 5) {
    flags.emplace_back(aspects[aspect]);
  }
  return sformat("%ux%u%s@%.2f %.3f %s %s %u (%.2f) %s %s", m.hdisplay,
                 m.vdisplay, interlaced ? "i" : "", hz, m.clock / 1000.0,
                 h.c_str(), v.c_str(), m.vrefresh, hz, join(types).c_str(),
                 join(flags).c_str());
}

// LCOV_EXCL_START needs a DRM device
uint32_t KmsDisplay::prop(uint32_t object, uint32_t type,
                          const char *name) const {
  drmModeObjectProperties *props =
      drmModeObjectGetProperties(fd_, object, type);
  uint32_t found = 0;
  for (uint32_t k = 0; props != nullptr && k < props->count_props && !found;
       ++k) {
    if (drmModePropertyRes *p = drmModeGetProperty(fd_, props->props[k])) {
      if (std::strcmp(p->name, name) == 0) {
        found = p->prop_id;
      }
      drmModeFreeProperty(p);
    }
  }
  drmModeFreeObjectProperties(props);
  if (!found) {
    throw Fault(std::string("the display offers no property ") + name);
  }
  return found;
}

namespace {

uint64_t plane_type(int fd, uint32_t plane) {
  drmModeObjectProperties *props =
      drmModeObjectGetProperties(fd, plane, DRM_MODE_OBJECT_PLANE);
  uint64_t type = UINT64_MAX;
  for (uint32_t k = 0; props != nullptr && k < props->count_props; ++k) {
    if (drmModePropertyRes *p = drmModeGetProperty(fd, props->props[k])) {
      if (std::strcmp(p->name, "type") == 0) {
        type = props->prop_values[k];
      }
      drmModeFreeProperty(p);
    }
  }
  drmModeFreeObjectProperties(props);
  return type;
}

} // namespace

KmsDisplay::KmsDisplay(const std::string &device, const std::string &connector,
                       uint32_t width, uint32_t height, Rate fps)
    : frame_bytes(static_cast<size_t>(width) * height * BYTES_PER_PIXEL),
      width_(width), height_(height) {
  fd_ = open(device.c_str(), O_RDWR | O_CLOEXEC);
  if (fd_ < 0) {
    throw Fault("cannot open " + device + ": " + std::strerror(errno));
  }
  drmSetClientCap(fd_, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
  if (drmSetClientCap(fd_, DRM_CLIENT_CAP_ATOMIC, 1) != 0) {
    throw Fault(device + " offers no atomic modesetting");
  }
  drmModeRes *res = drmModeGetResources(fd_);
  if (res == nullptr) {
    throw Fault(device + " offers no mode resources");
  }
  drmModeConnector *conn = nullptr;
  for (int k = 0; k < res->count_connectors && conn == nullptr; ++k) {
    drmModeConnector *c = drmModeGetConnector(fd_, res->connectors[k]);
    if (c == nullptr) {
      continue;
    }
    const char *type = drmModeGetConnectorTypeName(c->connector_type);
    if (type != nullptr &&
        connector ==
            std::string(type) + "-" + std::to_string(c->connector_type_id)) {
      conn = c;
    } else {
      drmModeFreeConnector(c);
    }
  }
  if (conn == nullptr) {
    throw Fault(device + " has no connector " + connector);
  }
  conn_ = conn->connector_id;
  if (conn->encoder_id != 0) {
    if (drmModeEncoder *enc = drmModeGetEncoder(fd_, conn->encoder_id)) {
      crtc_ = enc->crtc_id;
      drmModeFreeEncoder(enc);
    }
  }
  for (int e = 0; e < conn->count_encoders && crtc_ == 0; ++e) {
    if (drmModeEncoder *enc = drmModeGetEncoder(fd_, conn->encoders[e])) {
      for (int k = 0; k < res->count_crtcs && crtc_ == 0; ++k) {
        if ((enc->possible_crtcs & (1u << k)) != 0) {
          crtc_ = res->crtcs[k];
        }
      }
      drmModeFreeEncoder(enc);
    }
  }
  if (crtc_ == 0) {
    throw Fault(connector + " has no CRTC to drive it");
  }
  int crtc_index = 0;
  while (crtc_index < res->count_crtcs && res->crtcs[crtc_index] != crtc_) {
    ++crtc_index;
  }
  drmModeFreeResources(res);

  drmModePlaneRes *planes = drmModeGetPlaneResources(fd_);
  for (int pass = 0; pass < 2 && plane_ == 0; ++pass) {
    for (uint32_t k = 0;
         planes != nullptr && k < planes->count_planes && plane_ == 0; ++k) {
      drmModePlane *p = drmModeGetPlane(fd_, planes->planes[k]);
      if (p == nullptr) {
        continue;
      }
      if ((p->possible_crtcs & (1u << crtc_index)) != 0 &&
          (pass == 1 || p->crtc_id == crtc_) &&
          plane_type(fd_, p->plane_id) == DRM_PLANE_TYPE_PRIMARY) {
        plane_ = p->plane_id;
      }
      drmModeFreePlane(p);
    }
  }
  drmModeFreePlaneResources(planes);
  if (plane_ == 0) {
    throw Fault(connector + "'s CRTC has no primary plane");
  }

  drmModeCrtc *crtc = drmModeGetCrtc(fd_, crtc_);
  std::optional<drmModeModeInfo> current;
  if (crtc != nullptr && crtc->mode_valid) {
    current = crtc->mode;
  }
  drmModeFreeCrtc(crtc);
  std::vector<drmModeModeInfo> infos;
  std::vector<Mode> modes;
  infos.push_back(current.value_or(drmModeModeInfo{}));
  for (int k = 0; k < conn->count_modes; ++k) {
    infos.push_back(conn->modes[k]);
  }
  drmModeFreeConnector(conn);
  for (const drmModeModeInfo &m : infos) {
    modes.push_back(as_mode(m));
  }
  std::vector<const Mode *> offered;
  for (size_t k = 0; k < modes.size(); ++k) {
    offered.push_back(k == 0 && !current ? nullptr : &modes[k]);
  }
  Picked picked = pick_mode(offered, width, height, fps);
  info_ = infos[picked.index];
  hold = picked.hold;
  modeset = picked.index != 0;
  mode = mode_name(info_);
  refresh_hz = refresh(modes[picked.index]);
  period_ns = Rate{1000000000, 1} / refresh_hz;
  if (drmModeCreatePropertyBlob(fd_, &info_, sizeof info_, &blob_) != 0) {
    throw Fault(std::string("cannot make a mode blob: ") +
                std::strerror(errno));
  }

  conn_crtc_ = prop(conn_, DRM_MODE_OBJECT_CONNECTOR, "CRTC_ID");
  crtc_active_ = prop(crtc_, DRM_MODE_OBJECT_CRTC, "ACTIVE");
  crtc_mode_ = prop(crtc_, DRM_MODE_OBJECT_CRTC, "MODE_ID");
  const char *names[10] = {"FB_ID", "CRTC_ID", "SRC_X",  "SRC_Y",  "SRC_W",
                           "SRC_H", "CRTC_X",  "CRTC_Y", "CRTC_W", "CRTC_H"};
  uint32_t ids[10];
  for (int k = 0; k < 10; ++k) {
    ids[k] = prop(plane_, DRM_MODE_OBJECT_PLANE, names[k]);
  }
  for (int b = 0; b < BUFFERS; ++b) {
    Buffer &buf = buffers_[b];
    if (drmModeCreateDumbBuffer(fd_, width, height, 8 * BYTES_PER_PIXEL, 0,
                                &buf.handle, &buf.pitch, &buf.size) != 0) {
      throw Fault(std::string("cannot make a dumb buffer: ") +
                  std::strerror(errno));
    }
    uint32_t handles[4] = {buf.handle}, pitches[4] = {buf.pitch},
             offsets[4] = {0};
    if (drmModeAddFB2(fd_, width, height, DRM_FORMAT_RGB565, handles, pitches,
                      offsets, &buf.fb, 0) != 0) {
      throw Fault(std::string("cannot make an RG16 framebuffer: ") +
                  std::strerror(errno));
    }
    uint64_t offset = 0;
    if (drmModeMapDumbBuffer(fd_, buf.handle, &offset) != 0) {
      throw Fault(std::string("cannot map a dumb buffer: ") +
                  std::strerror(errno));
    }
    void *map = mmap(nullptr, buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
                     static_cast<off_t>(offset));
    if (map == MAP_FAILED) {
      throw Fault(std::string("cannot map a dumb buffer: ") +
                  std::strerror(errno));
    }
    buf.map = static_cast<uint8_t *>(map);
    Flip &f = flips_[b];
    const uint64_t values[10] = {buf.fb,
                                 crtc_,
                                 0,
                                 0,
                                 static_cast<uint64_t>(width) << 16,
                                 static_cast<uint64_t>(height) << 16,
                                 0,
                                 0,
                                 width,
                                 height};
    f.objs[0] = plane_;
    f.counts[0] = 10;
    std::memcpy(f.props, ids, sizeof ids);
    std::memcpy(f.values, values, sizeof values);
    f.req = {};
    f.req.flags = DRM_MODE_PAGE_FLIP_EVENT | DRM_MODE_ATOMIC_NONBLOCK;
    f.req.count_objs = 1;
    f.req.objs_ptr = reinterpret_cast<uintptr_t>(f.objs);
    f.req.count_props_ptr = reinterpret_cast<uintptr_t>(f.counts);
    f.req.props_ptr = reinterpret_cast<uintptr_t>(f.props);
    f.req.prop_values_ptr = reinterpret_cast<uintptr_t>(f.values);
  }
}

KmsDisplay::~KmsDisplay() {
  if (fd_ >= 0) {
    close(fd_);
  }
}

void KmsDisplay::fill(int buf, const uint8_t *data) {
  const Buffer &b = buffers_[buf];
  size_t row = static_cast<size_t>(width_) * BYTES_PER_PIXEL;
  if (b.pitch == row) {
    std::memcpy(b.map, data, frame_bytes);
    return;
  }
  for (uint32_t y = 0; y < height_; ++y) {
    std::memcpy(b.map + static_cast<size_t>(y) * b.pitch, data + y * row, row);
  }
}

std::vector<uint8_t> KmsDisplay::frame(int buf) const {
  const Buffer &b = buffers_[buf];
  size_t row = static_cast<size_t>(width_) * BYTES_PER_PIXEL;
  std::vector<uint8_t> out(frame_bytes);
  for (uint32_t y = 0; y < height_; ++y) {
    std::memcpy(out.data() + y * row, b.map + static_cast<size_t>(y) * b.pitch,
                row);
  }
  return out;
}

void KmsDisplay::commit_sync(int buf, std::optional<bool> active,
                             bool allow_modeset, const char *what) {
  drmModeAtomicReq *req = drmModeAtomicAlloc();
  if (active) {
    drmModeAtomicAddProperty(req, conn_, conn_crtc_, crtc_);
    drmModeAtomicAddProperty(req, crtc_, crtc_active_, *active ? 1 : 0);
    drmModeAtomicAddProperty(req, crtc_, crtc_mode_, blob_);
  }
  const Flip &f = flips_[buf];
  for (int k = 0; k < 10; ++k) {
    drmModeAtomicAddProperty(req, plane_, f.props[k], f.values[k]);
  }
  int done = drmModeAtomicCommit(
      fd_, req, allow_modeset ? DRM_MODE_ATOMIC_ALLOW_MODESET : 0, nullptr);
  drmModeAtomicFree(req);
  if (done != 0) {
    std::string why = std::strerror(-done);
    if (what == nullptr) {
      throw Fault("the first commit was refused (" + why +
                  "): is another DRM master running?");
    }
    throw Fault(std::string(what) + " was refused: " + why);
  }
}

void KmsDisplay::start(int buf) {
  commit_sync(buf, modeset ? std::optional<bool>(true) : std::nullopt, modeset,
              nullptr);
}

void KmsDisplay::set_active(bool active, int buf) {
  commit_sync(buf, active, true,
              active ? "turning the link up" : "turning the link down");
}

void KmsDisplay::flip(int buf, int64_t v) {
  Flip &f = flips_[buf];
  f.req.user_data = static_cast<uint64_t>(v + 1);
  if (drmIoctl(fd_, DRM_IOCTL_MODE_ATOMIC, &f.req) != 0) {
    throw Fault("flip for vblank " + std::to_string(v) +
                " was refused: " + std::strerror(errno));
  }
}

std::optional<Event> KmsDisplay::wait(double timeout_s) {
  using namespace std::chrono;
  auto deadline = steady_clock::now() + duration<double>(timeout_s);
  for (;;) {
    auto left = duration_cast<milliseconds>(deadline - steady_clock::now());
    pollfd fds{fd_, POLLIN, 0};
    int ready =
        poll(&fds, 1, static_cast<int>(std::max<int64_t>(left.count(), 0)));
    if (ready < 0 && errno == EINTR) {
      continue;
    }
    if (ready <= 0) {
      return std::nullopt;
    }
    alignas(8) char buf[1024];
    ssize_t n = read(fd_, buf, sizeof buf);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    for (ssize_t at = 0; at + static_cast<ssize_t>(sizeof(drm_event)) <= n;) {
      const auto *e = reinterpret_cast<const drm_event *>(buf + at);
      if (e->type == DRM_EVENT_FLIP_COMPLETE) {
        const auto *vbl = reinterpret_cast<const drm_event_vblank *>(e);
        return Event{static_cast<int64_t>(vbl->sequence),
                     static_cast<int64_t>(vbl->tv_sec) * 1000000000 +
                         static_cast<int64_t>(vbl->tv_usec) * 1000,
                     static_cast<int64_t>(vbl->user_data) - 1};
      }
      at += e->length;
    }
    return std::nullopt;
  }
}
// LCOV_EXCL_STOP

} // namespace kmsplay
