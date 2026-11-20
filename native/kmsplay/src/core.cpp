#include "kmsplay.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <linux/futex.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace kmsplay {

namespace {

__int128 gcd128(__int128 a, __int128 b) {
  a = a < 0 ? -a : a;
  b = b < 0 ? -b : b;
  while (b != 0) {
    __int128 t = a % b;
    a = b;
    b = t;
  }
  return a;
}

std::string format(const char *fmt, double x) {
  char buf[64];
  std::snprintf(buf, sizeof buf, fmt, x);
  return buf;
}

} // namespace

Rate make_rate(__int128 num, __int128 den) {
  if (den == 0 || num <= 0 || den < 0) {
    throw std::invalid_argument("a rate must be positive");
  }
  __int128 g = gcd128(num, den);
  num /= g;
  den /= g;
  if (num > INT64_MAX || den > INT64_MAX) {
    throw std::invalid_argument("a rate's terms overflow");
  }
  return {static_cast<int64_t>(num), static_cast<int64_t>(den)};
}

Rate operator*(Rate a, Rate b) {
  return make_rate(static_cast<__int128>(a.num) * b.num,
                   static_cast<__int128>(a.den) * b.den);
}

Rate operator/(Rate a, Rate b) {
  return make_rate(static_cast<__int128>(a.num) * b.den,
                   static_cast<__int128>(a.den) * b.num);
}

double to_double(Rate r) {
  return static_cast<double>(r.num) / static_cast<double>(r.den);
}

std::string to_string(Rate r) {
  return r.den == 1 ? std::to_string(r.num)
                    : std::to_string(r.num) + "/" + std::to_string(r.den);
}

Rate parse_rate(const std::string &text) {
  auto bad = [&] {
    return std::invalid_argument("frame rate must be positive, got " + text);
  };
  auto whole = [&](std::string_view s) {
    int64_t out = 0;
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), out);
    if (s.empty() || ec != std::errc() || end != s.data() + s.size()) {
      throw bad();
    }
    return out;
  };
  std::string_view s = text;
  try {
    if (auto slash = s.find('/'); slash != std::string_view::npos) {
      return make_rate(whole(s.substr(0, slash)), whole(s.substr(slash + 1)));
    }
    auto dot = s.find('.');
    if (dot == std::string_view::npos) {
      return make_rate(whole(s), 1);
    }
    std::string digits(s.substr(0, dot));
    std::string_view frac = s.substr(dot + 1);
    if (frac.size() > 18 ||
        frac.find_first_not_of("0123456789") != std::string_view::npos) {
      throw bad();
    }
    __int128 scale = 1;
    for (size_t k = 0; k < frac.size(); ++k) {
      scale *= 10;
    }
    __int128 num =
        static_cast<__int128>(digits.empty() ? 0 : whole(digits)) * scale +
        (frac.empty() ? 0 : whole(frac));
    return make_rate(num, scale);
  } catch (const std::invalid_argument &) {
    throw bad();
  }
}

__int128 floordiv(__int128 n, __int128 d) {
  __int128 q = n / d;
  return (n % d != 0 && ((n < 0) != (d < 0))) ? q - 1 : q;
}

__int128 round_half_even(__int128 n, __int128 d) {
  __int128 q = floordiv(n, d);
  __int128 twice = 2 * (n - q * d);
  if (twice > d || (twice == d && (q & 1) != 0)) {
    ++q;
  }
  return q;
}

std::string py_float(double x) {
  char buf[64];
  auto [end, ec] = std::to_chars(buf, buf + sizeof buf, x);
  std::string out(buf, end);
  if (std::isfinite(x) && out.find_first_of(".en") == std::string::npos) {
    out += ".0";
  }
  return out;
}

Rate refresh(const Mode &mode) {
  return make_rate(static_cast<__int128>(mode.clock) * 1000,
                   static_cast<__int128>(mode.htotal) * mode.vtotal);
}

Picked pick_mode(const std::vector<const Mode *> &modes, uint32_t width,
                 uint32_t height, Rate fps) {
  std::vector<Rate> seen;
  for (size_t k = 0; k < modes.size(); ++k) {
    const Mode *mode = modes[k];
    if (mode == nullptr || mode->hdisplay != width ||
        mode->vdisplay != height || (mode->flags & INTERLACE) != 0) {
      continue;
    }
    Rate rate = refresh(*mode);
    if (std::find(seen.begin(), seen.end(), rate) == seen.end()) {
      seen.push_back(rate);
    }
    Rate hold = rate / fps;
    if (hold.den == 1) {
      return {k, hold.num};
    }
  }
  std::string said;
  for (const Rate &r : seen) {
    said += (said.empty() ? "" : ", ") + format("%.4f", to_double(r));
  }
  throw Fault("no " + std::to_string(width) + "x" + std::to_string(height) +
              " progressive mode refreshes at a whole multiple of " +
              format("%g", to_double(fps)) + " fps; offered " +
              (said.empty() ? "none" : said) + " Hz");
}

Timeline::Timeline(int64_t hold_, Rate period)
    : hold(hold_), period_ns(period) {
  phases_ns.reserve(1 << 12);
}

int64_t Timeline::at(int64_t t_ns) const {
  return static_cast<int64_t>(round_half_even(
      static_cast<__int128>(t_ns - *t0) * period_ns.den, period_ns.num));
}

int64_t Timeline::anchor(int64_t seq, int64_t t_ns) {
  if (!t0) {
    t0 = t_ns;
  }
  int64_t v = at(t_ns);
  base = seq - v;
  start_v = v;
  start_t = t_ns;
  __int128 off = static_cast<__int128>(t_ns - *t0) * period_ns.den -
                 static_cast<__int128>(v) * period_ns.num;
  phases_ns.push_back(static_cast<int64_t>(off / period_ns.den));
  return v;
}

void Timeline::check(int64_t seq, int64_t t_ns, int64_t v) {
  if (seq - base != v) {
    throw Fault("flip for vblank " + std::to_string(v) + " (frame " +
                std::to_string(static_cast<int64_t>(floordiv(v, hold))) +
                ") landed on vblank " + std::to_string(seq - base) + ": " +
                std::to_string(seq - base - v) + " vblank(s) missed");
  }
  __int128 late = static_cast<__int128>(t_ns - start_t) * period_ns.den -
                  static_cast<__int128>(v - start_v) * period_ns.num;
  late = late < 0 ? -late : late;
  worst_ns = std::max(worst_ns, static_cast<int64_t>(late / period_ns.den));
  if (2 * late >= period_ns.num) {
    double ms = static_cast<double>(late) / period_ns.den / 1e6;
    throw Fault("vblank " + std::to_string(v) + " came " + format("%.3f", ms) +
                " ms off a grid of " +
                format("%.3f", to_double(period_ns) / 1e6) +
                " ms vblanks: the display clock and the event clock disagree");
  }
}

int64_t MonotonicClock::now() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

void MonotonicClock::sleep_ns(int64_t ns) {
  timespec ts{static_cast<time_t>(ns / 1000000000),
              static_cast<long>(ns % 1000000000)};
  while (clock_nanosleep(CLOCK_MONOTONIC, 0, &ts, &ts) == EINTR) {
  }
}

Player::Player(Display &display, Frames &frames, Clock &clock, Sink &sink,
               Signals &signals, int64_t hold, Rate period_ns,
               double event_timeout_s)
    : timeline(hold, period_ns), display_(display), frames_(frames),
      clock_(clock), sink_(sink), signals_(signals),
      event_timeout_s_(event_timeout_s) {}

int Player::free_buffer(int a, int b) {
  for (int k = 0; k < BUFFERS; ++k) {
    if (k != a && k != b) {
      return k;
    }
  }
  return -1;
}

std::optional<Player::Slot> Player::prepare(int64_t frame, int buf) {
  const uint8_t *data = frames_.get(frame);
  if (data == nullptr) {
    return std::nullopt;
  }
  display_.fill(buf, data);
  return Slot{frame, buf};
}

Event Player::event() {
  std::optional<Event> e = display_.wait(event_timeout_s_);
  if (!e) {
    throw Fault("no flip completed within " + py_float(event_timeout_s_) + "s");
  }
  current_ = &trace_[traced_++ % TRACE_DEPTH];
  *current_ = {e->v, e->t_ns, clock_.now(), -1, -1};
  return *e;
}

void Player::land(const Event &e, int64_t v, int64_t flags) {
  ++flips;
  sink_.record({v, e.seq, shown->frame, e.t_ns, flags});
}

void Player::flip(int buf, int64_t v) {
  display_.flip(buf, v);
  if (current_ != nullptr) {
    current_->commit_ns = clock_.now();
    worst_commit_ns = std::max(worst_commit_ns.value_or(INT64_MIN),
                               current_->commit_ns - current_->t_ns);
  }
  sink_.wake();
}

void Player::mark_ready() {
  if (current_ != nullptr) {
    current_->ready_ns = clock_.now();
  }
}

void Player::commit(int64_t v) {
  int64_t frame = v / timeline.hold;
  if (frame == shown->frame) {
    flip(shown->buf, v);
    mark_ready();
    return;
  }
  while (ready_ && ready_->frame < frame) {
    ready_ = prepare(frame, ready_->buf);
  }
  if (!ready_) {
    ended = true;
    return;
  }
  Slot pending = *ready_;
  flip(pending.buf, v);
  ready_ = prepare(frame + 1, free_buffer(shown->buf, pending.buf));
  shown = pending;
  mark_ready();
}

void Player::skip_to(int64_t frame) {
  while (ready_ && ready_->frame < frame) {
    ready_ = prepare(ready_->frame + 1, ready_->buf);
  }
}

void Player::down() {
  display_.set_active(false, shown->buf);
  const Rate p = timeline.period_ns;
  while (!Signals::pop(signals_.up) && !signals_.stop) {
    int64_t now = clock_.now();
    __int128 next =
        floordiv(static_cast<__int128>(now - *timeline.t0) * p.den, p.num) + 1;
    clock_.sleep_ns(*timeline.t0 - now -
                    static_cast<int64_t>(floordiv(-next * p.num, p.den)));
    skip_to(static_cast<int64_t>(
        floordiv(timeline.at(clock_.now()), timeline.hold)));
  }
  if (signals_.stop) {
    return;
  }
  display_.set_active(true, shown->buf);
  current_ = nullptr;
  display_.flip(shown->buf, -1);
  Event e = event();
  int64_t v = timeline.anchor(e.seq, e.t_ns);
  current_->v = v;
  ++segments;
  land(e, v, SEGMENT);
  skip_to(static_cast<int64_t>(floordiv(v, timeline.hold)));
  commit(v + 1);
}

void Player::hold_last() {
  if (Signals::pop(signals_.down)) {
    display_.set_active(false, shown->buf);
  }
  if (Signals::pop(signals_.up)) {
    display_.set_active(true, shown->buf);
  }
  clock_.sleep_ns(timeline.period_ns.num / timeline.period_ns.den);
}

void Player::run() {
  shown = prepare(0, 0);
  if (!shown) {
    throw Fault("the clip holds no frames");
  }
  ready_ = prepare(1, 1);
  display_.start(0);
  display_.flip(0, 0);
  Event e = event();
  timeline.anchor(e.seq, e.t_ns);
  segments = 1;
  land(e, 0, SEGMENT);
  commit(1);
  while (!signals_.stop) {
    if (ended) {
      hold_last();
      continue;
    }
    e = event();
    timeline.check(e.seq, e.t_ns, e.v);
    land(e, e.v, 0);
    if (Signals::pop(signals_.down)) {
      down();
    } else {
      commit(e.v + 1);
    }
  }
}

std::vector<Trace> Player::trace() const {
  std::vector<Trace> out;
  size_t n = std::min(traced_, TRACE_DEPTH);
  for (size_t k = traced_ - n; k < traced_; ++k) {
    out.push_back(trace_[k % TRACE_DEPTH]);
  }
  return out;
}

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t));

void Bell::park(uint32_t seen) {
  syscall(SYS_futex, reinterpret_cast<uint32_t *>(&word_), FUTEX_WAIT_PRIVATE,
          seen, nullptr, nullptr, 0);
}

void Bell::force() {
  word_.fetch_add(1, std::memory_order_seq_cst);
  syscall(SYS_futex, reinterpret_cast<uint32_t *>(&word_), FUTEX_WAKE_PRIVATE,
          1, nullptr, nullptr, 0);
}

Prerolled::Prerolled(std::unique_ptr<Source> source, size_t depth_,
                     size_t frame_bytes)
    : depth(depth_), source_(std::move(source)), bytes_(frame_bytes),
      slots_(depth_ + 1) {
  if (depth < 1) {
    throw std::invalid_argument(
        "preroll depth must be at least one frame, got " +
        std::to_string(depth));
  }
  storage_.reset(new uint8_t[slots_ * bytes_]);
  thread_ = std::thread([this] { fill(); });
}

Prerolled::~Prerolled() {
  closing_ = true;
  space_.force();
  source_->cancel();
  thread_.join();
}

void Prerolled::fill() {
  try {
    while (!closing_) {
      uint32_t h = head_.load(std::memory_order_relaxed);
      auto room = [&] {
        return h - tail_.load(std::memory_order_seq_cst) < slots_ || closing_;
      };
      while (!room()) {
        space_.wait_unless(room);
      }
      if (closing_ || !source_->next(slot(h))) {
        break;
      }
      head_.store(h + 1, std::memory_order_release);
    }
  } catch (const std::exception &err) {
    error_ = err.what();
  }
  done_.store(true, std::memory_order_release);
}

size_t Prerolled::queued() const {
  return head_.load(std::memory_order_acquire) -
         tail_.load(std::memory_order_acquire) - (holding_ ? 1 : 0);
}

void Prerolled::wait_full(double timeout_s,
                          const std::function<double()> &clock) {
  double deadline = clock() + timeout_s;
  while (queued() < depth && !done_.load(std::memory_order_acquire)) {
    if (clock() >= deadline) {
      throw Fault("preroll held " + std::to_string(queued()) + " of " +
                  std::to_string(depth) + " frames after " +
                  py_float(timeout_s) + "s");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

void Prerolled::wait_full(double timeout_s) {
  wait_full(timeout_s, [] {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  });
}

const uint8_t *Prerolled::get(int64_t index) {
  while (!ended_) {
    if (index < next_) {
      throw Fault("frame " + std::to_string(index) +
                  " was asked for after frame " + std::to_string(next_ - 1));
    }
    uint32_t t = tail_.load(std::memory_order_relaxed);
    if (holding_) {
      tail_.store(++t, std::memory_order_seq_cst);
      holding_ = false;
      space_.ring();
    }
    if (head_.load(std::memory_order_acquire) == t) {
      if (!done_.load(std::memory_order_acquire)) {
        throw Fault("frame " + std::to_string(index) +
                    " was not decoded when it was due");
      }
      if (head_.load(std::memory_order_acquire) == t) {
        ended_ = true;
        if (!error_.empty()) {
          throw Fault("the frame source failed: " + error_);
        }
        return nullptr;
      }
    }
    holding_ = true;
    if (next_++ == index) {
      return slot(t);
    }
  }
  return nullptr;
}

void LastFlip::store(const Record &rec) {
  uint32_t s = seq_.load(std::memory_order_relaxed);
  seq_.store(s + 1, std::memory_order_relaxed);
  std::atomic_thread_fence(std::memory_order_release);
  const int64_t values[5] = {rec.v, rec.seq, rec.frame, rec.t_ns, rec.flags};
  for (size_t k = 0; k < fields_.size(); ++k) {
    fields_[k].store(values[k], std::memory_order_relaxed);
  }
  seq_.store(s + 2, std::memory_order_release);
}

std::optional<Record> LastFlip::load() const {
  for (;;) {
    uint32_t before = seq_.load(std::memory_order_acquire);
    if (before == 0) {
      return std::nullopt;
    }
    int64_t v[5];
    for (size_t k = 0; k < fields_.size(); ++k) {
      v[k] = fields_[k].load(std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if ((before & 1) == 0 && seq_.load(std::memory_order_relaxed) == before) {
      return Record{v[0], v[1], v[2], v[3], v[4]};
    }
  }
}

void fifo(const std::string &path) {
  struct stat st{};
  if (stat(path.c_str(), &st) == 0) {
    if (S_ISFIFO(st.st_mode)) {
      return;
    }
    unlink(path.c_str());
  }
  if (mkfifo(path.c_str(), 0666) != 0) {
    throw Fault("cannot make the pipe " + path + ": " + std::strerror(errno));
  }
}

LiveFeed::LiveFeed(std::string path) : path_(std::move(path)) { fifo(path_); }

LiveFeed::~LiveFeed() { close(); }

void LiveFeed::release() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

void LiveFeed::write(const Record &rec) {
  std::lock_guard<std::mutex> hold(lock_);
  if (fd_ < 0) {
    fd_ = open(path_.c_str(), O_WRONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) {
      return;
    }
  }
  if (::write(fd_, &rec, sizeof rec) < 0 && errno != EAGAIN) {
    release();
  }
}

void LiveFeed::close() {
  std::lock_guard<std::mutex> hold(lock_);
  release();
}

void answer(int fd, const std::function<std::optional<Record>()> &last,
            const std::function<void(const Record &)> &reply, Rate period_ns,
            int64_t hold, Clock &clock) {
  char byte = 0;
  for (;;) {
    ssize_t got = read(fd, &byte, 1);
    if (got < 0 && errno == EINTR) {
      continue;
    }
    if (got <= 0) {
      return;
    }
    std::optional<Record> rec = last();
    if (!rec) {
      continue;
    }
    int64_t passed = std::max<int64_t>(
        static_cast<int64_t>(floordiv(
            static_cast<__int128>(clock.now() - rec->t_ns) * period_ns.den,
            period_ns.num)),
        0);
    int64_t now = rec->v + passed;
    int64_t frame = rec->frame + static_cast<int64_t>(floordiv(now, hold) -
                                                      floordiv(rec->v, hold));
    reply({now, rec->seq + passed, frame, rec->t_ns, rec->flags | REPLY});
  }
}

Recorder::Recorder(int log_fd, LiveFeed *live, size_t capacity)
    : fd_(log_fd), live_(live), ring_(capacity) {
  if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
    throw std::invalid_argument("the record ring must be a power of two");
  }
  writer_ = std::thread([this] { drain(); });
}

Recorder::~Recorder() { close(); }

void Recorder::record(const Record &rec) {
  last_.store(rec);
  uint32_t h = head_.load(std::memory_order_relaxed);
  if (h - taken_.load(std::memory_order_acquire) >= ring_.size()) {
    throw Fault("the flip log fell " + std::to_string(ring_.size()) +
                " records behind");
  }
  ring_[h % ring_.size()] = rec;
  head_.store(h + 1, std::memory_order_seq_cst);
}

void Recorder::wake() { bell_.ring(); }

void Recorder::drain() {
  for (;;) {
    uint32_t h = head_.load(std::memory_order_acquire);
    if (h == tail_) {
      if (closing_) {
        return;
      }
      bell_.wait_unless([&] {
        return head_.load(std::memory_order_seq_cst) != tail_ || closing_;
      });
      continue;
    }
    for (; tail_ != h; ++tail_) {
      const Record &rec = ring_[tail_ % ring_.size()];
      if (fd_ >= 0) {
        const char *p = reinterpret_cast<const char *>(&rec);
        size_t left = sizeof rec;
        while (left > 0) {
          ssize_t n = ::write(fd_, p, left);
          if (n < 0 && errno == EINTR) {
            continue;
          }
          if (n <= 0) {
            break;
          }
          p += n;
          left -= static_cast<size_t>(n);
        }
      }
      if (live_ != nullptr) {
        live_->write(rec);
      }
      taken_.store(tail_ + 1, std::memory_order_release);
    }
  }
}

void Recorder::close() {
  if (writer_.joinable()) {
    closing_ = true;
    bell_.force();
    writer_.join();
  }
}

void Recorder::serve_asks(const std::string &path, Rate period_ns,
                          int64_t hold) {
  fifo(path);
  std::thread([this, path, period_ns, hold] {
    MonotonicClock clock;
    for (;;) {
      int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
      if (fd < 0) {
        if (errno == EINTR) {
          continue;
        }
        return;
      }
      answer(
          fd, [this] { return last_.load(); },
          [this](const Record &rec) {
            if (live_ != nullptr) {
              live_->write(rec);
            }
          },
          period_ns, hold, clock);
      ::close(fd);
    }
  }).detach();
}

} // namespace kmsplay
