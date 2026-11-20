// A simulated display on an exact virtual clock, and frame helpers.
#pragma once

#include "kmsplay.hpp"

#include <cstring>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <map>
#include <set>

namespace kmsplay::test {

#define EXPECT_FAULT(stmt, re)                                                 \
  EXPECT_THAT([&] { stmt; }, ::testing::ThrowsMessage<::kmsplay::Fault>(       \
                                 ::testing::ContainsRegex(re)))

// An exact rational time, for vblank grids that no integer ns period hits.
struct Q {
  __int128 n = 0, d = 1;
  Q() = default;
  Q(__int128 num, __int128 den = 1) : n(num), d(den) {
    if (d < 0) {
      n = -n;
      d = -d;
    }
    __int128 a = n < 0 ? -n : n, b = d;
    while (b != 0) {
      __int128 t = a % b;
      a = b;
      b = t;
    }
    if (a > 1) {
      n /= a;
      d /= a;
    }
  }
  explicit Q(Rate r) : Q(r.num, r.den) {}
  Q operator+(Q o) const { return {n * o.d + o.n * d, d * o.d}; }
  Q operator-(Q o) const { return {n * o.d - o.n * d, d * o.d}; }
  Q operator*(Q o) const { return {n * o.n, d * o.d}; }
  Q operator/(Q o) const { return {n * o.d, d * o.n}; }
  __int128 floor() const { return floordiv(n, d); }
  Rate rate() const { return make_rate(n, d); }
};

inline const Q PERIOD{1000000000, 30};

// Vblanks every period on a virtual clock; a flip lands on the vblank after
// its commit. late names clip vblanks whose commit misses its vblank and lands
// one later; lost names ones whose event never arrives. While inactive no
// vblanks occur, and reactivating starts a grid phase off the old one.
class FakeDisplay : public Display, public Clock {
public:
  explicit FakeDisplay(Q period = PERIOD, std::set<int64_t> late = {},
                       std::set<int64_t> lost = {}, Q phase = {},
                       int64_t seq0 = 1000)
      : period(period), seq_at_grid(seq0), late(std::move(late)),
        lost(std::move(lost)), phase(phase) {}

  int64_t now() override { return static_cast<int64_t>(t.floor()); }
  void sleep_ns(int64_t ns) override {
    block();
    t = t + Q(ns);
    if (on_sleep) {
      on_sleep();
    }
  }
  void fill(int buf, const uint8_t *data) override {
    int64_t frame;
    std::memcpy(&frame, data, sizeof frame);
    filled[buf] = frame;
  }
  void start(int buf) override {
    started = buf;
    screen = filled.at(buf);
  }
  void set_active(bool on, int buf) override {
    block();
    actives.push_back(on);
    if (on && !active) {
      grid = t + phase;
    }
    active = on;
    screen = filled.at(buf);
  }
  void flip(int buf, int64_t v) override {
    EXPECT_FALSE(pending) << "a commit while one is pending is refused by KMS";
    pending = {buf, v, filled.at(buf)};
  }
  std::optional<Event> wait(double timeout_s) override {
    block();
    EXPECT_TRUE(active && pending);
    auto [buf, v, data] = *pending;
    pending.reset();
    __int128 ticks = ((t - grid) / period).floor() + 1;
    if (late.count(v) != 0) {
      ++ticks;
    }
    if (lost.count(v) != 0) {
      t = t + Q(static_cast<int64_t>(timeout_s * 1e9));
      return std::nullopt;
    }
    t = grid + Q(ticks) * period;
    int64_t seq = seq_at_grid + static_cast<int64_t>(ticks);
    grid = t;
    seq_at_grid = seq;
    screen = data;
    shown[v] = {buf, data};
    return Event{seq, static_cast<int64_t>(t.floor()), v};
  }

  Q period, t, grid;
  int64_t seq_at_grid;
  std::set<int64_t> late, lost;
  Q phase;
  bool active = true;
  std::optional<std::tuple<int, int64_t, int64_t>> pending;
  std::map<int64_t, std::pair<int, int64_t>> shown;
  std::map<int, int64_t> filled;
  int64_t screen = -1;
  int started = -1;
  std::vector<bool> actives;
  std::function<void()> on_sleep, on_block;

private:
  void block() {
    if (on_block) {
      on_block();
    }
  }
};

// Frames 0..count-1, each eight bytes holding its own index.
class Counting : public Source {
public:
  explicit Counting(int64_t count) : count_(count) {}
  bool next(uint8_t *dst) override {
    if (at_ >= count_) {
      return false;
    }
    std::memcpy(dst, &at_, sizeof at_);
    ++at_;
    return true;
  }

private:
  int64_t count_, at_ = 0;
};

inline int64_t index_of(const uint8_t *frame) {
  int64_t k;
  std::memcpy(&k, frame, sizeof k);
  return k;
}

// A read-ahead with everything already decoded.
inline std::unique_ptr<Prerolled> decoded(int64_t count) {
  auto pre = std::make_unique<Prerolled>(std::make_unique<Counting>(count),
                                         10000, sizeof(int64_t));
  pre->wait_full(5.0);
  return pre;
}

struct FnSink : Sink {
  std::function<void(const Record &)> fn;
  std::function<void()> woken;
  explicit FnSink(std::function<void(const Record &)> f,
                  std::function<void()> w = {})
      : fn(std::move(f)), woken(std::move(w)) {}
  void record(const Record &rec) override { fn(rec); }
  void wake() override {
    if (woken) {
      woken();
    }
  }
};

} // namespace kmsplay::test
