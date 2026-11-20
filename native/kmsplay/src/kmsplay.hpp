// Vsync-locked clip player for a DRM/KMS display: the pacing core, its frame
// read-ahead, and the flip record hand-off to the non-real-time threads.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace kmsplay {

// One flip: clip vblank, display vblank sequence, clip frame shown, event time
// (ns), flags; the host reads it as struct "<qqqqq".
struct Record {
  int64_t v, seq, frame, t_ns, flags;
};
static_assert(sizeof(Record) == 40, "a record is five little-endian int64s");

// Flag on the first flip of a segment, after the start or after the link came
// back up.
constexpr int64_t SEGMENT = 1;
// Flag on a copy of the last flip done, sent in reply to a request on the ask
// pipe.
constexpr int64_t REPLY = 2;
// DRM_MODE_FLAG_INTERLACE: such a mode has no single frame per vblank.
constexpr uint32_t INTERLACE = 0x10;
constexpr int BYTES_PER_PIXEL = 2;
// Displayed, committed and being filled.
constexpr int BUFFERS = 3;
// SCHED_FIFO priority of the pacing thread, above every normal thread and below
// the kernel's own.
constexpr int REALTIME_PRIORITY = 10;
// Flips kept for the timing trace reported on a fault.
constexpr size_t TRACE_DEPTH = 64;

// The display did not show the frame due on a vblank, or could not be made to.
struct Fault : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// A positive exact rational, always in lowest terms.
struct Rate {
  int64_t num = 1, den = 1;
  bool operator==(const Rate &) const = default;
};
Rate make_rate(__int128 num, __int128 den);
Rate operator*(Rate a, Rate b);
Rate operator/(Rate a, Rate b);
double to_double(Rate r);
// "n" when whole, else "n/d", as Python's str(Fraction).
std::string to_string(Rate r);
// An integer, a decimal or num/den; throws std::invalid_argument unless
// positive.
Rate parse_rate(const std::string &text);

__int128 floordiv(__int128 n, __int128 d);
// n / d rounded to the nearest integer, ties to even, for d > 0.
__int128 round_half_even(__int128 n, __int128 d);
// A double as Python's repr prints it: shortest round trip, ".0" when whole.
std::string py_float(double x);

struct Mode {
  uint32_t hdisplay, vdisplay, clock, htotal, vtotal, flags;
};
// A mode's exact vertical refresh in Hz, from its pixel clock (kHz) and totals.
Rate refresh(const Mode &mode);
struct Picked {
  size_t index;
  int64_t hold;
};
// The first progressive mode of this size refreshing at a whole multiple of
// fps, and that multiple. modes is in preference order, the current mode first
// (nullptr when there is none), so a mode already driven is kept.
Picked pick_mode(const std::vector<const Mode *> &modes, uint32_t width,
                 uint32_t height, Rate fps);

// Maps the display's vblanks to the clip's, and checks every flip against it.
//
// Vblank v counts from the first flip at t0 and shows frame v / hold. Within a
// segment each flip must land on the vblank it was committed for, within half a
// vblank of the segment's own grid; phases_ns keeps where each segment's grid
// sits against t0's.
class Timeline {
public:
  Timeline(int64_t hold, Rate period_ns);
  // The clip vblank nearest a time, once the timeline is anchored.
  int64_t at(int64_t t_ns) const;
  // Start the first segment on this flip, or a later one at its time's vblank.
  int64_t anchor(int64_t seq, int64_t t_ns);
  // Throw unless the flip committed for vblank v landed on it, on its grid.
  void check(int64_t seq, int64_t t_ns, int64_t v);

  int64_t hold;
  Rate period_ns;
  std::optional<int64_t> t0;
  int64_t base = 0, start_v = 0, start_t = 0;
  std::vector<int64_t> phases_ns;
  int64_t worst_ns = 0;
};

// Pending stop, link-down and link-up requests, set from signal handlers.
struct Signals {
  std::atomic<bool> stop{false}, down{false}, up{false};
  static bool pop(std::atomic<bool> &flag) { return flag.exchange(false); }
};
static_assert(std::atomic<bool>::is_always_lock_free);

struct Event {
  int64_t seq, t_ns, v;
};

class Display {
public:
  virtual ~Display() = default;
  // Copy one frame into a buffer.
  virtual void fill(int buf, const uint8_t *data) = 0;
  // Put buf up, setting the mode only when it is not the one already driven.
  virtual void start(int buf) = 0;
  // Turn the CRTC, and so the HDMI link, off or back on with buf showing.
  virtual void set_active(bool active, int buf) = 0;
  // Commit buf for the next vblank, tagged with its clip vblank.
  virtual void flip(int buf, int64_t v) = 0;
  // The next flip-complete event, or none on timeout.
  virtual std::optional<Event> wait(double timeout_s) = 0;
};

class Clock {
public:
  virtual ~Clock() = default;
  virtual int64_t now() = 0;
  virtual void sleep_ns(int64_t ns) = 0;
};

// CLOCK_MONOTONIC, the clock DRM stamps its events with.
class MonotonicClock : public Clock {
public:
  int64_t now() override;
  void sleep_ns(int64_t ns) override;
};

// Where the pacer hands each flip; record() must neither block nor allocate.
class Sink {
public:
  virtual ~Sink() = default;
  virtual void record(const Record &rec) = 0;
  // Called once a commit is made, or before the pacer pauses with no commit to
  // make, so every record is passed on off the deadline.
  virtual void wake() {}
};

class Frames {
public:
  virtual ~Frames() = default;
  // Clip frame index, or nullptr once the source has ended; throws on underrun.
  virtual const uint8_t *get(int64_t index) = 0;
};

// When the pacer woke for a flip-complete event, committed the next flip and
// had the frame after it in a buffer, against the event's own time.
struct Trace {
  int64_t v, t_ns, wake_ns, commit_ns, ready_ns;
};

// Paces prerolled frames onto a display, one atomic flip per vblank. While the
// link is down the timeline runs on the event clock and due frames are
// discarded.
class Player {
public:
  Player(Display &display, Frames &frames, Clock &clock, Sink &sink,
         Signals &signals, int64_t hold, Rate period_ns,
         double event_timeout_s = 1.0);
  // Play until a stop is asked for, holding the last frame once a clip ends.
  void run();
  // The last flips, oldest first.
  std::vector<Trace> trace() const;

  struct Slot {
    int64_t frame;
    int buf;
  };
  Timeline timeline;
  int64_t flips = 0, segments = 0;
  bool ended = false;
  std::optional<Slot> shown;
  // The longest time from a flip-complete event's own time to the next commit.
  // DRM stamps an event with the start of scanout after its vblank, after
  // the event is delivered, so a pacer quick to commit makes this negative.
  std::optional<int64_t> worst_commit_ns;

private:
  static int free_buffer(int a, int b);
  std::optional<Slot> prepare(int64_t frame, int buf);
  Event event();
  void land(const Event &e, int64_t v, int64_t flags);
  void flip(int buf, int64_t v);
  void commit(int64_t v);
  void skip_to(int64_t frame);
  void down();
  void hold_last();
  void mark_ready();

  Display &display_;
  Frames &frames_;
  Clock &clock_;
  Sink &sink_;
  Signals &signals_;
  double event_timeout_s_;
  std::optional<Slot> ready_;
  std::array<Trace, TRACE_DEPTH> trace_{};
  size_t traced_ = 0;
  Trace *current_ = nullptr;
};

// Fills a destination with the next frame; false at the end, throws on
// failure. cancel() is called from another thread to unblock next() for good.
// A futex one thread parks on until another rings it. ring() costs the ringer
// a syscall only while the other is parked, and never blocks.
class Bell {
public:
  // Park unless ready() holds, until rung; ready() must read with seq_cst.
  template <class Ready> void wait_unless(Ready ready) {
    waiting_.store(true, std::memory_order_seq_cst);
    uint32_t seen = word_.load(std::memory_order_seq_cst);
    if (!ready()) {
      park(seen);
    }
    waiting_.store(false, std::memory_order_relaxed);
  }
  // Wake the parked thread, if there is one.
  void ring() {
    if (waiting_.load(std::memory_order_seq_cst)) {
      force();
    }
  }
  // Wake the parked thread, or keep the one about to park awake.
  void force();

private:
  void park(uint32_t seen);
  std::atomic<uint32_t> word_{0};
  std::atomic<bool> waiting_{false};
};

class Source {
public:
  virtual ~Source() = default;
  virtual bool next(uint8_t *dst) = 0;
  virtual void cancel() {}
};

// Frames from a source, read ahead into a single-producer single-consumer ring
// of preallocated frames by a normal-priority thread. get(index) returns clip
// frame index, discarding any before it; a frame not yet decoded when it is due
// is an underrun, which is a fault. The frame returned stays valid until the
// next get().
class Prerolled : public Frames {
public:
  Prerolled(std::unique_ptr<Source> source, size_t depth, size_t frame_bytes);
  ~Prerolled() override;
  Prerolled(const Prerolled &) = delete;
  Prerolled &operator=(const Prerolled &) = delete;
  // Block until the read-ahead is full or the source has ended.
  void wait_full(double timeout_s, const std::function<double()> &clock);
  void wait_full(double timeout_s);
  const uint8_t *get(int64_t index) override;
  bool ended() const { return ended_; }
  size_t queued() const;
  const size_t depth;

private:
  void fill();
  uint8_t *slot(uint32_t n) { return storage_.get() + (n % slots_) * bytes_; }

  std::unique_ptr<Source> source_;
  const size_t bytes_, slots_;
  std::unique_ptr<uint8_t[]> storage_;
  std::atomic<uint32_t> head_{0}, tail_{0};
  std::atomic<bool> done_{false}, closing_{false};
  Bell space_;
  std::string error_;
  int64_t next_ = 0;
  bool ended_ = false, holding_ = false;
  std::thread thread_;
};

// The last flip, written wait-free by the pacer and read by any thread.
class LastFlip {
public:
  void store(const Record &rec);
  std::optional<Record> load() const;

private:
  std::atomic<uint32_t> seq_{0};
  std::array<std::atomic<int64_t>, 5> fields_{};
};

// A named pipe at path: the one there, else a new one in place of whatever is.
void fifo(const std::string &path);

// A named pipe offered every flip record as its flip completes, never blocking
// or raising. The FIFO at path is kept if one is there, so a reader that opened
// it before the player started gets every record. A record is dropped while no
// reader has the pipe open or while the pipe is full; once a reader leaves, the
// next record waits for another. Never called by the pacer.
class LiveFeed {
public:
  explicit LiveFeed(std::string path);
  ~LiveFeed();
  void write(const Record &rec);
  void close();

private:
  void release();
  std::string path_;
  std::mutex lock_;
  int fd_ = -1;
};

// For each byte read from fd, reply with the frame on screen then, flagged
// REPLY. The last flip recorded is carried on by the vblanks its event time
// puts before now, so a flip completed but not yet recorded still counts.
// Returns when fd ends.
void answer(int fd, const std::function<std::optional<Record>()> &last,
            const std::function<void(const Record &)> &reply, Rate period_ns,
            int64_t hold, Clock &clock);

// The pacer's Sink: records go into a lock-free ring a writer thread drains to
// the flip log and the live feed, and the last one is kept for --ask replies.
class Recorder : public Sink {
public:
  Recorder(int log_fd, LiveFeed *live, size_t capacity = 1 << 14);
  ~Recorder() override;
  Recorder(const Recorder &) = delete;
  Recorder &operator=(const Recorder &) = delete;
  void record(const Record &rec) override;
  void wake() override;
  // Write out every record taken and stop the writer.
  void close();
  // Answer requests on the named pipe at path from a thread of its own.
  void serve_asks(const std::string &path, Rate period_ns, int64_t hold);
  const LastFlip &last() const { return last_; }

private:
  void drain();
  int fd_;
  LiveFeed *live_;
  std::vector<Record> ring_;
  std::atomic<uint32_t> head_{0}, taken_{0};
  uint32_t tail_ = 0;
  std::atomic<bool> closing_{false};
  Bell bell_;
  LastFlip last_;
  std::thread writer_;
};

} // namespace kmsplay
