// Mode choice, exact rates, the read-ahead and the flip hand-off.
#include "fake.hpp"

#include <condition_variable>
#include <csignal>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace kmsplay::test {
namespace {

Mode mode(uint32_t w = 1920, uint32_t h = 1080, uint32_t clock = 74250,
          uint32_t htotal = 2200, uint32_t vtotal = 1125, uint32_t flags = 0) {
  return {w, h, clock, htotal, vtotal, flags};
}

TEST(Modes, RefreshIsExact) {
  EXPECT_EQ(refresh(mode()), (Rate{30, 1}));
  EXPECT_EQ(refresh(mode(1920, 1080, 74176)), make_rate(74176000, 2475000));
  EXPECT_EQ(to_string(refresh(mode(1920, 1080, 74176))), "74176/2475");
}

TEST(Modes, TheCurrentModeIsKeptWhenItDivides) {
  Mode current = mode(), faster = mode(1920, 1080, 148500);
  Picked p = pick_mode({&current, &faster}, 1920, 1080, {30, 1});
  EXPECT_EQ(p.index, 0u);
  EXPECT_EQ(p.hold, 1);
}

TEST(Modes, AWholeMultipleIsHeldFor) {
  Mode faster = mode(1920, 1080, 148500);
  Picked p = pick_mode({nullptr, &faster}, 1920, 1080, {30, 1});
  EXPECT_EQ(p.index, 1u);
  EXPECT_EQ(p.hold, 2);
}

TEST(Modes, NoDividingModeIsAFaultNamingWhatIsOffered) {
  Mode a = mode(), b = mode(),
       i = mode(1920, 1080, 74250, 2200, 1125, INTERLACE),
       small = mode(1280, 720);
  EXPECT_FAULT(pick_mode({nullptr, &a, &b, &i, &small}, 1920, 1080, {12, 1}),
               "12 fps; offered 30.0000 Hz$");
  EXPECT_FAULT(pick_mode({&small}, 1920, 1080, {12, 1}), "offered none Hz");
}

TEST(Rates, ParseExactly) {
  EXPECT_EQ(parse_rate("30000/1001"), (Rate{30000, 1001}));
  EXPECT_EQ(parse_rate("30"), (Rate{30, 1}));
  EXPECT_EQ(parse_rate("29.97"), (Rate{2997, 100}));
  EXPECT_EQ(parse_rate(".5"), (Rate{1, 2}));
  EXPECT_EQ(to_string(parse_rate("60/2")), "30");
  for (const char *bad : {"0", "-3", "x", "1/0", "3/", "1.2.3", "", "2.x"}) {
    EXPECT_THROW(parse_rate(bad), std::invalid_argument) << bad;
  }
}

TEST(Rates, RoundingMatchesPython) {
  EXPECT_EQ(round_half_even(5, 2), 2);
  EXPECT_EQ(round_half_even(7, 2), 4);
  EXPECT_EQ(round_half_even(-5, 2), -2);
  EXPECT_EQ(round_half_even(-7, 3), -2);
  EXPECT_EQ(floordiv(-7, 2), -4);
  EXPECT_EQ(py_float(1.0), "1.0");
  EXPECT_EQ(py_float(0.06), "0.06");
  EXPECT_EQ(py_float(1e22), "1e+22");
  EXPECT_DOUBLE_EQ(to_double(Rate{1, 4}), 0.25);
  EXPECT_THROW(make_rate(0, 1), std::invalid_argument);
  EXPECT_THROW(make_rate(static_cast<__int128>(1) << 70, 1),
               std::invalid_argument);
}

TEST(Timeline, AnchorsSegmentsOnTheFirstFlipsGrid) {
  Timeline tl(1, PERIOD.rate());
  EXPECT_EQ(tl.anchor(500, 1000), 0);
  EXPECT_EQ(tl.at(1000 + 33333333 * 3), 3);
  tl.check(503, 1000 + 100000000, 3);
  EXPECT_EQ(tl.worst_ns, 0);
  EXPECT_EQ(tl.anchor(900, 1000 + 333333333 + 1000), 10);
  EXPECT_EQ(tl.phases_ns, (std::vector<int64_t>{0, 999}));
  EXPECT_FAULT(tl.check(902, 1000 + 400000000, 11),
               "flip for vblank 11 \\(frame 11\\) landed on vblank 12: "
               "1 vblank\\(s\\) missed");
}

TEST(Prerolled, FramesComeInOrderAndSkipForward) {
  Prerolled pre(std::make_unique<Counting>(5), 8, sizeof(int64_t));
  pre.wait_full(5.0);
  EXPECT_EQ(index_of(pre.get(0)), 0);
  EXPECT_EQ(index_of(pre.get(3)), 3);
  EXPECT_FAULT(pre.get(2), "frame 2 was asked for after frame 3");
}

TEST(Prerolled, EndIsNull) {
  Prerolled pre(std::make_unique<Counting>(1), 4, sizeof(int64_t));
  pre.wait_full(5.0);
  EXPECT_NE(pre.get(0), nullptr);
  EXPECT_EQ(pre.get(1), nullptr);
  EXPECT_TRUE(pre.ended());
  EXPECT_EQ(pre.get(2), nullptr);
}

TEST(Prerolled, AReadAheadOfOneStillPlaysEveryFrame) {
  Prerolled pre(std::make_unique<Counting>(20), 1, sizeof(int64_t));
  for (int64_t k = 0; k < 20; ++k) {
    pre.wait_full(5.0);
    ASSERT_EQ(index_of(pre.get(k)), k);
  }
  pre.wait_full(5.0);
  EXPECT_EQ(pre.get(20), nullptr);
}

// A source that yields nothing until it is cancelled.
class Stall : public Source {
public:
  bool next(uint8_t *) override {
    std::unique_lock<std::mutex> hold(lock_);
    woken_.wait(hold, [&] { return cancelled_; });
    return false;
  }
  void cancel() override {
    std::lock_guard<std::mutex> hold(lock_);
    cancelled_ = true;
    woken_.notify_all();
  }

private:
  std::mutex lock_;
  std::condition_variable woken_;
  bool cancelled_ = false;
};

TEST(Prerolled, AnUnderrunIsAFault) {
  Prerolled pre(std::make_unique<Stall>(), 2, 8);
  EXPECT_FAULT(pre.get(0), "frame 0 was not decoded when it was due");
}

class Broken : public Source {
public:
  bool next(uint8_t *dst) override {
    if (sent_++ == 0) {
      dst[0] = 'x';
      return true;
    }
    throw std::runtime_error("disk gone");
  }

private:
  int sent_ = 0;
};

TEST(Prerolled, AFailingSourceIsAFault) {
  Prerolled pre(std::make_unique<Broken>(), 4, 1);
  pre.wait_full(5.0);
  EXPECT_NE(pre.get(0), nullptr);
  EXPECT_FAULT(pre.get(1), "the frame source failed: disk gone");
}

TEST(Prerolled, PrerollThatNeverFillsIsAFault) {
  Prerolled pre(std::make_unique<Stall>(), 2, 8);
  double tick = 0;
  EXPECT_FAULT(pre.wait_full(3.0, [&] { return tick++; }),
               "preroll held 0 of 2 frames after 3.0s");
  EXPECT_EQ(pre.queued(), 0u);
}

TEST(Prerolled, DepthMustBePositive) {
  EXPECT_THROW(Prerolled(std::make_unique<Counting>(0), 0, 8),
               std::invalid_argument);
}

TEST(LastFlip, IsEmptyUntilStoredAndThenTheLastWhole) {
  LastFlip last;
  EXPECT_FALSE(last.load());
  std::atomic<bool> done{false};
  std::thread writer([&] {
    for (int64_t k = 1; k <= 100000; ++k) {
      last.store({k, k, k, k, k});
    }
    done = true;
  });
  int64_t seen = 0;
  while (!done) {
    if (auto r = last.load()) {
      ASSERT_TRUE(r->v == r->seq && r->seq == r->frame && r->frame == r->t_ns &&
                  r->t_ns == r->flags);
      ASSERT_GE(r->v, seen) << "never an older flip than one already read";
      seen = r->v;
    }
  }
  writer.join();
  EXPECT_EQ(last.load()->v, 100000);
}

class Pipes : public ::testing::Test {
protected:
  void SetUp() override {
    signal(SIGPIPE, SIG_IGN);
    dir = std::filesystem::temp_directory_path() /
          ("kmsplay-" + std::to_string(getpid()) + "-" +
           ::testing::UnitTest::GetInstance()->current_test_info()->name());
    std::filesystem::create_directories(dir);
    path = (dir / "live").string();
  }
  void TearDown() override { std::filesystem::remove_all(dir); }
  static int reader(const std::string &p) {
    return open(p.c_str(), O_RDONLY | O_NONBLOCK);
  }
  // Everything a non-blocking reader can take now.
  static std::string drain(int fd) {
    std::string out;
    char buf[1 << 16];
    for (;;) {
      ssize_t n = read(fd, buf, sizeof buf);
      if (n <= 0) {
        return out;
      }
      out.append(buf, static_cast<size_t>(n));
    }
  }
  static std::string bytes(std::initializer_list<int64_t> frames) {
    std::string out;
    for (int64_t f : frames) {
      Record r{f, f, f, f, 0};
      out.append(reinterpret_cast<const char *>(&r), sizeof r);
    }
    return out;
  }
  static Record rec(int64_t f) { return {f, f, f, f, 0}; }
  std::filesystem::path dir;
  std::string path;
};

TEST_F(Pipes, TheLiveFeedDropsRecordsUntilAReaderComesThenDeliversThemWhole) {
  LiveFeed feed(path);
  feed.write(rec(0));
  feed.write(rec(1));
  int fd = reader(path);
  for (int64_t f : {2, 3, 4}) {
    feed.write(rec(f));
  }
  EXPECT_EQ(drain(fd), bytes({2, 3, 4}));
  feed.close();
  EXPECT_EQ(drain(fd), "");
  close(fd);
}

TEST_F(Pipes, AFullLivePipeDropsWholeRecordsInsteadOfBlocking) {
  LiveFeed feed(path);
  int fd = reader(path);
  std::string sent;
  for (int64_t f = 0; f < 4 * (1 << 16) / 40; ++f) {
    feed.write(rec(f));
    sent += bytes({f});
  }
  std::string got = drain(fd);
  EXPECT_GT(got.size(), 0u);
  EXPECT_LT(got.size(), sent.size());
  EXPECT_EQ(got.size() % sizeof(Record), 0u);
  EXPECT_EQ(got, sent.substr(0, got.size()));
  feed.write(rec(0));
  EXPECT_EQ(drain(fd), bytes({0}));
  close(fd);
}

TEST_F(Pipes, TheLiveFeedWaitsForANewReaderOnceOneLeaves) {
  LiveFeed feed(path);
  int fd = reader(path);
  feed.write(rec(0));
  close(fd);
  feed.write(rec(1));
  fd = reader(path);
  feed.write(rec(2));
  EXPECT_EQ(drain(fd), bytes({2}));
  close(fd);
}

TEST_F(Pipes, ALiveFeedKeepsThePipeAReaderOpenedBeforeItStarted) {
  fifo(path);
  int early = reader(path);
  LiveFeed feed(path);
  struct stat a{}, b{};
  stat(path.c_str(), &a);
  fstat(early, &b);
  EXPECT_EQ(a.st_ino, b.st_ino);
  feed.write(rec(0));
  EXPECT_EQ(drain(early), bytes({0}));
  close(early);
}

TEST_F(Pipes, AFileWhereThePipeBelongsIsReplacedByOne) {
  {
    std::ofstream(path) << "stale";
  }
  fifo(path);
  struct stat st{};
  stat(path.c_str(), &st);
  EXPECT_TRUE(S_ISFIFO(st.st_mode));
  EXPECT_FAULT(fifo((dir / "none" / "p").string()), "cannot make the pipe");
}

class Ticks : public Clock {
public:
  explicit Ticks(std::vector<int64_t> at) : at_(std::move(at)) {}
  int64_t now() override { return at_.at(k_++); }
  void sleep_ns(int64_t) override {}

private:
  std::vector<int64_t> at_;
  size_t k_ = 0;
};

TEST(Answer, EachRequestIsAnsweredWithTheFrameOnScreenFlaggedAsAReply) {
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  ASSERT_EQ(write(fds[1], "???", 3), 3);
  close(fds[1]);
  std::vector<std::optional<Record>> last = {
      std::nullopt, Record{4, 4, 4, 1000, SEGMENT}, Record{4, 4, 4, 1000, 0}};
  std::vector<Record> sent;
  Ticks clock({1000 + 99, 1000 + 250});
  size_t asked = 0;
  answer(
      fds[0], [&] { return last.at(asked++); },
      [&](const Record &r) { sent.push_back(r); }, {100, 1}, 1, clock);
  close(fds[0]);
  ASSERT_EQ(sent.size(), 2u) << "nothing to answer before the first flip";
  EXPECT_EQ(sent[0].v, 4);
  EXPECT_EQ(sent[0].flags, SEGMENT | REPLY);
  EXPECT_EQ(sent[1].v, 6) << "two vblanks past the flip recorded, unrecorded";
  EXPECT_EQ(sent[1].seq, 6);
  EXPECT_EQ(sent[1].frame, 6);
}

TEST(Answer, AHeldFrameAdvancesOnlyOnItsOwnBoundary) {
  int fds[2];
  ASSERT_EQ(pipe(fds), 0);
  ASSERT_EQ(write(fds[1], "??", 2), 2);
  close(fds[1]);
  std::vector<Record> sent;
  Ticks clock({1000 + 150, 1000 - 500});
  answer(
      fds[0], [&] { return std::optional<Record>(Record{5, 9, 2, 1000, 0}); },
      [&](const Record &r) { sent.push_back(r); }, {100, 1}, 2, clock);
  close(fds[0]);
  ASSERT_EQ(sent.size(), 2u);
  EXPECT_EQ(sent[0].v, 6);
  EXPECT_EQ(sent[0].frame, 3);
  EXPECT_EQ(sent[1].v, 5) << "a clock before the flip carries nothing on";
}

TEST_F(Pipes, EachFlipIsLoggedOfferedLiveAndKeptForReplies) {
  fifo(path);
  int fd = reader(path);
  LiveFeed live(path);
  std::string log_path = (dir / "flips.bin").string();
  int log = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  Recorder recorder(log, &live, 4);
  EXPECT_FALSE(recorder.last().load());
  for (int64_t f = 0; f < 3; ++f) {
    recorder.record(rec(f));
    recorder.wake();
  }
  EXPECT_EQ(recorder.last().load()->frame, 2);
  recorder.close();
  close(log);
  std::ifstream in(log_path, std::ios::binary);
  std::string logged((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
  EXPECT_EQ(logged, bytes({0, 1, 2}));
  EXPECT_EQ(drain(fd), bytes({0, 1, 2}));
  close(fd);
}

TEST_F(Pipes, AFlipLogThatFallsBehindIsAFault) {
  Recorder recorder(-1, nullptr, 2);
  recorder.close();
  recorder.record(rec(0));
  recorder.record(rec(1));
  EXPECT_FAULT(recorder.record(rec(2)), "fell 2 records behind");
  EXPECT_THROW(Recorder(-1, nullptr, 3), std::invalid_argument);
}

TEST_F(Pipes, AsksAreAnsweredOnTheLiveFeed) {
  fifo(path);
  int fd = reader(path);
  LiveFeed live(path);
  Recorder recorder(-1, &live, 4);
  std::string ask = (dir / "ask").string();
  recorder.serve_asks(ask, {1000000000, 1}, 1);
  recorder.record({7, 70, 7, MonotonicClock().now(), 0});
  recorder.wake();
  int asker = open(ask.c_str(), O_WRONLY);
  ASSERT_EQ(write(asker, "?", 1), 1);
  std::string got;
  for (int k = 0; k < 500 && got.size() < 2 * sizeof(Record); ++k) {
    got += drain(fd);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  close(asker);
  ASSERT_EQ(got.size(), 2 * sizeof(Record)) << "the record, then the reply";
  Record reply;
  std::memcpy(&reply, got.data() + sizeof(Record), sizeof reply);
  EXPECT_EQ(reply.frame, 7);
  EXPECT_EQ(reply.flags, REPLY);
  recorder.close();
  close(fd);
}

TEST(Clock, SleepsAtLeastAsLongAsAsked) {
  MonotonicClock clock;
  int64_t t0 = clock.now();
  clock.sleep_ns(2000000);
  EXPECT_GE(clock.now() - t0, 2000000);
}

} // namespace
} // namespace kmsplay::test
