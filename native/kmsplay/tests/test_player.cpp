// The player's pacing and miss detection, against a simulated display.
#include "fake.hpp"

namespace kmsplay::test {
namespace {

struct Played {
  std::unique_ptr<Player> player;
  std::vector<Record> records;
};

// Run a player over count frames, stopping after stop_after flips.
Played play(FakeDisplay &display, int64_t count, int64_t hold = 1,
            std::optional<size_t> stop_after = std::nullopt,
            std::optional<Q> period = std::nullopt) {
  Played out;
  Signals signals;
  size_t limit = stop_after.value_or(static_cast<size_t>(count * hold));
  FnSink sink([&](const Record &rec) {
    out.records.push_back(rec);
    if (out.records.size() >= limit) {
      signals.stop = true;
    }
  });
  auto frames = decoded(count);
  out.player =
      std::make_unique<Player>(display, *frames, display, sink, signals, hold,
                               period.value_or(display.period).rate());
  out.player->run();
  return out;
}

std::vector<int64_t> shown(const std::vector<Record> &records) {
  std::vector<int64_t> out;
  for (const Record &r : records) {
    out.push_back(r.frame);
  }
  return out;
}

std::vector<int64_t> upto(int64_t n) {
  std::vector<int64_t> out;
  for (int64_t k = 0; k < n; ++k) {
    out.push_back(k);
  }
  return out;
}

TEST(Player, EachFrameIsFlippedOnConsecutiveVblanks) {
  FakeDisplay display;
  auto [player, records] = play(display, 20);
  EXPECT_EQ(shown(records), upto(20));
  for (int64_t k = 0; k < 20; ++k) {
    EXPECT_EQ(records[k].seq, 1001 + k);
    EXPECT_EQ(records[k].v, k);
    EXPECT_EQ(records[k].flags, k == 0 ? SEGMENT : 0);
  }
  EXPECT_EQ(player->flips, 20);
  EXPECT_EQ(player->segments, 1);
}

TEST(Player, AFrameIsHeldForExactlyHoldVblanks) {
  FakeDisplay display(PERIOD / Q(2));
  auto [player, records] = play(display, 5, 2);
  EXPECT_EQ(shown(records),
            (std::vector<int64_t>{0, 0, 1, 1, 2, 2, 3, 3, 4, 4}));
  for (size_t k = 1; k < records.size(); ++k) {
    EXPECT_EQ(records[k].seq - records[k - 1].seq, 1);
  }
}

TEST(Player, TheScreenShowsTheFrameThatWasDue) {
  FakeDisplay display;
  play(display, 6);
  std::set<int> bufs;
  for (int64_t v = 0; v < 6; ++v) {
    EXPECT_EQ(display.shown[v].second, v);
    if (v < 3) {
      bufs.insert(display.shown[v].first);
    }
  }
  EXPECT_EQ(bufs.size(), 3u) << "three buffers in rotation";
}

TEST(Player, ALateFlipIsAFaultNotASlip) {
  FakeDisplay display(PERIOD, {7});
  EXPECT_FAULT(play(display, 20), "vblank 7 \\(frame 7\\).*1 vblank.*missed");
}

TEST(Player, AFlipThatNeverCompletesIsAFault) {
  FakeDisplay display(PERIOD, {}, {4});
  EXPECT_FAULT(play(display, 20), "no flip completed within 1.0s");
}

TEST(Player, ADriftingEventClockIsAFault) {
  FakeDisplay display(PERIOD * Q(101, 100));
  EXPECT_FAULT(play(display, 200, 1, std::nullopt, PERIOD), "off a grid");
  EXPECT_LT(Q(display.now()).floor(), (Q(100) * display.period).floor());
}

TEST(Player, AClipThatEndsHoldsItsLastFrame) {
  FakeDisplay display;
  Signals signals;
  std::vector<Record> records;
  FnSink sink([&](const Record &r) { records.push_back(r); });
  display.on_sleep = [&] { signals.stop = display.now() > 1000000000; };
  auto frames = decoded(4);
  Player player(display, *frames, display, sink, signals, 1, PERIOD.rate());
  player.run();
  EXPECT_TRUE(player.ended);
  EXPECT_EQ(shown(records), upto(4));
  EXPECT_EQ(display.screen, 3);
}

TEST(Player, AnEmptyClipIsAFault) {
  FakeDisplay display;
  Signals signals;
  FnSink sink([](const Record &) {});
  auto frames = decoded(0);
  Player player(display, *frames, display, sink, signals, 1, PERIOD.rate());
  EXPECT_FAULT(player.run(), "no frames");
}

TEST(Player, TheLinkDropKeepsTheTimelineRunning) {
  FakeDisplay display(PERIOD, {}, {}, PERIOD / Q(3));
  Signals signals;
  std::vector<Record> records;
  int sleeps = 0;
  display.on_sleep = [&] {
    if (++sleeps == 30) {
      signals.up = true;
    }
  };
  FnSink sink([&](const Record &r) {
    records.push_back(r);
    signals.stop = records.size() >= 60;
    if (records.size() == 10) {
      signals.down = true;
    }
  });
  auto frames = decoded(200);
  Player player(display, *frames, display, sink, signals, 1, PERIOD.rate());
  player.run();
  EXPECT_EQ(display.actives, (std::vector<bool>{false, true}));
  size_t resumed = 1;
  while (resumed < records.size() && (records[resumed].flags & SEGMENT) == 0) {
    ++resumed;
  }
  ASSERT_LT(resumed, records.size());
  const Record &r = records[resumed];
  EXPECT_EQ(r.v, static_cast<int64_t>(
                     round_half_even((Q(r.t_ns - records[0].t_ns) / PERIOD).n,
                                     (Q(r.t_ns - records[0].t_ns) / PERIOD).d)))
      << "the segment sits on the timeline, not where it paused";
  EXPECT_GT(r.v, 30);
  EXPECT_EQ(r.frame, 9) << "the first flip after the link returns shows the "
                           "frame left up";
  for (size_t k = resumed + 1; k < records.size(); ++k) {
    EXPECT_EQ(records[k].frame, records[k].v)
        << "and every later flip the frame due on it";
  }
  EXPECT_EQ(player.segments, 2);
  ASSERT_EQ(player.timeline.phases_ns.size(), 2u);
  EXPECT_NEAR(static_cast<double>(player.timeline.phases_ns[1]), 1e9 / 30 / 3,
              1.0);
}

TEST(Player, StopDuringTheLinkDropEndsThePlayer) {
  FakeDisplay display;
  Signals signals;
  display.on_sleep = [&] { signals.stop = true; };
  FnSink sink([&](const Record &r) { signals.down = r.v == 3; });
  auto frames = decoded(50);
  Player player(display, *frames, display, sink, signals, 1, PERIOD.rate());
  player.run();
  EXPECT_EQ(display.actives, std::vector<bool>{false});
}

TEST(Player, LinkRequestsAfterTheClipEndsStillDriveTheLink) {
  FakeDisplay display;
  Signals signals;
  int ticks = 0;
  display.on_sleep = [&] {
    switch (++ticks) {
    case 1:
      signals.down = true;
      break;
    case 2:
      signals.up = true;
      break;
    default:
      signals.stop = true;
    }
  };
  FnSink sink([](const Record &) {});
  auto frames = decoded(3);
  Player player(display, *frames, display, sink, signals, 1, PERIOD.rate());
  player.run();
  EXPECT_EQ(display.actives, (std::vector<bool>{false, true}));
}

TEST(Player, EachFlipIsTracedFromItsEventToTheNextCommitAndFrame) {
  FakeDisplay display;
  auto [player, records] = play(display, 100);
  std::vector<Trace> trace = player->trace();
  ASSERT_EQ(trace.size(), TRACE_DEPTH);
  EXPECT_EQ(trace.back().v, 99);
  EXPECT_EQ(trace.front().v, 100 - static_cast<int64_t>(TRACE_DEPTH));
  for (const Trace &t : trace) {
    EXPECT_EQ(t.wake_ns, t.t_ns) << "the fake clock stands still between "
                                    "an event and the pacer's work";
    EXPECT_TRUE(t.v == 99 || (t.commit_ns == t.t_ns && t.ready_ns == t.t_ns));
  }
  EXPECT_EQ(trace.back().commit_ns, -1) << "the clip ended, nothing committed";
  EXPECT_EQ(player->worst_commit_ns, 0);
  FakeDisplay idle;
  Signals signals;
  FnSink sink([](const Record &) {});
  auto frames = decoded(1);
  EXPECT_FALSE(Player(idle, *frames, idle, sink, signals, 1, PERIOD.rate())
                   .worst_commit_ns);
}

TEST(Player, TheTraceKeepsAllFlipsWhileFewerThanItsDepth) {
  FakeDisplay display;
  auto [player, records] = play(display, 5);
  EXPECT_EQ(player->trace().size(), 5u);
}

class SlowClock : public FakeDisplay {
public:
  int64_t now() override {
    t = t + Q(1000);
    return FakeDisplay::now();
  }
};

TEST(Player, TheLongestCommitIsTheWorstFromAnyEventToItsNextCommit) {
  SlowClock display;
  auto [player, records] = play(display, 10);
  EXPECT_GT(*player->worst_commit_ns, 0);
  for (const Trace &t : player->trace()) {
    EXPECT_GT(t.wake_ns, t.t_ns);
    if (t.commit_ns >= 0) {
      EXPECT_GT(t.commit_ns, t.wake_ns);
      EXPECT_GE(t.ready_ns, t.commit_ns);
      EXPECT_LE(t.commit_ns - t.t_ns, *player->worst_commit_ns);
    }
  }
}

} // namespace
} // namespace kmsplay::test
