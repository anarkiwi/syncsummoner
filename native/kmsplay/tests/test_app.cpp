// The command line, the status file and the mode's name.
#include "app.hpp"
#include "fake.hpp"
#include "kms.hpp"

#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace kmsplay::test {
namespace {

std::variant<Args, int> parse(std::vector<const char *> argv) {
  argv.insert(argv.begin(), "kmsplay");
  return parse_args(static_cast<int>(argv.size()), argv.data());
}

TEST(Arguments, NameOneSource) {
  auto got = parse({"--clip", "c.mkv", "--fps", "30", "--flips", "f",
                    "--status", "s", "--loop"});
  ASSERT_TRUE(std::holds_alternative<Args>(got));
  const Args &args = std::get<Args>(got);
  EXPECT_EQ(args.fps, (Rate{30, 1}));
  EXPECT_TRUE(args.loop);
  EXPECT_FALSE(args.frames);
  EXPECT_FALSE(args.live);
  EXPECT_EQ(args.device, "/dev/dri/card1");
  EXPECT_EQ(args.connector, "HDMI-A-1");
  EXPECT_EQ(args.fb, "/dev/fb0");
  EXPECT_EQ(args.width, 1920);
  EXPECT_EQ(args.height, 1080);
  EXPECT_DOUBLE_EQ(args.preroll_s, 2.0);
  EXPECT_TRUE(args.spans.empty());
  EXPECT_EQ(std::get<int>(parse({"--clip", "c", "--frames", "d", "--fps", "30",
                                 "--flips", "f", "--status", "s"})),
            2);
  EXPECT_EQ(
      std::get<int>(parse({"--fps", "30", "--flips", "f", "--status", "s"})),
      2);
  auto live = parse({"--frames", "d", "--fps", "30", "--flips", "f", "--status",
                     "s", "--live", "p"});
  EXPECT_EQ(std::get<Args>(live).live, "p");
}

TEST(Arguments, TakeSpansAndExactRates) {
  auto got = parse({"--clip", "c", "--fps", "30000/1001", "--flips", "f",
                    "--status", "s", "--spans", "395:6503", "--ask", "a",
                    "--preroll-s", "0.5", "--width", "8", "--height", "6"});
  const Args &args = std::get<Args>(got);
  EXPECT_EQ(args.fps, (Rate{30000, 1001}));
  EXPECT_EQ(args.spans, (std::vector<Span>{{395, 6503}}));
  EXPECT_EQ(args.ask, "a");
  EXPECT_EQ(args.width, 8);
  EXPECT_DOUBLE_EQ(args.preroll_s, 0.5);
  EXPECT_EQ(std::get<int>(parse({"--clip", "c", "--fps", "0", "--flips", "f",
                                 "--status", "s"})),
            2);
  EXPECT_EQ(std::get<int>(parse({"--clip", "c", "--fps", "30", "--flips", "f",
                                 "--status", "s", "--spans", "4:2"})),
            2);
  EXPECT_EQ(std::get<int>(parse({"--help"})), 0);
}

TEST(Status, IsReplacedWhole) {
  auto dir = std::filesystem::temp_directory_path() /
             ("kmsplay-status-" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  std::string path = (dir / "s.json").string();
  write_status(path, {{"state", "playing"}});
  std::ifstream in(path);
  EXPECT_EQ(Status::parse(in), (Status{{"state", "playing"}}));
  EXPECT_FALSE(std::filesystem::exists(path + ".tmp"));
  EXPECT_FAULT(write_status((dir / "none" / "s.json").string(), {}),
               "cannot write");
  std::filesystem::remove_all(dir);
}

TEST(Status, PrerollsTheFramesOfItsSecondsAtLeastTwo) {
  EXPECT_EQ(preroll_depth(2.0, {30, 1}), 60u);
  EXPECT_EQ(preroll_depth(0.01, {30, 1}), 2u);
  EXPECT_EQ(preroll_depth(2.0, {30000, 1001}), 60u);
}

TEST(Status, ReportsCountsAndOnAFaultTheTrace) {
  FakeDisplay display(PERIOD, {7});
  Signals signals;
  FnSink sink([](const Record &) {});
  auto frames = decoded(20);
  Player player(display, *frames, display, sink, signals, 1, PERIOD.rate());
  EXPECT_THROW(player.run(), Fault);
  Status status;
  report(status, player, true);
  EXPECT_EQ(status["flips"], 7);
  EXPECT_EQ(status["segments"], 1);
  EXPECT_EQ(status["ended"], false);
  EXPECT_EQ(status["worst_us"], 0.0);
  EXPECT_EQ(status["worst_commit_us"], 0.0);
  EXPECT_EQ(status["phases_us"], Status::array({0.0}));
  EXPECT_EQ(status["t0_ns"], 33333333);
  const Status &trace = status["trace"];
  ASSERT_EQ(trace.size(), 8u);
  EXPECT_EQ(trace.back()["v"], 7);
  EXPECT_TRUE(trace.back()["commit_us"].is_null());
  EXPECT_EQ(trace[0]["v"], 0);
  EXPECT_EQ(trace[0]["wake_us"], 0.0);
  EXPECT_EQ(trace[0]["ready_us"], 0.0);
  Status clean;
  report(clean, player, false);
  EXPECT_FALSE(clean.contains("trace"));
  Player idle(display, *frames, display, sink, signals, 1, PERIOD.rate());
  report(clean, idle, false);
  EXPECT_TRUE(clean["t0_ns"].is_null());
  EXPECT_TRUE(clean["worst_commit_us"].is_null());
}

TEST(Mode, IsNamedAsKmsxxNamesIt) {
  drmModeModeInfo m{};
  m.clock = 74250;
  m.hdisplay = 1920;
  m.hsync_start = 2008;
  m.hsync_end = 2052;
  m.htotal = 2200;
  m.vdisplay = 1080;
  m.vsync_start = 1084;
  m.vsync_end = 1089;
  m.vtotal = 1125;
  m.vrefresh = 30;
  m.flags = DRM_MODE_FLAG_NHSYNC | DRM_MODE_FLAG_NVSYNC;
  m.type = DRM_MODE_TYPE_PREFERRED | DRM_MODE_TYPE_DRIVER;
  EXPECT_EQ(mode_name(m), "1920x1080@30.00 74.250 1920/88/44/148/- "
                          "1080/4/5/36/- 30 (30.00) P|D ");
  m.flags = DRM_MODE_FLAG_PHSYNC | DRM_MODE_FLAG_PVSYNC |
            DRM_MODE_FLAG_INTERLACE | DRM_MODE_FLAG_PIC_AR_16_9;
  m.vsync_end = 1094;
  m.vrefresh = 60;
  m.type = DRM_MODE_TYPE_DRIVER;
  EXPECT_EQ(mode_name(m), "1920x1080i@60.00 74.250 1920/88/44/148/+ "
                          "1080/4/10/31/+ 60 (60.00) D 16:9");
  m.flags = DRM_MODE_FLAG_DBLCLK | DRM_MODE_FLAG_PIC_AR_4_3;
  m.type = DRM_MODE_TYPE_USERDEF;
  EXPECT_THAT(mode_name(m), ::testing::EndsWith("/? 60 (30.00) U 2x|4:3"));
}

} // namespace
} // namespace kmsplay::test
