// Decoder argv, span parsing, decoder sources and frame directories.
#include "decode.hpp"
#include "fake.hpp"

#include <filesystem>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>

namespace kmsplay::test {
namespace {

size_t find(const Argv &argv, const std::string &s) {
  auto it = std::find(argv.begin(), argv.end(), s);
  return it == argv.end() ? std::string::npos
                          : static_cast<size_t>(it - argv.begin());
}

// A decoder that writes bytes to stdout, and optionally text to stderr and an
// exit code.
Argv sh(const std::string &script) { return {"sh", "-c", script}; }

std::vector<std::string> all(Source &src, size_t size) {
  std::vector<std::string> out;
  std::string buf(size, '\0');
  while (src.next(reinterpret_cast<uint8_t *>(buf.data()))) {
    out.push_back(buf);
  }
  return out;
}

TEST(DecodeArgv, KeepsEveryFrameAtDisplayGeometry) {
  Argv argv = decode_argv("c.mkv", 1280, 720, true);
  EXPECT_EQ(find(argv, "-re"), std::string::npos);
  EXPECT_EQ(find(argv, "-r"), std::string::npos);
  EXPECT_EQ(argv.at(find(argv, "-stream_loop") + 1), "-1");
  EXPECT_NE(find(argv, "scale=1280:720"), std::string::npos);
  EXPECT_NE(find(argv, "bgr565le"), std::string::npos);
  EXPECT_EQ(find(decode_argv("c.mkv", 8, 8, false), "-stream_loop"),
            std::string::npos);
}

TEST(DecodeArgv, ASpanIsSoughtHalfAFrameEarlyAndTrimmedBeforeItIsScaled) {
  Argv argv = decode_argv("c.mkv", 8, 6, false, Span{3, 9}, {30, 1});
  EXPECT_EQ(argv.at(find(argv, "-vf") + 1), "trim=end_frame=6,scale=8:6");
  EXPECT_EQ(argv.at(find(argv, "-ss") + 1), "0.083333");
  EXPECT_LT(find(argv, "-ss"), find(argv, "-i"));
  EXPECT_EQ(find(decode_argv("c.mkv", 8, 6, false, Span{0, 9}, {30, 1}), "-ss"),
            std::string::npos);
  Argv ntsc = decode_argv("c.mkv", 8, 6, false, Span{395, 6503}, {30000, 1001});
  EXPECT_EQ(ntsc.at(find(ntsc, "-ss") + 1), "13.163150");
}

TEST(Spans, ParseAsRangesAndRefuseEmptyOrBackwardOnes) {
  EXPECT_EQ(parse_spans("0:5,2:9,"), (std::vector<Span>{{0, 5}, {2, 9}}));
  for (const char *bad :
       {"", "5:5", "3:1", "-1:4", "1:2:3", "a:b", "1:2x", "4"}) {
    EXPECT_THROW(parse_spans(bad), std::invalid_argument) << bad;
  }
}

TEST(Decoder, ReadsWholeFramesAndReportsADecoderError) {
  DecoderSource good({sh("printf abcdefghi")}, 4);
  EXPECT_EQ(all(good, 4), (std::vector<std::string>{"abcd", "efgh"}));
  DecoderSource bad({sh("echo 'bad input' >&2; exit 1")}, 4);
  EXPECT_FAULT(all(bad, 4), "bad input");
}

TEST(Decoder, AFailingDecoderFaultsWithItsLastLine) {
  DecoderSource bad({sh("printf 'first\\nlast words\\n\\n' >&2; exit 3")}, 2);
  EXPECT_FAULT(all(bad, 2), "decoding failed: last words$");
  DecoderSource unterminated({sh("printf 'only words' >&2; exit 3")}, 2);
  EXPECT_FAULT(all(unterminated, 2), "decoding failed: only words$");
  DecoderSource quiet({sh("exit 3")}, 2);
  EXPECT_TRUE(all(quiet, 2).empty()) << "a silent failure is just an end";
}

TEST(Decoder, ADecoderThatCannotStartIsAFault) {
  DecoderSource missing({{"/nonexistent/ffmpeg"}}, 2);
  EXPECT_FAULT(all(missing, 2), "cannot run /nonexistent/ffmpeg");
}

TEST(Decoder, ADecoderChatteringOnStderrNeverStallsItsFrames) {
  DecoderSource chatty(
      {sh("i=0; while [ $i -lt 20000 ]; do echo warning $i >&2; "
          "i=$((i+1)); done; printf abcdef")},
      2);
  EXPECT_EQ(all(chatty, 2), (std::vector<std::string>{"ab", "cd", "ef"}));
}

TEST(Decoder, SpansPlayAsOneStreamEachNextDecoderStartedBeforeItIsDue) {
  DecoderSource spans({sh("printf ab"), sh("printf cde"), sh("printf f")}, 1);
  std::string buf(1, '\0');
  auto *dst = reinterpret_cast<uint8_t *>(buf.data());
  ASSERT_TRUE(spans.next(dst));
  EXPECT_EQ(buf, "a");
  EXPECT_EQ(spans.started(), 2u)
      << "the next span decodes while this one plays";
  std::string rest;
  while (spans.next(dst)) {
    rest += buf;
  }
  EXPECT_EQ(rest, "bcdef");
  EXPECT_EQ(spans.started(), 3u);
}

pid_t pid_in(const std::string &path) {
  for (int k = 0; k < 500; ++k) {
    std::ifstream in(path);
    pid_t pid = 0;
    if (in >> pid) {
      return pid;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return 0;
}

bool alive(pid_t pid) { return kill(pid, 0) == 0; }

TEST(Decoder, ClosingEarlyReapsBothDecoders) {
  auto dir = std::filesystem::temp_directory_path() /
             ("kmsplay-reap-" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  std::string a = (dir / "a").string(), b = (dir / "b").string();
  pid_t pa = 0, pb = 0;
  {
    DecoderSource spans({sh("echo $$ > " + a + "; printf xy; exec sleep 60"),
                         sh("echo $$ > " + b + "; exec sleep 60")},
                        1);
    std::string buf(1, '\0');
    ASSERT_TRUE(spans.next(reinterpret_cast<uint8_t *>(buf.data())));
    pa = pid_in(a);
    pb = pid_in(b);
    ASSERT_TRUE(pa > 0 && pb > 0 && alive(pa) && alive(pb));
  }
  EXPECT_FALSE(alive(pa));
  EXPECT_FALSE(alive(pb));
  std::filesystem::remove_all(dir);
}

TEST(Decoder, CancellingUnblocksAWaitingRead) {
  auto pre = std::make_unique<Prerolled>(
      std::make_unique<DecoderSource>(std::vector<Argv>{sh("exec sleep 60")},
                                      4),
      2, 4);
  auto t0 = std::chrono::steady_clock::now();
  pre.reset();
  EXPECT_LT(std::chrono::steady_clock::now() - t0, std::chrono::seconds(30));
}

TEST(Decoder, ADecoderIsKilledWhenTheThreadThatStartedItExits) {
  pid_t pid = 0;
  std::thread([&] { pid = spawn(sh("exec sleep 60")).pid; }).join();
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_TRUE(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
}

TEST(ClipRate, ReadsTheContainer) {
  auto dir = std::filesystem::temp_directory_path() /
             ("kmsplay-probe-" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  std::string probe = (dir / "ffprobe").string();
  {
    std::ofstream(probe) << "#!/bin/sh\n"
                            "case \"$*\" in *bad*) echo 'no such clip' >&2; "
                            "exit 1;; *odd*) echo 'N/A';; *) echo 30000/1001;; "
                            "esac\n";
  }
  std::filesystem::permissions(probe, std::filesystem::perms::owner_all);
  EXPECT_EQ(clip_rate("x.mkv", probe), (Rate{30000, 1001}));
  EXPECT_FAULT(clip_rate("bad.mkv", probe), "no such clip");
  EXPECT_FAULT(clip_rate("odd.mkv", probe), "odd.mkv: frame rate");
  std::filesystem::remove_all(dir);
}

std::string ffmpeg_path() {
  for (const char *dir : {"/usr/bin", "/usr/local/bin", "/bin"}) {
    std::string p = std::string(dir) + "/ffmpeg";
    if (access(p.c_str(), X_OK) == 0) {
      return p;
    }
  }
  return "";
}

TEST(Decoder, ASoughtSpanDecodesTheFramesAFullDecodeTrimsTo) {
  if (ffmpeg_path().empty()) {
    GTEST_SKIP() << "needs ffmpeg";
  }
  auto dir = std::filesystem::temp_directory_path() /
             ("kmsplay-span-" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  std::string clip = (dir / "c.mkv").string();
  Child enc = spawn({"ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i",
                     "testsrc2=size=64x48:rate=30", "-frames:v", "90", "-c:v",
                     "libx264", "-g", "25", "-pix_fmt", "yuv420p", clip});
  ASSERT_EQ(reap(enc, false), 0);
  const size_t size = 64 * 48 * 2;
  DecoderSource whole_src({decode_argv(clip, 64, 48, false)}, size);
  std::string whole;
  for (const std::string &f : all(whole_src, size)) {
    whole += f;
  }
  ASSERT_EQ(whole.size(), 90 * size);
  for (Span span : {Span{0, 7}, Span{1, 4}, Span{26, 40}, Span{52, 90}}) {
    DecoderSource src({decode_argv(clip, 64, 48, false, span, {30, 1})}, size);
    std::string got;
    for (const std::string &f : all(src, size)) {
      got += f;
    }
    EXPECT_EQ(
        got, whole.substr(span.first * size, (span.second - span.first) * size))
        << span.first << ":" << span.second;
  }
  std::filesystem::remove_all(dir);
}

TEST(DirFrames, CycleWhenLooping) {
  auto dir = std::filesystem::temp_directory_path() /
             ("kmsplay-dir-" + std::to_string(getpid()));
  std::filesystem::create_directories(dir);
  for (char k = 0; k < 3; ++k) {
    std::ofstream(dir / ("frame.00" + std::to_string(k))) << k;
  }
  DirSource looped(dir.string(), true, 1);
  std::string seen;
  uint8_t byte = 0;
  for (int k = 0; k < 7; ++k) {
    ASSERT_TRUE(looped.next(&byte));
    seen.push_back(static_cast<char>(byte));
  }
  EXPECT_EQ(seen, std::string("\0\1\2\0\1\2\0", 7));
  DirSource once(dir.string(), false, 1);
  EXPECT_EQ(all(once, 1).size(), 3u);
  DirSource none((dir / "none").string(), true, 1);
  EXPECT_FALSE(none.next(&byte));
  DirSource wrong(dir.string(), false, 2);
  uint8_t two[2];
  EXPECT_FAULT(wrong.next(two), "frame of 1 bytes, buffer of 2");
  std::filesystem::remove_all(dir);
}

} // namespace
} // namespace kmsplay::test
