#include "app.hpp"

#include <CLI/CLI.hpp>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace kmsplay {

std::variant<Args, int> parse_args(int argc, const char *const *argv) {
  Args args;
  CLI::App app{"Vsync-locked clip player for a DRM/KMS display"};
  auto *source = app.add_option_group("source");
  source->add_option("--clip", args.clip, "clip to decode");
  source->add_option("--frames", args.frames, "directory of raw frame.* files");
  source->require_option(1);
  std::string fps, spans;
  app.add_option("--fps", fps)->required();
  app.add_flag("--loop", args.loop);
  app.add_option("--spans", spans,
                 "play only these clip frame ranges, a:b,a:b");
  app.add_option("--width", args.width);
  app.add_option("--height", args.height);
  app.add_option("--device", args.device);
  app.add_option("--connector", args.connector);
  app.add_option("--preroll-s", args.preroll_s);
  app.add_option("--flips", args.flips, "binary log of every flip")->required();
  app.add_option("--live", args.live,
                 "named pipe offered every flip record as it completes");
  app.add_option("--ask", args.ask,
                 "named pipe whose every byte is answered on --live with the "
                 "last flip");
  app.add_option("--status", args.status, "JSON status file")->required();
  app.add_option("--fb", args.fb, "fbdev left holding the last frame");
  try {
    app.parse(argc, argv);
    args.fps = parse_rate(fps);
    if (!spans.empty()) {
      args.spans = parse_spans(spans);
    }
  } catch (const CLI::CallForHelp &e) {
    return app.exit(e);
  } catch (const CLI::ParseError &e) {
    app.exit(e);
    return 2;
  } catch (const std::invalid_argument &e) {
    std::fprintf(stderr, "%s\n%s\n", app.help().c_str(), e.what());
    return 2;
  }
  return args;
}

void write_status(const std::string &path, const Status &status) {
  std::string tmp = path + ".tmp", text = status.dump();
  int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
  if (fd < 0) {
    throw Fault("cannot write " + tmp + ": " + std::strerror(errno));
  }
  bool whole =
      write(fd, text.data(), text.size()) == static_cast<ssize_t>(text.size());
  close(fd);
  if (!whole || rename(tmp.c_str(), path.c_str()) != 0) {
    throw Fault("cannot write " + path + ": " + std::strerror(errno));
  }
}

size_t preroll_depth(double preroll_s, Rate fps) {
  return static_cast<size_t>(
      std::max(2.0, std::nearbyint(preroll_s * to_double(fps))));
}

void report(Status &status, const Player &player, bool faulted) {
  const Timeline &tl = player.timeline;
  status["flips"] = player.flips;
  status["segments"] = player.segments;
  status["ended"] = player.ended;
  status["worst_us"] = static_cast<double>(tl.worst_ns) / 1e3;
  status["worst_commit_us"] =
      player.worst_commit_ns
          ? Status(static_cast<double>(*player.worst_commit_ns) / 1e3)
          : Status(nullptr);
  Status phases = Status::array();
  for (int64_t p : tl.phases_ns) {
    phases.push_back(static_cast<double>(p) / 1e3);
  }
  status["phases_us"] = phases;
  status["t0_ns"] = tl.t0 ? Status(*tl.t0) : Status(nullptr);
  if (!faulted) {
    return;
  }
  auto after = [](const Trace &t, int64_t at) {
    return at < 0 ? Status(nullptr)
                  : Status(static_cast<double>(at - t.t_ns) / 1e3);
  };
  Status trace = Status::array();
  for (const Trace &t : player.trace()) {
    trace.push_back({{"v", t.v},
                     {"t_ns", t.t_ns},
                     {"wake_us", after(t, t.wake_ns)},
                     {"commit_us", after(t, t.commit_ns)},
                     {"ready_us", after(t, t.ready_ns)}});
  }
  status["trace"] = trace;
}

} // namespace kmsplay
