// Plays a clip or a frame directory until stopped; exits 2 on any fault.
#include "app.hpp"
#include "kms.hpp"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace kmsplay;

namespace {

Signals signals;

void install(int sig, std::atomic<bool> &flag) {
  static std::atomic<bool> *flags[NSIG] = {};
  flags[sig] = &flag;
  struct sigaction sa{};
  sa.sa_handler = [](int s) { flags[s]->store(true); };
  sigemptyset(&sa.sa_mask);
  sigaction(sig, &sa, nullptr);
}

// Lock every page now and to come, but only when nothing caps how much may be
// locked: a capped MCL_FUTURE fails later allocations instead.
bool lock_memory() {
  rlimit lim{};
  return getrlimit(RLIMIT_MEMLOCK, &lim) == 0 &&
         lim.rlim_cur == RLIM_INFINITY &&
         mlockall(MCL_CURRENT | MCL_FUTURE) == 0;
}

bool realtime() {
  sched_param param{};
  param.sched_priority = REALTIME_PRIORITY;
  return sched_setscheduler(0, SCHED_FIFO, &param) == 0;
}

std::unique_ptr<Source> source_for(const Args &args, size_t frame_bytes) {
  if (args.frames) {
    return std::make_unique<DirSource>(*args.frames, args.loop, frame_bytes);
  }
  std::vector<Argv> argvs;
  if (args.spans.empty()) {
    argvs.push_back(
        decode_argv(*args.clip, args.width, args.height, args.loop));
  }
  for (const Span &span : args.spans) {
    argvs.push_back(decode_argv(*args.clip, args.width, args.height, false,
                                span, args.fps));
  }
  return std::make_unique<DecoderSource>(std::move(argvs), frame_bytes);
}

} // namespace

int main(int argc, char **argv) {
  auto parsed = parse_args(argc, argv);
  if (auto *code = std::get_if<int>(&parsed)) {
    return *code;
  }
  const Args &args = std::get<Args>(parsed);
  install(SIGTERM, signals.stop);
  install(SIGINT, signals.stop);
  install(SIGUSR1, signals.down);
  install(SIGUSR2, signals.up);
  signal(SIGPIPE, SIG_IGN);

  Status status = {{"state", "starting"}, {"pid", getpid()}};
  write_status(args.status, status);
  std::unique_ptr<KmsDisplay> display;
  std::unique_ptr<Prerolled> preroll;
  std::unique_ptr<Player> player;
  // The asks thread outlives main, so what it reads is never destroyed.
  LiveFeed *live = nullptr;
  Recorder *recorder = nullptr;
  int code = 0;
  bool faulted = false;
  try {
    if (args.clip) {
      Rate rate = clip_rate(*args.clip);
      if (!(rate == args.fps)) {
        throw Fault(*args.clip + " runs at " + to_string(rate) + " fps, not " +
                    to_string(args.fps));
      }
    }
    display = std::make_unique<KmsDisplay>(args.device, args.connector,
                                           args.width, args.height, args.fps);
    if (!args.spans.empty() && !args.clip) {
      throw Fault("spans play a clip, not a frame directory");
    }
    preroll = std::make_unique<Prerolled>(
        source_for(args, display->frame_bytes),
        preroll_depth(args.preroll_s, args.fps), display->frame_bytes);
    bool locked = lock_memory();
    preroll->wait_full(60.0);
    if (args.live) {
      live = new LiveFeed(*args.live);
    }
    int log = open(args.flips.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                   0666);
    if (log < 0) {
      throw Fault("cannot write " + args.flips + ": " + std::strerror(errno));
    }
    recorder = new Recorder(log, live);
    if (live != nullptr && args.ask) {
      recorder->serve_asks(*args.ask, display->period_ns, display->hold);
    }
    MonotonicClock clock;
    player =
        std::make_unique<Player>(*display, *preroll, clock, *recorder, signals,
                                 display->hold, display->period_ns);
    status["state"] = "playing";
    status["mode"] = display->mode;
    status["modeset"] = display->modeset;
    status["hold"] = display->hold;
    status["refresh"] = to_string(display->refresh_hz);
    status["fps"] = to_string(args.fps);
    status["preroll"] = preroll->depth;
    status["locked"] = locked;
    status["realtime"] = realtime();
    write_status(args.status, status);
    player->run();
    status["state"] = "stopped";
  } catch (const std::exception &err) {
    std::string said = dynamic_cast<const Fault *>(&err) != nullptr
                           ? err.what()
                           : std::string("error: ") + err.what();
    std::fprintf(stderr, "PLAYOUT FAULT: %s\n", said.c_str());
    std::fflush(stderr);
    status["state"] = "failed";
    status["error"] = said;
    code = 2;
    faulted = true;
  }
  if (recorder != nullptr) {
    recorder->close();
  }
  if (live != nullptr) {
    live->close();
  }
  if (player) {
    report(status, *player, faulted);
    struct stat st{};
    if (player->shown && stat(args.fb.c_str(), &st) == 0) {
      int fb = open(args.fb.c_str(), O_WRONLY | O_CLOEXEC);
      if (fb >= 0) {
        std::vector<uint8_t> last = display->frame(player->shown->buf);
        if (write(fb, last.data(), last.size()) < 0) {
          std::fprintf(stderr, "cannot write %s: %s\n", args.fb.c_str(),
                       std::strerror(errno));
        }
        close(fb);
      }
    }
  }
  preroll.reset();
  try {
    write_status(args.status, status);
  } catch (const Fault &err) {
    std::fprintf(stderr, "%s\n", err.what());
  }
  std::fflush(nullptr);
  _exit(code);
}
