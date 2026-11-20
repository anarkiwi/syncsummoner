#include "decode.hpp"

#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <poll.h>
#include <sched.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace kmsplay {

Argv decode_argv(const std::string &path, int width, int height, bool loop,
                 const std::optional<Span> &span, Rate rate,
                 const std::string &ffmpeg) {
  Argv argv = {ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"};
  if (loop) {
    argv.insert(argv.end(), {"-stream_loop", "-1"});
  }
  std::string trim;
  if (span) {
    if (span->first != 0) {
      double at = static_cast<double>(2 * span->first - 1) *
                  static_cast<double>(rate.den) /
                  (2.0 * static_cast<double>(rate.num));
      char buf[64];
      std::snprintf(buf, sizeof buf, "%.6f", at);
      argv.insert(argv.end(), {"-ss", buf});
    }
    trim = "trim=end_frame=" + std::to_string(span->second - span->first) + ",";
  }
  argv.insert(argv.end(), {"-i", path, "-map", "0:v:0", "-vf",
                           trim + "scale=" + std::to_string(width) + ":" +
                               std::to_string(height),
                           "-fps_mode", "passthrough", "-pix_fmt", "rgb565le",
                           "-f", "rawvideo", "pipe:1"});
  return argv;
}

std::vector<Span> parse_spans(const std::string &text) {
  std::vector<Span> spans;
  std::stringstream parts(text);
  std::string part;
  auto bad = [&] {
    return std::invalid_argument("spans must be a:b with 0 <= a < b, got '" +
                                 text + "'");
  };
  while (std::getline(parts, part, ',')) {
    if (part.empty()) {
      continue;
    }
    auto colon = part.find(':');
    if (colon == std::string::npos ||
        part.find(':', colon + 1) != std::string::npos) {
      throw bad();
    }
    try {
      size_t used_a = 0, used_b = 0;
      std::string a = part.substr(0, colon), b = part.substr(colon + 1);
      int64_t lo = std::stoll(a, &used_a), hi = std::stoll(b, &used_b);
      if (used_a != a.size() || used_b != b.size() || lo < 0 || lo >= hi) {
        throw bad();
      }
      spans.emplace_back(lo, hi);
    } catch (const std::logic_error &) {
      throw bad();
    }
  }
  if (spans.empty()) {
    throw bad();
  }
  return spans;
}

namespace {

std::string resolve(const std::string &name) {
  if (name.find('/') != std::string::npos) {
    return name;
  }
  const char *path = std::getenv("PATH");
  std::stringstream dirs(path != nullptr ? path : "/usr/bin:/bin");
  std::string dir;
  while (std::getline(dirs, dir, ':')) {
    std::string full = (dir.empty() ? "." : dir) + "/" + name;
    if (access(full.c_str(), X_OK) == 0) {
      return full;
    }
  }
  return name;
}

struct Exec {
  const char *path;
  char *const *argv;
  int out, err;
  pid_t parent;
  sigset_t mask;
  int error;
};

// Runs in the child, sharing the parent's memory until execve, so it touches
// only its own stack and the Exec it is handed.
int exec_child(void *arg) {
  auto *exec = static_cast<Exec *>(arg);
  prctl(PR_SET_PDEATHSIG, SIGKILL);
  if (getppid() != exec->parent) {
    _exit(127);
  }
  struct sigaction dfl{};
  dfl.sa_handler = SIG_DFL;
  for (int sig = 1; sig < NSIG; ++sig) {
    sigaction(sig, &dfl, nullptr);
  }
  sigprocmask(SIG_SETMASK, &exec->mask, nullptr);
  if (dup2(exec->out, 1) < 0 || dup2(exec->err, 2) < 0) {
    exec->error = errno;
    _exit(127);
  }
  execve(exec->path, exec->argv, environ);
  exec->error = errno;
  _exit(127);
}

} // namespace

Child spawn(const Argv &argv) {
  int out[2], err[2];
  if (pipe2(out, O_CLOEXEC) != 0) {
    throw Fault(std::string("cannot make a pipe: ") + std::strerror(errno));
  }
  if (pipe2(err, O_CLOEXEC) != 0) {
    close(out[0]);
    close(out[1]);
    throw Fault(std::string("cannot make a pipe: ") + std::strerror(errno));
  }
  std::string path = resolve(argv.at(0));
  std::vector<char *> args;
  for (const std::string &a : argv) {
    args.push_back(const_cast<char *>(a.c_str()));
  }
  args.push_back(nullptr);
  constexpr size_t stack_bytes = 1 << 16;
  void *stack = mmap(nullptr, stack_bytes, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
  if (stack == MAP_FAILED) {
    throw Fault(std::string("cannot map a stack: ") + std::strerror(errno));
  }
  Exec exec{path.c_str(), args.data(), out[1], err[1], getpid(), {}, 0};
  sigset_t all;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &exec.mask);
  pid_t pid = clone(exec_child, static_cast<char *>(stack) + stack_bytes,
                    CLONE_VM | CLONE_VFORK | SIGCHLD, &exec);
  int cloned = errno;
  pthread_sigmask(SIG_SETMASK, &exec.mask, nullptr);
  munmap(stack, stack_bytes);
  close(out[1]);
  close(err[1]);
  Child child{pid, out[0], err[0]};
  if (pid < 0 || exec.error != 0) {
    if (pid > 0) {
      reap(child, false);
    } else {
      close(out[0]);
      close(err[0]);
    }
    throw Fault("cannot run " + argv[0] + ": " +
                std::strerror(pid < 0 ? cloned : exec.error));
  }
  return child;
}

int reap(Child &child, bool kill_first) {
  int status = 0;
  if (child.pid > 0) {
    if (kill_first) {
      kill(child.pid, SIGKILL);
    }
    while (waitpid(child.pid, &status, 0) < 0 && errno == EINTR) {
    }
    child.pid = -1;
  }
  for (int *fd : {&child.out, &child.err}) {
    if (*fd >= 0) {
      close(*fd);
      *fd = -1;
    }
  }
  return status;
}

namespace {

std::string read_all(int fd) {
  std::string out;
  char buf[4096];
  for (;;) {
    ssize_t n = read(fd, buf, sizeof buf);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    if (n <= 0) {
      return out;
    }
    out.append(buf, static_cast<size_t>(n));
  }
}

std::string strip(const std::string &s) {
  auto a = s.find_first_not_of(" \t\r\n");
  if (a == std::string::npos) {
    return "";
  }
  return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

} // namespace

Rate clip_rate(const std::string &path, const std::string &ffprobe) {
  Child child =
      spawn({ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate", "-of", "csv=p=0", path});
  std::string out = read_all(child.out);
  std::string err = read_all(child.err);
  int status = reap(child, false);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw Fault(ffprobe + " could not read " + path + ": " + strip(err));
  }
  try {
    return parse_rate(strip(out));
  } catch (const std::invalid_argument &e) {
    throw Fault(path + ": " + e.what());
  }
}

DecoderSource::DecoderSource(std::vector<Argv> argvs, size_t frame_bytes)
    : argvs_(std::move(argvs)), bytes_(frame_bytes) {
  if (pipe2(cancel_, O_CLOEXEC) != 0) {
    throw Fault(std::string("cannot make a pipe: ") + std::strerror(errno));
  }
}

DecoderSource::~DecoderSource() {
  stop_all();
  close(cancel_[0]);
  close(cancel_[1]);
}

void DecoderSource::stop_all() {
  for (std::optional<Decoder> *dec : {&current_, &upcoming_}) {
    if (*dec) {
      reap((*dec)->child, true);
      dec->reset();
    }
  }
}

void DecoderSource::cancel() {
  char byte = 0;
  while (write(cancel_[1], &byte, 1) < 0 && errno == EINTR) {
  }
}

DecoderSource::Decoder DecoderSource::start(size_t index) {
  ++started_;
  return Decoder{spawn(argvs_[index]), "", ""};
}

void DecoderSource::hear(Decoder &dec) {
  char buf[4096];
  ssize_t n = read(dec.child.err, buf, sizeof buf);
  if (n <= 0) {
    if (n == 0 || errno != EINTR) {
      close(dec.child.err);
      dec.child.err = -1;
    }
    return;
  }
  for (ssize_t k = 0; k < n; ++k) {
    if (buf[k] == '\n') {
      if (!strip(dec.line).empty()) {
        dec.said = strip(dec.line);
      }
      dec.line.clear();
    } else {
      dec.line.push_back(buf[k]);
    }
  }
}

void DecoderSource::finish(Decoder &dec) {
  while (dec.child.err >= 0) {
    hear(dec);
  }
  if (!strip(dec.line).empty()) {
    dec.said = strip(dec.line);
  }
  int status = reap(dec.child, false);
  bool failed = WIFEXITED(status) ? WEXITSTATUS(status) != 0
                                  : WTERMSIG(status) != SIGKILL;
  if (failed && !dec.said.empty()) {
    throw Fault("decoding failed: " + dec.said);
  }
}

bool DecoderSource::next(uint8_t *dst) {
  try {
    for (;;) {
      if (!current_) {
        if (index_ >= argvs_.size()) {
          return false;
        }
        if (upcoming_) {
          current_ = std::move(upcoming_);
          upcoming_.reset();
        } else {
          current_ = start(index_);
        }
        if (index_ + 1 < argvs_.size()) {
          upcoming_ = start(index_ + 1);
        }
      }
      size_t got = 0;
      while (got < bytes_) {
        pollfd fds[4] = {{current_->child.out, POLLIN, 0},
                         {current_->child.err, POLLIN, 0},
                         {upcoming_ ? upcoming_->child.err : -1, POLLIN, 0},
                         {cancel_[0], POLLIN, 0}};
        if (poll(fds, 4, -1) < 0) {
          if (errno == EINTR) {
            continue;
          }
          throw Fault(std::string("poll: ") + std::strerror(errno));
        }
        if (fds[3].revents != 0) {
          stop_all();
          return false;
        }
        if (fds[1].revents != 0) {
          hear(*current_);
        }
        if (fds[2].revents != 0) {
          hear(*upcoming_);
        }
        if (fds[0].revents == 0) {
          continue;
        }
        ssize_t n = read(current_->child.out, dst + got, bytes_ - got);
        if (n < 0 && errno == EINTR) {
          continue;
        }
        if (n <= 0) {
          break;
        }
        got += static_cast<size_t>(n);
      }
      if (got == bytes_) {
        return true;
      }
      Decoder done = std::move(*current_);
      current_.reset();
      ++index_;
      finish(done);
    }
  } catch (...) {
    stop_all();
    throw;
  }
}

DirSource::DirSource(const std::string &directory, bool loop,
                     size_t frame_bytes)
    : loop_(loop), bytes_(frame_bytes) {
  std::vector<std::string> names;
  if (DIR *dir = opendir(directory.c_str())) {
    while (dirent *entry = readdir(dir)) {
      if (std::strncmp(entry->d_name, "frame.", 6) == 0) {
        names.emplace_back(entry->d_name);
      }
    }
    closedir(dir);
  }
  std::sort(names.begin(), names.end());
  for (const std::string &name : names) {
    std::ifstream in(directory + "/" + name, std::ios::binary);
    frames_.emplace_back(std::istreambuf_iterator<char>(in),
                         std::istreambuf_iterator<char>());
  }
}

bool DirSource::next(uint8_t *dst) {
  if (frames_.empty() || (at_ >= frames_.size() && !loop_)) {
    return false;
  }
  const std::string &frame = frames_[at_++ % frames_.size()];
  if (frame.size() != bytes_) {
    throw Fault("frame of " + std::to_string(frame.size()) +
                " bytes, buffer of " + std::to_string(bytes_));
  }
  std::memcpy(dst, frame.data(), bytes_);
  return true;
}

} // namespace kmsplay
