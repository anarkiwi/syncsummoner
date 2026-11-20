// Frame sources for the read-ahead: ffmpeg decoders, one per span, and raw
// frame directories.
#pragma once

#include "kmsplay.hpp"

#include <sys/types.h>
#include <utility>

namespace kmsplay {

using Argv = std::vector<std::string>;
using Span = std::pair<int64_t, int64_t>;

// ffmpeg decoding a clip to raw frames at the display's geometry and layout, as
// fast as it goes. span keeps the clip's frames a..b-1 only, for a clip at
// rate: the input is sought half a frame before frame a, so the decoder starts
// at the keyframe before it and drops every frame earlier than a however its
// timestamps round; trim then counts from a.
Argv decode_argv(const std::string &path, int width, int height, bool loop,
                 const std::optional<Span> &span = std::nullopt,
                 Rate rate = {1, 1}, const std::string &ffmpeg = "ffmpeg");

// Clip frame ranges given as a:b,a:b, each a..b-1; throws
// std::invalid_argument.
std::vector<Span> parse_spans(const std::string &text);

// A child process, its stdout and stderr piped back. It is killed when the
// thread that started it exits, so a dying player cannot leak it.
struct Child {
  pid_t pid = -1;
  int out = -1, err = -1;
};
Child spawn(const Argv &argv);
// Reap a child, killing it first when asked; its exit status as waitpid says.
int reap(Child &child, bool kill_first);

// The clip's frame rate as its container states it.
Rate clip_rate(const std::string &path, const std::string &ffprobe = "ffprobe");

// Whole frames from a decoder per argv in turn, as one stream with no seam.
// Each argv's decoder is started while the one before plays, so its decoding
// up to its first frame is done by the time that frame is due. A decoder that
// exits failing with something on stderr is a fault naming its last line.
class DecoderSource : public Source {
public:
  DecoderSource(std::vector<Argv> argvs, size_t frame_bytes);
  ~DecoderSource() override;
  bool next(uint8_t *dst) override;
  void cancel() override;
  // Decoders started so far.
  size_t started() const { return started_; }

private:
  struct Decoder {
    Child child;
    std::string line, said;
  };
  Decoder start(size_t index);
  void finish(Decoder &dec);
  static void hear(Decoder &dec);
  void stop_all();

  std::vector<Argv> argvs_;
  size_t bytes_, index_ = 0, started_ = 0;
  std::optional<Decoder> current_, upcoming_;
  int cancel_[2] = {-1, -1};
};

// Raw frames from frame.* files in a directory, read into memory once and
// cycled when looping.
class DirSource : public Source {
public:
  DirSource(const std::string &directory, bool loop, size_t frame_bytes);
  bool next(uint8_t *dst) override;
  size_t count() const { return frames_.size(); }

private:
  std::vector<std::string> frames_;
  bool loop_;
  size_t at_ = 0, bytes_;
};

} // namespace kmsplay
