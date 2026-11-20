// The player's command line and status file.
#pragma once

#include "decode.hpp"

#include <nlohmann/json.hpp>
#include <variant>

namespace kmsplay {

using Status = nlohmann::ordered_json;

struct Args {
  std::optional<std::string> clip, frames, live, ask;
  Rate fps;
  bool loop = false;
  std::vector<Span> spans;
  int width = 1920, height = 1080;
  std::string device = "/dev/dri/card1", connector = "HDMI-A-1";
  double preroll_s = 2.0;
  std::string flips, status, fb = "/dev/fb0";
};

// The command line parsed, or the exit code a refused one ends the player
// with, its usage printed.
std::variant<Args, int> parse_args(int argc, const char *const *argv);

// Replace the status file atomically, so a reader never sees half of one.
void write_status(const std::string &path, const Status &status);

// The frames read ahead: refresh * preroll_s, at least two.
size_t preroll_depth(double preroll_s, Rate fps);

// What a finished or faulted play reports: its counts, timeline and, after a
// fault, the trace of its last flips, all times in microseconds.
void report(Status &status, const Player &player, bool faulted);

} // namespace kmsplay
