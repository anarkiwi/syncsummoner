"""Standalone entry point: ``python -m syncsummoner.aesthetics score clip.mp4``.

Doubles as the extraction-seam test — it must run with the device stack absent.
"""

import argparse
import dataclasses
import enum
import json
import sys

import numpy as np

from syncsummoner.aesthetics import describe_clip, score_clip
from syncsummoner.aesthetics.io import read_clip


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, enum.Enum):
        return obj.value
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syncsummoner.aesthetics")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="print the ClipDescriptor and aggregate score as JSON")
    score.add_argument("clip")
    score.add_argument("--max-frames", type=int, default=None)
    score.add_argument("--stride", type=int, default=1)
    score.add_argument("--max-width", type=int, default=None)
    score.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, describe the clip and print the result as JSON."""
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    frames, fps = read_clip(
        args.clip, max_frames=args.max_frames, stride=args.stride, max_width=args.max_width
    )
    descriptor = describe_clip(frames, fps=fps, rng=np.random.default_rng(args.seed))
    payload = dataclasses.asdict(descriptor) | {"score": score_clip(descriptor)}
    print(json.dumps(payload, indent=2, default=_jsonable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
