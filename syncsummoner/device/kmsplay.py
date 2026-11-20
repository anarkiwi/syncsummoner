"""The flip record the Pi's player writes, as the host reads it.

The player itself is native (``native/kmsplay``): one ``SCHED_FIFO`` thread flips one frame per vblank
and checks every flip-complete event against the vblank it was committed for. Each flip is logged to
``--flips`` and offered on ``--live`` as one :data:`RECORD`.
"""

from __future__ import annotations

import struct
from typing import Any

#: One flip: clip vblank, display vblank sequence, clip frame shown, event time (ns), flags.
RECORD = struct.Struct("<qqqqq")
#: Flag on the first flip of a segment, after the start or after the link came back up.
SEGMENT = 1
#: Flag on a copy of the last flip done, sent in reply to a request on the ``--ask`` pipe.
REPLY = 2


def read_exactly(stream: Any, size: int) -> bytes | None:
    """One whole record from a pipe, or None when it ends first."""
    buf = bytearray(size)
    view, got = memoryview(buf), 0
    while got < size:
        more = stream.readinto(view[got:])
        if not more:
            return None
        got += more
    return bytes(buf)
