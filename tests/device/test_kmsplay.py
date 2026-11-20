"""The flip record the host reads from the Pi's player."""

# pylint: disable=missing-function-docstring

import io

from syncsummoner.device import kmsplay as K


def test_read_exactly_reassembles_short_reads():
    class Trickle(io.RawIOBase):
        """A pipe handing back at most three bytes per read."""

        def __init__(self, data):
            super().__init__()
            self.data = data

        def readinto(self, b):
            n = min(3, len(b), len(self.data))
            b[:n], self.data = self.data[:n], self.data[n:]
            return n

    stream = Trickle(b"0123456789")
    assert K.read_exactly(stream, 4) == b"0123"
    assert K.read_exactly(stream, 4) == b"4567"
    assert K.read_exactly(stream, 4) is None


def test_a_record_is_five_little_endian_int64s_with_distinct_flags():
    assert K.RECORD.size == 40
    assert K.RECORD.pack(1, 2, 3, 4, K.SEGMENT | K.REPLY)[32:] == bytes([3, 0, 0, 0, 0, 0, 0, 0])
