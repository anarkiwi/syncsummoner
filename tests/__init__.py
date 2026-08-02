"""Fixtures shared across the suite, chiefly the capture card's measured byte order."""

import numpy as np

#: Measured (Y, first chroma byte, second chroma byte) and the RGB a YVYU decode recovers.
MEASURED_YVYU = {
    "red": ((41, 241, 109), (209, 0, 0)),
    "green": ((144, 53, 34), (29, 247, 0)),
    "blue": ((82, 89, 241), (15, 64, 255)),
}


def native_frame(planes, *, height=4, width=4):
    """Packed native ``(H, W, 2)`` uint8 buffer of one constant ``(Y, first, second)``."""
    y, first, second = planes
    frame = np.empty((height, width, 2), dtype=np.uint8)
    frame[:, :, 0] = y
    frame[:, 0::2, 1] = first
    frame[:, 1::2, 1] = second
    return frame
