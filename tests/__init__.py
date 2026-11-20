"""Fixtures shared across the suite, chiefly the capture card's measured bytes."""

import numpy as np

#: Measured (Y, U, V) bytes the card sent for each primary on screen, and the RGB a YUYV decode gives.
MEASURED_YUYV = {
    "blue": ((41, 241, 109), (0, 0, 255)),
    "green": ((144, 53, 34), (0, 255, 0)),
    "red": ((82, 89, 241), (255, 0, 0)),
}


def native_frame(planes, *, height=4, width=4):
    """Packed native ``(H, W, 2)`` uint8 buffer of one constant ``(Y, first, second)``."""
    y, first, second = planes
    frame = np.empty((height, width, 2), dtype=np.uint8)
    frame[:, :, 0] = y
    frame[:, 0::2, 1] = first
    frame[:, 1::2, 1] = second
    return frame
