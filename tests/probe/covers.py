"""A program model cover stand-in for harvest tests."""

# pylint: disable=missing-function-docstring

import numpy as np


class FakeCover:
    """Two base states: P2 stepped in the first, P12 in the second, as a program model's cover would say."""

    def __init__(self):
        self.states = [np.full(12, 512, dtype=np.int64), np.r_[np.full(11, 512), 0].astype(np.int64)]

    def vectors(self):
        return [(self.states[0], self.states[0]), (self.states[0], self.states[1])]

    def where(self, param):
        return {2: [0], 12: [1]}.get(param, [])

    def schedule(self, index, param):
        return np.array([0, 341, 682, 1023]) if (index, param) in ((0, 2), (1, 12)) else np.zeros(0)


def fake_plan(info):
    return FakeCover() if info.declared is not None else None
