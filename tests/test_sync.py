"""Tests for session_analysis.light_sync: the cameras' sync checked from lights both cameras saw come on."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import session_analysis as sa  # noqa: E402


class LightSync(unittest.TestCase):
    def lights(self, n=40, seed=0):
        rng = np.random.default_rng(seed)
        t = np.cumsum(rng.uniform(1300, 1900, n))
        return [(int(rng.integers(1, 6)), float(x)) for x in t]

    def test_two_frames_off(self):
        """Camera B sees every light 40 ms early (two frames at 50 fps): B must move 40 ms later."""
        a = self.lights()
        b = [(s, t - 40 + np.random.default_rng(1).choice([-20, 0, 0, 0, 20])) for s, t in a]
        corr, n, share = sa.light_sync(a, b, 50.0)
        self.assertAlmostEqual(corr, 0.040, places=3)
        self.assertEqual(n, len(a))
        self.assertGreaterEqual(share, 0.9)

    def test_in_sync_and_too_few(self):
        a = self.lights()
        corr, _, _ = sa.light_sync(a, list(a), 50.0)
        self.assertEqual(corr, 0.0)
        self.assertIsNone(sa.light_sync(a[:5], a[:5], 50.0))

    def test_other_switch_not_paired(self):
        """A light of another switch at the same moment is not taken as the same light."""
        a = [(1, 1000.0 + 1500 * i) for i in range(30)]
        b = [(2, 1000.0 + 1500 * i) for i in range(30)]
        self.assertIsNone(sa.light_sync(a, b, 50.0))


if __name__ == '__main__':
    unittest.main()
