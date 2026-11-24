"""Tests for system_times.py: the light-switch system's logged times (optional) used for hit / miss and timing."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import experiment_design as ed  # noqa: E402
import system_times as stt  # noqa: E402

FPS = 50.0


def block(pattern, presses, offset=40.0, broken=()):
    """One block of trials as build_trials makes them, with the video's lights (seen `offset` ms longer than the
    system's times). presses: per trial the system's light-on -> press (ms), None = no press (a 1.06 s timeout).
    broken: trials whose light the video saw wrongly (a short false blip 0.3 s late)."""
    times = np.arange(0, 60000, 1000 / FPS)
    rows, t = [], 1000.0
    for i, (target, v) in enumerate(zip(pattern, presses)):
        on, off = t, t + (v + offset if v is not None else 1060.0)
        if i in broken:
            on, off = t + 300, t + 340
        rows.append(dict(Part='right_arm', Block=1, Sequence=i // 6 + 1, Trial=i % 6 + 1, Target=target, Limb='upper',
                         Side='right', block_index=0, Outcome='hit' if v is not None else 'miss',
                         Light_on_ms=on, Light_off_ms=off, Hit_ms=off if v is not None else None, Trial_end_ms=off,
                         Touch_ms=None, Light_source='video', on_frame=int(on / 1000 * FPS),
                         off_frame=int(off / 1000 * FPS), window_start=0))
        t = (t + v if v is not None else t + 1000.0) + 1000.0       # next light 1 s after the press / timeout
    return rows, times


class ApplySystemTimes(unittest.TestCase):
    def setUp(self):
        self.pattern = [1, 4, 3, 5, 4, 2]
        self.touch = {'upper': np.full((5, 3000), np.inf), 'lower': np.full((5, 3000), np.inf)}

    def sheets(self, presses):
        cols = {f'LED{i}': [] for i in range(1, 7)}
        for target, v in zip(self.pattern, presses):
            if v is not None:
                cols[stt.LED_OF_SWITCH[target]].append(v)
        return {'AL-RT-H1': dict(time='11:25', cols=cols)}

    def test_sheet_names(self):
        self.assertEqual(stt.sheet_key('AL-RT-H1'), ('AL', 'right', 'upper', 1))
        self.assertEqual(stt.sheet_key('YE-LT-l3'), ('YE', 'left', 'lower', 3))
        self.assertIsNone(stt.sheet_key('AL-practice1'))

    def test_hits_misses_and_times(self):
        presses = [500, 450, None, 520, 480, 600]
        rows, times = block(self.pattern, presses)
        # the video saw the timeout of trial 3 as 1.06 s; a late press measured 1.00 s would be a hit
        notes = stt.apply_system_times(rows, dict(sheets=self.sheets(presses)), times, FPS, 1.0, self.touch,
                                       ed.Settings())
        self.assertEqual([r['Outcome'] for r in rows], ['hit', 'hit', 'miss', 'hit', 'hit', 'hit'])
        for r, v in zip(rows, presses):
            if v is not None:
                self.assertAlmostEqual(r['Hit_ms'] - r['Light_on_ms'], v)
                self.assertEqual(r['System_ms'], v)
        self.assertIn('1 block', notes[0])

    def test_wrong_light_placed_by_timing(self):
        presses = [500, 450, 470, 520, 480, 600]
        rows, times = block(self.pattern, presses, broken=(5,))     # switch 2's light seen wrongly
        true_on = 1000.0 + sum(p + 1000 for p in presses[:5])
        stt.apply_system_times(rows, dict(sheets=self.sheets(presses)), times, FPS, 1.0, self.touch, ed.Settings())
        r = rows[5]
        self.assertEqual(r['Light_source'], 'timing+system')
        self.assertAlmostEqual(r['Light_on_ms'], true_on, delta=45)   # previous press (seen + offset) + 1 s
        self.assertAlmostEqual(r['Hit_ms'] - r['Light_on_ms'], 600)

    def test_no_sheet_keeps_the_video(self):
        presses = [500, 450, 470, 520, 480, 600]
        rows, times = block(self.pattern, presses)
        before = [dict(r) for r in rows]
        notes = stt.apply_system_times(rows, dict(sheets={}), times, FPS, 1.0, self.touch, ed.Settings())
        self.assertEqual(rows, before)
        self.assertTrue(any('No system sheet' in n for n in notes))


if __name__ == '__main__':
    unittest.main()
