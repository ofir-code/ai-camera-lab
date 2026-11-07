"""Tests for pipeline.py: participants and days on disk, step states and what an edit makes to-do again, the
one-camera fallback when the videos have no timecode, running a day (stops at an error, resumes after it),
and the queue going on after a day fails.

    venv/bin/python -m unittest tests.test_pipeline -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as pl  # noqa: E402
import session_analysis as sa  # noqa: E402


def write_video(path, n=30, fps=30.0, size=(320, 180)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 4 % 255, np.uint8))
    w.release()


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.results = os.path.join(self.dir, 'results')
        p = mock.patch.object(sa, 'RESULTS_DIR', self.results)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.va, self.vb = os.path.join(self.dir, 'a.mp4'), os.path.join(self.dir, 'b.mp4')
        write_video(self.va)
        write_video(self.vb)


class TestParticipants(Base):
    def test_new_and_load(self):
        p = pl.new_participant('S01', design=dict(tested_side='left', contralateral=True))
        self.assertEqual(p['design']['tested_side'], 'left')
        self.assertEqual(p['design']['practice_limb'], 'upper')     # default filled in
        q = pl.load_participant('S01')
        self.assertEqual(q['design'], p['design'])
        self.assertEqual(pl.list_participants(), ['S01'])
        with self.assertRaises(ValueError):
            pl.new_participant('S01')
        for bad in ('', 'a/b', '.x', 'S:1'):
            with self.assertRaises(ValueError):
                pl.new_participant(bad)

    def test_day_design_retest(self):
        pl.new_participant('S02', design=dict(tested_side='left'))
        d1 = pl.set_videos('S02', 'd1', self.va, self.vb)
        d2 = pl.set_videos('S02', 'd2', self.va)
        self.assertFalse(d1['design']['retest'])
        self.assertTrue(d2['design']['retest'])
        self.assertEqual(d2['design']['tested_side'], 'left')
        self.assertEqual(pl.load_day('S02', 'd2')['cam_b'], '')
        names = [p.name for p in sa.template_of(d2)]
        self.assertTrue(all(n.startswith('re_test') for n in names), names)

    def test_design_change_redoes_analysis_only(self):
        p = pl.new_participant('S03')
        st = pl.set_videos('S03', 'd1', self.va, self.vb)
        for s in pl.STEPS:
            pl.set_status(st, s, pl.STATUS_OK)
        pl.save_day(st)
        pl.update_participant(p, design=dict(tested_side='left'))
        st = pl.load_day('S03', 'd1')
        self.assertEqual([st['steps'][s]['status'] for s in pl.STEPS],
                         ['ok', 'ok', 'ok', 'ok', 'todo', 'todo'])
        pl.update_participant(p, square_mm=40.0)
        st = pl.load_day('S03', 'd1')
        self.assertEqual(st['steps']['calibration']['status'], 'todo')
        self.assertEqual(st['steps']['switches']['status'], 'ok')

    def test_earlier_app_folder(self):
        """A session folder made by the earlier app becomes a participant with step states."""
        st = sa.default_state('OLD', 'd1')
        st.update(cam_a=self.va, boxes_a=[[0, 0, .1, .1]] * 5)
        st['design']['tested_side'] = 'left'
        sa.save_state(st)
        self.assertEqual(pl.list_participants(), ['OLD'])
        self.assertEqual(pl.load_participant('OLD')['design']['tested_side'], 'left')
        day = pl.load_day('OLD', 'd1')
        self.assertEqual(day['steps']['videos']['status'], 'ok')
        self.assertEqual(day['steps']['switches']['status'], 'ok')
        self.assertEqual(day['steps']['processing']['status'], 'todo')   # no cache


class TestSteps(Base):
    def setUp(self):
        super().setUp()
        pl.new_participant('S01')

    def test_edits_mark_later_steps(self):
        st = pl.set_videos('S01', 'd1', self.va, self.vb)
        for s in pl.STEPS:
            pl.set_status(st, s, pl.STATUS_OK)
        pl.edited(st, 'switches', 'Boxes drawn')
        self.assertEqual([st['steps'][s]['status'] for s in pl.STEPS], ['ok', 'ok', 'ok', 'todo', 'todo', 'todo'])
        self.assertEqual(pl.next_step(st), 'processing')
        for s in pl.STEPS:
            pl.set_status(st, s, pl.STATUS_OK)
        pl.mark_todo(st, 'calibration')
        self.assertEqual([st['steps'][s]['status'] for s in pl.STEPS], ['ok', 'todo', 'ok', 'ok', 'todo', 'todo'])
        # New videos: everything again, the old boxes are dropped
        st['boxes_a'] = [[0, 0, 1, 1]]
        pl.save_day(st)
        st = pl.set_videos('S01', 'd1', self.vb, self.va)
        self.assertTrue(all(st['steps'][s]['status'] == 'todo' for s in pl.STEPS))
        self.assertIsNone(st['boxes_a'])

    def test_no_timecode_one_camera(self):
        """Videos without a timecode track: camera B is left out, the day goes on with camera A (2D)."""
        st = pl.set_videos('S01', 'd1', self.va, self.vb)
        status, msg, details = pl.step_videos(st, lambda *a: None, lambda: False)
        self.assertEqual(status, pl.STATUS_WARN)
        self.assertIn('camera A only', msg)
        self.assertEqual(st['cam_b'], '')
        self.assertEqual(st['cam_b_unsynced'], self.vb)
        self.assertEqual(pl.cameras_of(st), ['a'])
        self.assertEqual(pl.step_calibration(st, lambda *a: None, lambda: False)[0], pl.STATUS_SKIPPED)
        # An offset typed by hand brings camera B back
        pl.set_offset(st, 1.5)
        status, msg, _ = pl.step_videos(st, lambda *a: None, lambda: False)
        self.assertEqual(st['cam_b'], self.vb)
        self.assertEqual(st['offset_s'], 1.5)
        self.assertIn('by hand', msg)

    def test_swap(self):
        st = pl.set_videos('S01', 'd1', self.va, self.vb)
        st.update(boxes_a=[[1]], boxes_b=[[2]], offset_s=2.0)
        pl.swap_cameras(st)
        self.assertEqual((st['cam_a'], st['cam_b'], st['boxes_a'], st['offset_s']), (self.vb, self.va, [[2]], -2.0))

    def test_calibration_file(self):
        st = pl.set_videos('S01', 'd1', self.va, self.vb)
        cal = os.path.join(self.dir, 'cams.json')
        with open(cal, 'w') as f:
            json.dump(dict(method='board', units='cm', distance_cm=150.0, warnings=['w1'],
                           camera_a=dict(K=np.eye(3).tolist(), dist=[0] * 5, size=[320, 180]),
                           camera_b=dict(K=np.eye(3).tolist(), dist=[0] * 5, size=[320, 180],
                                         R=np.eye(3).tolist(), t=[150, 0, 0])), f)
        st.update(calibration=cal, calibration_source='file')
        status, msg, details = pl.step_calibration(st, lambda *a: None, lambda: False)
        self.assertEqual(status, pl.STATUS_OK)
        self.assertIn('Cameras apart (cm): 150.00', details)
        self.assertIn('! w1', details)

    def test_calibration_board_not_found(self):
        """No board in either window -> 2D with a warning, not an error."""
        st = pl.set_videos('S01', 'd1', self.va, self.vb)
        with mock.patch('calibrate_cameras.calibrate_board', side_effect=ValueError('The board was found in only 0')):
            status, msg, details = pl.step_calibration(st, lambda *a: None, lambda: False)
        self.assertEqual(status, pl.STATUS_WARN)
        self.assertEqual(len(details), len(pl.CALIB_WINDOWS))
        self.assertEqual(st['calibration'], '')


def fake_steps(fail=None, calls=None):
    """Patch the step functions with quick fakes; `fail` = the step that raises."""
    calls = calls if calls is not None else []

    def make(name):
        def f(st, progress, should_stop, *a, **k):
            calls.append((st['subject'], st['session'], name))
            progress(0.5, 'half')
            if name == fail:
                raise ValueError(f'{name} broke\nsecond line')
            return (pl.STATUS_OK, name, [], None) if name == 'analysis' else (pl.STATUS_OK, name, [])
        return f
    return [mock.patch.object(pl, f'step_{n}', make(n)) for n in pl.STEPS], calls


class TestRun(Base):
    def setUp(self):
        super().setUp()
        pl.new_participant('S01')
        pl.new_participant('S02')
        pl.set_videos('S01', 'd1', self.va, self.vb)
        pl.set_videos('S01', 'd2', self.va, self.vb)
        pl.set_videos('S02', 'd1', self.va, self.vb)

    def run_with(self, fn, fail=None):
        patches, calls = fake_steps(fail)
        for p in patches:
            p.start()
        try:
            return fn(), calls
        finally:
            for p in patches:
                p.stop()

    def test_run_day_and_resume(self):
        seen = []
        st, calls = self.run_with(lambda: pl.run_day('S01', 'd1', progress=lambda *a: seen.append(a)), 'switches')
        self.assertEqual([c[2] for c in calls], ['videos', 'calibration', 'switches'])
        self.assertEqual(st['steps']['switches']['status'], 'error')
        self.assertEqual(st['steps']['switches']['message'], 'switches broke')
        self.assertEqual(st['steps']['switches']['details'], ['second line'])
        self.assertEqual(st['steps']['processing']['status'], 'todo')
        self.assertIn(('switches', 0.5, 'half'), seen)
        self.assertEqual(pl.day_summary(st), 'error')
        # Again: stops at the error without re-running the earlier steps
        st, calls = self.run_with(lambda: pl.run_day('S01', 'd1'))
        self.assertEqual(calls, [])
        # The user fixes the switches by hand -> the rest runs
        pl.edited(st, 'switches', 'drawn')
        st, calls = self.run_with(lambda: pl.run_day('S01', 'd1'))
        self.assertEqual([c[2] for c in calls], ['processing', 'analysis', 'results'])
        self.assertEqual(pl.day_summary(st), 'done')

    def test_run_one_step_again(self):
        st, _ = self.run_with(lambda: pl.run_day('S01', 'd1'))
        pl.mark_todo(st, 'calibration')
        pl.save_day(st)
        st, calls = self.run_with(lambda: pl.run_day('S01', 'd1'))
        self.assertEqual([c[2] for c in calls], ['calibration', 'analysis', 'results'])

    def test_stop(self):
        flag = []

        def stop():
            return bool(flag)
        orig = pl.step_videos

        def videos(st, progress, should_stop):
            flag.append(1)
            return pl.STATUS_OK, 'v', []
        patches, calls = fake_steps()
        for p in patches:
            p.start()
        try:
            with mock.patch.object(pl, 'step_videos', videos):
                st = pl.run_day('S01', 'd1', should_stop=stop)
        finally:
            for p in patches:
                p.stop()
        del orig
        self.assertEqual(st['steps']['videos']['status'], 'ok')
        self.assertEqual(st['steps']['calibration']['status'], 'todo')

    def test_queue_goes_on(self):
        items = pl.queue_items()
        self.assertEqual(items, [('S01', 'd1'), ('S01', 'd2'), ('S02', 'd1')])
        # S01 d2 has a broken session.json: the queue goes on with S02
        with open(os.path.join(pl.day_dir('S01', 'd2'), 'session.json'), 'w') as f:
            f.write('{broken')
        out, calls = self.run_with(lambda: pl.run_queue(items))
        self.assertEqual([o[:2] for o in out], items)
        self.assertEqual(out[0][2], 'done')
        self.assertTrue(out[1][2].startswith('error'))
        self.assertEqual(out[2][2], 'done')
        self.assertEqual(pl.queue_items(), [])          # done days and the broken one are not queued again

    def test_running_left_over(self):
        """The app closed while a step ran: it is to-do again when the day is opened."""
        st = pl.load_day('S01', 'd1')
        pl.set_status(st, 'videos', pl.STATUS_RUNNING)
        pl.save_day(st)
        self.assertEqual(pl.load_day('S01', 'd1')['steps']['videos']['status'], 'todo')


if __name__ == '__main__':
    unittest.main()
