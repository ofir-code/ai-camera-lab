"""Tests for the participant area (MediaPipe looks only there) and the 3D marker track that also uses points
MediaPipe is unsure of when both cameras agree.

    venv/bin/python -m unittest tests.test_area_and_3d -v
"""
import os
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import calibrate_cameras as cc  # noqa: E402
import session_engine as se  # noqa: E402
import trial_measures as tm  # noqa: E402


class TestArea(unittest.TestCase):
    def test_roi_pixels(self):
        self.assertIsNone(se.roi_pixels(None, 3840, 2160))
        self.assertEqual(se.roi_pixels([0.25, 0.5, 0.75, 1.0], 3840, 2160), (960, 1080, 2880, 2160))
        self.assertEqual(se.roi_pixels([0.75, 1.2, 0.25, 0.5], 3840, 2160), (960, 1080, 2880, 2160))  # any order, clipped
        self.assertIsNone(se.roi_pixels([0.5, 0.5, 0.501, 0.9], 3840, 2160))                      # too small

    def test_landmarks_back_to_frame(self):
        lm = [SimpleNamespace(x=0.5, y=0.25, visibility=0.9), SimpleNamespace(x=0.0, y=1.0, visibility=0.1)]
        p = se.landmarks_to_pixels(lm, 3840, 2160, (960, 1080, 2880, 2160))
        np.testing.assert_allclose(p, [[1920, 1350, 0.9], [960, 2160, 0.1]])
        np.testing.assert_allclose(se.landmarks_to_pixels(lm, 3840, 2160)[0], [1920, 540, 0.9])

    def test_pose_input_is_the_full_size_crop(self):
        frame = np.zeros((2160, 3840, 3), np.uint8)
        frame[1500, 2000] = (0, 0, 255)                     # a red dot inside the area
        small = cv2.resize(frame, (1280, 720))
        self.assertIs(se.pose_input(frame, small, None), small)
        crop = se.pose_input(frame, small, (1500, 1000, 2500, 2000))
        self.assertEqual(crop.shape, (1000, 1000, 3))         # 1000 px: not shrunk, more pixels than in `small`
        self.assertEqual(tuple(crop[500, 500]), (255, 0, 0))  # RGB, same place


def cameras():
    K = np.array([[1500.0, 0, 960], [0, 1500, 540], [0, 0, 1]])
    ang = np.radians(40)
    R = np.array([[np.cos(ang), 0, -np.sin(ang)], [0, 1, 0], [np.sin(ang), 0, np.cos(ang)]])
    t = np.array([-60.0, 0, 15])
    ca = dict(K=K, dist=np.zeros(5), size=[1920, 1080])
    cb = dict(K=K, dist=np.zeros(5), size=[1920, 1080], R=R, t=t)
    return ca, cb


class Test3DAgreement(unittest.TestCase):
    def test_track_3d(self):
        ca, cb = cameras()
        X = np.column_stack([np.linspace(-20, 20, 6), np.linspace(-5, 5, 6), np.full(6, 150.0)])
        ra = cc.project(X, ca['K'], ca['dist'], np.eye(3), np.zeros(3))
        rb = cc.project(X, cb['K'], cb['dist'], cb['R'], cb['t'])
        rb[4] += 300                                     # frame 4: camera B is on something else (e.g. a person)
        rb[5] += 3                                       # frame 5: 3 px off - still agrees
        sure_a, sure_b = ra.copy(), rb.copy()
        sure_a[[1, 2]] = np.nan                          # frames 1-2: MediaPipe unsure in camera A
        sure_b[2] = np.nan
        X3, how = cc.track_3d(sure_a, sure_b, ra, rb, ca, cb)
        self.assertEqual(list(how), ['sure', 'agree', 'agree', 'sure', '', 'sure'])
        np.testing.assert_allclose(X3[[0, 1, 2, 3]], X[[0, 1, 2, 3]], atol=1e-6)
        self.assertTrue(np.isnan(X3[4]).all())              # sure in both views, but they don't meet: dropped
        # unsure AND not agreeing: not used
        rb2 = rb.copy()
        rb2[1] += 100
        X3, how = cc.track_3d(sure_a, sure_b, ra, rb2, ca, cb)
        self.assertEqual(how[1], '')

    def test_marker_track_min_visibility(self):
        pose = np.zeros((10, 33, 3))
        pose[:, 16, :2] = 100
        pose[:, 16, 2] = 0.1
        self.assertTrue(np.isnan(tm.marker_track(pose, 'upper', 'right')).all())
        self.assertFalse(np.isnan(tm.marker_track(pose, 'upper', 'right', min_visibility=0.0)).any())


if __name__ == '__main__':
    unittest.main()
