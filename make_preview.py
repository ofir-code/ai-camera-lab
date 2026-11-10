"""A preview video of an analysed day: both cameras side by side, the first trials of every block, with the
MediaPipe skeleton, the switch boxes (target in yellow), the tracked wrist / ankle (red cross) and the trial's
part / block / sequence / trial / target / outcome / RT / MT. A bar at the bottom shows the blocks of the session.

    venv/bin/python make_preview.py <participant> [day] [trials per block]
    -> results/<participant>/<day>/<participant>_<day>_preview.mp4
"""
import os
import sys
import time

import cv2
import numpy as np

import extract_kinematics as ek
import pipeline as pl
import session_analysis as sa
import trial_measures as tm
import video_source as vs

# MediaPipe pose: the lines of the skeleton (landmark pairs)
POSE_LINES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
              (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (0, 11), (0, 12),
              (15, 17), (15, 19), (16, 18), (16, 20)]
PART_COLOR = {'Familiarization': (180, 180, 180), 'pre_test': (255, 160, 60), 'practice': (80, 200, 80),
              'post_test': (60, 120, 255), 're_test': (200, 80, 200)}
OUT_W = 1920             # output width (two cameras side by side)
OUT_FPS = 30.0
PAD_S = 0.6              # seconds shown before the first light and after the last trial of a clip


class Reader:
    """Reads one camera's frames by frame number, going forward (jumps only when far ahead)."""

    def __init__(self, path):
        self.src = vs.VideoSource(path, hardware=False)
        self.pos = 0
        self.frame = None

    def get(self, i):
        if self.frame is not None and i == self.pos - 1:
            return self.frame
        if i < self.pos or i > self.pos + 90:
            self.src.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            self.pos = i
        while self.pos <= i:
            ok, f = self.src.read()
            if not ok:
                return None
            self.frame = f
            self.pos += 1
        return self.frame


def draw_camera(frame, cache, cf, boxes, r, cam_name):
    """Skeleton, switch boxes (target yellow), the tracked marker, the camera name."""
    h, w = frame.shape[:2]
    sc = w / 1920
    pose = cache['pose'][cf]
    vis = pose[:, 2] if pose.shape[1] > 2 else np.ones(len(pose))
    for a, b in POSE_LINES:
        if not (np.isnan(pose[a, 0]) or np.isnan(pose[b, 0])) and min(vis[a], vis[b]) > 0.3:
            cv2.line(frame, (int(pose[a, 0]), int(pose[a, 1])), (int(pose[b, 0]), int(pose[b, 1])),
                     (255, 255, 255), max(2, int(4 * sc)), cv2.LINE_AA)
    for j in range(11, 33):
        if not np.isnan(pose[j, 0]) and vis[j] > 0.3:
            cv2.circle(frame, (int(pose[j, 0]), int(pose[j, 1])), max(3, int(6 * sc)), (0, 200, 255), -1, cv2.LINE_AA)
    W, H = cache['meta']['size']
    M = cache['motion'][cf]
    for k, box in enumerate(boxes):
        c = ek.to_frame(ek.box_corners(box, W, H, 0.0), M).astype(np.int32)
        target = r is not None and k + 1 == r['Target']
        cv2.polylines(frame, [c], True, (0, 230, 255) if target else (160, 160, 160),
                      max(2, int((7 if target else 3) * sc)), cv2.LINE_AA)
        cv2.putText(frame, str(k + 1), (int(c[:, 0].min()), int(c[:, 1].min() - 10 * sc)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6 * sc, (0, 230, 255) if target else (200, 200, 200), max(2, int(4 * sc)), cv2.LINE_AA)
    if r is not None:
        j = tm.MARKER[r['Limb']][r['Side']]
        p = pose[j]
        if not np.isnan(p[0]):
            cv2.drawMarker(frame, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, int(70 * sc),
                           max(3, int(8 * sc)))
    cv2.putText(frame, cam_name, (int(30 * sc), int(80 * sc)), cv2.FONT_HERSHEY_SIMPLEX, 2.2 * sc, (255, 255, 255),
                max(3, int(6 * sc)), cv2.LINE_AA)
    return frame


def text_lines(img, lines, x, y, size, color=(255, 255, 255), gap=1.45):
    for k, (t, c) in enumerate(lines):
        cv2.putText(img, t, (x, int(y + k * size * 30 * gap)), cv2.FONT_HERSHEY_SIMPLEX, size, c or color, 2,
                    cv2.LINE_AA)


def make_preview(pid, day='d1', per_block=2, out=None, progress=print):
    st = pl.load_day(pid, day)
    s = sa.Session(st)
    res = s.analyze()
    trials = [r for r in res['trials'] if r['on_frame'] is not None and r['Part'] != 'ignore']
    times = s.times
    cams = [c for c in ('a', 'b') if c in s.cams]
    readers = {c: Reader(st[f'cam_{c}']) for c in cams}
    # Clips: the first `per_block` trials of every block
    clips = []
    for bi in sorted({r['block_index'] for r in trials}):
        rows = [r for r in trials if r['block_index'] == bi][:per_block]
        a = rows[0]['on_frame']
        last = rows[-1]
        b = last['off_frame'] if last['off_frame'] is not None else last['on_frame']
        clips.append((rows, max(0, a - int(PAD_S * s.fps)), min(len(times) - 1, b + int(PAD_S * s.fps))))
    blocks_t = []                        # for the bottom bar: (start s, end s, part)
    for bi, (part, no) in enumerate(res['parts']):
        rows = [r for r in res['trials'] if r['block_index'] == bi and r['on_frame'] is not None]
        if rows and part != 'ignore':
            blocks_t.append((times[rows[0]['on_frame']] / 1000, times[rows[-1]['on_frame']] / 1000, part, no))
    total_s = times[-1] / 1000

    half = OUT_W // len(cams)
    size = {c: s.cams[c]['meta']['size'] for c in cams}
    vid_h = max(int(half * size[c][1] / size[c][0]) for c in cams)
    info_h, bar_h = 150, 60
    H = vid_h + info_h + bar_h
    out = out or os.path.join(pl.day_dir(pid, day), f'{pid}_{day}_preview.mp4')
    writer = None
    for codec in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*codec), OUT_FPS, (OUT_W, H))
        if writer.isOpened():
            break
    step = max(1, int(round(s.fps / OUT_FPS)))
    n_total = sum((b - a) // step + 1 for _, a, b in clips)
    done, t0 = 0, time.time()
    for rows, a, b in clips:
        for f in range(a, b + 1, step):
            t = times[f]
            # the trial shown: the last one whose light came on by now
            r = [x for x in rows if x['on_frame'] <= f]
            r = r[-1] if r else rows[0]
            canvas = np.zeros((H, OUT_W, 3), np.uint8)
            for k, c in enumerate(cams):
                cf = int(s.idx[c][f])
                if cf < 0:
                    continue
                frame = readers[c].get(cf)
                if frame is None:
                    continue
                frame = draw_camera(frame.copy(), s.cams[c], cf, st[f'boxes_{c}'], r, f'camera {c.upper()}')
                sw, sh = size[c]
                small = cv2.resize(frame, (half, int(half * sh / sw)), interpolation=cv2.INTER_AREA)
                canvas[:small.shape[0], k * half:k * half + half] = small
            # info
            y0 = vid_h + 45
            lit = r['on_frame'] <= f <= (r['off_frame'] if r['off_frame'] is not None else r['on_frame'])
            flags = []
            if r['anticipation']:
                flags.append((f"started {r['anticipation_ms']:.0f} ms before the light", (80, 220, 255)))
            if r.get('start_from') not in ('', 'home', None):
                flags.append((f"not from home ({r['start_from']})", (220, 140, 255)))
            if r['corrected_from']:
                flags.append((f"corrected (went to {r['corrected_from']} first)", (255, 220, 120)))
            fmt = lambda v: '-' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.3f} s'
            text_lines(canvas, [(f"{r['Part']}   block {r['Block']}   sequence {r['Sequence']}   trial {r['Trial']}"
                                 f"   ->  switch {r['Target']}   ({r['Limb']} {r['Side']})",
                                 PART_COLOR.get(r['Part'].replace('_contralateral', ''))),
                                (f"{r['Outcome']}    RT {fmt(r['responsetime'])}    MT {fmt(r['movementtime'])}"
                                 f"    data: {r.get('Data') or '-'}    light: {r.get('Light_source', '')}", None)],
                       30, y0, 1.0)
            text_lines(canvas, flags, 1180, y0, 0.8)
            cv2.putText(canvas, f'{int(t // 60000)}:{(t / 1000) % 60:05.2f}', (OUT_W - 200, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            if lit:
                cv2.circle(canvas, (OUT_W - 240, y0 - 10), 14, (0, 230, 255), -1, cv2.LINE_AA)
            # bar: the blocks, where we are
            by = H - bar_h + 10
            for bs, be, part, no in blocks_t:
                x0, x1 = int(bs / total_s * (OUT_W - 40)) + 20, int(be / total_s * (OUT_W - 40)) + 20
                cv2.rectangle(canvas, (x0, by), (max(x0 + 3, x1), by + 30),
                              PART_COLOR.get(part.replace('_contralateral', ''), (150, 150, 150)), -1)
            xc = int(t / 1000 / total_s * (OUT_W - 40)) + 20
            cv2.line(canvas, (xc, by - 6), (xc, by + 36), (255, 255, 255), 3)
            writer.write(canvas)
            done += 1
            if done % 150 == 0:
                left = (time.time() - t0) / done * (n_total - done)
                progress(f'{done}/{n_total} frames, {left / 60:.1f} min left')
    writer.release()
    return out


if __name__ == '__main__':
    args = sys.argv[1:]
    path = make_preview(args[0], args[1] if len(args) > 1 else 'd1', int(args[2]) if len(args) > 2 else 2)
    print('Saved', path)
