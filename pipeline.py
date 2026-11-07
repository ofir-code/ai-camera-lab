"""The automatic steps of the Reaching Lab app - no window code, so it can run from the app, a queue, or a test.

A participant has a folder results/<id>/ with participant.json (design, settings, board square size) and one
folder per day: d1 (day 1) and d2 (re-test day). Each day folder holds session.json (see session_analysis.py)
plus the state of the steps below. Each day is 2 videos (camera A and B, GoPro chapters are followed
automatically) that cover the whole session.

Steps of a day, in order (STEPS):
  videos       find the GoPro chapters, check the camera settings, line up the 2 videos by their timecodes
               (one camera only / no timecode -> the day is analysed in 2D with the camera that has video)
  calibration  find the ChArUco board in both videos and compute where the cameras are (3D) - or use a file
  switches     find the 5 switches (their lights) in each video
  processing   MediaPipe + lights + touches on every frame (the slow step: about 1.5 x the video length)
  analysis     lights -> parts / blocks / sequences / trials synced with the design -> RT, MT and flags
  results      <id>_<day>_trials.csv, <id>_<day>_analysis.xlsx, and the all-participants Excel

Each step has a status (STATUS_*) with a message and details. Editing a step by hand (e.g. redrawing a switch
box, choosing another calibration file, changing the parts of the blocks) sets that step's result and marks
the steps that depend on it (DEPENDS) as to-do, so running the day again redoes only those.
"""
import json
import os
import shutil
import time
import traceback
from dataclasses import asdict

import experiment_design as ed
import session_analysis as sa

DAYS = ('d1', 'd2')
DAY_NAMES = {'d1': 'Day 1', 'd2': 'Day 2 (re-test)'}
STEPS = ('videos', 'calibration', 'switches', 'processing', 'analysis', 'results')
STEP_NAMES = {'videos': 'Videos & sync', 'calibration': 'Calibration', 'switches': 'Switches',
              'processing': 'Processing', 'analysis': 'Blocks & trials', 'results': 'Results'}
# Steps that must be redone when a step changes
DEPENDS = {'videos': STEPS[1:], 'calibration': ('analysis', 'results'),
           'switches': ('processing', 'analysis', 'results'), 'processing': ('analysis', 'results'),
           'analysis': ('results',), 'results': ()}
STATUS_TODO, STATUS_RUNNING, STATUS_OK, STATUS_WARN, STATUS_ERROR, STATUS_SKIPPED = (
    'todo', 'running', 'ok', 'warn', 'error', 'skipped')
DONE = (STATUS_OK, STATUS_WARN, STATUS_SKIPPED)

# Where the automatic steps look (seconds into camera A's video)
CALIB_WINDOWS = ((0.0, 180.0), (180.0, 420.0))      # the board is filmed at the start of the session
CALIB_PER_SECOND = 2.0
SWITCH_WINDOWS = ((0.0, 5.0), (5.0, 5.0))           # (from minute, for minutes): the first trials light them up

DESIGN_DEFAULT = dict(tested_side='right', contralateral=False, practice_limb='upper', practice_side='right')


class Stopped(Exception):
    """The user stopped the run."""


# -----------------------------
# Participants
# -----------------------------
def participant_dir(pid, results_dir=None):
    """The folder of a participant: results/<id>."""
    return os.path.join(results_dir or sa.RESULTS_DIR, pid)


def check_id(pid):
    """A participant id must be usable as a folder name. Returns: the cleaned id. Raises ValueError if not."""
    pid = (pid or '').strip()
    if not pid or any(c in pid for c in '/\\:*?"<>|') or pid.startswith('.'):
        raise ValueError(f'"{pid}" can\'t be a participant id - use letters, digits, - or _ (e.g. S01).')
    return pid


def new_participant(pid, design=None, settings=None, square_mm=None, results_dir=None):
    """Create a participant (results/<id>/participant.json). Raises ValueError if the id is taken or not valid.

    Inputs: pid - the id (e.g. 'S01'); design - dict like DESIGN_DEFAULT; settings - experiment_design.Settings
    as a dict (None = the defaults); square_mm - the calibration board's square size (None = the standard board).
    Returns: the participant dict.
    """
    import calibrate_cameras as cc
    pid = check_id(pid)
    if os.path.exists(os.path.join(participant_dir(pid, results_dir), 'participant.json')):
        raise ValueError(f'Participant {pid} already exists.')
    p = dict(id=pid, design=dict(DESIGN_DEFAULT, **(design or {})),
             settings=settings or asdict(ed.Settings()), square_mm=square_mm or cc.BOARD_SQUARE_MM, created=_now())
    save_participant(p, results_dir)
    return p


def save_participant(p, results_dir=None):
    """Write participant.json."""
    folder = participant_dir(p['id'], results_dir)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'participant.json'), 'w') as f:
        json.dump(p, f, indent=1)


def load_participant(pid, results_dir=None):
    """The participant dict. An older folder made by the earlier app (only <day>/session.json) gets one made from
    its sessions' design. Raises FileNotFoundError if there is neither."""
    folder = participant_dir(pid, results_dir)
    path = os.path.join(folder, 'participant.json')
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
        p['design'] = dict(DESIGN_DEFAULT, **p.get('design', {}))
        p.setdefault('settings', asdict(ed.Settings()))
        p.setdefault('square_mm', 38.0)
        p['id'] = pid
        return p
    for day in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        if os.path.exists(os.path.join(folder, day, 'session.json')):
            st = sa.load_state(os.path.join(folder, day))
            d = {k: st['design'][k] for k in DESIGN_DEFAULT if k in st['design']}
            p = dict(id=pid, design=dict(DESIGN_DEFAULT, **d), settings=st['settings'], square_mm=38.0,
                     created=_now())
            save_participant(p, results_dir)
            return p
    raise FileNotFoundError(f'No participant {pid} in {os.path.dirname(folder)}.')


def list_participants(results_dir=None):
    """The ids of all participants in the results folder (folders with participant.json or a day's session.json)."""
    root = results_dir or sa.RESULTS_DIR
    out = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        folder = os.path.join(root, name)
        if not os.path.isdir(folder) or name.startswith('.'):
            continue
        if os.path.exists(os.path.join(folder, 'participant.json')) or any(
                os.path.exists(os.path.join(folder, d, 'session.json')) for d in os.listdir(folder)):
            out.append(name)
    return out


def update_participant(p, design=None, settings=None, square_mm=None, results_dir=None):
    """Change a participant's design / settings / board size; the days' steps that use them become to-do.

    Returns: the updated participant dict."""
    changed = set()
    if design is not None and dict(DESIGN_DEFAULT, **design) != p['design']:
        p['design'] = dict(DESIGN_DEFAULT, **design)
        changed.add('analysis')
    if settings is not None and settings != p['settings']:
        p['settings'] = settings
        changed.add('analysis')
    if square_mm is not None and abs(float(square_mm) - p['square_mm']) > 1e-9:
        p['square_mm'] = float(square_mm)
        changed.add('calibration')
    save_participant(p, results_dir)
    for day in DAYS:
        if day_exists(p['id'], day, results_dir):
            st = load_day(p['id'], day, results_dir)
            for step in changed:
                if step == 'calibration' and st.get('calibration_source') == 'file':
                    continue
                mark_todo(st, step)
            save_day(st, results_dir)
    return p


def delete_participant(pid, results_dir=None):
    """Remove a participant's folder (videos are not touched - they are only referenced)."""
    shutil.rmtree(participant_dir(check_id(pid), results_dir))


# -----------------------------
# Days
# -----------------------------
def day_dir(pid, day, results_dir=None):
    """The folder of one day: results/<id>/<day>."""
    return os.path.join(participant_dir(pid, results_dir), day)


def day_exists(pid, day, results_dir=None):
    """True if the day has been set up (it has a session.json)."""
    return os.path.exists(os.path.join(day_dir(pid, day, results_dir), 'session.json'))


def new_day_state(p, day):
    """The settings of a day before it has videos (session_analysis.default_state + the steps)."""
    st = sa.default_state(subject=p['id'], session=day)
    st.update(steps={s: _status(STATUS_TODO) for s in STEPS}, offset_source='timecode', calibration_source='auto',
              calib=dict(start_s=None, seconds=None), scan_custom=False, video_notes={})
    _apply_participant(st, p)
    return st


def _apply_participant(st, p):
    """Copy the participant's design and settings into a day's state (day 2 is the re-test)."""
    st['design'] = dict(p['design'], retest=st['session'] != 'd1')
    st['settings'] = dict(p['settings'])
    st['square_mm'] = p['square_mm']


def load_day(pid, day, results_dir=None):
    """A day's state (session.json); a day made by the earlier app gets step states from what is on disk."""
    folder = day_dir(pid, day, results_dir)
    with open(os.path.join(folder, 'session.json')) as f:
        raw = json.load(f)
    st = sa.default_state(subject=pid, session=day)
    st.update(raw)
    st['subject'], st['session'] = pid, day
    st.setdefault('offset_source', 'timecode')
    st.setdefault('calibration_source', 'file' if st.get('calibration') else 'auto')
    st.setdefault('calib', dict(start_s=None, seconds=None))
    st.setdefault('scan_custom', False)
    st.setdefault('video_notes', {})
    if 'steps' not in raw:
        st['steps'] = _infer_steps(st, folder)
    for s in STEPS:
        st['steps'].setdefault(s, _status(STATUS_TODO))
        if st['steps'][s]['status'] == STATUS_RUNNING:      # the app was closed while it ran
            st['steps'][s] = _status(STATUS_TODO, 'It was stopped before it finished.')
    return st


def save_day(st, results_dir=None):
    """Write a day's state to results/<id>/<day>/session.json."""
    folder = day_dir(st['subject'], st['session'], results_dir)
    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, 'session.json.tmp')
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(folder, 'session.json'))


def _infer_steps(st, folder):
    """Step states for a session made by the earlier app, from what it saved."""
    import session_engine as se
    steps = {s: _status(STATUS_TODO) for s in STEPS}
    if st.get('cam_a'):
        steps['videos'] = _status(STATUS_OK, 'From the earlier app.')
        steps['calibration'] = (_status(STATUS_OK, 'Calibration file from the earlier app.') if st.get('calibration')
                                else _status(STATUS_SKIPPED, 'No calibration (2D).'))
        cams = [c for c in 'ab' if st.get(f'cam_{c}')]
        if all(st.get(f'boxes_{c}') for c in cams):
            steps['switches'] = _status(STATUS_OK, 'From the earlier app.')
            caches = [se.load_cache(st[f'cam_{c}'], folder, st[f'boxes_{c}']) for c in cams]
            if all(c is not None and c['meta'].get('complete') for c in caches):
                steps['processing'] = _status(STATUS_OK, 'From the earlier app.')
                if os.path.exists(os.path.join(folder, 'trials.json')):
                    steps['analysis'] = _status(STATUS_OK, 'From the earlier app.')
                    steps['results'] = _status(STATUS_OK, 'From the earlier app.')
    return steps


def set_videos(pid, day, cam_a, cam_b=None, results_dir=None):
    """Choose a day's videos (camera A and B; B may be empty). The first GoPro chapter is used. If the videos
    changed, every step of the day becomes to-do. Returns: the day's state (saved)."""
    import video_source as vs
    p = load_participant(pid, results_dir)
    st = load_day(pid, day, results_dir) if day_exists(pid, day, results_dir) else new_day_state(p, day)
    paths = []
    for v in (cam_a, cam_b):
        v = (v or '').strip()
        paths.append(vs.first_chapter(v) if v and os.path.exists(v) else v)
    if not paths[0] and paths[1]:
        paths = [paths[1], '']
    if (paths[0], paths[1]) != (st['cam_a'], st.get('cam_b') or ''):
        st['cam_a'], st['cam_b'] = paths
        st.update(boxes_a=None, boxes_b=None, offset_s=0.0, offset_source='timecode', calibration='',
                  calibration_source='auto', parts_override=None, touse_override={}, video_notes={})
        mark_todo(st, 'videos')
    _apply_participant(st, p)
    save_day(st, results_dir)
    return st


def swap_cameras(st, results_dir=None):
    """Make camera A camera B and the other way round (keeps the switch boxes with their video)."""
    if not st.get('cam_b'):
        return st
    st['cam_a'], st['cam_b'] = st['cam_b'], st['cam_a']
    st['boxes_a'], st['boxes_b'] = st.get('boxes_b'), st.get('boxes_a')
    st['offset_s'] = -st.get('offset_s', 0.0)
    if st.get('calibration_source') == 'auto':
        st['calibration'] = ''
    mark_todo(st, 'videos')        # processing is quick again: the per-video caches go with their videos
    save_day(st, results_dir)
    return st


def set_offset(st, offset_s=None, results_dir=None):
    """Line up the videos by hand (offset_s = seconds into video A at which video B started), or by the timecodes
    again (offset_s=None). Camera B left out for lack of a timecode is brought back."""
    if st.get('cam_b_unsynced') and not st.get('cam_b'):
        st['cam_b'] = st.pop('cam_b_unsynced')
    if offset_s is None:
        st['offset_source'] = 'timecode'
    else:
        st['offset_source'], st['offset_s'] = 'manual', float(offset_s)
    mark_todo(st, 'videos')
    save_day(st, results_dir)
    return st


# -----------------------------
# Step states
# -----------------------------
def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def _status(status, message='', details=None):
    return dict(status=status, message=message, details=list(details or []), time=_now())


def set_status(st, step, status, message='', details=None):
    """Set a step's status (with a short message and detail lines)."""
    st['steps'][step] = _status(status, message, details)


def mark_todo(st, step, include_self=True):
    """A step changed: it (if include_self) and the steps that depend on it become to-do."""
    for s in ((step,) if include_self else ()) + tuple(DEPENDS[step]):
        if st['steps'][s]['status'] != STATUS_TODO:
            st['steps'][s] = _status(STATUS_TODO)


def edited(st, step, message, results_dir=None):
    """A step was edited by hand: it is done (with the message) and the steps after it become to-do."""
    set_status(st, step, STATUS_OK, message, ['Edited by hand.'])
    mark_todo(st, step, include_self=False)
    save_day(st, results_dir)


def next_step(st):
    """The first step that is not done yet (None if all are)."""
    for s in STEPS:
        if st['steps'][s]['status'] not in DONE:
            return s
    return None


def day_summary(st):
    """One word for the whole day: 'done', 'warn' (done with warnings), 'error', 'todo' or 'empty' (no videos)."""
    if not st.get('cam_a'):
        return 'empty'
    statuses = [st['steps'][s]['status'] for s in STEPS]
    if STATUS_ERROR in statuses:
        return 'error'
    if all(x in DONE for x in statuses):
        return 'warn' if STATUS_WARN in statuses else 'done'
    return 'todo'


# -----------------------------
# The steps
# -----------------------------
def cameras_of(st):
    """The cameras of the day that are used: ['a', 'b'], ['a'] (one camera)."""
    return [c for c in 'ab' if st.get(f'cam_{c}')]


def step_videos(st, progress, should_stop):
    """Check the videos and line them up by the GoPro timecodes. Returns (status, message, details)."""
    import timecode
    import video_source as vs
    details, status = [], STATUS_OK
    if not st.get('cam_a'):
        raise ValueError('Choose the videos of this day first.')
    if st.get('cam_b_unsynced') and not st.get('cam_b'):     # left out last time: try again
        st['cam_b'] = st.pop('cam_b_unsynced')
    info, notes = {}, {}
    for c in cameras_of(st):
        path = st[f'cam_{c}']
        if not os.path.exists(path):
            raise FileNotFoundError(f'Camera {c.upper()}: the video is not there any more: {path}')
        src = vs.VideoSource(path, hardware=False)
        ch = src.info[0]
        info[c] = ch
        fmt = vs.video_format(src.paths[0])
        length = src.total_ms / 60000
        codec = f', {fmt["codec"].upper()}{" 10-bit" if fmt["ten_bit"] else ""}' if fmt else ''
        line = (f'Camera {c.upper()}: {os.path.basename(path)} - {ch["width"]}x{ch["height"]}, {ch["fps"]:.3f} fps'
                f'{codec}, {length:.1f} min' + (f', {len(src.paths)} chapters' if len(src.paths) > 1 else ''))
        details.append(line)
        warn = src.check() + timecode.chapter_gaps(path)
        hints = vs.camera_notes(ch, fmt)
        notes[c] = dict(minutes=round(length, 2), fps=ch['fps'], size=[ch['width'], ch['height']],
                        chapters=[os.path.basename(x) for x in src.paths], warnings=warn, hints=hints)
        for w in warn:
            details.append(f'  ! {w}')
            status = STATUS_WARN
        for h in hints:
            details.append(f'  - {h}')
        src.release()
    st['video_notes'] = notes
    progress(0.5, 'timecodes')
    if len(info) == 2:
        a, b = info['a'], info['b']
        if (a['width'], a['height']) != (b['width'], b['height']) or abs(a['fps'] - b['fps']) > 0.01:
            details.append('! The two videos differ in resolution or frame rate - record both cameras with the same '
                           'settings.')
            status = STATUS_WARN
        if st.get('offset_source') == 'manual':
            message = f'Offset set by hand: camera B started {st["offset_s"]:+.3f} s after camera A.'
        else:
            try:
                off, ta, tb = timecode.timecode_offset(st['cam_a'], st['cam_b'])
            except Exception as e:
                # One camera only: keep camera B's video for the record, analyse with camera A
                st['cam_b_unsynced'], st['cam_b'], st['boxes_b'] = st['cam_b'], '', None
                st['offset_s'] = 0.0
                details.append(f'! {str(e).splitlines()[0]}')
                details.append('Camera B is left out: this day is analysed in 2D with camera A only. If you know the '
                               'offset, enter it by hand on this page.')
                return STATUS_WARN, 'The videos can\'t be lined up by their timecodes - camera A only (2D).', details
            st['offset_s'] = off
            later = 'B' if off >= 0 else 'A'
            message = (f'Lined up by the timecodes: A starts {ta["timecode"]}, B starts {tb["timecode"]} - camera '
                       f'{later} started {abs(off):.3f} s later.')
        ma, mb = notes['a']['minutes'], notes['b']['minutes']
        if abs(ma - mb) > 1.0:
            details.append(f'Note: the videos have different lengths ({ma:.1f} / {mb:.1f} min) - where only one '
                           'camera has video, that camera alone is used.')
    else:
        message = 'One camera: this day is analysed in 2D with camera A.'
        if status == STATUS_OK:
            status = STATUS_WARN
    return status, message, details


def step_calibration(st, progress, should_stop, results_dir=None):
    """Find the board in both videos and compute the cameras (or check the chosen calibration file)."""
    import calibrate_cameras as cc
    if not st.get('cam_b'):
        st['calibration'] = ''
        return STATUS_SKIPPED, 'One camera - no calibration needed (2D).', []
    if st.get('calibration_source') == 'file':
        path = st.get('calibration')
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f'The calibration file is not there: {path}')
        cc.load_cameras(path)
        with open(path) as f:
            meta = json.load(f)
        lines = [f'Using the file {path}.']
        for k, label in (('distance_cm', 'Cameras apart (cm)'), ('square_mean_mm', 'Squares measure (mm)'),
                         ('reprojection_px', 'Reprojection error (px)')):
            if isinstance(meta.get(k), (int, float)):
                lines.append(f'{label}: {meta[k]:.2f}')
        lines += [f'! {w}' for w in meta.get('warnings') or []]
        return STATUS_OK, 'Calibration from a file (cameras not moved since).', lines
    c = st.get('calib') or {}
    windows = ([(c['start_s'], c['seconds'])] if c.get('start_s') is not None and c.get('seconds')
               else list(CALIB_WINDOWS))
    errors = []
    for k, (start, length) in enumerate(windows):
        if should_stop():
            raise Stopped()
        try:
            res = cc.calibrate_board(st['cam_a'], st['cam_b'], st.get('offset_s', 0.0), st.get('square_mm', 38.0),
                                     start, length, CALIB_PER_SECOND,
                                     progress=lambda f, t='': progress((k + f) / len(windows), t),
                                     should_stop=should_stop)
        except ValueError as e:
            if should_stop():
                raise Stopped()
            errors.append(f'{start / 60:.0f}-{(start + length) / 60:.0f} min: {e}')
            continue
        path = os.path.join(day_dir(st['subject'], st['session'], results_dir), 'cameras_board.json')
        cc.save_board_calibration(path, res, st['cam_a'], st['cam_b'])
        st['calibration'] = path
        details = [f'Board searched {start / 60:.0f}-{(start + length) / 60:.0f} min.'] + cc.board_report(res).splitlines()
        if res['ok']:
            return STATUS_OK, f'3D: cameras {res["distance_cm"]:.0f} cm apart, squares {res["square_mean_mm"]:.1f} mm.', details
        return STATUS_WARN, f'3D with warnings: cameras {res["distance_cm"]:.0f} cm apart.', details
    st['calibration'] = ''
    return (STATUS_WARN, 'The board was not found - each camera is analysed in 2D. Set the time window where the '
            'board is filmed, or use the calibration file of an earlier recording.', errors)


def _scan_switches(path, start_min, minutes, progress, should_stop):
    """Scan part of a video for the switch lights. Returns the switch boxes found."""
    import cv2
    import extract_kinematics as ek
    import session_engine as se
    cap = se.open_video(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = cap.read()
        if not ok:
            raise ValueError(f'Can\'t read {path}')
        start = int(start_min * 60 * fps)
        if start >= total - fps:
            return []
        bg, pink = ek.scan_video(cap, first, total, fps, lambda f: progress(f), start_frame=start,
                                 max_frames=int(minutes * 60 * fps), should_stop=should_stop)
    finally:
        cap.release()
    if should_stop():
        raise Stopped()
    return ek.find_switches(bg, pink)


def step_switches(st, progress, should_stop):
    """Find the switches of each camera: the first minutes, then later minutes if not all 5 were seen."""
    import extract_kinematics as ek
    cams = cameras_of(st)
    windows = ([(st.get('scan_start_s', 0) / 60, st.get('scan_minutes', 5))] if st.get('scan_custom')
               else list(SWITCH_WINDOWS))
    details, missing = [], []
    for i, c in enumerate(cams):
        best = []
        for k, (start, minutes) in enumerate(windows):
            boxes = _scan_switches(st[f'cam_{c}'], start, minutes,
                                   lambda f, i=i, k=k: progress((i + (k + f) / len(windows)) / len(cams),
                                                                f'camera {c.upper()}'), should_stop)
            if len(boxes) > len(best):
                best = boxes
                details.append(f'Camera {c.upper()}: {len(boxes)} switches seen in minutes {start:g}-{start + minutes:g}.')
            if len(best) >= ek.N_SWITCHES:
                break
        st[f'boxes_{c}'] = best or None
        if len(best) < ek.N_SWITCHES:
            missing.append(f'camera {c.upper()}: {len(best)} of {ek.N_SWITCHES}')
    if missing:
        raise ValueError(f'Not all switches were found ({"; ".join(missing)}). Draw the missing switch(es) on the '
                         'Switches page, or set the minutes where the lights turn on and run it again.')
    return STATUS_OK, f'{ek.N_SWITCHES} switches found in ' + ' and '.join(f'camera {c.upper()}' for c in cams) + \
        '. Check the numbering: 1-5 from the subject\'s RIGHT to LEFT.', details


def step_processing(st, progress, should_stop, results_dir=None):
    """MediaPipe and the lights / touches on every frame of each video (skips videos already processed)."""
    import session_engine as se
    out = day_dir(st['subject'], st['session'], results_dir)
    cams = cameras_of(st)
    details, status = [], STATUS_OK
    for i, c in enumerate(cams):
        path, boxes = st[f'cam_{c}'], st.get(f'boxes_{c}')
        if not boxes:
            raise ValueError(f'Camera {c.upper()} has no switches marked.')
        cache = se.load_cache(path, out, boxes)
        if cache is None or not cache['meta'].get('complete'):
            t0 = time.time()
            cache = se.process_video(path, boxes, out, should_stop=should_stop,
                                     progress=lambda n, tot, left, i=i: progress(
                                         (i + n / max(tot, 1)) / len(cams),
                                         f'camera {c.upper()}' + (f', {left / 60:.0f} min left' if left else '')))
            if should_stop():
                raise Stopped()
            details.append(f'Camera {c.upper()}: {cache["meta"]["frames"]} frames in {(time.time() - t0) / 60:.1f} min.')
        else:
            details.append(f'Camera {c.upper()}: {cache["meta"]["frames"]} frames (processed before).')
        gaps = [x for x in cache['meta'].get('seams') or [] if x.get('missing')]
        if gaps:
            details.append(f'! Camera {c.upper()}: frames missing where the parallel parts meet ({len(gaps)} places) '
                           '- run with LAB_WORKERS=1 to avoid it.')
            status = STATUS_WARN
    return status, 'Every frame processed.', details


def step_analysis(st, progress, should_stop, results_dir=None):
    """Lights -> blocks -> trials -> measures. Returns (status, message, details, session)."""
    s = sa.Session(st)
    if not s.ready:
        raise ValueError('No processed video - run the Processing step first.')
    res = s.analyze()
    trials = res['trials']
    found = [r for r in trials if r['on_frame'] is not None]
    hits = sum(1 for r in found if r['Outcome'] == 'hit')
    flags = dict(miss=sum(1 for r in found if r['Outcome'] == 'miss'),
                 anticipation=sum(1 for r in found if r.get('anticipation')),
                 corrected=sum(1 for r in found if r.get('corrected_from')),
                 not_from_home=sum(1 for r in found if r.get('start_from') not in ('', 'home', None)),
                 not_detected=len(trials) - len(found))
    parts = {}
    for part, _ in res['parts']:
        if part != ed.IGNORE:
            parts[part] = parts.get(part, 0) + 1
    planned = {p.name: p.blocks for p in sa.template_of(st)}
    details = [f"{len(res['blocks'])} groups of lights; switch order {', '.join(map(str, res['pattern'])) or '?'}.",
               'Blocks per part: ' + ', '.join(f'{k} {v}' + ('' if planned.get(k) == v else f' (plan {planned.get(k, 0)})')
                                               for k, v in parts.items()),
               f'{len(found)} trials found: {hits} hits, ' + ', '.join(f'{v} {k.replace("_", " ")}'
                                                                     for k, v in flags.items() if v),
               'Data: ' + s.units]
    details += [f'! {w}' for w in res['warnings']]
    status = STATUS_WARN if res['warnings'] or parts != {k: v for k, v in planned.items() if v} else STATUS_OK
    return status, f'{len(found)} trials in {sum(parts.values())} blocks.', details, s


def step_results(st, progress, should_stop, session=None, results_dir=None):
    """Write the day's CSV + Excel, and the Excel of all participants."""
    s = session
    if s is None or s.result is None:
        s = sa.Session(st)
        s.analyze()
    csv, xlsx = s.write_outputs()
    details = [csv, xlsx]
    try:
        path, n = sa.export_all(results_dir or sa.RESULTS_DIR)
        details.append(f'{path} ({n} participants)')
    except PermissionError:            # the all-participants file is open in Excel
        details.append('! The all-participants Excel is open in another program - not updated.')
    return STATUS_OK, 'Saved the trials CSV and the analysis Excel.', details


# -----------------------------
# Running
# -----------------------------
def run_day(pid, day, progress=None, should_stop=None, on_change=None, results_dir=None, steps=None):
    """Run the steps of a day that are not done yet, in order. A step that fails is marked (error) and the steps
    after it wait. Inputs: progress(step, fraction, text); should_stop() -> True stops; on_change(state) after
    each step; steps - only these steps (None = all). Returns: the day's state."""
    progress = progress or (lambda *a: None)
    should_stop = should_stop or (lambda: False)
    on_change = on_change or (lambda st: None)
    st = load_day(pid, day, results_dir)
    _apply_participant(st, load_participant(pid, results_dir))
    session = None
    for step in STEPS:
        if steps is not None and step not in steps:
            if st['steps'][step]['status'] not in DONE:
                break
            continue
        if st['steps'][step]['status'] in DONE:
            continue
        if st['steps'][step]['status'] == STATUS_ERROR and steps is None:
            break                           # waits for an edit or "Run again"
        set_status(st, step, STATUS_RUNNING, 'Running...')
        save_day(st, results_dir)
        on_change(st)
        t0 = time.time()
        prog = lambda f, text='', step=step: progress(step, max(0.0, min(1.0, f)), text)
        try:
            if step == 'videos':
                out = step_videos(st, prog, should_stop)
            elif step == 'calibration':
                out = step_calibration(st, prog, should_stop, results_dir)
            elif step == 'switches':
                out = step_switches(st, prog, should_stop)
            elif step == 'processing':
                out = step_processing(st, prog, should_stop, results_dir)
            elif step == 'analysis':
                *out, session = step_analysis(st, prog, should_stop, results_dir)
            else:
                out = step_results(st, prog, should_stop, session, results_dir)
            status, message, details = out
            took = time.time() - t0
            if took > 60:
                details = list(details) + [f'Took {took / 60:.1f} min.']
            set_status(st, step, status, message, details)
        except Stopped:
            set_status(st, step, STATUS_TODO, 'Stopped.')
            save_day(st, results_dir)
            on_change(st)
            return st
        except Exception as e:
            traceback.print_exc()
            set_status(st, step, STATUS_ERROR, str(e).splitlines()[0] if str(e) else type(e).__name__,
                       str(e).splitlines()[1:])
            save_day(st, results_dir)
            on_change(st)
            return st
        save_day(st, results_dir)
        on_change(st)
        if should_stop():
            return st
    return st


def queue_items(results_dir=None):
    """Every (participant, day) that has videos and steps still to do (not stopped on an error)."""
    out = []
    for pid in list_participants(results_dir):
        for day in DAYS:
            if day_exists(pid, day, results_dir):
                try:
                    st = load_day(pid, day, results_dir)
                except (ValueError, OSError):          # a broken session.json: shown in the app, not queued
                    continue
                if day_summary(st) == 'todo':
                    out.append((pid, day))
    return out


def run_queue(items, progress=None, should_stop=None, on_change=None, results_dir=None):
    """Run several days one after the other (e.g. overnight). A day that fails doesn't stop the others.
    progress(index, pid, day, step, fraction, text). Returns: list of (pid, day, summary)."""
    progress = progress or (lambda *a: None)
    should_stop = should_stop or (lambda: False)
    done = []
    for i, (pid, day) in enumerate(items):
        if should_stop():
            break
        try:
            st = run_day(pid, day, lambda step, f, text, i=i, pid=pid, day=day: progress(i, pid, day, step, f, text),
                         should_stop, on_change, results_dir)
            done.append((pid, day, day_summary(st)))
        except Exception as e:      # e.g. a broken session.json - the queue goes on
            traceback.print_exc()
            done.append((pid, day, f'error: {e}'))
    return done


def main():
    """Command line: python pipeline.py [participant [day]] - run the queue (all, one participant, or one day)."""
    import sys
    args = sys.argv[1:]
    if args:
        items = [(args[0], d) for d in ((args[1],) if len(args) > 1 else DAYS) if day_exists(args[0], d)]
    else:
        items = queue_items()
    print(f'{len(items)} day(s) to run: ' + ', '.join(f'{p}/{d}' for p, d in items))

    def show(i, pid, day, step, f, text):
        print(f'\r[{i + 1}/{len(items)}] {pid} {DAY_NAMES[day]} - {STEP_NAMES[step]} {f:4.0%} {text:40.40s}',
              end='', flush=True)

    for pid, day, summary in run_queue(items, show):
        st = load_day(pid, day)
        print(f'\n{pid} {DAY_NAMES[day]}: {summary}')
        for s in STEPS:
            x = st['steps'][s]
            print(f'  {STEP_NAMES[s]:16s} {x["status"]:8s} {x["message"]}')


if __name__ == '__main__':
    main()
