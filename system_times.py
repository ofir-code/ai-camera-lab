"""Optional: the light-switch system's own times (its Excel file), used instead of the lights seen in the video.

The system logs, for every press, the time from its light coming on to the press (ms). Its Excel file has one sheet
per block, named <code>-<RT|LT>-<H|L><block>, e.g. AL-RT-H1 (subject AL, right hand, block 1) or YE-LT-L3; in a sheet
each LED channel's column lists the times of that switch's presses in order (a light nobody pressed is not listed).
Channels -> switches (as numbered in the app): LED6 = switch 1, LED2..LED5 = switches 2..5 (LED1 unused) - checked
against the video (LED_OF_SWITCH).

With it on (state['system']['use']), for every trial of a block that has a sheet:
  - hit or miss comes from the system (a trial without a logged press is a miss);
  - the press = light on + the system's time;
  - the light-on stays the one seen in the video when the video agrees with the system (its light-on -> light-off
    matches the logged time within VIDEO_AGREES_MS, after the usual offset); otherwise (e.g. a switch whose light
    the cameras see badly) it comes from the timing: the previous press + the system's pause between trials.
This is optional - without it the video's lights are used, as before.
"""
import re

import numpy as np

import extract_kinematics as ek

LED_OF_SWITCH = {1: 'LED6', 2: 'LED2', 3: 'LED3', 4: 'LED4', 5: 'LED5'}
SHEET = re.compile(r'^\s*([A-Za-z]+)-(RT|LT)-(H|L)(\d+)\s*$', re.I)
VIDEO_AGREES_MS = 80        # the video's light-on -> light-off within this of the logged time (after the offset)
TIMEOUT_MS = 1020           # a video light on at least this long (when the system logged no press) = a timeout


def sheet_key(name):
    """'AL-RT-H1' -> ('AL', 'right', 'upper', 1); None if the name doesn't follow that pattern."""
    m = SHEET.match(name)
    if not m:
        return None
    code, side, limb, n = m.groups()
    return code.upper(), 'right' if side.upper() == 'RT' else 'left', 'upper' if limb.upper() == 'H' else 'lower', int(n)


def read_system_file(path, code):
    """The sheets of one subject (code = the initials at the start of the sheet names, e.g. 'AL'):
    {sheet name: {'time': start time text, 'cols': {'LED1': [ms, ...], ...}}}."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {}
    for name in wb.sheetnames:
        k = sheet_key(name)
        if k is None or k[0] != code.strip().upper():
            continue
        rows = list(wb[name].iter_rows(values_only=True))
        head = next((i for i, r in enumerate(rows) if r and str(r[0]).strip().lower() == 'cycle'), None)
        if head is None:
            continue
        names = [str(c).strip() for c in rows[head]]
        cols = {}
        for j, c in enumerate(names):
            if c.upper().startswith('LED'):
                cols[c.upper()] = [float(r[j]) for r in rows[head + 1:] if r and j < len(r)
                                   and isinstance(r[j], (int, float))]
        when = rows[1][1] if len(rows) > 1 and len(rows[1]) > 1 else ''
        out[name] = dict(time=str(when), cols=cols)
    wb.close()
    return out


def sheet_for(sheets, limb, side, block):
    """The sheet of one block (by limb, side and the block's number), or None."""
    for name, sh in sheets.items():
        k = sheet_key(name)
        if k and k[1:] == (side, limb, block):
            return name, sh
    return None, None


def _align(durs, vals, offset):
    """Which trial each logged press belongs to (index into durs per value), allowing trials without a press
    (misses) - dynamic programming on how well the video's light duration fits each logged time."""
    n, m = len(durs), len(vals)
    INF = 1e18
    f = np.full((n + 1, m + 1), INF)
    f[0, 0] = 0
    back = {}

    def match_cost(d, v):
        return 60.0 if d is None else min(abs(d - v - offset), 300.0)

    def skip_trial(d):     # a trial the system has no press for: cheap if it looks like a timeout
        return 0.0 if d is not None and d >= TIMEOUT_MS else 250.0

    for i in range(n + 1):
        for j in range(m + 1):
            if f[i, j] >= INF:
                continue
            if i < n and j < m and f[i, j] + match_cost(durs[i], vals[j]) < f[i + 1, j + 1]:
                f[i + 1, j + 1] = f[i, j] + match_cost(durs[i], vals[j])
                back[(i + 1, j + 1)] = (i, j, 'match')
            if i < n and f[i, j] + skip_trial(durs[i]) < f[i + 1, j]:
                f[i + 1, j] = f[i, j] + skip_trial(durs[i])
                back[(i + 1, j)] = (i, j, 'miss')
            if j < m and f[i, j] + 400.0 < f[i, j + 1]:     # a logged press with no trial (rare)
                f[i, j + 1] = f[i, j] + 400.0
                back[(i, j + 1)] = (i, j, 'extra')
    pairs = [None] * n
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj, kind = back[(i, j)]
        if kind == 'match':
            pairs[pi] = pj
        i, j = pi, pj
    return pairs


def apply_system_times(trials, system, times_ms, fps, delay_s, touch, settings):
    """Use the system's logged times for the trials of every block that has a sheet (in place). Returns notes."""
    sheets = system.get('sheets') or {}
    led_of = {int(k): v for k, v in (system.get('led_of_switch') or LED_OF_SWITCH).items()}
    times_ms = np.asarray(times_ms, float)
    n = len(times_ms)
    frame_at = lambda ms: int(np.clip(np.searchsorted(times_ms, ms), 0, n - 1))
    delay_ms = 1000.0 * (delay_s or 1.0)
    window_ms = 1000.0 * settings.light_window_s
    found, missing, from_timing, outcome_changed = 0, [], 0, 0
    blocks = {}
    for r in trials:
        blocks.setdefault(r['block_index'], []).append(r)
    for bi, rows in blocks.items():
        r0 = rows[0]
        name, sh = sheet_for(sheets, r0['Limb'], r0['Side'], r0['Block'])
        if sh is None:
            missing.append(f"{r0['Part']} block {r0['Block']}")
            continue
        found += 1
        rows = sorted(rows, key=lambda r: (r['Sequence'], r['Trial']))
        dur = lambda r: (r['Light_off_ms'] - r['Light_on_ms']) if r.get('Light_on_ms') is not None \
            and r.get('Light_off_ms') is not None else None
        # the usual video - system offset in this block (switches whose counts match one to one)
        diffs = []
        for s, led in led_of.items():
            mine = [r for r in rows if r['Target'] == s]
            vals = sh['cols'].get(led, [])
            if len(vals) == len(mine) and all(dur(r) is not None for r in mine):
                diffs += [dur(r) - v for r, v in zip(mine, vals)]
        offset = float(np.median(diffs)) if diffs else 40.0
        logged = {}
        for s, led in led_of.items():
            mine = [r for r in rows if r['Target'] == s]
            vals = sh['cols'].get(led, [])
            for r, j in zip(mine, _align([dur(r) for r in mine], vals, offset)):
                logged[id(r)] = vals[j] if j is not None else None
        prev_press = None
        for r in rows:
            v = logged.get(id(r))
            d = dur(r)
            on_ms = r.get('Light_on_ms')
            video_ok = on_ms is not None and (d is None and v is None or
                                              (d is not None and v is not None and abs(d - v - offset) <= VIDEO_AGREES_MS)
                                              or (v is None and d is not None and d >= TIMEOUT_MS))
            if not video_ok and prev_press is not None:
                on_ms = prev_press + delay_ms
                r['Light_source'] = 'timing'
                from_timing += 1
            if on_ms is None:
                continue
            press = on_ms + v if v is not None else None
            off_ms = press if press is not None else on_ms + window_ms
            new = dict(Light_on_ms=round(on_ms, 1), Light_off_ms=round(off_ms, 1), Trial_end_ms=round(off_ms, 1),
                       Hit_ms=round(press, 1) if press is not None else None,
                       Outcome='hit' if press is not None else 'miss', System_ms=v,
                       on_frame=frame_at(on_ms), off_frame=frame_at(off_ms))
            if r.get('Outcome') != new['Outcome']:
                outcome_changed += 1
            r.update(new)
            r['Light_source'] = (r.get('Light_source') or 'video') + '+system'
            d_touch = touch[r['Limb']][r['Target'] - 1, r['on_frame']:min(n, r['off_frame'] + 2)]
            hit = np.nonzero(d_touch <= ek.TOUCH_DIST)[0]
            r['Touch_ms'] = round(float(times_ms[r['on_frame'] + int(hit[0])]), 1) if len(hit) else None
            prev_press = off_ms if press is not None else on_ms + window_ms
        # each trial's data window: from the previous trial's end (the home phase)
        done = [r for r in rows if r.get('on_frame') is not None]
        for k, r in enumerate(done):
            prev_end = done[k - 1]['off_frame'] if k > 0 else None
            r['window_start'] = prev_end if prev_end is not None else max(0, r['on_frame'] - int(2 * fps))
    notes = [f'Light-switch system times used for {found} block(s): hit / miss and press times from the system; '
             f'{outcome_changed} trial(s) changed hit / miss, {from_timing} light(s) placed by the timing (the video and '
             'the system disagreed, e.g. a light the cameras see badly).']
    if missing:
        notes.append('No system sheet for: ' + ', '.join(missing) + ' - the video\'s lights are used there.')
    return notes
