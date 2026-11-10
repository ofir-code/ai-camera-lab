"""Reaching Lab app: participants -> 2 GoPro videos per day (day 1 and the re-test day) -> everything runs
automatically (sync, board calibration, switches, MediaPipe, trials synced with the design, RT / MT) -> CSV + Excel.

Run:  venv/bin/python lab_app.py      (or double-click "Lab App.command" / "Lab App.bat")

Left: the participants and the queue (Run all: every day still to do, one after the other - e.g. overnight).
Right: the chosen participant's days. Each day shows its steps; the app runs them by itself, and each step's
page shows what it found and lets you change it (then the steps after it run again).
The steps themselves are in pipeline.py.
"""
import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal

import extract_kinematics as ek
import experiment_design as ed
import pipeline as pl
import session_analysis as sa
from lab_widgets import (THREAD_STACK, BlocksView, BoxView, TrialsView, Worker, group, read_frame, status_html)

STATUS_STYLE = {pl.STATUS_TODO: ('#e9ecef', '#495057', ''), pl.STATUS_RUNNING: ('#cfe2ff', '#084298', '⟳ '),
                pl.STATUS_OK: ('#d1e7dd', '#0f5132', '✓ '), pl.STATUS_WARN: ('#fff3cd', '#664d03', '! '),
                pl.STATUS_ERROR: ('#f8d7da', '#842029', '✗ '), pl.STATUS_SKIPPED: ('#f1f3f5', '#868e96', '– ')}
SUMMARY_MARK = {'done': '✓', 'warn': '!', 'error': '✗', 'todo': '…', 'empty': '–', 'running': '⟳'}
VIDEO_FILTER = 'Videos (*.mp4 *.MP4 *.mov *.MOV *.avi *.mkv)'


# -----------------------------
# Background runs of the pipeline
# -----------------------------
class Runner(QThread):
    """Runs days of the pipeline in the background (one at a time): a queue of (participant, day)."""
    progress = Signal(int, str, str, str, float, str)    # index, participant, day, step, fraction, text
    changed = Signal(str, str)                           # participant, day: a step's state changed
    finished_all = Signal(list)                          # [(participant, day, summary)]

    def __init__(self, items):
        super().__init__()
        self.items = list(items)
        self.stop_requested = False
        self.current = None
        self.setStackSize(THREAD_STACK)

    def run(self):
        def prog(i, pid, day, step, f, text):
            self.current = (pid, day)
            self.progress.emit(i, pid, day, step, f, text)
        def changed(st):
            self.current = (st['subject'], st['session'])
            self.changed.emit(st['subject'], st['session'])
        out = pl.run_queue(self.items, prog, lambda: self.stop_requested, changed)
        self.current = None
        self.finished_all.emit(out)


# -----------------------------
# Participant (design) dialog
# -----------------------------
class ParticipantDialog(QtWidgets.QDialog):
    """New participant (id, design, videos of the days) or the design of an existing one."""

    def __init__(self, parent, participant=None):
        super().__init__(parent)
        self.p = participant
        self.setWindowTitle('New participant' if participant is None else f'Participant {participant["id"]}')
        self.resize(760, 0)
        v = QtWidgets.QVBoxLayout(self)
        f = QtWidgets.QFormLayout()
        self.e_id = QtWidgets.QLineEdit()
        self.e_id.setPlaceholderText('e.g. S01')
        if participant is not None:
            self.e_id.setText(participant['id'])
            self.e_id.setEnabled(False)
        f.addRow('Participant ID', self.e_id)
        v.addWidget(group('Participant', f))

        d = (participant or {}).get('design') or pl.DESIGN_DEFAULT
        f = QtWidgets.QFormLayout()
        self.c_tested = QtWidgets.QComboBox()
        self.c_tested.addItems(['right', 'left'])
        self.c_tested.setCurrentText(d['tested_side'])
        self.k_contra = QtWidgets.QCheckBox('also test the contralateral (other) arm, pre, post and re-test')
        self.k_contra.setChecked(d['contralateral'])
        self.c_plimb = QtWidgets.QComboBox()
        self.c_plimb.addItems(['upper', 'lower'])
        self.c_plimb.setCurrentText(d['practice_limb'])
        self.c_pside = QtWidgets.QComboBox()
        self.c_pside.addItems(['right', 'left'])
        self.c_pside.setCurrentText(d['practice_side'])
        f.addRow('Tested arm (pre / post / re-test)', self.c_tested)
        f.addRow('', self.k_contra)
        prow = QtWidgets.QHBoxLayout()
        for lab, w in (('limb', self.c_plimb), ('side', self.c_pside)):
            prow.addWidget(QtWidgets.QLabel(lab))
            prow.addWidget(w)
        prow.addStretch()
        f.addRow('Practice', prow)
        self.c_fam = QtWidgets.QComboBox()
        self.c_fam.addItems(['with the practice limb', 'with the tested arm'])
        self.c_fam.setCurrentIndex(1 if d.get('familiarization') == 'test' else 0)
        self.c_fam.currentIndexChanged.connect(self.update_plan)
        f.addRow('Familiarization', self.c_fam)
        self.l_plan = QtWidgets.QLabel()
        self.l_plan.setWordWrap(True)
        f.addRow('Plan', self.l_plan)
        for w in (self.c_tested, self.c_plimb, self.c_pside):
            w.currentIndexChanged.connect(self.update_plan)
        self.k_contra.toggled.connect(self.update_plan)
        v.addWidget(group('Experimental design', f))

        st = (participant or {}).get('settings') or vars(ed.Settings())
        f = QtWidgets.QFormLayout()
        self.s_tps = QtWidgets.QSpinBox()
        self.s_tps.setRange(1, 20)
        self.s_tps.setValue(st['trials_per_sequence'])
        self.s_spb = QtWidgets.QSpinBox()
        self.s_spb.setRange(1, 20)
        self.s_spb.setValue(st['sequences_per_block'])
        self.s_window = QtWidgets.QDoubleSpinBox()
        self.s_window.setRange(0.2, 5)
        self.s_window.setSuffix(' s')
        self.s_window.setValue(st['light_window_s'])
        self.s_gap = QtWidgets.QDoubleSpinBox()
        self.s_gap.setRange(2, 120)
        self.s_gap.setSuffix(' s')
        self.s_gap.setValue(st['block_gap_s'])
        self.e_pattern = QtWidgets.QLineEdit(','.join(map(str, st.get('pattern') or [])))
        self.e_pattern.setPlaceholderText('auto (from the data), or e.g. 1,4,3,5,4,2')
        import calibrate_cameras as cc
        self.s_square = QtWidgets.QDoubleSpinBox()
        self.s_square.setRange(5, 200)
        self.s_square.setDecimals(1)
        self.s_square.setSuffix(' mm')
        self.s_square.setValue((participant or {}).get('square_mm', cc.BOARD_SQUARE_MM))
        f.addRow('Trials per sequence', self.s_tps)
        f.addRow('Sequences per block', self.s_spb)
        f.addRow('Light window (miss if not hit within)', self.s_window)
        f.addRow('New block after a pause longer than', self.s_gap)
        f.addRow('Switch order in a sequence', self.e_pattern)
        f.addRow('Calibration board square (measured)', self.s_square)
        self.adv = group('Settings (usually left as they are)', f)
        self.adv.setCheckable(True)
        self.adv.setChecked(participant is not None)
        v.addWidget(self.adv)

        self.videos = {}
        if participant is None:
            f = QtWidgets.QFormLayout()
            for day in pl.DAYS:
                row = QtWidgets.QHBoxLayout()
                e = QtWidgets.QLineEdit()
                e.setReadOnly(True)
                e.setPlaceholderText('the 2 videos of this day (can be added later)')
                b = QtWidgets.QPushButton('Choose…')
                b.clicked.connect(lambda _, day=day: self.choose(day))
                row.addWidget(e, 1)
                row.addWidget(b)
                self.videos[day] = (e, [])
                f.addRow(pl.DAY_NAMES[day], row)
            self.k_start = QtWidgets.QCheckBox('Start running now')
            self.k_start.setChecked(True)
            f.addRow('', self.k_start)
            v.addWidget(group('Videos (GoPro: choose the first chapter of each camera, GX01....MP4)', f))
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self.update_plan()

    def choose(self, day):
        paths = choose_videos(self)
        if paths is not None:
            e, lst = self.videos[day]
            lst[:] = paths
            e.setText('   +   '.join(os.path.basename(p) for p in paths))

    def design(self):
        return dict(tested_side=self.c_tested.currentText(), contralateral=self.k_contra.isChecked(),
                    practice_limb=self.c_plimb.currentText(), practice_side=self.c_pside.currentText(),
                    familiarization='test' if self.c_fam.currentIndex() == 1 else 'practice')

    def settings(self):
        base = dict((self.p or {}).get('settings') or vars(ed.Settings()))
        pattern = ed.parse_pattern(self.e_pattern.text())
        base.update(trials_per_sequence=self.s_tps.value(), sequences_per_block=self.s_spb.value(),
                    light_window_s=self.s_window.value(), block_gap_s=self.s_gap.value(), pattern=pattern)
        return base

    def update_plan(self, *_):
        d = self.design()
        lines = []
        for day in pl.DAYS:
            parts = ed.session_template(d['tested_side'], d['contralateral'], d['practice_limb'], d['practice_side'],
                                        day != 'd1', d['familiarization'])
            lines.append(f'<b>{pl.DAY_NAMES[day]}</b>: ' + ' → '.join(
                f'{p.name} ({p.blocks}, {"arm" if p.limb == "upper" else "leg"})' for p in parts)
                         + f' = {sum(p.blocks for p in parts)} blocks')
        self.l_plan.setText('<br>'.join(lines))

    def accept(self):
        try:
            problem = ed.pattern_problem(self.settings()['pattern'], ed.Settings(**self.settings()))
            if problem:
                raise ValueError(problem)
            if self.p is None:
                pl.check_id(self.e_id.text())
                if self.e_id.text().strip() in pl.list_participants():
                    raise ValueError(f'Participant {self.e_id.text().strip()} already exists.')
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'Participant', str(e))
            return
        super().accept()


def choose_videos(parent):
    """Let the user choose the 1 or 2 videos of a day (camera A and B). Returns their paths (first chapters),
    or None if cancelled / not valid."""
    import video_source as vs
    paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
        parent, 'The videos of this day: one per camera (GoPro: the first chapter, GX01....MP4)',
        os.path.expanduser('~'), VIDEO_FILTER)
    if not paths:
        return None
    firsts = list(dict.fromkeys(vs.first_chapter(p) for p in sorted(paths)))
    if len(firsts) > 2:
        QtWidgets.QMessageBox.warning(parent, 'Videos', 'Choose one video per camera (two at most). The chapters of '
                                      'a GoPro video (GX02…, GX03…) are found by themselves.')
        return None
    if len(firsts) < len(paths):
        QtWidgets.QMessageBox.information(parent, 'Videos', 'Some of the chosen files are chapters of the same '
                                          'video - it is used once: ' + ', '.join(map(os.path.basename, firsts)))
    return firsts


# -----------------------------
# One day
# -----------------------------
class DayView(QtWidgets.QWidget):
    """A participant's day: its videos, the step bar and the page of the chosen step."""
    run_requested = Signal(str, str)        # participant, day

    def __init__(self, app, day):
        super().__init__()
        self.app, self.day = app, day
        self.pid = None
        self.st = None
        self.session = None
        self.busy = False
        self.edit_widgets = []
        v = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        self.l_videos = QtWidgets.QLabel()
        self.l_videos.setTextFormat(Qt.RichText)
        self.l_videos.setWordWrap(True)
        top.addWidget(self.l_videos, 1)
        self.b_choose = QtWidgets.QPushButton('Choose videos…')
        self.b_choose.clicked.connect(self.choose_videos)
        self.b_swap = QtWidgets.QPushButton('Swap A / B')
        self.b_swap.setToolTip('Make camera A camera B and the other way round')
        self.b_swap.clicked.connect(self.swap)
        self.b_run = QtWidgets.QPushButton('Run')
        self.b_run.setStyleSheet('font-weight:bold; padding:6px 20px;')
        self.b_run.setToolTip('Run the steps of this day that are not done yet')
        self.b_run.clicked.connect(lambda: self.run_requested.emit(self.pid, self.day))
        for b in (self.b_choose, self.b_swap, self.b_run):
            top.addWidget(b)
        v.addLayout(top)

        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(4)
        self.step_buttons = {}
        for i, s in enumerate(pl.STEPS):
            b = QtWidgets.QPushButton(f'{i + 1}. {pl.STEP_NAMES[s]}'.replace('&', '&&'))
            b.setCheckable(True)
            b.setMinimumHeight(38)
            b.clicked.connect(lambda _, s=s: self.show_step(s))
            bar.addWidget(b, 1)
            self.step_buttons[s] = b
        v.addLayout(bar)

        self.l_status = QtWidgets.QLabel()
        self.l_status.setTextFormat(Qt.RichText)
        self.l_status.setWordWrap(True)
        self.l_status.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.l_status.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        self.l_status.setOpenExternalLinks(True)
        self.l_status.setStyleSheet('QLabel{background:#f8f9fa; border:1px solid #dee2e6; padding:8px;}')
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.l_status)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(240)
        scroll.setMinimumHeight(110)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        v.addWidget(scroll)

        self.pages = QtWidgets.QStackedWidget()
        self.page_of = {}
        for s, make in (('videos', self.page_videos), ('calibration', self.page_calibration),
                        ('switches', self.page_switches), ('processing', self.page_processing),
                        ('analysis', self.page_analysis), ('results', self.page_results)):
            self.page_of[s] = self.pages.addWidget(make())
        v.addWidget(self.pages, 1)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addStretch()
        self.b_again = QtWidgets.QPushButton('Run this step again')
        self.b_again.setToolTip('Run this step again automatically (and the steps after it)')
        self.b_again.clicked.connect(self.run_again)
        bottom.addWidget(self.b_again)
        v.addLayout(bottom)
        self.current_step = 'videos'
        self.edit_widgets += [self.b_choose, self.b_swap, self.b_again]

    # ---------- state ----------
    def set_participant(self, pid):
        self.pid = pid
        self.session = None
        self.reload()
        if self.st is not None:
            self.show_step(pl.next_step(self.st) or 'analysis')

    def reload(self):
        """Read the day's state from disk and refresh what is shown."""
        old = self.st['steps']['analysis'] if self.st else None
        self.st = (pl.load_day(self.pid, self.day) if self.pid and pl.day_exists(self.pid, self.day) else None)
        if self.session is not None:      # keep the loaded analysis unless the analysis step changed
            if self.st is None or self.st['steps']['analysis'] != old:
                self.session = None
            else:
                self.session.state = self.st
        self.refresh()

    def save(self):
        pl.save_day(self.st)

    def set_busy(self, busy):
        """The day is being run in the background: no edits meanwhile."""
        self.busy = busy
        for w in self.edit_widgets:
            w.setEnabled(not busy)
        self.b_run.setEnabled(not busy and not self.app.running() and self.st is not None)

    def refresh(self):
        st = self.st
        if st is None or not st.get('cam_a'):
            self.l_videos.setText('<b>No videos yet</b> - choose the 2 videos of this day (one per camera).')
        else:
            names = [f'<b>A</b>: {os.path.basename(st["cam_a"])}']
            if st.get('cam_b'):
                names.append(f'<b>B</b>: {os.path.basename(st["cam_b"])}')
            elif st.get('cam_b_unsynced'):
                names.append(f'<b>B</b>: <s>{os.path.basename(st["cam_b_unsynced"])}</s> (not lined up - left out)')
            else:
                names.append('<b>B</b>: none (2D)')
            self.l_videos.setText('&nbsp;&nbsp;&nbsp;'.join(names))
        for s, b in self.step_buttons.items():
            status = st['steps'][s]['status'] if st else pl.STATUS_TODO
            bg, fg, mark = STATUS_STYLE[status]
            b.setText(f'{mark}{pl.STEPS.index(s) + 1}. {pl.STEP_NAMES[s]}'.replace('&', '&&'))
            b.setStyleSheet(f'QPushButton{{background:{bg}; color:{fg}; border:1px solid #ced4da; border-radius:4px;}}'
                            f'QPushButton:checked{{border:2px solid #0d6efd; font-weight:bold;}}')
        self.b_swap.setEnabled(bool(st and (st.get('cam_b') or st.get('cam_b_unsynced'))) and not self.busy)
        self.b_run.setEnabled(bool(st and st.get('cam_a')) and not self.busy and not self.app.running())
        self.show_step(self.current_step)

    def show_step(self, step):
        self.current_step = step
        for s, b in self.step_buttons.items():
            b.setChecked(s == step)
        self.pages.setCurrentIndex(self.page_of[step])
        if self.st is None:
            self.l_status.setText('Choose the videos of this day first.')
        else:
            x = self.st['steps'][step]
            text = status_html(x) if x['status'] != pl.STATUS_TODO or x['message'] else (
                'Not run yet - press <b>Run</b> (or <b>Run all</b> on the left) and the app does it by itself.')
            self.l_status.setText(text + (f'<br><small style="color:#888">{x["time"]}</small>'
                                          if x['status'] not in (pl.STATUS_TODO, pl.STATUS_RUNNING) else ''))
        refresh = getattr(self, f'refresh_{step}', None)
        if refresh:
            refresh()

    def progress(self, step, f, text):
        if self.st and self.st['steps'][step]['status'] == pl.STATUS_RUNNING:
            self.step_buttons[step].setText(f'⟳ {pl.STEPS.index(step) + 1}. {pl.STEP_NAMES[step]} {f:.0%}'.replace('&', '&&'))

    def run_again(self):
        if self.st is None:
            return
        pl.mark_todo(self.st, self.current_step)
        self.save()
        self.session = None
        self.refresh()
        self.run_requested.emit(self.pid, self.day)

    def changed(self, step, message, run=True):
        """The user changed a step by hand: the steps after it run again."""
        pl.edited(self.st, step, message)
        self.session = None
        self.refresh()
        if run:
            self.run_requested.emit(self.pid, self.day)

    # ---------- top buttons ----------
    def choose_videos(self):
        paths = choose_videos(self)
        if paths is None:
            return
        if self.st and self.st.get('cam_a') and self.st['steps']['processing']['status'] in pl.DONE:
            if QtWidgets.QMessageBox.question(self, 'Videos', 'Other videos: this day\'s results are made again from '
                                              'the start. Go on?') != QtWidgets.QMessageBox.Yes:
                return
        self.st = pl.set_videos(self.pid, self.day, paths[0], paths[1] if len(paths) > 1 else '')
        self.session = None
        self.app.refresh_list()
        self.show_step('videos')
        self.refresh()
        self.run_requested.emit(self.pid, self.day)

    def swap(self):
        if self.st.get('cam_b_unsynced') and not self.st.get('cam_b'):
            self.st['cam_b'] = self.st.pop('cam_b_unsynced')
        pl.swap_cameras(self.st)
        self.session = None
        self.refresh()
        self.run_requested.emit(self.pid, self.day)

    # ---------- 1. videos ----------
    def page_videos(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QHBoxLayout()
        self.r_tc = QtWidgets.QRadioButton('by the GoPro timecodes (cameras synced in Quik)')
        self.r_manual = QtWidgets.QRadioButton('by hand: camera B started')
        self.s_offset = QtWidgets.QDoubleSpinBox()
        self.s_offset.setRange(-3600, 3600)
        self.s_offset.setDecimals(3)
        self.s_offset.setSuffix(' s after camera A')
        b = QtWidgets.QPushButton('Apply')
        b.clicked.connect(self.apply_offset)
        for x in (self.r_tc, self.r_manual, self.s_offset, b):
            f.addWidget(x)
        f.addStretch()
        g = group('Line up the two videos', f)
        v.addWidget(g)
        v.addWidget(QtWidgets.QLabel(
            'Before recording each day (and after changing a battery): sync both cameras in the GoPro Quik app '
            '(Timecode Sync). A video without a timecode is analysed with one camera (2D).'))
        v.addStretch()
        self.edit_widgets += [b]
        return w

    def refresh_videos(self):
        st = self.st
        manual = bool(st and st.get('offset_source') == 'manual')
        self.r_manual.setChecked(manual)
        self.r_tc.setChecked(not manual)
        self.s_offset.setValue(st.get('offset_s', 0.0) if st else 0.0)

    def apply_offset(self):
        if self.st is None:
            return
        pl.set_offset(self.st, self.s_offset.value() if self.r_manual.isChecked() else None)
        self.session = None
        self.refresh()
        self.run_requested.emit(self.pid, self.day)

    # ---------- 2. calibration ----------
    def page_calibration(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        f = QtWidgets.QHBoxLayout()
        self.r_cal_auto = QtWidgets.QRadioButton('Find the board in the session videos')
        self.s_cal_from = QtWidgets.QDoubleSpinBox()
        self.s_cal_from.setRange(0, 600)
        self.s_cal_from.setSuffix(' min')
        self.s_cal_from.setDecimals(1)
        self.s_cal_len = QtWidgets.QSpinBox()
        self.s_cal_len.setRange(0, 1800)
        self.s_cal_len.setSuffix(' s')
        self.s_cal_len.setSpecialValueText('automatic')
        self.s_cal_len.setToolTip('automatic: the first 3 minutes, then 3-7 minutes')
        for x in (self.r_cal_auto, QtWidgets.QLabel('from'), self.s_cal_from, QtWidgets.QLabel('for'), self.s_cal_len):
            f.addWidget(x)
        f.addStretch()
        v.addWidget(group('Where the ChArUco board is', f))
        row = QtWidgets.QHBoxLayout()
        b_apply = QtWidgets.QPushButton('Calibrate with this window')
        b_apply.clicked.connect(self.apply_calib_window)
        b_clips = QtWidgets.QPushButton('Board filmed in separate clips…')
        b_clips.setToolTip('The board was filmed in 2 short clips (one per camera) without moving the cameras')
        b_clips.clicked.connect(self.calib_clips)
        b_file = QtWidgets.QPushButton('Use a calibration file…')
        b_file.setToolTip('The calibration of an earlier recording - only if the cameras were not moved since')
        b_file.clicked.connect(self.calib_file)
        b_none = QtWidgets.QPushButton('No 3D (2D per camera)')
        b_none.clicked.connect(self.calib_none)
        for b in (b_apply, b_clips, b_file, b_none):
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)
        v.addWidget(QtWidgets.QLabel(
            'For a good calibration: at the start of the session hold the board where both cameras see it and slowly '
            'tilt and move it for 20-30 s (filling about 10-20 % of the picture).'))
        v.addStretch()
        self.edit_widgets += [b_apply, b_clips, b_file, b_none]
        return w

    def refresh_calibration(self):
        c = (self.st or {}).get('calib') or {}
        self.r_cal_auto.setChecked((self.st or {}).get('calibration_source', 'auto') == 'auto')
        self.s_cal_from.setValue((c.get('start_s') or 0) / 60)
        self.s_cal_len.setValue(int(c.get('seconds') or 0))

    def apply_calib_window(self):
        if self.st is None:
            return
        secs = self.s_cal_len.value()
        self.st['calib'] = dict(start_s=self.s_cal_from.value() * 60 if secs else None, seconds=secs or None)
        self.st['calibration_source'] = 'auto'
        self.run_again_step('calibration')

    def calib_clips(self):
        if self.st is None:
            return
        paths = choose_videos(self)
        if not paths or len(paths) != 2:
            if paths is not None:
                QtWidgets.QMessageBox.warning(self, 'Calibration', 'Choose the 2 board clips (one per camera).')
            return
        self.st['calib_clips'] = paths
        self.st['calibration_source'] = 'clips'
        self.run_again_step('calibration')

    def calib_file(self):
        if self.st is None:
            return
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Calibration file (cameras not moved since)',
                                                     pl.participant_dir(self.pid), 'JSON (*.json)')
        if p:
            self.st['calibration'] = p
            self.st['calibration_source'] = 'file'
            self.run_again_step('calibration')

    def calib_none(self):
        if self.st is None:
            return
        self.st['calibration'] = ''
        self.st['calibration_source'] = 'none'
        pl.set_status(self.st, 'calibration', pl.STATUS_SKIPPED, 'No 3D - each camera is analysed in 2D (chosen).')
        pl.mark_todo(self.st, 'calibration', include_self=False)
        self.save()
        self.session = None
        self.refresh()
        self.run_requested.emit(self.pid, self.day)

    def run_again_step(self, step):
        pl.mark_todo(self.st, step)
        self.save()
        self.session = None
        self.refresh()
        self.run_requested.emit(self.pid, self.day)

    # ---------- 3. switches ----------
    def page_switches(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        top = QtWidgets.QHBoxLayout()
        self.c_swcam = QtWidgets.QComboBox()
        self.c_swcam.addItems(['Camera A', 'Camera B'])
        self.c_swcam.currentIndexChanged.connect(self.refresh_switches)
        top.addWidget(self.c_swcam)
        self.l_sw = QtWidgets.QLabel()
        top.addWidget(self.l_sw, 1)
        v.addLayout(top)
        self.boxview = BoxView()
        self.boxview.box_drawn.connect(self.on_box_drawn)
        v.addWidget(self.boxview, 1)
        bot = QtWidgets.QHBoxLayout()
        b_flip = QtWidgets.QPushButton('Flip numbering')
        b_flip.clicked.connect(self.flip_switches)
        self.s_swn = QtWidgets.QSpinBox()
        self.s_swn.setRange(1, ek.N_SWITCHES)
        self.s_swn.setPrefix('switch ')
        b_redraw = QtWidgets.QPushButton('Redraw')
        b_redraw.clicked.connect(lambda: self.draw_box(self.s_swn.value() - 1))
        b_del = QtWidgets.QPushButton('Remove')
        b_del.clicked.connect(self.remove_switch)
        b_add = QtWidgets.QPushButton('Add a missing switch')
        b_add.clicked.connect(lambda: self.draw_box(None))
        self.b_sw_done = QtWidgets.QPushButton('Done - continue')
        self.b_sw_done.setStyleSheet('font-weight:bold;')
        self.b_sw_done.clicked.connect(self.switches_done)
        for b in (b_flip, self.s_swn, b_redraw, b_del, b_add):
            bot.addWidget(b)
        bot.addSpacing(20)
        bot.addWidget(QtWidgets.QLabel('Look for the lights from'))
        self.s_scan_from = QtWidgets.QDoubleSpinBox()
        self.s_scan_from.setRange(0, 600)
        self.s_scan_from.setSuffix(' min')
        self.s_scan_len = QtWidgets.QDoubleSpinBox()
        self.s_scan_len.setRange(0.5, 60)
        self.s_scan_len.setValue(5)
        self.s_scan_len.setSuffix(' min')
        b_find = QtWidgets.QPushButton('Find again')
        b_find.clicked.connect(self.find_again)
        for x in (self.s_scan_from, QtWidgets.QLabel('for'), self.s_scan_len, b_find):
            bot.addWidget(x)
        bot.addStretch()
        bot.addWidget(self.b_sw_done)
        v.addLayout(bot)
        self.redraw_index = None
        self.boxes_edited = False
        self.edit_widgets += [b_flip, b_redraw, b_del, b_add, b_find, self.b_sw_done]
        return w

    def sw_cam(self):
        return 'b' if self.c_swcam.currentIndex() == 1 else 'a'

    def refresh_switches(self):
        st = self.st
        self.c_swcam.setEnabled(bool(st and st.get('cam_b')))
        if st and not st.get('cam_b'):
            self.c_swcam.setCurrentIndex(0)
        cam = self.sw_cam()
        if st is None or not st.get(f'cam_{cam}'):
            self.boxview.show_frame(None, None)
            self.l_sw.setText('')
            return
        self.s_scan_from.setValue(st.get('scan_start_s', 0) / 60)
        self.s_scan_len.setValue(st.get('scan_minutes', 5))
        boxes = st.get(f'boxes_{cam}')
        key = (st[f'cam_{cam}'], 0)
        if getattr(self, '_frame_key', None) != key:
            self._frame = read_frame(st[f'cam_{cam}'])
            self._frame_key = key
        self.boxview.show_frame(self._frame, boxes)
        n = len(boxes or [])
        self.l_sw.setText(f'  {n} of {ek.N_SWITCHES} switches. Numbered 1-5 from the subject\'s RIGHT to LEFT - use '
                          'Flip if it\'s the other way. A box should include the light.')
        self.b_sw_done.setVisible(self.boxes_edited or st['steps']['switches']['status'] == pl.STATUS_ERROR)

    def set_boxes(self, boxes):
        self.st[f'boxes_{self.sw_cam()}'] = boxes
        self.boxes_edited = True
        pl.mark_todo(self.st, 'switches', include_self=False)
        self.save()
        self.session = None
        self.refresh_switches()

    def flip_switches(self):
        boxes = self.st.get(f'boxes_{self.sw_cam()}') if self.st else None
        if boxes:
            self.set_boxes(boxes[::-1])

    def draw_box(self, index):
        if self.st is None:
            return
        self.redraw_index = index
        self.boxview.start_drawing()
        self.l_sw.setText('  Drag a box around the switch (include its light).')

    def on_box_drawn(self, box):
        boxes = list(self.st.get(f'boxes_{self.sw_cam()}') or [])
        if self.redraw_index is None:
            boxes = ek.sort_switches(boxes + [box])
        elif self.redraw_index < len(boxes):
            boxes[self.redraw_index] = box
        else:
            boxes.append(box)
        self.set_boxes(boxes)

    def remove_switch(self):
        boxes = list(self.st.get(f'boxes_{self.sw_cam()}') or []) if self.st else []
        i = self.s_swn.value() - 1
        if i < len(boxes):
            boxes.pop(i)
            self.set_boxes(boxes)

    def switches_done(self):
        """The boxes were edited by hand: they are the switches step's result; go on."""
        missing = [c.upper() for c in pl.cameras_of(self.st) if len(self.st.get(f'boxes_{c}') or []) != ek.N_SWITCHES]
        if missing:
            QtWidgets.QMessageBox.warning(self, 'Switches', f'Camera {" and ".join(missing)} does not have '
                                          f'{ek.N_SWITCHES} switches yet.')
            return
        self.boxes_edited = False
        self.changed('switches', 'Switch boxes set by hand.')

    def find_again(self):
        if self.st is None:
            return
        self.st['scan_start_s'] = self.s_scan_from.value() * 60
        self.st['scan_minutes'] = self.s_scan_len.value()
        self.st['scan_custom'] = True
        self.boxes_edited = False
        self.run_again_step('switches')

    # ---------- 4. processing ----------
    def page_processing(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            'One pass over each video: MediaPipe on every frame, the switch lights, and where the hands / feet are.\n'
            'This is the slow step (on a Mac about 1.5 x the video\'s length per camera). It is saved, so it runs '
            'once per video - changing the switch boxes runs it again.'))
        v.addStretch()
        return w

    # ---------- 5. blocks & trials ----------
    def page_analysis(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        self.tabs = QtWidgets.QTabWidget()
        self.blocks = BlocksView()
        self.blocks.parts_changed.connect(self.parts_changed)
        self.trials = TrialsView()
        self.trials.touse_changed.connect(self.touse_changed)
        self.tabs.addTab(self.trials, 'Trials')
        self.tabs.addTab(self.blocks, 'Blocks && parts')
        v.addWidget(self.tabs)
        self.loader = None
        self.save_timer = QtCore.QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(1500)
        self.save_timer.timeout.connect(lambda: self.run_requested.emit(self.pid, self.day))
        return w

    def refresh_analysis(self):
        self.blocks.set_editable(not self.busy)
        if self.st is None or self.st['steps']['analysis']['status'] not in pl.DONE:
            self.session = None
            self.blocks.set_session(None)
            self.trials.set_session(None)
            return
        if self.session is not None:
            return
        if self.loader is not None and self.loader.isRunning():
            return
        st = dict(self.st)

        def job(progress, stop):
            s = sa.Session(st)
            s.analyze()
            return s

        def done(s):
            if self.st is None or s.state.get('subject') != self.pid:
                return
            s.state = self.st            # touse choices are saved into the day's state
            self.session = s
            self.blocks.set_session(s)
            self.trials.set_session(s)
        self.blocks.label.setText('Loading the analysis…')
        self.loader = Worker(job)
        self.loader.done.connect(done)
        self.loader.failed.connect(lambda m: self.blocks.label.setText(f'Could not load the analysis: {m}'))
        self.loader.start()

    def parts_changed(self, parts):
        self.st['parts_override'] = parts
        self.run_again_step('analysis')

    def touse_changed(self):
        pl.mark_todo(self.st, 'results')
        self.save()
        self.refresh_buttons_only()
        if not self.app.running():
            self.save_timer.start()

    def refresh_buttons_only(self):
        for s, b in self.step_buttons.items():
            bg, fg, mark = STATUS_STYLE[self.st['steps'][s]['status']]
            b.setText(f'{mark}{pl.STEPS.index(s) + 1}. {pl.STEP_NAMES[s]}'.replace('&', '&&'))
            b.setStyleSheet(f'QPushButton{{background:{bg}; color:{fg}; border:1px solid #ced4da; border-radius:4px;}}'
                            f'QPushButton:checked{{border:2px solid #0d6efd; font-weight:bold;}}')

    # ---------- 6. results ----------
    def page_results(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.addWidget(QtWidgets.QLabel(
            '<b>trials.csv</b>: every trial synced with the design - part, block, sequence, trial, target, light on, '
            'touch, hit, light off, trial end, outcome, started before the light, corrected, not from home.<br>'
            '<b>analysis.xlsx</b>: RT, MT (+ AT, DT, peaks, straightness) per trial, and mean / median / SD / IQR by '
            'part, block and sequence (familiarization and touse = 0 left out).<br>'
            '<b>analysis_all_subjects.xlsx</b> (in the results folder): every participant, both days.'))
        row = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton('Open this day\'s folder')
        b1.clicked.connect(lambda: open_folder(pl.day_dir(self.pid, self.day)))
        b2 = QtWidgets.QPushButton('Open the results folder')
        b2.clicked.connect(lambda: open_folder(sa.RESULTS_DIR))
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch()
        v.addLayout(row)
        v.addStretch()
        return w


def open_folder(path):
    os.makedirs(path, exist_ok=True)
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))


# -----------------------------
# Main window
# -----------------------------
class LabApp(QtWidgets.QMainWindow):
    """Participants and the queue on the left; the chosen participant's days on the right."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Reaching Lab')
        self.resize(1450, 920)
        self.runner = None
        split = QtWidgets.QSplitter()
        self.setCentralWidget(split)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.addWidget(QtWidgets.QLabel('<b>Participants</b>'))
        self.list = QtWidgets.QListWidget()
        self.list.currentItemChanged.connect(self.on_select)
        lv.addWidget(self.list, 1)
        b_new = QtWidgets.QPushButton('New participant…')
        b_new.clicked.connect(self.new_participant)
        lv.addWidget(b_new)
        lv.addSpacing(10)
        lv.addWidget(QtWidgets.QLabel('<b>Queue</b>'))
        self.b_run_all = QtWidgets.QPushButton('Run all')
        self.b_run_all.setToolTip('Run every day that still has steps to do, one after the other (e.g. overnight)')
        self.b_run_all.setStyleSheet('font-weight:bold; padding:6px;')
        self.b_run_all.clicked.connect(self.run_all)
        self.b_stop = QtWidgets.QPushButton('Stop')
        self.b_stop.setEnabled(False)
        self.b_stop.clicked.connect(self.stop)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.b_run_all, 1)
        row.addWidget(self.b_stop)
        lv.addLayout(row)
        self.l_queue = QtWidgets.QLabel('Idle.')
        self.l_queue.setWordWrap(True)
        lv.addWidget(self.l_queue)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(True)
        lv.addWidget(self.bar)
        b_all = QtWidgets.QPushButton('Make the all-participants Excel')
        b_all.clicked.connect(self.export_all)
        lv.addWidget(b_all)
        split.addWidget(left)

        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        head = QtWidgets.QHBoxLayout()
        self.l_head = QtWidgets.QLabel()
        self.l_head.setTextFormat(Qt.RichText)
        self.l_head.setWordWrap(True)
        head.addWidget(self.l_head, 1)
        self.b_design = QtWidgets.QPushButton('Design && settings…')
        self.b_design.clicked.connect(self.edit_design)
        head.addWidget(self.b_design)
        rv.addLayout(head)
        self.day_tabs = QtWidgets.QTabWidget()
        self.days = {}
        for day in pl.DAYS:
            dv = DayView(self, day)
            dv.run_requested.connect(self.run_day)
            self.days[day] = dv
            self.day_tabs.addTab(dv, pl.DAY_NAMES[day])
        rv.addWidget(self.day_tabs, 1)
        self.empty = QtWidgets.QLabel('Add a participant with "New participant…" on the left.')
        self.empty.setAlignment(Qt.AlignCenter)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self.empty)
        self.stack.addWidget(right)
        split.addWidget(self.stack)
        split.setSizes([280, 1170])
        QtGui.QShortcut(QtGui.QKeySequence('Escape'), self, activated=self.stop)
        pl.recover_interrupted()
        self.refresh_list()
        if self.list.count():
            self.list.setCurrentRow(0)

    # ---------- participants ----------
    def pid(self):
        it = self.list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def refresh_list(self):
        cur = self.pid()
        self.list.blockSignals(True)
        self.list.clear()
        running = self.runner.current if self.running() else None
        for pid in pl.list_participants():
            marks = []
            for day in pl.DAYS:
                try:
                    s = pl.day_summary(pl.load_day(pid, day)) if pl.day_exists(pid, day) else 'empty'
                except (ValueError, OSError):
                    s = 'error'
                if running == (pid, day):
                    s = 'running'
                marks.append(f'{"D1" if day == "d1" else "D2"} {SUMMARY_MARK[s]}')
            it = QtWidgets.QListWidgetItem(f'{pid}     ' + '   '.join(marks))
            it.setData(Qt.UserRole, pid)
            self.list.addItem(it)
            if pid == cur:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)

    def on_select(self, *_):
        pid = self.pid()
        self.stack.setCurrentIndex(1 if pid else 0)
        if not pid:
            return
        try:
            p = pl.load_participant(pid)
        except Exception as e:
            self.l_head.setText(f'<span style="color:#b00">{e}</span>')
            return
        d = p['design']
        self.l_head.setText(
            f'<span style="font-size:16pt"><b>{pid}</b></span>&nbsp;&nbsp; tested arm <b>{d["tested_side"]}</b>'
            + (' + contralateral' if d['contralateral'] else '')
            + f', practice <b>{d["practice_limb"]} {d["practice_side"]}</b>'
            + (', familiarization with the tested arm' if d.get('familiarization') == 'test' else ''))
        for dv in self.days.values():
            dv.set_participant(pid)
            dv.set_busy(self.running() and self.runner.current == (pid, dv.day))
        if not pl.day_exists(pid, 'd1') or not pl.load_day(pid, 'd1').get('cam_a'):
            self.day_tabs.setCurrentIndex(0)

    def new_participant(self):
        dlg = ParticipantDialog(self)
        if not dlg.exec():
            return
        try:
            p = pl.new_participant(dlg.e_id.text(), dlg.design(), dlg.settings(), dlg.s_square.value())
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, 'New participant', str(e))
            return
        todo = []
        for day, (_, paths) in dlg.videos.items():
            if paths:
                pl.set_videos(p['id'], day, paths[0], paths[1] if len(paths) > 1 else '')
                todo.append((p['id'], day))
            else:
                pl.save_day(pl.new_day_state(p, day))
        self.refresh_list()
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.UserRole) == p['id']:
                self.list.setCurrentRow(i)
        if todo and dlg.k_start.isChecked():
            self.start(todo)

    def edit_design(self):
        pid = self.pid()
        if not pid:
            return
        if self.running() and self.runner.current and self.runner.current[0] == pid:
            QtWidgets.QMessageBox.information(self, 'Design', 'Wait until this participant\'s run ends (or Stop it).')
            return
        p = pl.load_participant(pid)
        dlg = ParticipantDialog(self, p)
        if dlg.exec():
            pl.update_participant(p, dlg.design(), dlg.settings(), dlg.s_square.value())
            self.on_select()
            self.refresh_list()

    # ---------- running ----------
    def running(self):
        return self.runner is not None and self.runner.isRunning()

    def run_day(self, pid, day):
        if self.running():
            # The runner's queue is the list it goes through: a day added at its end runs after the others
            items, cur = self.runner.items, self.runner.current
            later = items[items.index(cur) + 1:] if cur in items else items
            if (pid, day) not in later and (pid, day) != cur:
                items.append((pid, day))
            return
        self.start([(pid, day)])

    def run_all(self):
        items = pl.queue_items()
        if not items:
            QtWidgets.QMessageBox.information(self, 'Run all', 'Nothing to run: every day with videos is done, or '
                                              'waits for an edit (✗).')
            return
        self.start(items)

    def start(self, items):
        self.runner = Runner(items)
        self.runner.progress.connect(self.on_progress)
        self.runner.changed.connect(self.on_changed)
        self.runner.finished_all.connect(self.on_finished)
        self.b_run_all.setEnabled(False)
        self.b_stop.setEnabled(True)
        self.l_queue.setText(f'Running {len(items)} day(s)…')
        for dv in self.days.values():
            dv.b_run.setEnabled(False)
        self.runner.start()

    def stop(self):
        if self.running():
            self.runner.stop_requested = True
            self.l_queue.setText('Stopping after the current step part…')

    def on_progress(self, i, pid, day, step, f, text):
        n = len(self.runner.items) if self.runner else 1
        self.l_queue.setText(f'[{i + 1}/{n}]  {pid}  {pl.DAY_NAMES[day]}\n{pl.STEP_NAMES[step]} {f:.0%}  {text}')
        self.bar.setValue(int(f * 1000))
        self.bar.setFormat(f'{pl.STEP_NAMES[step]} {f:.0%}')
        if pid == self.pid():
            self.days[day].progress(step, f, text)

    def on_changed(self, pid, day):
        self.refresh_list()
        if pid == self.pid():
            dv = self.days[day]
            dv.reload()
            dv.set_busy(self.running() and self.runner.current == (pid, day))

    def on_finished(self, out):
        self.b_run_all.setEnabled(True)
        self.b_stop.setEnabled(False)
        self.bar.setValue(0)
        self.bar.setFormat('')
        marks = ', '.join(f'{p} {"D1" if d == "d1" else "D2"} {SUMMARY_MARK.get(s, "✗")}' for p, d, s in out)
        self.l_queue.setText(f'Finished: {marks}' if out else 'Stopped.')
        self.runner.wait()
        self.runner = None
        self.refresh_list()
        if self.pid():
            for dv in self.days.values():
                dv.set_busy(False)
                dv.reload()

    def export_all(self):
        try:
            path, n = sa.export_all()
        except PermissionError:
            QtWidgets.QMessageBox.warning(self, 'Export', 'The all-participants Excel is open in another program - '
                                          'close it and try again.')
            return
        QtWidgets.QMessageBox.information(self, 'Export', f'Saved ({n} participants):\n{path}')

    def closeEvent(self, e):
        if self.running():
            if QtWidgets.QMessageBox.question(self, 'Quit', 'A run is going on. Stop it and quit? (What is finished '
                                              'is kept; the current step starts over next time.)') \
                    != QtWidgets.QMessageBox.Yes:
                e.ignore()
                return
            self.runner.stop_requested = True
            self.runner.wait(30000)
        e.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName('Reaching Lab')
    w = LabApp()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
