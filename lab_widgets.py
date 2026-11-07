"""Widgets of the Reaching Lab app (lab_app.py): video frames with switch boxes, a background worker, and the
blocks and trials views of a day's analysis.
"""
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThread, Signal

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import experiment_design as ed  # noqa: E402
import extract_kinematics as ek  # noqa: E402
import trial_measures as tm  # noqa: E402
import video_source as vs  # noqa: E402

PART_CHOICES = [ed.IGNORE] + ed.PART_ORDER
# Background threads get a big stack: on macOS a QThread has 512 KB, too little for numpy / LAPACK (bus error)
THREAD_STACK = 64 * 1024 * 1024
FLAG_COLOR = {'miss': '#f8d7da', 'not detected': '#e2e3e5', 'unknown': '#e2e3e5',
              'anticipation': '#fff3cd', 'corrected': '#d1ecf1', 'shortcut': '#e8daef'}


# -----------------------------
# Helpers
# -----------------------------
def to_pixmap(bgr):
    """Turn a video frame (OpenCV image, BGR colors) into a picture Qt can show on screen.

    Input: bgr - the frame. Returns: a QPixmap.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    img = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(img.copy())


def read_frame(path, index=0):
    """A frame by its frame number (across GoPro chapters)."""
    return vs.read_frame(path, index)


def mmss(ms):
    """Milliseconds -> 'minutes:seconds' text for the screen, e.g. 83500 -> '1:23.50'.

    Input: ms - time in milliseconds (or None). Returns: the text ('' for None).
    """
    if ms is None:
        return ''
    s = ms / 1000
    return f'{int(s // 60)}:{s % 60:05.2f}'


def fmt(v, digits=3):
    """A value -> short text for a table cell.

    Inputs: v - the value; digits - decimals for numbers.
    Returns: '' for missing values (None / NaN), numbers with `digits` decimals, anything else as text.
    """
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ''
    if isinstance(v, (float, np.floating)):
        return f'{0.0 if abs(v) < 0.5 * 10 ** -digits else v:.{digits}f}'
    return str(v)


class Worker(QThread):
    """Runs fn(progress, should_stop) in the background."""
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        """Input: fn - the work to do; it is called as fn(progress, should_stop) when the thread starts."""
        super().__init__()
        self.fn = fn
        self.stop_requested = False
        self.setStackSize(THREAD_STACK)

    def run(self):
        """The background thread: run the work, then send `done` with its result, or `failed` with the error
        message (the app shows it to the user).
        """
        try:
            self.done.emit(self.fn(lambda f, text='': self.progress.emit(f, text), lambda: self.stop_requested))
        except Exception as e:  # shown to the user
            import traceback
            traceback.print_exc()
            self.failed.emit(str(e))


class BoxView(QtWidgets.QGraphicsView):
    """A video frame with the numbered switch boxes; can draw one new box by dragging."""
    box_drawn = Signal(list)

    def __init__(self):
        """Set up an empty picture area."""
        super().__init__()
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHint(QtGui.QPainter.Antialiasing)
        self.setMinimumSize(640, 360)
        self.pix = None
        self.size_wh = (1, 1)
        self.drawing = False
        self.start = None
        self.rubber = None

    def show_frame(self, bgr, boxes):
        """Show a frame with the switch boxes drawn on it (yellow, numbered 1, 2, 3 ...).

        Inputs: bgr - the frame (None = show nothing); boxes - switch boxes as [x0, y0, x1, y1] fractions of the frame.
        """
        self.scene().clear()
        self.rubber = None
        if bgr is None:
            return
        self.size_wh = (bgr.shape[1], bgr.shape[0])
        self.pix = self.scene().addPixmap(to_pixmap(bgr))
        w, h = self.size_wh
        pen = QtGui.QPen(QtGui.QColor('#ffd400'), max(2, w // 600))
        font = QtGui.QFont()
        font.setPixelSize(max(18, h // 25))
        font.setBold(True)
        for k, (x0, y0, x1, y1) in enumerate(boxes or []):
            self.scene().addRect(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h, pen)
            t = self.scene().addSimpleText(str(k + 1), font)
            t.setBrush(QtGui.QColor('#ffd400'))
            t.setPos(x0 * w, y0 * h - font.pixelSize() * 1.3)
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, e):
        """When the window changes size, scale the picture to fit."""
        super().resizeEvent(e)
        if self.scene().items():
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def start_drawing(self):
        """Get ready for the user to drag a new box (the mouse pointer becomes a cross)."""
        self.drawing = True
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, e):
        """Mouse button pressed while drawing: start the box at this point."""
        if self.drawing:
            self.start = self.mapToScene(e.position().toPoint())
            self.rubber = self.scene().addRect(QtCore.QRectF(self.start, self.start),
                                               QtGui.QPen(QtGui.QColor('#00e5ff'), 4))
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        """Mouse moved while drawing: stretch the box to the pointer."""
        if self.drawing and self.start is not None and self.rubber is not None:
            self.rubber.setRect(QtCore.QRectF(self.start, self.mapToScene(e.position().toPoint())).normalized())
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        """Mouse button released while drawing: finish the box and send it (box_drawn) as [x0, y0, x1, y1]
        fractions of the frame. Very small boxes (a click without dragging) are ignored.
        """
        if self.drawing and self.start is not None:
            r = QtCore.QRectF(self.start, self.mapToScene(e.position().toPoint())).normalized()
            self.drawing, self.start = False, None
            self.setCursor(Qt.ArrowCursor)
            w, h = self.size_wh
            if r.width() > 5 and r.height() > 5:
                self.box_drawn.emit([round(max(0, r.left()) / w, 4), round(max(0, r.top()) / h, 4),
                                     round(min(w, r.right()) / w, 4), round(min(h, r.bottom()) / h, 4)])
        super().mouseReleaseEvent(e)


def group(title, layout):
    """A framed box with a title around some controls on a page.

    Inputs: title - the title shown on the frame; layout - the controls inside. Returns: the QGroupBox.
    """
    g = QtWidgets.QGroupBox(title)
    g.setLayout(layout)
    return g


def path_row(edit, on_browse, extra=()):
    """A row with a file path text field and a "Browse…" button (plus optional extra buttons).

    Inputs: edit - the text field; on_browse - what to do when Browse is clicked; extra - more widgets for the row.
    Returns: the row (a layout).
    """
    row = QtWidgets.QHBoxLayout()
    row.addWidget(edit, 1)
    b = QtWidgets.QPushButton('Browse…')
    b.clicked.connect(on_browse)
    row.addWidget(b)
    for w in extra:
        row.addWidget(w)
    return row


def status_html(step_state):
    """A step's message and details as rich text (lines starting with '!' in red)."""
    import html
    lines = [f'<b>{html.escape(step_state.get("message") or "")}</b>']
    for d in step_state.get('details') or []:
        d = html.escape(str(d))
        if d.lstrip().startswith('!'):
            lines.append(f'<span style="color:#b00">{d.lstrip()[1:].strip()}</span>')
        else:
            lines.append(d.replace('  ', '&nbsp;&nbsp;'))
    return '<br>'.join(lines)


# -----------------------------
# Blocks & parts
# -----------------------------
class BlocksView(QtWidgets.QWidget):
    """The groups of lights found (blocks): when each starts and ends, trials found, and the part and block
    number - which the user can change. Sends `parts_changed` with the new list ([part, block no] per block,
    or None for "back to the session plan")."""
    parts_changed = Signal(object)

    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.RichText)
        v.addWidget(self.label)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Starts at', 'Trials found', 'Ends at', 'Part', 'Block # in part'])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        v.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        self.b_apply = QtWidgets.QPushButton('Apply these parts')
        self.b_apply.clicked.connect(self.apply)
        self.b_reset = QtWidgets.QPushButton('Back to the session plan')
        self.b_reset.clicked.connect(lambda: self.parts_changed.emit(None))
        row.addWidget(self.b_apply)
        row.addWidget(self.b_reset)
        row.addStretch()
        v.addLayout(row)

    def set_session(self, session):
        """Fill the table from an analysed session (None = empty)."""
        self.table.setRowCount(0)
        if session is None or session.result is None:
            self.label.setText('Not analysed yet.')
            return
        res, times, ev = session.result, session.times, session.result['events']
        self.label.setText(
            f"<b>{len(res['blocks'])} groups of lights</b>, switch order in a sequence: "
            f"<b>{', '.join(map(str, res['pattern'])) or '?'}</b>. Change a block's part if it was put in the wrong "
            'one (e.g. a block that was stopped and restarted), then Apply.')
        self.table.setRowCount(len(res['blocks']))
        for i, (b, (part, no)) in enumerate(zip(res['blocks'], res['parts'])):
            on, off = ev[b[0]][1], ev[b[-1]][2] if ev[b[-1]][2] is not None else ev[b[-1]][1]
            rows = [r for r in res['trials'] if r['block_index'] == i and r['on_frame'] is not None]
            found = str(len(rows)) if part != ed.IGNORE else f'{len(b)} lights'
            if rows and rows[-1]['off_frame'] is not None:
                off = max(off, rows[-1]['off_frame'])
            n_timing = sum(1 for r in rows if r.get('Light_source') == 'timing')
            if n_timing:
                found += f'  ({n_timing} from timing)'
            for c, text in enumerate([mmss(times[on]), found, mmss(times[off])]):
                it = QtWidgets.QTableWidgetItem(text)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, c, it)
            combo = QtWidgets.QComboBox()
            combo.addItems(PART_CHOICES)
            combo.setCurrentText(part)
            self.table.setCellWidget(i, 3, combo)
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 50)
            spin.setValue(no)
            self.table.setCellWidget(i, 4, spin)

    def apply(self):
        """Send the parts chosen in the table."""
        n = self.table.rowCount()
        if n:
            self.parts_changed.emit([[self.table.cellWidget(i, 3).currentText(), self.table.cellWidget(i, 4).value()]
                                     for i in range(n)])

    def set_editable(self, on):
        self.b_apply.setEnabled(on)
        self.b_reset.setEnabled(on)


# -----------------------------
# Trials
# -----------------------------
class TrialsView(QtWidgets.QWidget):
    """All trials of a day (filter by part / block / sequence, or only the flagged ones), and the selected trial's
    graph and video frames. Sends `touse_changed` when a trial is ticked in or out of the summaries."""
    touse_changed = Signal()

    def __init__(self):
        super().__init__()
        self.session = None
        self.shown_rows = []
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        left = QtWidgets.QVBoxLayout()
        flt = QtWidgets.QHBoxLayout()
        self.f_part, self.f_block, self.f_seq = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for lab, c in (('Part', self.f_part), ('Block', self.f_block), ('Sequence', self.f_seq)):
            flt.addWidget(QtWidgets.QLabel(lab))
            flt.addWidget(c)
            c.currentIndexChanged.connect(self.fill)
        self.k_flagged = QtWidgets.QCheckBox('only marked trials')
        self.k_flagged.toggled.connect(self.fill)
        flt.addWidget(self.k_flagged)
        flt.addStretch()
        left.addLayout(flt)
        cols = ['Part', 'Blk', 'Seq', 'Trial', 'Target', 'Outcome', 'Light', 'Anticip.', 'Start', 'Corrected',
                'RT (s)', 'MT (s)', 'Data', 'touse']
        self.table = QtWidgets.QTableWidget(0, len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_trial)
        self.table.itemChanged.connect(self.on_touse)
        left.addWidget(self.table, 1)
        legend = QtWidgets.QLabel(
            '<span style="background:#f8d7da">&nbsp;miss&nbsp;</span> '
            '<span style="background:#fff3cd">&nbsp;started before the light (negative RT)&nbsp;</span> '
            '<span style="background:#e8daef">&nbsp;not from home (shortcut)&nbsp;</span> '
            '<span style="background:#d1ecf1">&nbsp;corrected (went to another switch first)&nbsp;</span> '
            '<span style="background:#e2e3e5">&nbsp;light not detected&nbsp;</span> - untick touse to leave a '
            'trial out of the summaries')
        legend.setTextFormat(Qt.RichText)
        legend.setWordWrap(True)
        left.addWidget(legend)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)

        right = QtWidgets.QVBoxLayout()
        g = QtWidgets.QHBoxLayout()
        self.c_graph = QtWidgets.QComboBox()
        self.c_graph.addItems(['speed', 'x', 'y', 'z', 'path'])
        self.c_graph.currentIndexChanged.connect(self.show_trial)
        b_win = QtWidgets.QPushButton('Open graph in a new window')
        b_win.clicked.connect(self.graph_window)
        g.addWidget(QtWidgets.QLabel('Graph'))
        g.addWidget(self.c_graph)
        g.addWidget(b_win)
        g.addStretch()
        right.addLayout(g)
        self.fig = Figure(figsize=(6, 3.2), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        right.addWidget(self.canvas, 2)
        fr = QtWidgets.QHBoxLayout()
        fr.addWidget(QtWidgets.QLabel('Video frame at:'))
        for key in ('light on', 'movement onset', 'movement end', 'trial end'):
            b = QtWidgets.QPushButton(key)
            b.clicked.connect(lambda _, k=key: self.show_video_frame(k))
            fr.addWidget(b)
        fr.addStretch()
        right.addLayout(fr)
        self.l_frame = QtWidgets.QLabel()
        self.l_frame.setAlignment(Qt.AlignCenter)
        self.l_frame.setMinimumHeight(240)
        right.addWidget(self.l_frame, 2)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)
        split = QtWidgets.QSplitter()
        split.addWidget(lw)
        split.addWidget(rw)
        split.setSizes([700, 600])
        h.addWidget(split)

    def set_session(self, session):
        """Show an analysed session's trials (None = empty)."""
        self.session = session
        if session is None or session.result is None:
            self.shown_rows = []
            self.table.setRowCount(0)
            self.fig.clear()
            self.canvas.draw_idle()
            self.l_frame.clear()
            return
        trials = session.result['trials']
        for c, key in ((self.f_part, 'Part'), (self.f_block, 'Block'), (self.f_seq, 'Sequence')):
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            vals = list(dict.fromkeys(str(r[key]) for r in trials))
            c.addItems(['All'] + vals)
            c.setCurrentText(cur if cur in vals else 'All')
            c.blockSignals(False)
        self.fill()

    def fill(self, *_):
        """Fill the table with the trials that match the filters, rows colored by flag."""
        if self.session is None or self.session.result is None:
            return
        sel = lambda c: None if c.currentText() in ('', 'All') else c.currentText()
        p, b, s = sel(self.f_part), sel(self.f_block), sel(self.f_seq)
        rows = [r for r in self.session.result['trials']
                if (p is None or r['Part'] == p) and (b is None or str(r['Block']) == b)
                and (s is None or str(r['Sequence']) == s)]
        if self.k_flagged.isChecked():
            rows = [r for r in rows if r['Outcome'] != 'hit' or r['anticipation'] or r['corrected_from']
                    or r.get('start_from') not in ('', 'home', None)]
        self.shown_rows = rows
        t = self.table
        t.blockSignals(True)
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [r['Part'], r['Block'], r['Sequence'], r['Trial'], r['Target'], r['Outcome'],
                    r.get('Light_source', ''),
                    f"yes ({r['anticipation_ms']:.0f} ms)" if r['anticipation'] else '',
                    r.get('start_from', '') if r.get('start_from') != 'home' else 'home',
                    f"from {r['corrected_from']}" if r['corrected_from'] else '',
                    fmt(r['responsetime']), fmt(r['movementtime']), r.get('Data', '')]
            shortcut = r.get('start_from') not in ('', 'home', None)
            color = (FLAG_COLOR.get(r['Outcome']) or (FLAG_COLOR['shortcut'] if shortcut else None)
                     or (FLAG_COLOR['anticipation'] if r['anticipation'] else None)
                     or (FLAG_COLOR['corrected'] if r['corrected_from'] else None))
            for c, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(val))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if color:
                    it.setBackground(QtGui.QColor(color))
                t.setItem(i, c, it)
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if r['touse'] else Qt.Unchecked)
            t.setItem(i, len(vals), it)
        t.blockSignals(False)

    def on_touse(self, item):
        """A touse box was ticked / unticked: save that choice for the trial."""
        if item.column() != self.table.columnCount() - 1 or item.row() >= len(self.shown_rows):
            return
        self.session.set_touse(self.shown_rows[item.row()], item.checkState() == Qt.Checked)
        self.touse_changed.emit()

    def current_trial(self):
        rows = self.table.selectionModel().selectedRows()
        return self.shown_rows[rows[0].row()] if rows and rows[0].row() < len(self.shown_rows) else None

    def plot_trial(self, fig, r):
        """One trial's graph: speed / x / y / z over time (light on, movement onset / end, hit marked), or the
        path seen from above with the switches."""
        fig.clear()
        ax = fig.add_subplot(111)
        if r is None:
            return
        k, m = self.session.trial_curves(r)
        if k is None:
            ax.text(0.5, 0.5, 'no movement data for this trial', ha='center', transform=ax.transAxes)
            return
        t = k['time'] - k['stim']
        rel = lambda x: x - k['stim'] if x is not None and not np.isnan(x) else None
        kind = self.c_graph.currentText()
        on_i = end_i = None
        if m and not np.isnan(m.get('onset_s', np.nan)):
            on_i, end_i = np.searchsorted(k['time'], m['onset_s']), np.searchsorted(k['time'], m['end_s'])
        if kind == 'path':
            xyz = k['xyz']
            ax.plot(xyz[:, 0], xyz[:, 1], '.', color='0.6', ms=3)
            if on_i is not None:
                ax.plot(xyz[on_i:end_i + 1, 0], xyz[on_i:end_i + 1, 1], '-', lw=2.5, color='C0', label='movement')
            sp = self.session.trial_switch_pos(r)
            ax.scatter(sp[:, 0], sp[:, 1], marker='s', s=60, color='0.3')
            for i, p in enumerate(sp):
                ax.annotate(str(i + 1), p[:2], color='C3' if i + 1 == r['Target'] else '0.3', fontsize=12,
                            fontweight='bold', xytext=(6, 6), textcoords='offset points')
            if xyz.shape[1] == 2:
                ax.invert_yaxis()
            ax.set_aspect('equal', 'datalim')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
        else:
            d = 'xyz'.index(kind) if kind in 'xyz' else None
            if d is not None and d >= k['xyz'].shape[1]:
                ax.text(0.5, 0.5, 'z needs two calibrated cameras', ha='center', transform=ax.transAxes)
                return
            y = k['tangvel'] if kind == 'speed' else k['xyz'][:, d]
            ax.plot(t, y, '.', color='0.5', ms=3)
            if on_i is not None:
                ax.plot(t[on_i:end_i + 1], y[on_i:end_i + 1], '-', lw=2.5, color='C0')
            ax.axvline(0, color='k', ls='--', label='light on')
            if m and rel(m.get('onset_s')) is not None:
                ax.axvline(rel(m['onset_s']), color='r', ls='--', label='movement onset')
                ax.axvline(rel(m['end_s']), color='b', ls='--', label='movement end')
            end_ms = r.get('Trial_end_ms')
            if end_ms is not None:
                ax.axvline(end_ms / 1000 - k['stim'], color='g', ls='-',
                           label='hit' if r['Outcome'] == 'hit' else 'light off')
            ax.set_xlabel('time from light on (s)')
            ax.set_ylabel('speed' if kind == 'speed' else kind)
            ax.legend(loc='upper left', fontsize=8)
        ax.set_title(f"{r['Part']}  block {r['Block']}  seq {r['Sequence']}  trial {r['Trial']}  → switch {r['Target']}"
                     f"   RT {fmt(r['responsetime'])} s  MT {fmt(r['movementtime'])} s", fontsize=9)
        if not r['touse']:
            ax.set_facecolor('0.85')

    def show_trial(self):
        r = self.current_trial()
        self.plot_trial(self.fig, r)
        self.canvas.draw_idle()
        if r is not None:
            self.show_video_frame('movement onset' if r.get('onset_s') else 'light on')

    def graph_window(self):
        r = self.current_trial()
        if r is None:
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('Trial graph')
        dlg.resize(1000, 600)
        fig = Figure(tight_layout=True)
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.addWidget(FigureCanvasQTAgg(fig))
        self.plot_trial(fig, r)
        dlg.show()

    def show_video_frame(self, which):
        """The selected trial's video frame at a moment, with the marker (red cross) and the target switch's box."""
        r = self.current_trial()
        s = self.session
        if r is None or r.get('on_frame') is None:
            self.l_frame.clear()
            return
        times = s.times
        if which == 'light on':
            f = r['on_frame']
        elif which == 'trial end':
            f = r['off_frame'] if r['off_frame'] is not None else r['on_frame']
        else:
            t = r.get('onset_s' if which == 'movement onset' else 'end_s')
            if t is None or np.isnan(t):
                return
            f = int(np.clip(np.searchsorted(times, t * 1000), 0, len(times) - 1))
        prefer = {'camera A': 'a', 'camera B': 'b'}.get(r.get('Data'), 'a')
        cam, cf = s.frame_ref(f, prefer)
        if cam is None:
            return
        frame = read_frame(s.state[f'cam_{cam}'], cf)
        if frame is None:
            return
        cache = s.cams[cam]
        j = tm.MARKER[r['Limb']][r['Side']]
        p = cache['pose'][cf, j]
        sc = max(frame.shape[:2]) / 1280
        if not np.isnan(p[0]):
            cv2.drawMarker(frame, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, int(40 * sc), int(4 * sc))
        w, h = cache['meta']['size']
        b = s.state[f'boxes_{cam}'][r['Target'] - 1]
        c = ek.to_frame(ek.box_corners(b, w, h, 0.0), cache['motion'][cf])
        cv2.polylines(frame, [c.astype(np.int32)], True, (0, 212, 255), int(3 * sc))
        cv2.putText(frame, f'camera {cam.upper()}: {which}  {mmss(times[f])}', (int(20 * sc), int(50 * sc)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * sc, (255, 255, 255), int(3 * sc))
        self.l_frame.setPixmap(to_pixmap(frame).scaled(self.l_frame.size(), Qt.KeepAspectRatio,
                                                       Qt.SmoothTransformation))
