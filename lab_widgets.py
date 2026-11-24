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

    def show_frame(self, bgr, boxes, area=None):
        """Show a frame with the switch boxes drawn on it (yellow, numbered 1, 2, 3 ...).

        Inputs: bgr - the frame (None = show nothing); boxes - switch boxes as [x0, y0, x1, y1] fractions of the frame;
        area - the participant area (same form, drawn dashed in cyan), or None.
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
        if area:
            x0, y0, x1, y1 = area
            apen = QtGui.QPen(QtGui.QColor('#00e5ff'), max(3, w // 400), Qt.DashLine)
            self.scene().addRect(x0 * w, y0 * h, (x1 - x0) * w, (y1 - y0) * h, apen)
            t = self.scene().addSimpleText('participant area', font)
            t.setBrush(QtGui.QColor('#00e5ff'))
            t.setPos(x0 * w + 8, y0 * h + 4)
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
def plot_trial(fig, session, r, kind='speed'):
    """One trial's graph: speed / x / y / z over time (light on, movement onset / end, hit marked), or the
    path seen from above with the switches."""
    fig.clear()
    ax = fig.add_subplot(111)
    if r is None:
        return
    k, m = session.trial_curves(r)
    if k is None:
        ax.text(0.5, 0.5, 'no movement data for this trial', ha='center', transform=ax.transAxes)
        return
    t = k['time'] - k['stim']
    rel = lambda x: x - k['stim'] if x is not None and not np.isnan(x) else None
    on_i = end_i = None
    if m and not np.isnan(m.get('onset_s', np.nan)):
        on_i, end_i = np.searchsorted(k['time'], m['onset_s']), np.searchsorted(k['time'], m['end_s'])
    if kind == 'path':
        xyz = k['xyz']
        ax.plot(xyz[:, 0], xyz[:, 1], '.', color='0.6', ms=3)
        if on_i is not None:
            ax.plot(xyz[on_i:end_i + 1, 0], xyz[on_i:end_i + 1, 1], '-', lw=2.5, color='C0', label='movement')
        sp = session.trial_switch_pos(r)
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
        """One trial's graph (see plot_trial), of the kind chosen in the Graph box."""
        plot_trial(fig, self.session, r, self.c_graph.currentText())

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


# -----------------------------
# Data cleaning
# -----------------------------
REC_COLOR = {'exclude': '#f8d7da', 'check': '#fff3cd'}
CLIP_BEFORE_S = 1.0      # the trial's video starts this long before its light (the home phase)
CLIP_AFTER_S = 0.6       # ... and ends this long after the light went off
CLIP_W = 1280            # width of the shown clip (both cameras side by side)


def default_touse(r):
    """Whether a trial is used when the user has not chosen: a hit with movement data."""
    return r['Outcome'] == 'hit' and bool(r.get('Data'))


def load_trial_clip(session, r, should_stop=lambda: False):
    """One trial's video, both cameras side by side with the skeleton, the switch boxes (target in yellow), the
    tracked wrist / ankle (red cross), and the time from the light. Returns (frames (BGR), session frame numbers)."""
    import make_preview as mp
    s = session
    fps = s.fps
    end = r['off_frame'] if r.get('off_frame') is not None else r['on_frame'] + int(fps)
    a = max(0, r['on_frame'] - int(CLIP_BEFORE_S * fps))
    b = min(len(s.times) - 1, end + int(CLIP_AFTER_S * fps))
    cams = [c for c in ('a', 'b') if c in s.cams]
    readers = {c: mp.Reader(s.state[f'cam_{c}']) for c in cams}
    half = CLIP_W // len(cams)
    tile_h = max(int(half * s.cams[c]['meta']['size'][1] / s.cams[c]['meta']['size'][0]) for c in cams)
    onset = int(np.searchsorted(s.times, r['onset_s'] * 1000)) if r.get('onset_s') and not np.isnan(r['onset_s']) else None
    m_end = int(np.searchsorted(s.times, r['end_s'] * 1000)) if r.get('end_s') and not np.isnan(r['end_s']) else None
    frames, nums = [], []
    for f in range(a, b + 1):
        if should_stop():
            return None
        canvas = np.zeros((tile_h + 44, CLIP_W, 3), np.uint8)
        for k, c in enumerate(cams):
            cf = int(s.idx[c][f])
            img = readers[c].get(cf) if cf >= 0 else None
            if img is None:
                continue
            img = mp.draw_camera(img.copy(), s.cams[c], cf, s.state[f'boxes_{c}'], r, f'camera {c.upper()}')
            w, h = s.cams[c]['meta']['size']
            small = cv2.resize(img, (half, int(half * h / w)), interpolation=cv2.INTER_AREA)
            canvas[:small.shape[0], k * half:k * half + half] = small
        t = (s.times[f] - s.times[r['on_frame']]) / 1000
        lit = r['on_frame'] <= f <= end
        label = f'{t:+.2f} s from the light'
        if onset is not None and abs(f - onset) <= 1:
            label += '   MOVEMENT ONSET'
        if m_end is not None and abs(f - m_end) <= 1:
            label += '   MOVEMENT END'
        if r.get('off_frame') is not None and f == r['off_frame']:
            label += '   LIGHT OFF' + (' (press)' if r['Outcome'] == 'hit' else '')
        cv2.putText(canvas, label, (12, tile_h + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if lit:
            cv2.circle(canvas, (CLIP_W - 30, tile_h + 22), 13, (0, 230, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, 'light on', (CLIP_W - 140, tile_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 230, 255), 2, cv2.LINE_AA)
        frames.append(canvas)
        nums.append(f)
    return frames, nums


class CleaningView(QtWidgets.QWidget):
    """Data cleaning: every trial (part / block / sequence / trial), its data and its video, and whether it is used
    in the statistics (touse). Trials are marked with a recommendation (exclude / check) and its reasons - nothing
    is left out automatically. Keys: up / down = trial, I = include, E = exclude, space = play / pause."""
    touse_changed = Signal()
    COLS = ['#', 'Part', 'Block', 'Seq', 'Trial', 'Switch', 'Outcome', 'Clean', 'Recommendation', 'RT (s)',
            'RT Jason (s)', 'MT (s)', 'Data', 'Use']
    SHOW = ['All trials', 'Recommended: exclude', 'Recommended: check', 'Recommended: exclude or check',
            'Not clean', 'Excluded (Use off)', 'Used but recommended to exclude']

    def __init__(self):
        super().__init__()
        self.session = None
        self.rows = []
        self.frames, self.frame_nums, self.pos = [], [], 0
        self.loader = None
        self.clip_of = None
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)

        # left: the list
        left = QtWidgets.QVBoxLayout()
        flt = QtWidgets.QHBoxLayout()
        self.f_part, self.f_block, self.f_show = QtWidgets.QComboBox(), QtWidgets.QComboBox(), QtWidgets.QComboBox()
        self.f_show.addItems(self.SHOW)
        for lab, c in (('Part', self.f_part), ('Block', self.f_block), ('Show', self.f_show)):
            flt.addWidget(QtWidgets.QLabel(lab))
            flt.addWidget(c)
            c.currentIndexChanged.connect(self.fill)
        flt.addStretch()
        left.addLayout(flt)
        self.l_counts = QtWidgets.QLabel()
        left.addWidget(self.l_counts)
        self.table = QtWidgets.QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self.show_trial)
        self.table.itemChanged.connect(self.on_use_box)
        left.addWidget(self.table, 1)
        bb = QtWidgets.QHBoxLayout()
        b_rec = QtWidgets.QPushButton('Exclude all recommended "exclude"')
        b_rec.clicked.connect(self.exclude_recommended)
        b_reset = QtWidgets.QPushButton('Undo all my choices')
        b_reset.clicked.connect(self.reset_choices)
        bb.addWidget(b_rec)
        bb.addWidget(b_reset)
        bb.addStretch()
        left.addLayout(bb)
        legend = QtWidgets.QLabel(
            f'<span style="background:{REC_COLOR["exclude"]}">&nbsp;recommended: exclude&nbsp;</span> '
            f'<span style="background:{REC_COLOR["check"]}">&nbsp;recommended: check the video&nbsp;</span> '
            '- the reasons are shown with the trial. Nothing is left out unless you untick <b>Use</b>.<br>'
            'Keys: <b>↑ ↓</b> trial, <b>I</b> include, <b>E</b> exclude, <b>space</b> play / pause.')
        legend.setTextFormat(Qt.RichText)
        legend.setWordWrap(True)
        left.addWidget(legend)
        lw = QtWidgets.QWidget()
        lw.setLayout(left)

        # right: the selected trial
        right = QtWidgets.QVBoxLayout()
        self.l_title = QtWidgets.QLabel()
        self.l_title.setStyleSheet('font-size:15px; font-weight:bold;')
        right.addWidget(self.l_title)
        self.l_reasons = QtWidgets.QLabel()
        self.l_reasons.setWordWrap(True)
        right.addWidget(self.l_reasons)
        ub = QtWidgets.QHBoxLayout()
        self.b_include = QtWidgets.QPushButton('Include (I)')
        self.b_exclude = QtWidgets.QPushButton('Exclude (E)')
        self.b_include.clicked.connect(lambda: self.set_current(True))
        self.b_exclude.clicked.connect(lambda: self.set_current(False))
        self.l_use = QtWidgets.QLabel()
        ub.addWidget(self.b_include)
        ub.addWidget(self.b_exclude)
        ub.addWidget(self.l_use)
        ub.addStretch()
        right.addLayout(ub)
        self.l_video = QtWidgets.QLabel('Select a trial')
        self.l_video.setAlignment(Qt.AlignCenter)
        self.l_video.setMinimumHeight(300)
        self.l_video.setStyleSheet('background:#111; color:#bbb;')
        right.addWidget(self.l_video, 3)
        pc = QtWidgets.QHBoxLayout()
        self.b_play = QtWidgets.QPushButton('Play')
        self.b_play.clicked.connect(self.toggle_play)
        self.c_speed = QtWidgets.QComboBox()
        self.c_speed.addItems(['0.25x', '0.5x', '1x'])
        self.c_speed.setCurrentText('0.5x')
        self.c_speed.currentIndexChanged.connect(self.set_speed)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.seek)
        pc.addWidget(self.b_play)
        pc.addWidget(QtWidgets.QLabel('Speed'))
        pc.addWidget(self.c_speed)
        pc.addWidget(self.slider, 1)
        right.addLayout(pc)
        low = QtWidgets.QHBoxLayout()
        self.data = QtWidgets.QTableWidget(0, 2)
        self.data.setHorizontalHeaderLabels(['Measure', 'Value'])
        self.data.verticalHeader().setVisible(False)
        self.data.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.data.horizontalHeader().setStretchLastSection(True)
        self.data.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        low.addWidget(self.data, 1)
        self.fig = Figure(figsize=(4, 2.6), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        low.addWidget(self.canvas, 1)
        right.addLayout(low, 2)
        rw = QtWidgets.QWidget()
        rw.setLayout(right)
        split = QtWidgets.QSplitter()
        split.addWidget(lw)
        split.addWidget(rw)
        split.setSizes([620, 760])
        h.addWidget(split)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.set_speed()
        for key, fn in (('I', lambda: self.set_current(True)), ('E', lambda: self.set_current(False)),
                        ('Space', self.toggle_play)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ----- list -----
    def set_session(self, session):
        """Show an analysed session's trials (None = empty)."""
        self.stop_clip()
        self.session = session
        if session is None or session.result is None:
            self.rows = []
            self.table.setRowCount(0)
            self.l_counts.clear()
            self.clear_trial()
            return
        trials = session.result['trials']
        for c, key in ((self.f_part, 'Part'), (self.f_block, 'Block')):
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            vals = list(dict.fromkeys(str(r[key]) for r in trials))
            c.addItems(['All'] + vals)
            c.setCurrentText(cur if cur in vals else 'All')
            c.blockSignals(False)
        self.fill()

    def fill(self, *_):
        """The trials that match the filters, colored by recommendation."""
        if self.session is None or self.session.result is None:
            return
        trials = self.session.result['trials']
        sel = lambda c: None if c.currentText() in ('', 'All') else c.currentText()
        p, b = sel(self.f_part), sel(self.f_block)
        show = self.f_show.currentText()
        keep = {'All trials': lambda r: True,
                'Recommended: exclude': lambda r: r.get('Recommendation') == 'exclude',
                'Recommended: check': lambda r: r.get('Recommendation') == 'check',
                'Recommended: exclude or check': lambda r: bool(r.get('Recommendation')),
                'Not clean': lambda r: not r.get('Clean'),
                'Excluded (Use off)': lambda r: not r['touse'],
                'Used but recommended to exclude': lambda r: r['touse'] and r.get('Recommendation') == 'exclude'}[show]
        self.rows = [(i + 1, r) for i, r in enumerate(trials)
                     if (p is None or r['Part'] == p) and (b is None or str(r['Block']) == b) and keep(r)]
        t = self.table
        cur = self.current_trial()
        t.blockSignals(True)
        t.setRowCount(len(self.rows))
        for i, (n, r) in enumerate(self.rows):
            vals = [n, r['Part'], r['Block'], r['Sequence'], r['Trial'], r['Target'], r['Outcome'],
                    'yes' if r.get('Clean') else 'no', r.get('Recommendation', ''), fmt(r.get('responsetime')),
                    fmt(r.get('responsetime_jason')), fmt(r.get('movementtime')), r.get('Data', '')]
            color = REC_COLOR.get(r.get('Recommendation'))
            for c, val in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(str(val))
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if color:
                    it.setBackground(QtGui.QColor(color))
                if r.get('Reasons'):
                    it.setToolTip(r['Reasons'])
                t.setItem(i, c, it)
            it = QtWidgets.QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if r['touse'] else Qt.Unchecked)
            t.setItem(i, len(vals), it)
        t.blockSignals(False)
        self.update_counts()
        if cur is not None:
            for i, (_, r) in enumerate(self.rows):
                if r is cur:
                    t.selectRow(i)
                    break

    def update_counts(self):
        trials = self.session.result['trials']
        n = len(trials)
        used = sum(1 for r in trials if r['touse'])
        ex = sum(1 for r in trials if r.get('Recommendation') == 'exclude')
        ch = sum(1 for r in trials if r.get('Recommendation') == 'check')
        clean = sum(1 for r in trials if r.get('Clean'))
        self.l_counts.setText(f'{n} trials: <b>{used} used</b>, {n - used} excluded  |  {clean} clean  |  '
                              f'recommended: <span style="background:{REC_COLOR["exclude"]}">&nbsp;exclude {ex}&nbsp;'
                              f'</span> <span style="background:{REC_COLOR["check"]}">&nbsp;check {ch}&nbsp;</span>')

    def current_trial(self):
        sel = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.rows[sel[0].row()][1] if sel and sel[0].row() < len(self.rows) else None

    # ----- include / exclude -----
    def on_use_box(self, item):
        if item.column() != len(self.COLS) - 1 or item.row() >= len(self.rows):
            return
        self.set_use(self.rows[item.row()][1], item.checkState() == Qt.Checked)

    def set_use(self, r, value, refresh=True):
        if r is None or self.session is None or bool(r['touse']) == bool(value):
            return
        self.session.set_touse(r, value)
        if refresh:
            self.fill()
            self.show_use(r)
            self.touse_changed.emit()

    def set_current(self, value):
        r = self.current_trial()
        self.set_use(r, value)

    def exclude_recommended(self):
        if self.session is None:
            return
        todo = [r for r in self.session.result['trials'] if r['touse'] and r.get('Recommendation') == 'exclude']
        if not todo:
            return
        for r in todo:
            self.set_use(r, False, refresh=False)
        self.fill()
        self.show_use(self.current_trial())
        self.touse_changed.emit()

    def reset_choices(self):
        """Forget the user's include / exclude choices: every trial back to the default (hits with data used)."""
        if self.session is None:
            return
        if QtWidgets.QMessageBox.question(self, 'Undo all choices',
                                          'Forget all include / exclude choices of this day?') != \
                QtWidgets.QMessageBox.Yes:
            return
        self.session.state['touse_override'] = {}
        for r in self.session.result['trials']:
            r['touse'] = default_touse(r)
        import session_analysis as sa
        sa.save_state(self.session.state)
        self.fill()
        self.show_use(self.current_trial())
        self.touse_changed.emit()

    # ----- the selected trial -----
    def clear_trial(self):
        self.l_title.clear()
        self.l_reasons.clear()
        self.l_use.clear()
        self.data.setRowCount(0)
        self.fig.clear()
        self.canvas.draw_idle()
        self.l_video.clear()
        self.l_video.setText('Select a trial')

    def show_use(self, r):
        if r is None:
            return
        self.l_use.setText('<b style="color:#198754">used in the statistics</b>' if r['touse'] else
                           '<b style="color:#dc3545">excluded from the statistics</b>')

    def show_trial(self):
        r = self.current_trial()
        if r is None:
            return
        n = next((n for n, x in self.rows if x is r), '')
        self.l_title.setText(f"#{n}   {r['Part']}  ·  block {r['Block']}  ·  sequence {r['Sequence']}  ·  "
                             f"trial {r['Trial']}   →  switch {r['Target']}")
        rec = r.get('Recommendation')
        if rec:
            self.l_reasons.setText(f'<span style="background:{REC_COLOR[rec]}">&nbsp;<b>Recommended: {rec}</b>'
                                   f'&nbsp;</span> {r.get("Reasons", "")}')
        else:
            self.l_reasons.setText('No problem found' + (' - a clean trial.' if r.get('Clean') else '.'))
        self.show_use(r)
        self.fill_data(r)
        plot_trial(self.fig, self.session, r, 'speed')
        self.canvas.draw_idle()
        self.load_clip(r)

    def fill_data(self, r):
        ms = lambda v: '' if v is None else mmss(v)
        start = r.get('start_from', '')
        if start and r.get('start_dist_pct') is not None:
            start += f" ({r['start_dist_pct']}% of the way from home)"
        items = [('Outcome', r['Outcome']), ('Clean', 'yes' if r.get('Clean') else 'no'),
                 ('Used in statistics', 'yes' if r['touse'] else 'no'),
                 ('Light on (video time)', ms(r.get('Light_on_ms'))), ('Light found from', r.get('Light_source', '')),
                 ('Data from', r.get('Data', '')),
                 ('System: light on → press (ms)', fmt(r.get('System_ms'), 0)),
                 ('RT (s)', fmt(r.get('responsetime'))), ('RT, Jason method (s)', fmt(r.get('responsetime_jason'))),
                 ('MT (s)', fmt(r.get('movementtime'))), ('MT, Jason method (s)', fmt(r.get('movementtime_jason'))),
                 ('AT (s)', fmt(r.get('AT'))), ('DT (s)', fmt(r.get('DT'))), ('AT / DT', fmt(r.get('AT_over_DT'), 2)),
                 ('Speed peaks', fmt(r.get('numpeaks'), 0)),
                 ('Straight-line deviation', fmt(r.get('straighlinedeviation'), 2)),
                 ('End-point distance', fmt(r.get('endpointdistance'), 2)),
                 ('Started before the light', f"{r['anticipation_ms']:.0f} ms" if r.get('anticipation') else 'no'),
                 ('Started from', start),
                 ('Corrected (went first to)', f"switch {r['corrected_from']}" if r.get('corrected_from') else 'no'),
                 ('On the switch → light off (s)', fmt(r.get('touch_to_hit_s'), 2)),
                 ('Pressed again?', 'possibly' if r.get('repress') else 'no')]
        self.data.setRowCount(len(items))
        for i, (k, v) in enumerate(items):
            self.data.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
            self.data.setItem(i, 1, QtWidgets.QTableWidgetItem(str(v)))

    # ----- video -----
    def stop_clip(self):
        self.timer.stop()
        self.b_play.setText('Play')
        if self.loader is not None and self.loader.isRunning():
            self.loader.stop_requested = True

    def load_clip(self, r):
        self.stop_clip()
        self.frames, self.frame_nums, self.pos = [], [], 0
        if r.get('on_frame') is None:
            self.l_video.setText('This trial was not found in the video.')
            return
        self.l_video.setText('Loading the video of this trial…')
        session = self.session
        self.clip_of = r
        loader = Worker(lambda progress, stop: load_trial_clip(session, r, stop))
        loader.done.connect(lambda res, r=r: self.clip_loaded(r, res))
        loader.failed.connect(lambda m: self.l_video.setText(f'Could not load the video: {m}'))
        self.loader = loader
        loader.start()

    def clip_loaded(self, r, res):
        if res is None or r is not self.clip_of:
            return
        self.frames, self.frame_nums = res
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.frames) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.pos = 0
        self.show_frame()
        self.toggle_play()

    def show_frame(self):
        if not self.frames:
            return
        self.l_video.setPixmap(to_pixmap(self.frames[self.pos]).scaled(self.l_video.size(), Qt.KeepAspectRatio,
                                                                       Qt.SmoothTransformation))

    def set_speed(self, *_):
        speed = float(self.c_speed.currentText().rstrip('x'))
        fps = self.session.fps if self.session is not None else 50.0
        self.timer.setInterval(max(5, int(1000 / (fps * speed))))

    def toggle_play(self):
        if not self.frames:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.b_play.setText('Play')
        else:
            if self.pos >= len(self.frames) - 1:
                self.pos = 0
            self.set_speed()
            self.timer.start()
            self.b_play.setText('Pause')

    def next_frame(self):
        if self.pos >= len(self.frames) - 1:
            self.timer.stop()
            self.b_play.setText('Play')
            return
        self.pos += 1
        self.slider.blockSignals(True)
        self.slider.setValue(self.pos)
        self.slider.blockSignals(False)
        self.show_frame()

    def seek(self, v):
        self.pos = int(v)
        self.show_frame()
