"""Canvas controls copied from FourStepAdj ``image_canvas`` for the HDR window."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QEvent, QObject, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QSpinBox,
)

COLOR_FLOOR = 256.0
COLOR_CEIL = 4096.0


def clip_color_scale(data: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    """<= lo becomes 0, >= hi becomes hi. A missing bound is not applied."""
    out = np.asarray(data, dtype=float).copy()
    use_lo = lo is not None and np.isfinite(lo)
    use_hi = hi is not None and np.isfinite(hi)
    if use_lo and use_hi and float(hi) < float(lo):
        lo, hi = hi, lo
    if use_lo:
        out = np.where(out <= float(lo), 0.0, out)
    if use_hi:
        out = np.where(out >= float(hi), float(hi), out)
    return out


class OptionalNumberBox(QLineEdit):
    """Number field that can be cleared. Empty means the bound is not used."""

    numberChanged = pyqtSignal()

    def __init__(self, default: float | None, tooltip: str, width: int = 88):
        super().__init__()
        self.setFixedWidth(width)
        self.setPlaceholderText("不採用")
        self.setToolTip(tooltip)
        self._number: float | None = None if default is None else float(default)
        self.setText(self._format(self._number))
        self.editingFinished.connect(self._commit)

    def number(self) -> float | None:
        return self._number

    def _format(self, value: float | None) -> str:
        if value is None:
            return ""
        if float(value).is_integer():
            return str(int(value))
        text = f"{float(value):.6f}".rstrip("0").rstrip(".")
        return text or "0"

    def _commit(self) -> None:
        text = self.text().strip()
        if text == "":
            new: float | None = None
        else:
            try:
                new = float(text)
            except ValueError:
                self.setText(self._format(self._number))
                return
            if not np.isfinite(new):
                self.setText(self._format(self._number))
                return
        changed = new != self._number
        self._number = new
        self.setText(self._format(new))
        if changed:
            self.numberChanged.emit()

    def set_number(self, value: float | None, emit: bool = True) -> None:
        new = None if value is None else float(value)
        if new is not None and not np.isfinite(new):
            return
        changed = new != self._number
        self._number = new
        self.setText(self._format(new))
        if changed and emit:
            self.numberChanged.emit()


def function_frame() -> tuple[QFrame, QHBoxLayout]:
    """One bordered box for a single canvas control group."""
    frame = QFrame()
    frame.setObjectName("canvasFunction")
    frame.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    frame.setStyleSheet(
        "QFrame#canvasFunction {"
        "background: #ffffff;"
        "border: 1px solid #c5ccd4;"
        "border-radius: 6px;"
        "}"
    )
    row = QHBoxLayout(frame)
    row.setContentsMargins(8, 3, 8, 3)
    row.setSpacing(6)
    return frame, row


def add_gaussian_controls(layout) -> tuple[QSpinBox, QSpinBox, QDoubleSpinBox]:
    """Horizontal kernel width, vertical kernel width, and one shared sigma."""

    def kernel_box(tip: str) -> QSpinBox:
        box = QSpinBox()
        box.setRange(1, 99)
        box.setValue(3)
        box.setEnabled(False)
        box.setKeyboardTracking(False)
        box.setFixedWidth(52)
        box.setToolTip(tip)
        return box

    horizontal = kernel_box("水平方向的 Gaussian 核寬。1 表示這個方向不平滑。")
    vertical = kernel_box("垂直方向的 Gaussian 核寬。1 表示這個方向不平滑。")
    sigma = QDoubleSpinBox()
    sigma.setRange(0.01, 100.0)
    sigma.setDecimals(2)
    sigma.setSingleStep(0.1)
    sigma.setValue(0.8)
    sigma.setEnabled(False)
    sigma.setKeyboardTracking(False)
    sigma.setFixedWidth(64)
    sigma.setToolTip("Gaussian blur 的 sigma。水平和垂直共用。預設 0.8。")
    for text, widget in (("水平", horizontal), ("垂直", vertical), ("sigma", sigma)):
        layout.addWidget(QLabel(text))
        layout.addWidget(widget)
    return horizontal, vertical, sigma


class _InputUndo(QObject):
    """Ctrl+Z restores the previous committed value in spin boxes and line edits."""

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._widgets: dict[int, QAbstractSpinBox | QLineEdit] = {}
        self._last: dict[int, object] = {}
        self._stack: dict[int, list[object]] = {}
        self._undoing: set[int] = set()
        app.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.FocusIn:
            target = self._input(obj)
            if target is not None:
                self._bind(target)
        if event.type() == QEvent.KeyPress and event.matches(QKeySequence.Undo):
            target = self._input(obj)
            if target is None:
                return False
            edit = target.lineEdit() if isinstance(target, QAbstractSpinBox) else target
            if isinstance(edit, QLineEdit) and edit.isUndoAvailable():
                return False
            return self._restore(target)
        return False

    def _input(self, obj):
        if isinstance(obj, QAbstractSpinBox):
            return obj
        if isinstance(obj, QLineEdit):
            parent = obj.parent()
            if isinstance(parent, QAbstractSpinBox):
                return parent
            return obj
        return None

    def _bind(self, widget) -> None:
        key = id(widget)
        if key in self._widgets:
            return
        self._widgets[key] = widget
        widget.destroyed.connect(lambda _obj=None, key=key: self._drop(key))
        if isinstance(widget, QAbstractSpinBox):
            self._last[key] = widget.value()
            widget.valueChanged.connect(lambda _value, key=key: self._on_spin_changed(key))
            widget.editingFinished.connect(lambda key=key: self._on_spin_done(key))
        else:
            self._last[key] = widget.text()
            widget.editingFinished.connect(lambda key=key: self._on_line_done(key))

    def _drop(self, key: int) -> None:
        self._widgets.pop(key, None)
        self._last.pop(key, None)
        self._stack.pop(key, None)
        self._undoing.discard(key)

    def _on_spin_changed(self, key: int) -> None:
        widget = self._widgets.get(key)
        if not isinstance(widget, QAbstractSpinBox) or key in self._undoing:
            return
        edit = widget.lineEdit()
        if edit is not None and edit.isUndoAvailable():
            return
        self._push(key, widget.value())

    def _on_spin_done(self, key: int) -> None:
        widget = self._widgets.get(key)
        if isinstance(widget, QAbstractSpinBox) and key not in self._undoing:
            self._push(key, widget.value())

    def _on_line_done(self, key: int) -> None:
        widget = self._widgets.get(key)
        if isinstance(widget, QLineEdit) and key not in self._undoing:
            self._push(key, widget.text())

    def _push(self, key: int, value: object) -> None:
        previous = self._last.get(key, None)
        if previous is None and key not in self._last:
            self._last[key] = value
            return
        if previous == value:
            return
        stack = self._stack.setdefault(key, [])
        stack.append(previous)
        if len(stack) > 40:
            del stack[0]
        self._last[key] = value

    def _restore(self, widget) -> bool:
        key = id(widget)
        stack = self._stack.get(key)
        if not stack:
            return False
        previous = stack.pop()
        self._undoing.add(key)
        try:
            if isinstance(widget, QAbstractSpinBox):
                widget.setValue(previous)
                self._last[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                widget.setText(str(previous))
                if isinstance(widget, OptionalNumberBox):
                    widget._commit()
                self._last[key] = widget.text()
        finally:
            self._undoing.discard(key)
        return True


_input_undo: _InputUndo | None = None


class _NoWheelValueChange(QObject):
    """Ignore mouse-wheel on spin boxes and combo boxes so scrolling a panel cannot nudge values."""

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Wheel:
            return False
        if isinstance(obj, (QAbstractSpinBox, QComboBox)):
            return True
        if isinstance(obj, QLineEdit) and isinstance(obj.parent(), QAbstractSpinBox):
            return True
        return False


_no_wheel: _NoWheelValueChange | None = None


def install_no_wheel_on_inputs(app: QApplication | None = None) -> None:
    """Mouse wheel over spin/combo boxes does not change their value."""
    global _no_wheel
    if _no_wheel is not None:
        return
    application = app or QApplication.instance()
    if application is None:
        return
    _no_wheel = _NoWheelValueChange(application)
    application.installEventFilter(_no_wheel)


def install_input_undo(app: QApplication | None = None) -> None:
    """Ctrl+Z in any spin box or line edit restores the previous committed value."""
    global _input_undo
    if _input_undo is not None:
        return
    application = app or QApplication.instance()
    if application is None:
        return
    _input_undo = _InputUndo(application)
    install_no_wheel_on_inputs(application)
