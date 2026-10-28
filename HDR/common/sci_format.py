"""Scientific-notation display for intensity values.

Copied from FourStepAdj ``sci_format.py``.
"""

from __future__ import annotations

import re

import numpy as np
from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QVBoxLayout,
)

_PARTIAL = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)?(?:[eE][+-]?\d*)?$")


def format_sci(value: float) -> str:
    number = float(value)
    if number == 0.0:
        return "0.0000e+00"
    text = str(np.format_float_scientific(number, unique=True, trim="k", exp_digits=2))
    mantissa, exponent = text.split("e")
    if mantissa.endswith("."):
        mantissa += "0"
    elif "." not in mantissa:
        mantissa += ".0"
    return f"{mantissa}e{exponent}"


class ScientificSpinBox(QDoubleSpinBox):
    def textFromValue(self, value: float) -> str:
        return format_sci(value)

    def valueFromText(self, text: str) -> float:
        return float(text.strip())

    def validate(self, text: str, pos: int):
        body = text.strip()
        if body in {"", "+", "-", ".", "+.", "-."}:
            return QValidator.Intermediate, text, pos
        if _PARTIAL.fullmatch(body):
            try:
                float(body)
            except ValueError:
                return QValidator.Intermediate, text, pos
            return QValidator.Acceptable, text, pos
        return QValidator.Invalid, text, pos


def ask_scientific(parent, title: str, label: str, value: float) -> tuple[float, bool]:
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(label))
    box = ScientificSpinBox()
    box.setRange(-1e15, 1e15)
    box.setDecimals(12)
    box.setKeyboardTracking(False)
    box.setButtonSymbols(QAbstractSpinBox.NoButtons)
    box.setValue(float(value))
    box.selectAll()
    layout.addWidget(box)
    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    box.setFocus()
    if dialog.exec_() != QDialog.Accepted:
        return float(value), False
    return float(box.value()), True
