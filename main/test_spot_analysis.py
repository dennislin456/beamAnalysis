#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""雙光斑波谷／散射抗干擾單元測試（見 docs/雙光斑分析模組功能需求書.md）。"""
from __future__ import annotations

import math
import sys

import numpy as np

from shared_components import (
    _find_dual_peak_valley_detail,
    find_dual_peak_valley_y,
    intersect_half_with_locate_band,
)


def _make_dual_spot(
    h: int = 120,
    w: int = 80,
    *,
    above=(40, 30),
    below=(45, 90),
    sigma: float = 6.0,
) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w), dtype=np.float64)
    for cx, cy, amp in ((above[0], above[1], 1000.0), (below[0], below[1], 900.0)):
        img += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    img += 20.0
    return img


def test_valley_between_strongest_dual_peaks_ignores_scatter():
    n = 512
    y = np.zeros(n, dtype=np.float64)
    y[210] = 100.0
    y[238] = 95.0
    for i in range(205, 216):
        y[i] = max(y[i], 55.0)
    for i in range(233, 244):
        y[i] = max(y[i], 50.0)
    for i in range(30, 100):
        y[i] = 70.0 * math.exp(-((i - 60) / 22.0) ** 2)
    y[60] = 92.0

    detail = _find_dual_peak_valley_detail(y, smooth_win=1, min_peak_distance=5)
    assert detail is not None
    assert 210 <= detail["peak_lo"] <= 238
    assert 210 <= detail["peak_hi"] <= 238
    assert detail["peak_lo"] < detail["valley"] < detail["peak_hi"]


def test_close_dual_peaks_honor_min_peak_distance():
    n = 220
    y = np.zeros(n, dtype=np.float64)
    y[80] = 100.0
    y[90] = 95.0
    y[200] = 40.0
    for i in (79, 81, 89, 91):
        y[i] = max(y[i], 60.0)
    detail = _find_dual_peak_valley_detail(y, smooth_win=1, min_peak_distance=5)
    assert detail is not None
    assert 80 < detail["valley"] < 90


def test_reject_dual_plus_scatter_rim_pair():
    n = 512
    y = np.zeros(n, dtype=np.float64)
    y[210] = 100.0
    y[238] = 96.0
    for i in range(206, 215):
        y[i] = max(y[i], 48.0)
    for i in range(234, 243):
        y[i] = max(y[i], 46.0)
    for i in range(70, 110):
        y[i] = max(y[i], 55.0 * math.exp(-((i - 90) / 18.0) ** 2))
    y[90] = 88.0

    detail = _find_dual_peak_valley_detail(y, smooth_win=1, min_peak_distance=5)
    assert detail is not None
    assert detail["peak_lo"] >= 200
    assert detail["peak_hi"] <= 245
    assert detail["peak_lo"] < detail["valley"] < detail["peak_hi"]
    assert (detail["peak_hi"] - detail["peak_lo"]) < 50


def test_scatter_frame_ok_without_user_roi():
    h, w = 200, 120
    img = _make_dual_spot(h, w, above=(60, 90), below=(62, 115), sigma=5.0)
    yy, xx = np.mgrid[0:h, 0:w]
    img = img + 1800.0 * np.exp(-((xx - 60) ** 2 + (yy - 25) ** 2) / (2 * 8.0**2))
    img = img + 1200.0 * np.exp(-((xx - 40) ** 2 + (yy - 35) ** 2) / (2 * 12.0**2))

    info = find_dual_peak_valley_y(
        img, roi=None, min_peak_distance=5, pixel_pitch_um=5.5,
    )
    valley = info["valley_y"]
    assert 90 < valley < 115
    peaks = info["peak_ys"]
    assert peaks is not None
    assert peaks[0] < valley < peaks[1]

    bounds = info["locate_bounds"]
    assert bounds is not None
    sub, x0, y0 = intersect_half_with_locate_band(img, valley, "below", bounds)
    assert sub is not None
    cy = int(np.argmax(sub.max(axis=1)))
    assert cy + y0 > 60


def test_search_roi_clamps_locate_band():
    h, w = 200, 120
    img = _make_dual_spot(h, w, above=(60, 70), below=(62, 95), sigma=5.0)
    yy, xx = np.mgrid[0:h, 0:w]
    img = img + 2000.0 * np.exp(-((xx - 60) ** 2 + (yy - 20) ** 2) / (2 * 4.0**2))

    info = find_dual_peak_valley_y(
        img,
        roi=(30, 50, 60, 70),
        min_peak_distance=5,
        pixel_pitch_um=5.5,
    )
    assert info["roi"] is not None
    bounds = info["locate_bounds"]
    assert bounds is not None
    rx0, ry0, rx1, ry1 = info["roi"]
    assert bounds[0] >= rx0 and bounds[2] <= rx1
    assert bounds[1] >= ry0 and bounds[3] <= ry1


def test_expected_distance_gates_peak_sep():
    h, w = 200, 120
    img = _make_dual_spot(h, w, above=(60, 90), below=(62, 115), sigma=5.0)
    yy, xx = np.mgrid[0:h, 0:w]
    img = img + 1600.0 * np.exp(-((xx - 60) ** 2 + (yy - 25) ** 2) / (2 * 8.0**2))

    info = find_dual_peak_valley_y(
        img,
        min_peak_distance=5,
        expected_distance_min_um=80.0,
        expected_distance_max_um=250.0,
        pixel_pitch_um=5.5,
    )
    peaks = info["peak_ys"]
    assert peaks is not None
    sep_um = abs(peaks[1] - peaks[0]) * 5.5
    assert 80.0 <= sep_um <= 250.0


if __name__ == "__main__":
    test_valley_between_strongest_dual_peaks_ignores_scatter()
    test_close_dual_peaks_honor_min_peak_distance()
    test_reject_dual_plus_scatter_rim_pair()
    test_scatter_frame_ok_without_user_roi()
    test_search_roi_clamps_locate_band()
    test_expected_distance_gates_peak_sep()
    print("OK: all spot_analysis tests passed")
