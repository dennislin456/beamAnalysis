from pathlib import Path

p = Path(r"d:\User\Documents\00_code\beamAnalysis\basler\tab_batch_basler_dual.py")
text = p.read_text(encoding="utf-8")
start = text.index("    def _make_profile_plot(self, title, *, which, dual_peaks=True):")
end = text.index("    def _data_range_from_block(self, block):")
new = '''    def _make_profile_plot(self, title, *, which, dual_peaks=True):
        plot = pg.PlotWidget(title=title)
        plot.setBackground("#FAFBFC")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.setLabel("bottom", "Y (pixels)")
        plot.setLabel("left", "Horizontal sum")
        plot.addLegend(offset=(10, 10))
        line1 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#2E7D32", width=2, style=Qt.DashLine),
        )
        line2 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=False,
            pen=pg.mkPen("#C62828", width=2, style=Qt.DashLine),
        )
        plot.addItem(line1)
        plot.addItem(line2)
        if not dual_peaks:
            line2.hide()
        self._scale_ctrls[which] = attach_waveform_axis_scale(
            plot,
            parent=self,
            x_label="Y (pixels)",
            y_label="Horizontal sum",
        )
        return plot, line1, line2

'''
text = text[:start] + new + text[end:]
old_apply = '''    def _apply_default_range(self, which, plot):
        rng = self._default_ranges.get(which)
        if rng is None:
            plot.enableAutoRange()
            return
        x0, x1, y0, y1 = rng
        plot.setXRange(x0, x1, padding=0.02)
        plot.setYRange(y0, y1, padding=0)

    def _update_default_ranges_from_profiles(self, *, apply=False):
        self._default_ranges["short"] = self._data_range_from_block(
            self._profiles.get("short")
        )
        self._default_ranges["long"] = self._data_range_from_block(
            self._profiles.get("long")
        )
        if apply:
            self._apply_default_range("short", self.plot_short)
            self._apply_default_range("long", self.plot_long)
'''
new_apply = '''    def _apply_default_range(self, which, plot):
        rng = self._default_ranges.get(which)
        ctrl = self._scale_ctrls.get(which)
        if rng is not None and ctrl is not None:
            ctrl.set_default_range(*rng)
            ctrl.reset_view()
            return
        if rng is None:
            plot.enableAutoRange()
            return
        x0, x1, y0, y1 = rng
        plot.setXRange(x0, x1, padding=0.02)
        plot.setYRange(y0, y1, padding=0)

    def _update_default_ranges_from_profiles(self, *, apply=False):
        self._default_ranges["short"] = self._data_range_from_block(
            self._profiles.get("short")
        )
        self._default_ranges["long"] = self._data_range_from_block(
            self._profiles.get("long")
        )
        for which in ("short", "long"):
            rng = self._default_ranges.get(which)
            ctrl = self._scale_ctrls.get(which)
            if rng is not None and ctrl is not None:
                ctrl.set_default_range(*rng)
        if apply:
            self._apply_default_range("short", self.plot_short)
            self._apply_default_range("long", self.plot_long)
'''
if old_apply not in text:
    raise SystemExit("apply block not found")
text = text.replace(old_apply, new_apply, 1)
p.write_text(text, encoding="utf-8")
print("ok")
