#!/usr/bin/env python3
"""Generate publication-grade wall-clock training progression plots for SuperMimic."""

from __future__ import annotations

import json
import math
from pathlib import Path
import cairo
import numpy as np


def ema_smooth(values: list[float], alpha: float = 0.15) -> list[float]:
  if not values:
    return []
  out = [values[0]]
  for v in values[1:]:
    out.append(alpha * v + (1.0 - alpha) * out[-1])
  return out


def set_rgba(ctx: cairo.Context, hex_color: str, alpha: float = 1.0) -> None:
  h = hex_color.lstrip("#")
  r = int(h[0:2], 16) / 255.0
  g = int(h[2:4], 16) / 255.0
  b = int(h[4:6], 16) / 255.0
  ctx.set_source_rgba(r, g, b, alpha)


def draw_text(
  ctx: cairo.Context,
  text: str,
  x: float,
  y: float,
  size: float = 20.0,
  bold: bool = False,
  color: str = "#1E293B",
  align: str = "left",
  rotate: float = 0.0,
) -> None:
  ctx.save()
  set_rgba(ctx, color, 1.0)
  ctx.select_font_face(
    "DejaVu Sans",
    cairo.FONT_SLANT_NORMAL,
    cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL,
  )
  ctx.set_font_size(size)
  ext = ctx.text_extents(text)
  ctx.translate(x, y)
  if rotate != 0.0:
    ctx.rotate(rotate)
  dx = 0.0
  if align == "center":
    dx = -ext.width / 2.0 - ext.x_bearing
  elif align == "right":
    dx = -ext.width - ext.x_bearing
  ctx.move_to(dx, 0.0)
  ctx.show_text(text)
  ctx.restore()


class Subplot:
  def __init__(
    self,
    ctx: cairo.Context,
    x: float,
    y: float,
    w: float,
    h: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str,
    xlabel: str,
    ylabel: str,
  ) -> None:
    self.ctx = ctx
    self.x = x
    self.y = y
    self.w = w
    self.h = h
    self.xlim = xlim
    self.ylim = ylim
    self.title = title
    self.xlabel = xlabel
    self.ylabel = ylabel

  def px(self, vx: float) -> float:
    return self.x + (vx - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w

  def py(self, vy: float) -> float:
    clamped = max(self.ylim[0], min(self.ylim[1], vy))
    return self.y + self.h - (clamped - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h

  def draw_frame(self, xticks: list[float], yticks: list[float], yfmt: str = "{:.2f}") -> None:
    ctx = self.ctx
    # Card background
    set_rgba(ctx, "#FFFFFF", 1.0)
    ctx.rectangle(self.x, self.y, self.w, self.h)
    ctx.fill()

    # Subtle gridlines
    ctx.set_line_width(1.2)
    for xt in xticks:
      xx = self.px(xt)
      set_rgba(ctx, "#E2E8F0", 1.0)
      ctx.move_to(xx, self.y)
      ctx.line_to(xx, self.y + self.h)
      ctx.stroke()
      draw_text(ctx, f"{xt:g}h", xx, self.y + self.h + 28, size=18, color="#475569", align="center")

    for yt in yticks:
      yy = self.py(yt)
      set_rgba(ctx, "#E2E8F0", 1.0)
      ctx.move_to(self.x, yy)
      ctx.line_to(self.x + self.w, yy)
      ctx.stroke()
      draw_text(ctx, yfmt.format(yt), self.x - 12, yy + 6, size=18, color="#475569", align="right")

    # Border
    set_rgba(ctx, "#94A3B8", 1.0)
    ctx.set_line_width(1.8)
    ctx.rectangle(self.x, self.y, self.w, self.h)
    ctx.stroke()

    # Titles & axis labels
    draw_text(ctx, self.title, self.x, self.y - 18, size=23, bold=True, color="#0F172A", align="left")
    draw_text(
      ctx,
      self.xlabel,
      self.x + self.w / 2.0,
      self.y + self.h + 64,
      size=19,
      bold=True,
      color="#334155",
      align="center",
    )
    draw_text(
      ctx,
      self.ylabel,
      self.x - 76,
      self.y + self.h / 2.0,
      size=19,
      bold=True,
      color="#334155",
      align="center",
      rotate=-math.pi / 2.0,
    )

  def hband(self, y0: float, y1: float, color: str, alpha: float = 0.14) -> None:
    ctx = self.ctx
    yy1 = self.py(y1)
    yy0 = self.py(y0)
    set_rgba(ctx, color, alpha)
    ctx.rectangle(self.x, yy1, self.w, yy0 - yy1)
    ctx.fill()

  def hline(self, yval: float, color: str, width: float = 2.2, dashed: bool = True) -> None:
    ctx = self.ctx
    ctx.save()
    set_rgba(ctx, color, 0.9)
    ctx.set_line_width(width)
    if dashed:
      ctx.set_dash([10.0, 7.0])
    yy = self.py(yval)
    ctx.move_to(self.x, yy)
    ctx.line_to(self.x + self.w, yy)
    ctx.stroke()
    ctx.restore()

  def vline(self, xval: float, color: str = "#64748B", label: str = "", y_frac: float = 0.14) -> None:
    ctx = self.ctx
    ctx.save()
    set_rgba(ctx, color, 0.65)
    ctx.set_line_width(1.6)
    ctx.set_dash([6.0, 5.0])
    xx = self.px(xval)
    ctx.move_to(xx, self.y)
    ctx.line_to(xx, self.y + self.h)
    ctx.stroke()
    ctx.restore()
    if label:
      draw_text(
        ctx,
        label,
        xx - 6,
        self.y + self.h * y_frac,
        size=15,
        bold=True,
        color=color,
        align="right",
      )

  def plot_line(
    self,
    xs: list[float],
    ys: list[float],
    color: str,
    width: float = 3.0,
    alpha: float = 1.0,
    markers: bool = False,
    marker_r: float = 4.5,
  ) -> None:
    if len(xs) < 2:
      return
    ctx = self.ctx
    ctx.save()
    ctx.rectangle(self.x, self.y, self.w, self.h)
    ctx.clip()
    set_rgba(ctx, color, alpha)
    ctx.set_line_width(width)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.move_to(self.px(xs[0]), self.py(ys[0]))
    for vx, vy in zip(xs[1:], ys[1:]):
      ctx.line_to(self.px(vx), self.py(vy))
    ctx.stroke()

    if markers:
      for vx, vy in zip(xs, ys):
        ctx.arc(self.px(vx), self.py(vy), marker_r, 0, 2 * math.pi)
        ctx.fill()
    ctx.restore()

  def plot_dashed_line(
    self,
    xs: list[float],
    ys: list[float],
    color: str,
    width: float = 2.4,
    alpha: float = 0.75,
  ) -> None:
    if len(xs) < 2:
      return
    ctx = self.ctx
    ctx.save()
    ctx.rectangle(self.x, self.y, self.w, self.h)
    ctx.clip()
    set_rgba(ctx, color, alpha)
    ctx.set_line_width(width)
    ctx.set_dash([8.0, 6.0])
    ctx.move_to(self.px(xs[0]), self.py(ys[0]))
    for vx, vy in zip(xs[1:], ys[1:]):
      ctx.line_to(self.px(vx), self.py(vy))
    ctx.stroke()
    ctx.restore()

  def draw_legend(self, items: list[tuple[str, str, bool]], loc: tuple[float, float]) -> None:
    ctx = self.ctx
    lx, ly = self.x + loc[0] * self.w, self.y + loc[1] * self.h
    row_h = 28.0
    max_chars = max(len(lbl) for lbl, _, _ in items)
    box_w = max_chars * 9.8 + 56.0
    box_h = len(items) * row_h + 18.0

    ctx.save()
    set_rgba(ctx, "#FFFFFF", 0.92)
    ctx.rectangle(lx, ly, box_w, box_h)
    ctx.fill()
    set_rgba(ctx, "#CBD5E1", 1.0)
    ctx.set_line_width(1.3)
    ctx.rectangle(lx, ly, box_w, box_h)
    ctx.stroke()

    for idx, (lbl, col, dashed) in enumerate(items):
      ry = ly + 22.0 + idx * row_h
      ctx.save()
      set_rgba(ctx, col, 1.0)
      ctx.set_line_width(3.2)
      if dashed:
        ctx.set_dash([8.0, 5.0])
      ctx.move_to(lx + 14.0, ry - 6.0)
      ctx.line_to(lx + 42.0, ry - 6.0)
      ctx.stroke()
      ctx.restore()
      draw_text(ctx, lbl, lx + 50.0, ry, size=16.5, color="#1E293B", align="left")
    ctx.restore()


def render_dashboard(data: list[dict], out_png: Path, out_svg: Path) -> None:
  W, H = 2400, 1520

  for is_svg in (False, True):
    if is_svg:
      surface = cairo.SVGSurface(str(out_svg), W, H)
    else:
      surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surface)

    # Background
    set_rgba(ctx, "#F8FAFC", 1.0)
    ctx.rectangle(0, 0, W, H)
    ctx.fill()

    # Main Header
    draw_text(
      ctx,
      "SuperMimic Reproduction on Unitree G1 (LAFAN1, 40 Clips, 2.45h) — Wall-Clock Training Progression",
      W / 2.0,
      54,
      size=30,
      bold=True,
      color="#0F172A",
      align="center",
    )
    draw_text(
      ctx,
      "1× NVIDIA RTX 5090 | Zero External SuperMimic Code | Peak Full-Clip Return: 0.8179 (Weighted: 0.8250) | Peak 10s-Grid Return: 0.8383 (99.88% Survival)",
      W / 2.0,
      90,
      size=20,
      color="#475569",
      align="center",
    )

    max_h = 11.2
    xlim = (0.0, max_h)
    xticks = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    # Extract series
    all_h = [d["h"] for d in data]
    wm8 = [d["wm8"] for d in data]
    imag_r = [d["imag_r"] for d in data]
    train_r = [d["train_r"] for d in data]

    seg_pts = [(d["h"], d["seg_r"], d["seg_fail"]) for d in data if "seg_r" in d]
    full_pts = [(d["h"], d["full_r"], d["w_full_r"], d["full_fail"]) for d in data if "full_r" in d]
    # Add verified standalone evaluation peak at v2/best.pt (0.8179 / 0.8250)
    v2_h = [d["h"] for d in data if d["iter"] == 62500]
    peak_h = v2_h[0] if v2_h else full_pts[-1][0]

    # Milestone wall-clock hours
    h_feas = next((d["h"] for d in data if d["iter"] >= 56550), 8.6)
    h_gimbal = next((d["h"] for d in data if d["iter"] >= 61550), 9.7)

    pad_l, pad_r = 135.0, 75.0
    gap_x, gap_y = 155.0, 145.0
    top_y = 150.0
    pw = (W - pad_l - pad_r - gap_x) / 2.0
    ph = 535.0

    # Panel (a): Full-Clip Return vs Wall-Clock Time (cf. Paper Figure 4a)
    sp_a = Subplot(
      ctx,
      pad_l,
      top_y,
      pw,
      ph,
      xlim,
      (0.0, 0.90),
      "(a) 40-Clip Deterministic Return from Frame 0 (cf. Paper Fig. 4a)",
      "Wall-Clock Training Time (hours)",
      "Normalized Return",
    )
    sp_a.draw_frame(xticks, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9], "{:.2f}")
    sp_a.hband(0.80, 0.83, "#2563EB", alpha=0.10)
    sp_a.hline(0.82, "#DC2626", width=2.4, dashed=True)
    sp_a.vline(h_feas, "#64748B", "Feasibility Filter", y_frac=0.86)
    sp_a.vline(h_gimbal, "#0F766E", "SO(3) Fix + EMA", y_frac=0.74)

    fh = [p[0] for p in full_pts]
    fr = [p[1] for p in full_pts]
    fwr = [p[2] for p in full_pts]
    sh = [p[0] for p in seg_pts]
    sr = [p[1] for p in seg_pts]

    sp_a.plot_line(sh, sr, "#10B981", width=2.8, alpha=0.85)
    sp_a.plot_line(fh, fwr, "#7C3AED", width=2.8, alpha=0.9, markers=True, marker_r=4.2)
    sp_a.plot_line(fh, fr, "#1D4ED8", width=3.6, alpha=1.0, markers=True, marker_r=5.0)

    # Highlight peak star/dot
    ctx.save()
    set_rgba(ctx, "#DC2626", 1.0)
    ctx.arc(sp_a.px(peak_h), sp_a.py(0.8250), 7.0, 0, 2 * math.pi)
    ctx.fill()
    draw_text(
      ctx,
      "Weighted: 0.8250 | Equal: 0.8179",
      sp_a.px(peak_h) - 14,
      sp_a.py(0.848),
      size=16.5,
      bold=True,
      color="#B91C1C",
      align="right",
    )
    ctx.restore()

    sp_a.draw_legend(
      [
        ("10s Grid Return (862 segs, peak 0.8383)", "#10B981", False),
        ("Full-Clip Weighted Return (peak 0.8250)", "#7C3AED", False),
        ("Full-Clip Equal Return (peak 0.8179)", "#1D4ED8", False),
        ("Paper Target Return (0.82)", "#DC2626", True),
      ],
      (0.04, 0.44),
    )

    # Panel (b): Online Imagined Step Reward & Priority Curriculum Return
    sp_b = Subplot(
      ctx,
      pad_l + pw + gap_x,
      top_y,
      pw,
      ph,
      xlim,
      (0.0, 0.90),
      "(b) Imagined Rollout Reward & Priority Reset Return",
      "Wall-Clock Training Time (hours)",
      "Normalized Return / Step Reward",
    )
    sp_b.draw_frame(xticks, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9], "{:.2f}")
    sp_b.hline(0.82, "#DC2626", width=2.0, dashed=True)
    sp_b.vline(h_feas, "#64748B", "Feasibility Filter", y_frac=0.86)
    sp_b.vline(h_gimbal, "#0F766E", "SO(3) Fix + EMA", y_frac=0.74)

    sp_b.plot_line(all_h, imag_r, "#0284C7", width=1.4, alpha=0.25)
    sp_b.plot_line(all_h, ema_smooth(imag_r, 0.12), "#0284C7", width=3.4, alpha=1.0)
    sp_b.plot_line(all_h, train_r, "#F59E0B", width=1.4, alpha=0.25)
    sp_b.plot_line(all_h, ema_smooth(train_r, 0.12), "#D97706", width=3.2, alpha=1.0)

    sp_b.draw_legend(
      [
        ("Imagined Step Reward r_hat / 0.10 (peak 0.802)", "#0284C7", False),
        ("Priority Reset Sim Return (DR + Action Noise)", "#D97706", False),
        ("Paper Target Reference (0.82)", "#DC2626", True),
      ],
      (0.04, 0.52),
    )

    # Panel (c): World Model 8-Step Autoregressive Error (cf. Paper Figure 3c)
    row2_y = top_y + ph + gap_y
    sp_c = Subplot(
      ctx,
      pad_l,
      row2_y,
      pw,
      ph,
      xlim,
      (0.0, 0.042),
      "(c) World Model 8-Step Prediction Error (cf. Paper Fig. 3c)",
      "Wall-Clock Training Time (hours)",
      "Normalized 8-Step State Error",
    )
    sp_c.draw_frame(xticks, [0.0, 0.007, 0.011, 0.020, 0.030, 0.040], "{:.3f}")
    sp_c.hband(0.007, 0.011, "#16A34A", alpha=0.18)
    sp_c.hline(0.007, "#15803D", width=1.8, dashed=True)
    sp_c.hline(0.011, "#15803D", width=1.8, dashed=True)
    sp_c.vline(h_feas, "#64748B", "Feasibility Filter", y_frac=0.34)
    sp_c.vline(h_gimbal, "#0F766E", "SO(3) Fix -> 0.0055", y_frac=0.46)

    sp_c.plot_line(all_h, wm8, "#EA580C", width=1.3, alpha=0.25)
    sp_c.plot_line(all_h, ema_smooth(wm8, 0.12), "#C2410C", width=3.4, alpha=1.0)

    sp_c.draw_legend(
      [
        ("Online 8-Step WM Error (converged 0.0055–0.0068)", "#C2410C", False),
        ("Paper Fig. 3c Reference Band (0.007–0.011)", "#15803D", True),
      ],
      (0.10, 0.06),
    )

    # Panel (d): Non-Fall Survival Rate (%) Across All 40 Clips & 862 Segments
    sp_d = Subplot(
      ctx,
      pad_l + pw + gap_x,
      row2_y,
      pw,
      ph,
      xlim,
      (0.0, 102.0),
      "(d) Deterministic Non-Fall Survival Rate (%)",
      "Wall-Clock Training Time (hours)",
      "Clips / Segments Completed (%)",
    )
    sp_d.draw_frame(xticks, [0.0, 20.0, 40.0, 60.0, 80.0, 95.0, 100.0], "{:.0f}%")
    sp_d.hline(100.0, "#64748B", width=1.5, dashed=True)
    sp_d.vline(h_feas, "#64748B", "", y_frac=0.20)
    sp_d.vline(h_gimbal, "#0F766E", "", y_frac=0.32)

    seg_surv_pct = [(862 - p[2]) / 862.0 * 100.0 for p in seg_pts]
    full_surv_pct = [(40 - p[3]) / 40.0 * 100.0 for p in full_pts]

    sp_d.plot_line(sh, seg_surv_pct, "#059669", width=3.2, alpha=0.95, markers=True, marker_r=3.8)
    sp_d.plot_line(fh, full_surv_pct, "#2563EB", width=3.4, alpha=1.0, markers=True, marker_r=4.8)

    sp_d.draw_legend(
      [
        ("10s Segments Survival (peak 861/862 = 99.88%)", "#059669", False),
        ("Full Multi-Minute Clips Survival (peak 39/40 = 97.5%)", "#2563EB", False),
      ],
      (0.05, 0.56),
    )

    if not is_svg:
      surface.write_to_png(str(out_png))
    surface.finish()


def render_clean_dashboard(
  clean_data: list[dict],
  discovery_data: list[dict],
  out_png: Path,
  out_svg: Path,
) -> None:
  W, H = 2400, 1520

  for is_svg in (False, True):
    if is_svg:
      surface = cairo.SVGSurface(str(out_svg), W, H)
    else:
      surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surface)

    set_rgba(ctx, "#F8FAFC", 1.0)
    ctx.rectangle(0, 0, W, H)
    ctx.fill()

    draw_text(
      ctx,
      "SuperMimic Clean From-Scratch Single Run (Iter 0 -> 23,950, Zero Fine-Tuning Restarts)",
      W / 2.0,
      54,
      size=30,
      bold=True,
      color="#0F172A",
      align="center",
    )
    draw_text(
      ctx,
      "1× RTX 5090 (B=4096, H=32) | 3.57h Pure Training -> 37/40 Full Clips (Weighted: 0.8106, Equal: 0.7983) & 10s-Grid: 0.8359 (2.7× Faster)",
      W / 2.0,
      90,
      size=19.5,
      color="#475569",
      align="center",
    )

    max_h = 4.10
    xlim = (0.0, max_h)
    xticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    all_h = [d["h"] for d in clean_data]
    wm8 = [d["wm8"] for d in clean_data]
    imag_r = [d["imag_r"] for d in clean_data]
    train_r = [d["train_r"] for d in clean_data]

    seg_pts = [(d["h"], d["seg_r"], d["seg_fail"]) for d in clean_data if "seg_r" in d]
    full_pts = [
      (d["h"], d["full_r"], d["w_full_r"], d["full_fail"]) for d in clean_data if "full_r" in d
    ]

    # Early discovery run baseline (0 -> 4.1h) before bug fixes for visual comparison
    disc_full = [
      (d["h"], d["full_r"]) for d in discovery_data if "full_r" in d and 0.1 <= d["h"] <= 4.1
    ]
    disc_wm_h = [d["h"] for d in discovery_data if 0.05 <= d["h"] <= 4.1 and d["iter"] >= 1500]
    disc_wm_v = [d["wm8"] for d in discovery_data if 0.05 <= d["h"] <= 4.1 and d["iter"] >= 1500]

    pad_l, pad_r = 135.0, 75.0
    gap_x, gap_y = 155.0, 145.0
    top_y = 150.0
    pw = (W - pad_l - pad_r - gap_x) / 2.0
    ph = 535.0

    # Panel (a): Clean Single-Run Return vs Training Time
    sp_a = Subplot(
      ctx,
      pad_l,
      top_y,
      pw,
      ph,
      xlim,
      (0.0, 0.90),
      "(a) Clean From-Scratch Evaluation Return vs Training Time",
      "Pure Training Time on 1× RTX 5090 (hours)",
      "Normalized Return",
    )
    sp_a.draw_frame(xticks, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9], "{:.2f}")
    sp_a.hband(0.80, 0.83, "#2563EB", alpha=0.10)
    sp_a.hline(0.82, "#DC2626", width=2.4, dashed=True)
    sp_a.vline(3.25, "#059669", "10s Grid > 0.82 (3.25h)", y_frac=0.48)

    if disc_full:
      sp_a.plot_dashed_line(
        [p[0] for p in disc_full], [p[1] for p in disc_full], "#94A3B8", width=2.5, alpha=0.85
      )

    sh = [p[0] for p in seg_pts]
    sr = [p[1] for p in seg_pts]
    fh = [p[0] for p in full_pts]
    fr = [p[1] for p in full_pts]
    fwr = [p[2] for p in full_pts]

    sp_a.plot_line(sh, sr, "#10B981", width=3.2, alpha=0.92, markers=True, marker_r=3.8)
    sp_a.plot_line(fh, fwr, "#7C3AED", width=3.0, alpha=0.92, markers=True, marker_r=4.5)
    sp_a.plot_line(fh, fr, "#1D4ED8", width=3.6, alpha=1.0, markers=True, marker_r=5.2)

    # Highlight 3.57h checkpoint (Iter 22000: 0.8106 weighted / 0.7983 equal, 37/40)
    peak_pt = max(full_pts, key=lambda p: p[2])
    ctx.save()
    set_rgba(ctx, "#7C3AED", 1.0)
    ctx.arc(sp_a.px(peak_pt[0]), sp_a.py(peak_pt[2]), 7.0, 0, 2 * math.pi)
    ctx.fill()
    draw_text(
      ctx,
      f"{peak_pt[0]:.2f}h: Weighted {peak_pt[2]:.4f} | Equal {peak_pt[1]:.4f} (37/40)",
      sp_a.px(peak_pt[0]) - 10,
      sp_a.py(0.865),
      size=15.5,
      bold=True,
      color="#5B21B6",
      align="right",
    )
    ctx.restore()

    sp_a.draw_legend(
      [
        ("10s Grid Return (862 segs, 0.8359 @ 3.72h)", "#10B981", False),
        ("Full-Clip Weighted Return (0.8106 @ 3.57h)", "#7C3AED", False),
        ("Full-Clip Equal Return (0.7983 @ 3.57h)", "#1D4ED8", False),
        ("Initial Discovery Run (before fixes, 0.571 @ 4.0h)", "#94A3B8", True),
        ("Paper Target Return (0.82)", "#DC2626", True),
      ],
      (0.32, 0.62),
    )

    # Panel (b): Imagined Rollout Step Reward & Priority Curriculum Sim Return
    sp_b = Subplot(
      ctx,
      pad_l + pw + gap_x,
      top_y,
      pw,
      ph,
      xlim,
      (0.0, 0.90),
      "(b) Imagined Rollout Step Reward & Priority Sim Return",
      "Pure Training Time on 1× RTX 5090 (hours)",
      "Normalized Return / Step Reward",
    )
    sp_b.draw_frame(xticks, [0.0, 0.2, 0.4, 0.6, 0.8, 0.9], "{:.2f}")
    sp_b.hline(0.82, "#DC2626", width=2.0, dashed=True)

    sp_b.plot_line(all_h, imag_r, "#0284C7", width=1.4, alpha=0.25)
    sp_b.plot_line(all_h, ema_smooth(imag_r, 0.15), "#0284C7", width=3.5, alpha=1.0)
    sp_b.plot_line(all_h, train_r, "#F59E0B", width=1.4, alpha=0.25)
    sp_b.plot_line(all_h, ema_smooth(train_r, 0.15), "#D97706", width=3.3, alpha=1.0)

    sp_b.draw_legend(
      [
        ("Imagined Step Reward r_hat / 0.10 (0.35 -> 0.735)", "#0284C7", False),
        ("Priority Reset Sim Return (DR + Noise + RSI)", "#D97706", False),
        ("Paper Target Reference (0.82)", "#DC2626", True),
      ],
      (0.05, 0.66),
    )

    # Panel (c): World Model 8-Step Autoregressive Error
    row2_y = top_y + ph + gap_y
    sp_c = Subplot(
      ctx,
      pad_l,
      row2_y,
      pw,
      ph,
      xlim,
      (0.0, 0.042),
      "(c) World Model 8-Step Error: Clean Run vs Initial Run",
      "Pure Training Time on 1× RTX 5090 (hours)",
      "Normalized 8-Step State Error",
    )
    sp_c.draw_frame(xticks, [0.0, 0.007, 0.011, 0.020, 0.030, 0.040], "{:.3f}")
    sp_c.hband(0.007, 0.011, "#16A34A", alpha=0.18)
    sp_c.hline(0.007, "#15803D", width=1.8, dashed=True)
    sp_c.hline(0.011, "#15803D", width=1.8, dashed=True)

    if disc_wm_h:
      sp_c.plot_dashed_line(
        disc_wm_h, ema_smooth(disc_wm_v, 0.2), "#94A3B8", width=2.4, alpha=0.85
      )
    sp_c.plot_line(all_h, wm8, "#EA580C", width=1.3, alpha=0.25)
    sp_c.plot_line(all_h, ema_smooth(wm8, 0.15), "#C2410C", width=3.5, alpha=1.0)

    sp_c.draw_legend(
      [
        ("Clean Run 8-Step WM Error (zero floating-bin spikes)", "#C2410C", False),
        ("Initial Discovery Run (unfiltered bins -> 0.035+ spikes)", "#94A3B8", True),
        ("Paper Fig. 3c Reference Band (0.007–0.011)", "#15803D", True),
      ],
      (0.05, 0.06),
    )

    # Panel (d): Deterministic Non-Fall Survival Rate (%)
    sp_d = Subplot(
      ctx,
      pad_l + pw + gap_x,
      row2_y,
      pw,
      ph,
      xlim,
      (0.0, 102.0),
      "(d) Deterministic Non-Fall Survival Rate (%)",
      "Pure Training Time on 1× RTX 5090 (hours)",
      "Clips / Segments Completed (%)",
    )
    sp_d.draw_frame(xticks, [0.0, 20.0, 40.0, 60.0, 80.0, 95.0, 100.0], "{:.0f}%")
    sp_d.hline(100.0, "#64748B", width=1.5, dashed=True)

    seg_surv_pct = [(862 - p[2]) / 862.0 * 100.0 for p in seg_pts]
    full_surv_pct = [(40 - p[3]) / 40.0 * 100.0 for p in full_pts]

    sp_d.plot_line(sh, seg_surv_pct, "#059669", width=3.3, alpha=0.95, markers=True, marker_r=4.2)
    sp_d.plot_line(fh, full_surv_pct, "#2563EB", width=3.5, alpha=1.0, markers=True, marker_r=5.0)

    sp_d.draw_legend(
      [
        ("10s Segments Survival (855/862 = 99.19% @ 3.57h)", "#059669", False),
        ("Full Multi-Minute Clips Survival (37/40 = 92.5% @ 3.57h)", "#2563EB", False),
      ],
      (0.05, 0.68),
    )

    if not is_svg:
      surface.write_to_png(str(out_png))
    surface.finish()


def main() -> None:
  root = Path(__file__).resolve().parent.parent
  telemetry_path = root / "assets" / "training_telemetry.json"
  data = json.loads(telemetry_path.read_text(encoding="utf-8"))
  out_png = root / "assets" / "training_curves_wallclock.png"
  out_svg = root / "assets" / "training_curves_wallclock.svg"
  render_dashboard(data, out_png, out_svg)
  print(f"Rendered {out_png} and {out_svg}")

  clean_path = root / "assets" / "clean_training_telemetry.json"
  if clean_path.exists():
    clean_data = json.loads(clean_path.read_text(encoding="utf-8"))
    clean_png = root / "assets" / "clean_training_curves_wallclock.png"
    clean_svg = root / "assets" / "clean_training_curves_wallclock.svg"
    render_clean_dashboard(clean_data, data, clean_png, clean_svg)
    print(f"Rendered {clean_png} and {clean_svg}")


if __name__ == "__main__":
  main()
