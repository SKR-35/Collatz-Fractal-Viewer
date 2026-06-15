#!/usr/bin/env python3
"""Collatz Fractal Viewer (Tkinter UI).

Interactive explorer for the complex Collatz extension

    C(z) = 1/4 * (2 + 7z - (2 + 5z) * cos(pi*z))

The UI is intentionally close to the earlier Mandelbrot viewer: toolbar,
rectangle zoom, pan, save, gamma, colormaps and auto-iteration scaling.

This version also includes several orbit/threshold modes inspired by
Collatz-fractal explorations around points 5+0i and 10+0i.
"""
from __future__ import annotations

import math
import os
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Optional, Tuple

import numpy as np

# Pillow for fast image blit
try:
    from PIL import Image, ImageTk
except Exception as exc:
    raise SystemExit("Pillow is required. Install with:  pip install pillow") from exc

# Colormap via matplotlib (for colors only + no GUI)
try:
    import matplotlib.cm as cm  # noqa: F401
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# --- Palettes ---------------------------------------------------------------
CMAPS = [
    "plasma", "viridis", "magma", "inferno", "turbo", "cividis", "twilight",
    "SoftSunset", "EarthAndSky", "Seashore", "Forest", "HotAndCold",
    "Pastel", "Grayscale",
]
DEFAULT_CMAP = "plasma"

# Custom soft palettes: list of hex stops (low -> high)
CUSTOM_STOPS = {
    "SoftSunset":  ["#2b1055", "#6a0572", "#ff6f91", "#ffc15e", "#ffe29a"],
    "EarthAndSky": ["#1a2a6c", "#28a0b0", "#84ffc9", "#f0f3bd", "#ffd166"],
    "Seashore":    ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6"],
    "Forest":      ["#0b3d0b", "#236e3c", "#4caf50", "#a8e6cf", "#f1f8e9"],
    "HotAndCold":  ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fee090", "#f46d43", "#d73027"],
    "Pastel":      ["#b3e5fc", "#c5cae9", "#e1bee7", "#f8bbd0", "#ffe0b2", "#dcedc8"],
    "Grayscale":   ["#0a0a0a", "#2f2f2f", "#5e5e5e", "#9a9a9a", "#cccccc", "#f2f2f2"],
}


def _build_lut(stops: list[str], n: int = 1024) -> np.ndarray:
    import matplotlib.colors as mcolors

    cmap = mcolors.LinearSegmentedColormap.from_list("custom", stops, N=n)
    return (cmap(np.linspace(0, 1, n))[:, :3] * 255.0).astype(np.uint8)


_CUSTOM_LUTS = {name: _build_lut(stops) for name, stops in CUSTOM_STOPS.items()}

MIN_SCALE = 1e-14
ESCAPE_RADIUS = 1.0e6
SMALL_RADIUS = 1.0
LARGE_RADIUS = 100.0

RENDER_MODES = [
    "Escape Time |z| > 1e6",
    "Hit Small Orbit |z| <= 1",
    "Hit Large Orbit |z| >= 100",
    "Point 5+0i, |z_n| <= 1",
    "Point 5+0i, |z_n| >= 100",
    "Point 10+0i, |z_n| <= 1",
    "Point 10+0i, |z_n| >= 100",
]
DEFAULT_RENDER_MODE = RENDER_MODES[0]

# View presets matching the example regions discussed in Collatz-fractal articles.
MODE_PRESETS = {
    RENDER_MODES[0]: (0.0, 0.0, 3.5),
    RENDER_MODES[1]: (0.0, 0.0, 3.5),
    RENDER_MODES[2]: (0.0, 0.0, 3.5),
    RENDER_MODES[3]: (5.0, 0.0, 1.0),
    RENDER_MODES[4]: (5.0, 0.0, 1.0),
    RENDER_MODES[5]: (10.0, 0.0, 1.0),
    RENDER_MODES[6]: (10.0, 0.0, 1.0),
}


@dataclass
class View:
    cx: float = 0.0
    cy: float = 0.0
    scale: float = 3.5  # half-height; at 1200x800 gives roughly x=[-5.25, 5.25]
    max_iter: int = 80

    def grid(self, w: int, h: int) -> np.ndarray:
        aspect = h / w
        x_min = self.cx - self.scale / aspect
        x_max = self.cx + self.scale / aspect
        y_min = self.cy - self.scale
        y_max = self.cy + self.scale
        xs = np.linspace(x_min, x_max, w, dtype=np.float64)
        ys = np.linspace(y_min, y_max, h, dtype=np.float64)
        x_grid, y_grid = np.meshgrid(xs, ys)
        return x_grid + 1j * y_grid


def collatz_step(z: np.ndarray) -> np.ndarray:
    """Complex Collatz extension used for the fractal."""
    return 0.25 * (2.0 + 7.0 * z - (2.0 + 5.0 * z) * np.cos(np.pi * z))


def collatz_escape(c: np.ndarray, max_iter: int, escape_radius: float = ESCAPE_RADIUS) -> np.ndarray:
    """Return escape-time values for the complex Collatz iteration.

    Points that do not escape within max_iter receive max_iter and are rendered
    as black, matching the usual Collatz-set convention.
    """
    z = c.copy().astype(np.complex128)
    vals = np.zeros(c.shape, dtype=np.float64)
    active = np.ones(c.shape, dtype=bool)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for n in range(max_iter):
            if not active.any():
                break

            z_active = z[active]
            z_next = collatz_step(z_active)
            z[active] = z_next

            mag = np.abs(z_next)
            escaped_local = (~np.isfinite(mag)) | (mag > escape_radius)

            if np.any(escaped_local):
                active_indices = np.flatnonzero(active)
                escaped_indices = active_indices[escaped_local]
                escaped_mag = mag[escaped_local]
                safe_mag = np.where(
                    np.isfinite(escaped_mag),
                    np.maximum(escaped_mag, escape_radius + 1.0),
                    escape_radius * 10.0,
                )
                smooth = n + 1.0 - np.log(np.log(safe_mag + 1e-16)) / np.log(2.0)
                vals.flat[escaped_indices] = np.nan_to_num(
                    np.clip(smooth, 0.0, float(max_iter)),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                active.flat[escaped_indices] = False

    vals[active] = float(max_iter)
    return vals


def collatz_threshold_time(
    c: np.ndarray,
    max_iter: int,
    threshold: float,
    relation: str,
    bailout: float = ESCAPE_RADIUS,
) -> np.ndarray:
    """Return first-hit time for |z_n| <= threshold or |z_n| >= threshold.

    Values equal to max_iter mean the condition was not reached. This supports
    the article-style explorations such as ``|z_n| <= 1`` and ``|z_n| >= 100``.
    """
    if relation not in {"le", "ge"}:
        raise ValueError("relation must be 'le' or 'ge'")

    z = c.copy().astype(np.complex128)
    vals = np.full(c.shape, float(max_iter), dtype=np.float64)
    active = np.ones(c.shape, dtype=bool)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for n in range(max_iter):
            if not active.any():
                break

            z_next = collatz_step(z[active])
            z[active] = z_next
            mag = np.abs(z_next)

            if relation == "le":
                hit_local = np.isfinite(mag) & (mag <= threshold)
                # If a point blows up while looking for a small orbit, stop it.
                dead_local = (~np.isfinite(mag)) | (mag > bailout)
            else:
                hit_local = (~np.isfinite(mag)) | (mag >= threshold)
                dead_local = np.zeros_like(hit_local, dtype=bool)

            active_indices = np.flatnonzero(active)
            if np.any(hit_local):
                hit_indices = active_indices[hit_local]
                vals.flat[hit_indices] = float(n + 1)
                active.flat[hit_indices] = False

            # Only deactivate non-hit dead points. Hit points were already handled.
            dead_only = dead_local & ~hit_local
            if np.any(dead_only):
                dead_indices = active_indices[dead_only]
                active.flat[dead_indices] = False

    return vals


def render_values(c: np.ndarray, max_iter: int, mode: str) -> np.ndarray:
    """Compute scalar rendering values for the selected mode."""
    if mode == "Escape Time |z| > 1e6":
        return collatz_escape(c, max_iter, ESCAPE_RADIUS)
    if mode in {"Hit Small Orbit |z| <= 1", "Point 5+0i, |z_n| <= 1", "Point 10+0i, |z_n| <= 1"}:
        return collatz_threshold_time(c, max_iter, SMALL_RADIUS, "le")
    if mode in {"Hit Large Orbit |z| >= 100", "Point 5+0i, |z_n| >= 100", "Point 10+0i, |z_n| >= 100"}:
        return collatz_threshold_time(c, max_iter, LARGE_RADIUS, "ge")
    return collatz_escape(c, max_iter, ESCAPE_RADIUS)


def escaped_contrast(vals: np.ndarray, max_iter: int) -> Tuple[float, float]:
    escaped = vals < (max_iter - 1e-9)
    if np.any(escaped):
        lo = float(np.percentile(vals[escaped], 0.5))
        hi = float(np.percentile(vals[escaped], 99.5))
        if hi <= lo:
            lo, hi = float(vals[escaped].min()), float(vals[escaped].max())
    else:
        lo, hi = 0.0, float(max_iter)
    return lo, hi


def map_to_rgb(
    vals: np.ndarray,
    cmap_name: str,
    lo: float,
    hi: float,
    max_iter: int,
    gamma: float = 1.25,
    smoothstep: bool = True,
) -> np.ndarray:
    """Map scalar field to RGB and render non-hit/non-escaped points as black."""
    interior = vals >= (max_iter - 1e-9)
    x = np.clip((vals - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

    if smoothstep:
        x = x * x * (3.0 - 2.0 * x)
    if gamma and gamma > 0:
        x = np.power(x, gamma)

    if cmap_name in _CUSTOM_LUTS:
        lut = _CUSTOM_LUTS[cmap_name]
        idx = np.minimum((x * (len(lut) - 1)).astype(np.int32), len(lut) - 1)
        rgb = lut[idx].copy()
    elif HAVE_MPL:
        import matplotlib
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        rgb = (cmap(x)[..., :3] * 255.0).astype(np.uint8)
    else:
        r = (0.6 + 0.4 * x) * 255
        g = (0.0 + 0.9 * x) * 255
        b = (0.6 - 0.5 * x) * 255
        rgb = np.dstack([r, g, b]).astype(np.uint8)

    rgb[interior] = 0
    return rgb


def auto_iters(scale: float) -> int:
    depth = max(0.0, -math.log10(max(scale, MIN_SCALE)))
    return int(80 + 45 * depth + 12 * depth * depth)


class TkCollatz:
    def __init__(self, w: int = 1200, h: int = 800):
        self.root = tk.Tk()
        self.root.title("Collatz Fractal Viewer — Tk")

        top = ttk.Frame(self.root, padding=(6, 4, 6, 4))
        top.pack(side=tk.TOP, fill=tk.X)

        self.mode = tk.StringVar(value="zoom")
        ttk.Button(top, text="Zoom", command=lambda: self.mode.set("zoom")).pack(side=tk.LEFT)
        ttk.Button(top, text="Pan", command=lambda: self.mode.set("pan")).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        ttk.Label(top, text="Render:").pack(side=tk.LEFT, padx=(0, 4))
        self.render_mode_var = tk.StringVar(value=DEFAULT_RENDER_MODE)
        ttk.OptionMenu(
            top,
            self.render_mode_var,
            DEFAULT_RENDER_MODE,
            *RENDER_MODES,
            command=lambda _=None: self._on_render_mode_change(),
        ).pack(side=tk.LEFT)
        ttk.Button(top, text="Preset View", command=self._apply_mode_preset).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)

        self.auto_iter = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="AutoIter", variable=self.auto_iter, command=self._render).pack(side=tk.LEFT)
        ttk.Button(top, text="Iter +", command=lambda: self._bump_iter(1.25)).pack(side=tk.LEFT, padx=(6, 2))
        ttk.Button(top, text="Iter −", command=lambda: self._bump_iter(1 / 1.25)).pack(side=tk.LEFT, padx=(2, 6))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(top, text="Save", command=self._save_dialog).pack(side=tk.LEFT)
        ttk.Button(top, text="Reset", command=self._reset).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(top, text="Colormap:").pack(side=tk.LEFT, padx=(0, 4))
        self.cmap_var = tk.StringVar(value=DEFAULT_CMAP)
        ttk.OptionMenu(top, self.cmap_var, DEFAULT_CMAP, *CMAPS, command=lambda _=None: self._render()).pack(side=tk.LEFT)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Label(top, text="Gamma:").pack(side=tk.LEFT, padx=(0, 4))
        self.gamma_var = tk.DoubleVar(value=1.25)
        ttk.Scale(
            top,
            from_=0.6,
            to=2.2,
            value=1.25,
            command=lambda _=None: self._render(),
            variable=self.gamma_var,
            length=120,
        ).pack(side=tk.LEFT)

        self.canvas = tk.Canvas(self.root, width=w, height=h, highlightthickness=0, bg="#ffffff", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(self.root, anchor="w", font=("Consolas", 10))
        self.status.pack(fill=tk.X)

        self.w, self.h = w, h
        self.view = View()
        self._imgtk: Optional[ImageTk.PhotoImage] = None
        self._rb_start: Optional[Tuple[int, int]] = None
        self._rb_rect_id: Optional[int] = None
        self._pan_start_px: Optional[Tuple[int, int]] = None
        self._pan_start_center: Optional[Tuple[float, float]] = None
        self._render_job: Optional[str] = None

        self.root.bind("<Configure>", self._on_resize)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(0.85, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(1 / 0.85, e.x, e.y))
        self.canvas.bind("<ButtonPress-1>", self._on_press_left)
        self.canvas.bind("<B1-Motion>", self._on_drag_left)
        self.canvas.bind("<ButtonRelease-1>", self._on_release_left)
        self.canvas.bind("<Button-3>", lambda e: self._reset())

        self.root.bind("<KeyPress-z>", lambda e: self._zoom_at(0.85, self.w / 2, self.h / 2))
        self.root.bind("<KeyPress-x>", lambda e: self._zoom_at(1 / 0.85, self.w / 2, self.h / 2))
        self.root.bind("<KeyPress-plus>", lambda e: self._bump_iter(1.25))
        self.root.bind("<KeyPress-equal>", lambda e: self._bump_iter(1.25))
        self.root.bind("<KeyPress-minus>", lambda e: self._bump_iter(1 / 1.25))
        self.root.bind("<KeyPress-c>", lambda e: self._cycle_cmap())
        self.root.bind("<KeyPress-i>", lambda e: self._toggle_autoiter())
        self.root.bind("<KeyPress-p>", lambda e: self._apply_mode_preset())
        self.root.bind("<KeyPress-r>", lambda e: self._reset())
        self.root.bind("<KeyPress-s>", lambda e: self._save_dialog())
        self.root.bind("<Escape>", lambda e: self.root.quit())

        self._render()

    def _status_text(self) -> str:
        mode = self.render_mode_var.get()
        return (
            f"Mode={mode}   "
            f"Center=({self.view.cx:.6f}, {self.view.cy:.6f})   "
            f"Scale={self.view.scale:.6e}   Iter={self.view.max_iter}   "
            f"AutoIter={'ON' if self.auto_iter.get() else 'OFF'}   "
            f"Size={self.w}x{self.h}"
        )

    def _set_status(self) -> None:
        self.status.config(text=self._status_text())

    def _on_render_mode_change(self) -> None:
        self._render()

    def _apply_mode_preset(self) -> None:
        cx, cy, scale = MODE_PRESETS.get(self.render_mode_var.get(), MODE_PRESETS[DEFAULT_RENDER_MODE])
        self.view.cx = cx
        self.view.cy = cy
        self.view.scale = scale
        self.view.max_iter = max(80, self.view.max_iter)
        self._render()

    def _on_resize(self, event) -> None:
        if event.widget is self.root:
            self.w, self.h = max(100, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())
            # Debounce resize renders so dragging the window does not queue hundreds of renders.
            if self._render_job is not None:
                self.root.after_cancel(self._render_job)
            self._render_job = self.root.after(120, self._render)

    def _cycle_cmap(self) -> None:
        idx = CMAPS.index(self.cmap_var.get())
        self.cmap_var.set(CMAPS[(idx + 1) % len(CMAPS)])
        self._render()

    def _toggle_autoiter(self) -> None:
        self.auto_iter.set(not self.auto_iter.get())
        self._render()

    def _on_press_left(self, event) -> None:
        if self.mode.get() == "pan":
            self._pan_start_px = (event.x, event.y)
            self._pan_start_center = (self.view.cx, self.view.cy)
        else:
            self._rb_start = (event.x, event.y)
            if self._rb_rect_id:
                self.canvas.delete(self._rb_rect_id)
                self._rb_rect_id = None

    def _on_drag_left(self, event) -> None:
        if self.mode.get() == "pan" and self._pan_start_px:
            sx, sy = self._pan_start_px
            dx, dy = event.x - sx, event.y - sy
            aspect = self.h / self.w
            span_x = 2 * self.view.scale / aspect
            span_y = 2 * self.view.scale
            dcx = -dx / self.w * span_x
            dcy = dy / self.h * span_y
            cx0, cy0 = self._pan_start_center or (self.view.cx, self.view.cy)
            self.view.cx = cx0 + dcx
            self.view.cy = cy0 + dcy
            self._render()
        elif self.mode.get() == "zoom" and self._rb_start:
            x0, y0 = self._rb_start
            x1, y1 = event.x, event.y
            if self._rb_rect_id:
                self.canvas.coords(self._rb_rect_id, x0, y0, x1, y1)
            else:
                self._rb_rect_id = self.canvas.create_rectangle(
                    x0, y0, x1, y1, outline="#00e0ff", width=2, dash=(4, 2)
                )

    def _on_release_left(self, event) -> None:
        if self.mode.get() == "pan":
            self._pan_start_px = self._pan_start_center = None
            return
        if not self._rb_start:
            return

        x0, y0 = self._rb_start
        x1, y1 = event.x, event.y
        self._rb_start = None
        if self._rb_rect_id:
            self.canvas.delete(self._rb_rect_id)
            self._rb_rect_id = None

        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:
            self._zoom_at(0.7, x1, y1)
            return

        left, right = sorted([x0, x1])
        bottom, top = sorted([y0, y1])
        aspect = self.h / self.w
        x_min = self.view.cx - self.view.scale / aspect
        x_max = self.view.cx + self.view.scale / aspect
        y_min = self.view.cy - self.view.scale
        y_max = self.view.cy + self.view.scale

        rx_min = x_min + (left / self.w) * (x_max - x_min)
        rx_max = x_min + (right / self.w) * (x_max - x_min)
        ry_min = y_min + ((self.h - top) / self.h) * (y_max - y_min)
        ry_max = y_min + ((self.h - bottom) / self.h) * (y_max - y_min)

        rect_w = rx_max - rx_min
        rect_h = ry_max - ry_min
        target_aspect = self.h / self.w
        cx = (rx_min + rx_max) / 2.0
        cy = (ry_min + ry_max) / 2.0

        if rect_h / rect_w > target_aspect:
            rect_w = rect_h / target_aspect
        else:
            rect_h = rect_w * target_aspect

        self.view.cx = cx
        self.view.cy = cy
        self.view.scale = max(MIN_SCALE, rect_h / 2.0)
        self._render()

    def _on_wheel(self, event) -> None:
        steps = int(event.delta / 120) if event.delta else 0
        if steps > 0:
            factor = 0.85 ** steps
        elif steps < 0:
            factor = (1 / 0.85) ** (-steps)
        else:
            return
        self._zoom_at(factor, event.x, event.y)

    def _zoom_at(self, factor: float, px: float, py: float) -> None:
        aspect = self.h / self.w
        x_min = self.view.cx - self.view.scale / aspect
        x_max = self.view.cx + self.view.scale / aspect
        y_min = self.view.cy - self.view.scale
        y_max = self.view.cy + self.view.scale
        target_x = x_min + (px / self.w) * (x_max - x_min)
        target_y = y_min + ((self.h - py) / self.h) * (y_max - y_min)
        self.view.cx = target_x + (self.view.cx - target_x) * factor
        self.view.cy = target_y + (self.view.cy - target_y) * factor
        self.view.scale = max(MIN_SCALE, self.view.scale * factor)
        self._render()

    def _bump_iter(self, mul: float) -> None:
        self.view.max_iter = max(20, int(self.view.max_iter * mul + 1))
        self._render()

    def _reset(self) -> None:
        self.view = View()
        self._render()

    def _render(self) -> None:
        self._render_job = None
        if self.auto_iter.get():
            self.view.max_iter = max(self.view.max_iter, auto_iters(self.view.scale))

        grid = self.view.grid(self.w, self.h)
        vals = render_values(grid, self.view.max_iter, self.render_mode_var.get())
        lo, hi = escaped_contrast(vals, self.view.max_iter)
        rgb = map_to_rgb(vals, self.cmap_var.get(), lo, hi, self.view.max_iter, gamma=self.gamma_var.get())

        img = Image.fromarray(rgb, mode="RGB")
        self._imgtk = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._imgtk, anchor="nw")

        if self._rb_rect_id:
            self.canvas.tag_raise(self._rb_rect_id)

        self._set_status()

    def _save_dialog(self) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_mode = self.render_mode_var.get().lower().replace(" ", "_").replace("|", "").replace(">", "gt").replace("<", "lt")
        safe_mode = "".join(ch for ch in safe_mode if ch.isalnum() or ch in "_-+=")[:50]
        default = f"collatz_{safe_mode}_{ts}.png"
        path = filedialog.asksaveasfilename(
            title="Save PNG",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=default,
        )
        if not path:
            return
        try:
            self._save_png(path)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _save_png(self, path: str) -> None:
        w, h = self.w * 2, self.h * 2
        grid = self.view.grid(w, h)
        save_iter = max(self.view.max_iter, int(self.view.max_iter * 1.25))
        vals = render_values(grid, save_iter, self.render_mode_var.get())
        lo, hi = escaped_contrast(vals, save_iter)
        rgb = map_to_rgb(vals, self.cmap_var.get(), lo, hi, save_iter, gamma=self.gamma_var.get())
        Image.fromarray(rgb, "RGB").save(path, optimize=True)
        print(f"Saved: {os.path.abspath(path)}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TkCollatz().run()
