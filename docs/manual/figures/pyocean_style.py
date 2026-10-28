"""
Shared figure style of the pyOCEAN manual.

Every figure script in ``docs/manual/figures/`` imports this module so that all figures read as one set:

- one sans typeface (Helvetica Neue on macOS, DejaVu Sans elsewhere) with matching sans math;
- one width (7.5 in at 200 dpi ≈ the content column of the site and the PDF) and one type scale;
- a fixed colour for each model and each pyOCEAN configuration, the same in every chart (``MODELS``);
- one cmocean colormap per physical variable (``CMAPS``), so temperature always looks like temperature;
- no figure titles: the Markdown caption says what the figure shows; panels get letters (``panel_labels``)
  and, at most, a short panel heading.

Usage::

    import pyocean_style as ps
    ps.use()
    fig, axs = ps.figure(ncols=2, aspect=0.45)
    axs[0].plot(x, y, **ps.model("pyocean_zstar"))
    ps.panel_labels(axs)
    ps.save(fig, "vertical-ale", "coordinates_section")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cmocean  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

IMG = Path(__file__).resolve().parents[1] / "img"

WIDTH = 7.5          # in: full content width
DPI = 200

# ── colours ──────────────────────────────────────────────────────────────────────────────────────────────
# Neutrals carry a slight blue bias (the accent family) instead of pure grey.
INK = "#1d2733"       # text, axes, primary lines
MUTED = "#5f6b78"     # secondary text, reference lines
FAINT = "#c9d1da"     # grid, separators
PAPER = "#ffffff"
ACCENT = "#1f5fa8"    # the single highlight colour (pyOCEAN default configuration)
WARN = "#c2410c"      # "error", "unstable", "bad" annotations

# Physical materials in schematics and sections
WATER = "#d6e9f5"     # water fill
WATER_DEEP = "#8fbcd9"
BED = "#c8b89a"       # sea bed / sediment
BED_EDGE = "#8a7a5c"
LAND = "#e3dccd"      # dry land
DRY = "#efe6d2"       # dried cells (wetting–drying)

# Models and pyOCEAN configurations: one colour + marker + line style each, used in every chart.
# Colours from the Okabe–Ito colour-blind-safe set, assigned once.
MODELS: dict[str, dict] = {
    "pyocean":        dict(color="#1f5fa8", marker="o", ls="-",  label="pyOCEAN"),
    "pyocean_zstar":  dict(color="#1f5fa8", marker="o", ls="-",  label="pyOCEAN z*"),
    "pyocean_sigma":  dict(color="#56b4e9", marker="s", ls="-",  label="pyOCEAN σ"),
    "pyocean_hybrid": dict(color="#0b2e59", marker="D", ls="-",  label="pyOCEAN hybrid"),
    "mom6":           dict(color="#e69f00", marker="^", ls="--", label="MOM6"),
    "mom6_rho":       dict(color="#e69f00", marker="v", ls=":",  label="MOM6 ρ"),
    "mom6_hycom1":    dict(color="#e69f00", marker="v", ls=":",  label="MOM6 HYCOM1"),
    "croco":          dict(color="#009e73", marker="P", ls="--", label="CROCO"),
    "mitgcm":         dict(color="#d55e00", marker="X", ls="--", label="MITgcm"),
    "hycom":          dict(color="#cc79a7", marker="*", ls="--", label="HYCOM"),
    "tpxo":           dict(color="#7f7f7f", marker="h", ls="--", label="TPXO10"),
    "obs":            dict(color=INK,       marker=".", ls="none", label="observations"),
    "exact":          dict(color=INK,       marker="",  ls=":",  label="exact"),
}

# A neutral qualitative cycle for curves that are not models (parameter sweeps, constituents, …).
CYCLE = ["#1f5fa8", "#e69f00", "#009e73", "#d55e00", "#cc79a7", "#56b4e9", "#0b2e59", "#7f7f7f"]

# One colormap per variable (cmocean, Thyng et al. 2016).
CMAPS = {
    "temperature": cmocean.cm.thermal,
    "density": cmocean.cm.dense,
    "velocity": cmocean.cm.balance,       # signed: centre on zero
    "anomaly": cmocean.cm.balance,
    "speed": cmocean.cm.speed,
    "bathymetry": cmocean.cm.deep,
    "amplitude": cmocean.cm.amp,
    "phase": cmocean.cm.phase,
    "error": cmocean.cm.matter,
    "fraction": cmocean.cm.ice,
    "ssh": cmocean.cm.balance,
}


def _font_family() -> list[str]:
    have = {f.name for f in font_manager.fontManager.ttflist}
    return [f for f in ("Helvetica Neue", "DejaVu Sans") if f in have] or ["sans-serif"]


def use() -> None:
    """Apply the manual's rcParams (call once at the top of every figure script)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": _font_family(),
        "mathtext.fontset": "stixsans",
        "font.size": 10.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 6,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "grid.color": FAINT,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "patch.linewidth": 0.8,
        "image.cmap": "cmo.thermal",
        "figure.dpi": 100,
        "figure.facecolor": PAPER,
        "savefig.dpi": DPI,
        "savefig.facecolor": PAPER,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "text.color": INK,
    })


def figure(nrows: int = 1, ncols: int = 1, aspect: float = 0.5, width: float = WIDTH, **kw):
    """Figure of the standard width; ``aspect`` = height / width of the whole figure."""
    kw.setdefault("constrained_layout", True)
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, width * aspect), **kw)
    return fig, axs


def model(key: str, markers: bool = True, **override) -> dict:
    """Plot keywords (color, marker, linestyle, label) of a model / configuration."""
    m = dict(MODELS[key])
    kw = dict(color=m["color"], linestyle=m["ls"], label=m["label"])
    if markers and m["marker"]:
        kw["marker"] = m["marker"]
    kw.update(override)
    return kw


def panel_labels(axs, x: float = -0.02, y: float = 1.02, start: str = "a") -> None:
    """Bold lower-case panel letters at the top-left corner of each axes."""
    for i, ax in enumerate(np.atleast_1d(axs).ravel()):
        ax.text(x, y, chr(ord(start) + i), transform=ax.transAxes, fontsize=11.5, fontweight="bold",
                va="bottom", ha="right", color=INK)


def section_axes(ax, xlabel: str = "x [km]", ylabel: str = "z [m]") -> None:
    """Axes for vertical sections: no grid, labels set, depth increasing downwards as data are negative."""
    ax.grid(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def fill_bed(ax, x, zb, zmin=None, **kw) -> None:
    """Shade the sea bed below the depth profile ``zb`` (negative, metres)."""
    zb = np.asarray(zb, float)
    lo = np.nanmin(zb) * 1.05 if zmin is None else zmin
    ax.fill_between(x, zb, lo, color=kw.pop("color", BED), lw=0, zorder=kw.pop("zorder", 3), **kw)
    ax.plot(x, zb, color=BED_EDGE, lw=0.9, zorder=4)


def colorbar(fig, mappable, ax, label: str, **kw):
    """Slim colour bar with the standard label style."""
    cb = fig.colorbar(mappable, ax=ax, shrink=kw.pop("shrink", 0.9), aspect=kw.pop("aspect", 28),
                      pad=kw.pop("pad", 0.02), **kw)
    cb.set_label(label)
    cb.outline.set_linewidth(0.5)
    cb.outline.set_edgecolor(MUTED)
    cb.ax.tick_params(labelsize=9)
    return cb


def note(ax, x, y, text: str, **kw):
    """Small annotation in the muted colour (data coordinates unless ``transform`` given)."""
    kw.setdefault("fontsize", 9)
    kw.setdefault("color", MUTED)
    return ax.text(x, y, text, **kw)


def save(fig, chapter: str, name: str) -> Path:
    """Save ``img/<chapter>/<name>.png`` at the manual's resolution and close the figure."""
    out = IMG / chapter
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path.relative_to(IMG.parent))
    return path


def freeze_layout(fig) -> None:
    """Run the constrained layout once and keep it fixed, so animation frames do not jitter."""
    fig.canvas.draw()
    fig.set_layout_engine("none")


def save_animation(fig, update, n_frames: int, stem: Path, fps: int = 12, dpi: int = 110,
                   gif_fps: int = 8, gif_width: int = 960) -> tuple[Path, Path]:
    """
    Render ``update(n)`` for n = 0 … n_frames − 1 to ``stem``.mp4 (H.264, every frame) and ``stem``.gif (from the
    MP4 with a per-file palette, ``gif_fps`` frames per second, ``gif_width`` pixels wide); close the figure.
    Needs ffmpeg on the PATH.
    """
    import subprocess

    from matplotlib.animation import FFMpegWriter, FuncAnimation

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    mp4, gif = stem.parent / f"{stem.name}.mp4", stem.parent / f"{stem.name}.gif"     # names may hold dots (nu0.1)
    writer = FFMpegWriter(fps=fps, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "20",
                                      "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    FuncAnimation(fig, update, frames=n_frames, blit=False).save(mp4, writer=writer, dpi=dpi)
    plt.close(fig)
    graph = (f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
             "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-vf", graph, str(gif)], check=True)
    print("wrote", mp4, "and", gif.name)
    return mp4, gif


def swatches() -> None:
    """Reference sheet of the palette and colormaps (img/style/style_sheet.png)."""
    use()
    fig, (a, b) = figure(1, 2, aspect=0.42, gridspec_kw=dict(width_ratios=[1, 1.3]))
    for i, (k, m) in enumerate(MODELS.items()):
        a.plot([0, 1], [-i, -i], color=m["color"], ls=m["ls"] if m["ls"] != "none" else "-",
               marker=m["marker"] or None, markevery=[1])
        a.text(1.15, -i, f"{m['label']}  ({k})", va="center", fontsize=9)
    a.set_xlim(-0.1, 3.4)
    a.axis("off")
    grad = np.linspace(0, 1, 256)[None, :]
    for i, (k, cm) in enumerate(CMAPS.items()):
        b.imshow(grad, extent=(0, 1, -i - 0.4, -i + 0.4), cmap=cm, aspect="auto")
        b.text(1.04, -i, k, va="center", fontsize=9)
    b.set_xlim(0, 1.5)
    b.set_ylim(-len(CMAPS) + 0.4, 0.6)
    b.axis("off")
    save(fig, "style", "style_sheet")


if __name__ == "__main__":
    swatches()
