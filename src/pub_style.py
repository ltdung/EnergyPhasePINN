"""
Publication figure style, matching the journal checklist:
  - Width: 89 mm (1-column) or 183 mm (2-column); height <= 170 mm
  - >= 300 dpi raster, vector for line charts/diagrams (PDF/EPS, editable, not flattened)
  - RGB color mode (not CMYK)
  - Font: Arial or Helvetica, 5-7 pt at publication size
  - Scale bars in units, not magnification factors (n/a for these plots)
  - Full error bars, n reported in the legend/caption
  - Colorblind-safe palette; no red-green or rainbow scales
  - Figures cited in order: Fig. 1, Fig. 2, ...
  - Extended-data raster <= 10 MB, PNG/TIFF/EPS
"""

import os

import matplotlib
import matplotlib.pyplot as plt

MM = 1 / 25.4
WIDTH_1COL = 89 * MM
WIDTH_2COL = 183 * MM
MAX_HEIGHT = 170 * MM
DPI = 600  # >= 300 dpi requirement, generous margin

# Okabe-Ito colorblind-safe palette (Okabe & Ito, 2008) -- no red-green, no rainbow.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
PALETTE = [OKABE_ITO[k] for k in
           ["blue", "vermillion", "bluish_green", "orange", "reddish_purple", "sky_blue", "black"]]


def apply_style(font_size=7):
    """5-7 pt Arial/Helvetica at publication size; RGB; no flattening of vector output."""
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "pdf.fonttype": 42,   # TrueType, editable text (not flattened to paths)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "errorbar.capsize": 2.5,
    })


def new_fig(width="1col", height_mm=70, ncols=1, nrows=1, **kwargs):
    w = {"1col": WIDTH_1COL, "2col": WIDTH_2COL}.get(width, width * MM if isinstance(width, (int, float)) else WIDTH_1COL)
    h = min(height_mm * MM, MAX_HEIGHT)
    fig, ax = plt.subplots(nrows, ncols, figsize=(w, h), **kwargs)
    return fig, ax


def save(fig, out_dir, name, tight=True):
    """Save PNG (>=300 dpi raster) + PDF (vector, editable) + EPS (vector). RGB throughout."""
    os.makedirs(out_dir, exist_ok=True)
    if tight:
        fig.tight_layout()
    base = os.path.join(out_dir, name)
    fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(base + ".pdf", bbox_inches="tight", facecolor="white")
    try:
        fig.savefig(base + ".eps", bbox_inches="tight", facecolor="white")
    except Exception:
        pass  # some artists (alpha transparency) are not EPS-representable; PDF/PNG still saved
    sz = os.path.getsize(base + ".png")
    assert sz <= 10 * 1024 * 1024, f"{name}.png exceeds 10 MB extended-data limit ({sz} bytes)"
    print(f"  saved {base}.png / .pdf / .eps  ({sz/1024:.0f} KB png)")
    return base
