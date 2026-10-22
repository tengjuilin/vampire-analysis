"""Generate the vampire-analysis logo, icon, and favicon as clean vector files.

The mark is a stack of closed contours that morph from an upright, tilted
oval (back, blue) into a bat-wing shape with a pointed left tip and a small
loop on the right (front, dark red), colored along a truncated ``turbo``
colormap. It evokes VAMPIRE's interpolation between shape modes.

The wordmark is set in Inter (SIL Open Font License) and converted to path
outlines with fontTools, so the generated SVGs render identically without
the font installed. The font is only needed to regenerate the files; get the
variable TTF from the Google Fonts repository::

    https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz,wght%5D.ttf

Usage::

    uv run python tools/make_logo.py --font /path/to/Inter[opsz,wght].ttf

or set ``INTER_FONT=/path/to/Inter[opsz,wght].ttf``. Without a font only the
icon files are written. Outputs go to ``docs/_static/``:

- ``vampire-icon.svg`` / ``vampire-icon.png``: square mark (favicon).
- ``vampire-logo.svg``: mark + wordmark for light backgrounds.
- ``vampire-logo-dark.svg``: mark + wordmark for dark backgrounds.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

OUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "_static"

# --------------------------------------------------------------------------
# Mark geometry (y axis points down, roughly the 363 x 412 original canvas).
# Each front control point is paired with a parameter on the back oval so the
# two closed curves have matching point counts and phase.
# --------------------------------------------------------------------------

# (x, y, s): front point and the fraction s of the way around the back oval
# (s = 0 left, 0.25 top, 0.5 right, 0.75 bottom, traversed clockwise).
FRONT = [
    (16, 182, 0.00),  # pointed left tip
    (72, 166, 0.06),
    (150, 164, 0.13),
    (222, 182, 0.20),
    (296, 236, 0.30),  # loop bottom (after the crossing)
    (338, 226, 0.38),
    (350, 194, 0.45),  # loop right
    (322, 170, 0.51),  # loop top
    (284, 178, 0.57),
    (242, 248, 0.66),  # back down through the crossing
    (200, 316, 0.74),  # bottom of the lower lobe
    (140, 282, 0.82),
    (64, 212, 0.92),
]

# Back oval: center, semi-axes, and tilt (degrees; positive leans top right).
OVAL_CENTER = (296.0, 206.0)
OVAL_AXES = (48.0, 194.0)
OVAL_TILT = 9.0

CMAP_RANGE = (0.11, 1.0)  # turbo, blue -> dark red


def _back_points() -> list[tuple[float, float]]:
    """Return back-oval points matched to the front control points."""
    cx, cy = OVAL_CENTER
    a, b = OVAL_AXES
    t = math.radians(OVAL_TILT)
    pts = []
    for _, _, s in FRONT:
        th = math.pi + 2 * math.pi * s
        x, y = a * math.cos(th), b * math.sin(th)
        # rotate so the top (negative y) leans right
        pts.append(
            (
                cx + x * math.cos(t) - y * math.sin(t),
                cy + x * math.sin(t) + y * math.cos(t),
            )
        )
    return pts


def contours(n: int) -> list[list[tuple[float, float]]]:
    """Return ``n`` control polygons interpolated from back (0) to front (1)."""
    back = _back_points()
    front = [(x, y) for x, y, _ in FRONT]
    out = []
    for i in range(n):
        u = i / (n - 1)
        out.append(
            [
                (bx + (fx - bx) * u, by + (fy - by) * u)
                for (bx, by), (fx, fy) in zip(back, front, strict=True)
            ]
        )
    return out


def colors(n: int) -> list[str]:
    """Return ``n`` hex colors along the truncated turbo colormap."""
    lo, hi = CMAP_RANGE
    cmap = mpl.colormaps["turbo"]
    return [mpl.colors.to_hex(cmap(lo + (hi - lo) * i / (n - 1))) for i in range(n)]


def bezier_segments(pts):
    """Convert a closed Catmull-Rom polygon to cubic Bezier segments."""
    k = len(pts)
    segs = []
    for i in range(k):
        p0, p1, p2, p3 = (pts[(i + j - 1) % k] for j in range(4))
        c1 = (p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6)
        c2 = (p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6)
        segs.append((p1, c1, c2, p2))
    return segs


def _sample(segs, m: int = 24):
    for p0, c1, c2, p3 in segs:
        for j in range(m):
            t = j / m
            a, b, c, d = (1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3
            yield (
                a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0],
                a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1],
            )


def _fmt(v: float, nd: int = 1) -> str:
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


class Mark:
    """The contour mark, fitted into a box.

    Parameters
    ----------
    n : int
        Number of contours.
    stroke : float
        Stroke width in source units (original canvas is ~412 tall).
    """

    def __init__(self, n: int, stroke: float):
        self.stroke = stroke
        self.colors = colors(n)
        self.segs = [bezier_segments(c) for c in contours(n)]
        pts = [p for s in self.segs for p in _sample(s)]
        h = stroke / 2
        self.x0 = min(p[0] for p in pts) - h
        self.y0 = min(p[1] for p in pts) - h
        self.w = max(p[0] for p in pts) + h - self.x0
        self.h = max(p[1] for p in pts) + h - self.y0

    def placed(self, x: float, y: float, height: float):
        """Return (segments, stroke width) scaled to ``height`` at (x, y)."""
        k = height / self.h

        def tf(p):
            return (x + (p[0] - self.x0) * k, y + (p[1] - self.y0) * k)

        segs = [[tuple(tf(p) for p in seg) for seg in s] for s in self.segs]
        return segs, self.stroke * k

    def svg(self, x: float, y: float, height: float, nd: int = 1) -> str:
        """Return an SVG ``<g>`` drawing the mark (back to front)."""
        segs, sw = self.placed(x, y, height)
        lines = [
            f'<g fill="none" stroke-width="{_fmt(sw, nd)}" '
            'stroke-linecap="round" stroke-linejoin="round">'
        ]
        for s, col in zip(segs, self.colors, strict=True):
            d = [f"M{_fmt(s[0][0][0], nd)} {_fmt(s[0][0][1], nd)}"]
            for _, c1, c2, p in s:
                d.append(
                    "C"
                    + " ".join(
                        f"{_fmt(v[0], nd)} {_fmt(v[1], nd)}" for v in (c1, c2, p)
                    )
                )
            lines.append(f'<path stroke="{col}" d="{"".join(d)}Z"/>')
        lines.append("</g>")
        return "\n".join(lines)


# Icon: fewer, bolder contours so the mark survives 32 px favicons.
ICON_MARK = Mark(n=7, stroke=20.0)
# Logo: closer to the original density.
LOGO_MARK = Mark(n=9, stroke=14.0)

ICON_SIZE = 512
ICON_PAD = 0.08  # fraction of the icon size left empty on the tighter axis


def icon_layout():
    """Return (x, y, height) placing the icon mark centered in the square."""
    m = ICON_MARK
    avail = ICON_SIZE * (1 - 2 * ICON_PAD)
    height = min(avail, avail * m.h / m.w)
    width = height * m.w / m.h
    return (ICON_SIZE - width) / 2, (ICON_SIZE - height) / 2, height


def write_icon_svg(path: Path) -> None:
    """Write the square icon SVG."""
    x, y, h = icon_layout()
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON_SIZE} {ICON_SIZE}"'
        ' role="img" aria-labelledby="t">\n'
        '<title id="t">vampire-analysis</title>\n'
        f"{ICON_MARK.svg(x, y, h)}\n</svg>\n"
    )
    path.write_text(svg)


def write_icon_png(path: Path, px: int = 256) -> None:
    """Rasterize the icon with matplotlib using the same Bezier geometry."""
    x, y, h = icon_layout()
    segs, sw = ICON_MARK.placed(x, y, h)
    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, ICON_SIZE)
    ax.set_ylim(ICON_SIZE, 0)
    ax.set_axis_off()
    lw = sw * px / ICON_SIZE * 72 / dpi
    for s, col in zip(segs, ICON_MARK.colors, strict=True):
        verts = [s[0][0]]
        codes = [MplPath.MOVETO]
        for _, c1, c2, p in s:
            verts += [c1, c2, p]
            codes += [MplPath.CURVE4] * 3
        verts.append(s[0][0])
        codes.append(MplPath.CLOSEPOLY)
        ax.add_patch(
            PathPatch(
                MplPath(verts, codes),
                fill=False,
                edgecolor=col,
                lw=lw,
                capstyle="round",
                joinstyle="round",
            )
        )
    fig.savefig(path, dpi=dpi, transparent=True, metadata={"Software": None})
    plt.close(fig)


# --------------------------------------------------------------------------
# Wordmark
# --------------------------------------------------------------------------

LIGHT_TEXT = ("#1f2937", "#4b5563")  # "vampire", "analysis" on white
DARK_TEXT = ("#e5e7eb", "#9ca3af")  # on dark backgrounds

WORDS = (("vampire", 700), ("analysis", 400))
OPSZ = 32  # Inter display optical size: tighter spacing for large text
LOGO_HEIGHT = 120.0  # viewBox height; the mark fills it
TEXT_SIZE = 80.0  # font size in viewBox units
WORD_GAP = 0.26  # space between words, in em
MARK_GAP = 0.42  # space between mark and text, in em
TRACKING = 0.0  # extra letter spacing, in em


def _instance(font_path: str, wght: float):
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont

    font = TTFont(font_path)
    return instantiateVariableFont(font, {"wght": wght, "opsz": OPSZ})


def _kerning(font) -> dict[tuple[str, str], int]:
    """Collect pair kerning (GPOS PairPos formats 1 and 2) from ``font``."""
    pairs: dict[tuple[str, str], int] = {}
    if "GPOS" not in font:
        return pairs
    gpos = font["GPOS"].table
    for lookup in gpos.LookupList.Lookup:
        for st in lookup.SubTable:
            if lookup.LookupType == 9:
                st = st.ExtSubTable
            if getattr(st, "LookupType", lookup.LookupType) not in (2,):
                continue
            first = st.Coverage.glyphs
            if st.Format == 1:
                for g1, ps in zip(first, st.PairSet, strict=True):
                    for rec in ps.PairValueRecord:
                        v = getattr(rec.Value1, "XAdvance", 0) or 0
                        pairs.setdefault((g1, rec.SecondGlyph), v)
            elif st.Format == 2:
                c1 = st.ClassDef1.classDefs
                c2 = st.ClassDef2.classDefs
                for g1 in first:
                    row = st.Class1Record[c1.get(g1, 0)].Class2Record
                    for g2, k2 in c2.items():
                        v = getattr(row[k2].Value1, "XAdvance", 0) or 0
                        if v:
                            pairs.setdefault((g1, g2), v)
    return pairs


def text_path(font, text: str, size: float, x: float, baseline: float):
    """Return (SVG path data, advance) for ``text`` set in ``font``."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen

    upm = font["head"].unitsPerEm
    k = size / upm
    cmap = font.getBestCmap()
    gs = font.getGlyphSet()
    hmtx = font["hmtx"]
    kern = _kerning(font)
    names = [cmap[ord(c)] for c in text]
    pen = SVGPathPen(gs, ntos=_fmt)
    cursor = 0.0
    for i, g in enumerate(names):
        gs[g].draw(TransformPen(pen, (k, 0, 0, -k, x + cursor * k, baseline)))
        cursor += hmtx[g][0] + TRACKING * upm
        if i + 1 < len(names):
            cursor += kern.get((g, names[i + 1]), 0)
    return pen.getCommands(), cursor * k


def write_logos(font_path: str) -> None:
    """Write the light and dark horizontal lockups."""
    fonts = [(_instance(font_path, w), word) for word, w in WORDS]
    ref = fonts[0][0]
    upm = ref["head"].unitsPerEm
    xh = ref["OS/2"].sxHeight / upm * TEXT_SIZE
    asc = ref["OS/2"].sCapHeight / upm * TEXT_SIZE

    mh = LOGO_HEIGHT
    mw = mh * LOGO_MARK.w / LOGO_MARK.h
    # Optically center the lowercase body (x-height plus a share of the
    # ascenders) on the mark's vertical center.
    baseline = mh / 2 + (xh + 0.35 * (asc - xh)) / 2
    x = mw + MARK_GAP * TEXT_SIZE
    paths = []
    for i, (font, word) in enumerate(fonts):
        d, adv = text_path(font, word, TEXT_SIZE, x, baseline)
        paths.append(d)
        x += adv + (WORD_GAP * TEXT_SIZE if i == 0 else 0)
    width = math.ceil(x + 2)
    mark = LOGO_MARK.svg(0, 0, mh)

    for name, (c1, c2) in (
        ("vampire-logo.svg", LIGHT_TEXT),
        ("vampire-logo-dark.svg", DARK_TEXT),
    ):
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {_fmt(mh)}"'
            ' role="img" aria-labelledby="t">\n'
            '<title id="t">vampire analysis</title>\n'
            f"{mark}\n"
            f'<path fill="{c1}" d="{paths[0]}"/>\n'
            f'<path fill="{c2}" d="{paths[1]}"/>\n'
            "</svg>\n"
        )
        (OUT_DIR / name).write_text(svg)


def main() -> None:
    """Parse arguments and write all logo files."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--font",
        default=os.environ.get("INTER_FONT"),
        help="path to Inter[opsz,wght].ttf (or set INTER_FONT)",
    )
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_icon_svg(OUT_DIR / "vampire-icon.svg")
    write_icon_png(OUT_DIR / "vampire-icon.png")
    if args.font:
        write_logos(args.font)
    else:
        print("No --font/INTER_FONT given: skipped the text logos.")


if __name__ == "__main__":
    main()
