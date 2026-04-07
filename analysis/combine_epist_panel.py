"""Combine epist.pdf and overall_pattern_bars_horizontal_individual.pdf into one figure.

Uses PyMuPDF's show_pdf_page to embed source pages as vector graphics,
preserving full quality without rasterization.

Layout:
  The canvas size equals epist.pdf.  The bars figure is overlaid on top of it,
  centred horizontally, at a tuneable Y position.  Labels A, B and C are also
  overlaid at tuneable coordinates.

Usage:
    python analysis/combine_epist_panel.py
"""

from __future__ import annotations

from pathlib import Path

import fitz
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
EPIST_PDF = REPO_ROOT / "analysis" / "epist.pdf"
BARS_PDF = (
    REPO_ROOT
    / "analysis"
    / "results"
    / "figures"
    / "fig_epistemology"
    / "overall_pattern_bars_horizontal_individual.pdf"
)
OUTPUT_PDF = (
    REPO_ROOT
    / "analysis"
    / "results"
    / "figures"
    / "fig_epistemology"
    / "epst_combined.pdf"
)

CMU_SANS_SERIF = Path.home() / "Library" / "Fonts" / "cmunss.otf"

# Bars figure: centred in X; top edge at BARS_Y.
# Width is scaled to BARS_WIDTH_FRACTION of the canvas width.
BARS_Y = 365.0  # adjust this to move the bars figure up/down
BARS_WIDTH_FRACTION = 0.95  # fraction of canvas width for the bars figure

# Labels A and B share the same Y coordinate.
LABEL_AB_Y = 13.0  # adjust this to move A and B up/down
LABEL_A_X = 9.0  # adjust this to move A left/right
LABEL_B_X = 0.505  # fraction of canvas width (0.50 = start of second half)

# Label C has the same X as A but its own Y.
LABEL_C_X = LABEL_A_X  # same X as A
LABEL_C_Y = 362.0  # adjust this to move C up/down

LABEL_FONTSIZE = 16

_FONT_REGISTERED = False


def _register_font(page) -> str:
    """Register CMU Sans Serif on the page and return the fontname tag."""
    global _FONT_REGISTERED  # noqa: PLW0603
    tag = "F0"
    if not _FONT_REGISTERED:
        if not CMU_SANS_SERIF.exists():
            raise FileNotFoundError(
                f"CMU Sans Serif font not found at {CMU_SANS_SERIF}. "
                "Please install the CMU fonts."
            )
        page.insert_font(fontname=tag, fontfile=str(CMU_SANS_SERIF))
        _FONT_REGISTERED = True
    return tag


def _insert_label(page, x: float, y: float, label: str, description: str = "") -> None:
    """Insert a label followed by a description, all in CMU Sans Serif."""
    tag = _register_font(page)
    text = label + description
    page.insert_text(
        fitz.Point(x, y),
        text,
        fontsize=LABEL_FONTSIZE,
        fontname=tag,
        color=(0, 0, 0),
    )


def main() -> None:
    """Compose the combined epistemology panel PDF from its constituent figures.

    Embeds epist.pdf as a full-page background and overlays the individual
    pattern bar chart at the configured position, then inserts panel labels
    A, B, and C. Writes the result to OUTPUT_PDF.
    """
    for pdf_path in (EPIST_PDF, BARS_PDF):
        if not pdf_path.exists():
            raise FileNotFoundError(f"Required input figure not found: {pdf_path}")

    doc_epist = fitz.open(str(EPIST_PDF))
    doc_bars = fitz.open(str(BARS_PDF))

    page_epist = doc_epist[0]
    page_bars = doc_bars[0]

    canvas_w = page_epist.rect.width
    canvas_h = page_epist.rect.height

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=canvas_w, height=canvas_h)

    out_page.show_pdf_page(fitz.Rect(0, 0, canvas_w, canvas_h), doc_epist, 0)

    bars_w = canvas_w * BARS_WIDTH_FRACTION
    bars_h = page_bars.rect.height * (bars_w / page_bars.rect.width)
    bars_x0 = (canvas_w - bars_w) / 2.0
    bars_rect = fitz.Rect(bars_x0, BARS_Y, bars_x0 + bars_w, BARS_Y + bars_h)
    out_page.show_pdf_page(bars_rect, doc_bars, 0)

    _insert_label(out_page, LABEL_A_X, LABEL_AB_Y, "A")
    _insert_label(out_page, canvas_w * LABEL_B_X, LABEL_AB_Y, "B")
    _insert_label(out_page, LABEL_C_X, LABEL_C_Y, "C")

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(OUTPUT_PDF), garbage=3, deflate=True)
    out_doc.close()
    doc_epist.close()
    doc_bars.close()
    logger.info(f"Saved combined figure to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
