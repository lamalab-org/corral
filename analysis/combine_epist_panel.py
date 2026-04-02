"""Combine epist.pdf and overall_pattern_bars_horizontal.pdf into one labeled figure.

Uses PyMuPDF's show_pdf_page() to embed source pages as *vector* graphics,
preserving full quality (no rasterization).

Layout:
  - Label A ("From trace to motif") top-left, above epist.pdf content
  - Label B ("Canonical graph motifs") top-center (starts at horizontal midpoint)
  - epist.pdf image
  - Label C ("Prevalence and weak adaptation across settings") centered between images
  - overall_pattern_bars_horizontal.pdf image

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
    / "overall_pattern_bars_horizontal.pdf"
)
OUTPUT_PDF = (
    REPO_ROOT
    / "analysis"
    / "results"
    / "figures"
    / "fig_epistemology"
    / "epst_combined.pdf"
)

CMU_SANS_SERIF = Path.home() / "Library" / "Fonts" / "cmunso.otf"

LABEL_FONTSIZE = 14
LABEL_FONTSIZE_SMALL = 12
LABEL_GAP = 14  # vertical space reserved for labels above each panel
PANEL_GAP = 2  # vertical gap between the two panels

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
    # Verify input figures exist
    for pdf_path in (EPIST_PDF, BARS_PDF):
        if not pdf_path.exists():
            raise FileNotFoundError(f"Required input figure not found: {pdf_path}")

    doc_top = fitz.open(str(EPIST_PDF))
    doc_bot = fitz.open(str(BARS_PDF))

    page_top = doc_top[0]
    page_bot = doc_bot[0]

    w_top, h_top = page_top.rect.width, page_top.rect.height
    w_bot, h_bot = page_bot.rect.width, page_bot.rect.height

    # Scale bottom panel so its width matches the top panel
    scale = w_top / w_bot
    h_bot_scaled = h_bot * scale

    # Total canvas dimensions
    total_width = w_top
    total_height = LABEL_GAP + h_top + PANEL_GAP + LABEL_GAP + h_bot_scaled

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=total_width, height=total_height)

    top_y0 = LABEL_GAP
    top_rect = fitz.Rect(0, top_y0, w_top, top_y0 + h_top)
    out_page.show_pdf_page(top_rect, doc_top, 0)

    # Labels A and B above the top panel
    label_y = LABEL_GAP - 2
    _insert_label(out_page, 9, label_y, "A.", " From trace to motif")
    _insert_label(
        out_page, total_width * 0.50, label_y, "B.", " Canonical graph motifs"
    )

    bot_y0 = top_y0 + h_top + PANEL_GAP + LABEL_GAP
    bot_rect = fitz.Rect(0, bot_y0, w_top, bot_y0 + h_bot_scaled)
    out_page.show_pdf_page(bot_rect, doc_bot, 0)

    # Label C above the bottom panel
    label_c_y = bot_y0 - 2
    _insert_label(
        out_page,
        9,
        label_c_y,
        "C.",
        " Prevalence and weak adaptation across settings",
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    out_doc.save(str(OUTPUT_PDF), garbage=3, deflate=True)
    out_doc.close()
    doc_top.close()
    doc_bot.close()
    logger.info(f"Saved combined figure to {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
