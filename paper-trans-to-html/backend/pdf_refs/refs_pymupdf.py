"""PyMuPDF citation extraction strategy."""
from __future__ import annotations

from .refs import _build_standard_refs

__all__ = ["build_pymupdf_refs"]


def build_pymupdf_refs(doc: dict) -> dict:
    return _build_standard_refs(doc)
