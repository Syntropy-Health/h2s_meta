"""E2 — member-facing passport constants."""

PASSPORT_TITLE = "Pattern passport"
WITHHELD_MESSAGE = "Not enough data yet."
FALLBACK_TITLE = "Personal pattern"
PENDING: tuple[str, ...] = ("R10", "R11")  # citations (evidence) and RM patterns are not built yet

__all__ = ["FALLBACK_TITLE", "PASSPORT_TITLE", "PENDING", "WITHHELD_MESSAGE"]
