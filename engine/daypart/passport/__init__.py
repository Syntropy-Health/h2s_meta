"""passport — the member-facing pattern passport (CONTRACT slice E, E2; I6, D11).

One card per mapped metric node: a number only when the node is ready and its
text passes the LHM claim + narration guards; otherwise a withheld card.
"""

from daypart.passport.constants import (
    FALLBACK_TITLE,
    PASSPORT_TITLE,
    PENDING,
    WITHHELD_MESSAGE,
)
from daypart.passport.models import Flag, Passport
from daypart.passport.render import (
    PassportRenderer,
    flags_from_state,
    metric_key,
    passport_from_state,
)

__all__ = [
    "FALLBACK_TITLE",
    "PASSPORT_TITLE",
    "PENDING",
    "WITHHELD_MESSAGE",
    "Flag",
    "Passport",
    "PassportRenderer",
    "flags_from_state",
    "metric_key",
    "passport_from_state",
]
