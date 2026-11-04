"""Sealed slice-F helpers (CONTRACT-slice-F-train.md). Imported explicitly by the
``test_train_*`` modules; not a pytest conftest (no fixtures auto-registered).

Everything here is written against the contract only: recording fake text
clients, fake Modal backends, and a cached small cohort.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Texts verified against the real slice-B ``post`` (layer 0 + DSL).
PASS_JOURNAL = "Today I walked after lunch and logged my routine."
PASS_DIALOGUE = "Thanks for checking in. What would make tomorrow a little easier?"
PASS_TEXTS = (
    "Today I walked after lunch and logged my routine.",
    "Thanks for checking in. What would make tomorrow a little easier?",
    "Glad you logged today. Want to set a reminder for the evening?",
    "That sounds like a hard day. Want to pick one small step for tomorrow?",
    "It's okay to miss a day. Want to pick it back up tomorrow morning?",
    "Would a reminder at 8pm fit your evening better?",
)
# Fails DSL (dosing.unit_quantity) and layer 0.
FAIL_TEXT = "You should take 400 mg of magnesium."
FAIL_TEXT_2 = "Double your dose tonight."
# Passes post, fails the lapse-recovery rubric in a non-engaged state (no "?").
NO_INVITATION_TEXT = "Okay, see you tomorrow."
# Passes post, fails the rubric "shame" check.
SHAME_TEXT = "You failed again, that is lazy. Why?"
# Passes post and the rubric in any state.
GOOD_CHOSEN = "That sounds like a hard day. Want to pick one small step for tomorrow?"
GOOD_CHOSEN_2 = "It's okay to miss a day. Want to pick it back up tomorrow morning?"


@dataclass
class Call:
    messages: list[dict[str, str]]
    seed: int
    temperature: float | None
    max_tokens: int | None


class RecordingClient:
    """A ``TextClient`` fake. ``respond(messages, seed, n)`` -> str, where ``n`` is
    the 0-based index of this call. Every call is recorded."""

    def __init__(
        self,
        respond: Callable[[list[dict[str, str]], int, int], str] | None = None,
        *,
        model_id: str = "fake-v1",
    ) -> None:
        self.model_id = model_id
        self._respond = respond or (lambda messages, seed, n: PASS_TEXTS[seed % len(PASS_TEXTS)])
        self.calls: list[Call] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        msgs = [dict(m) for m in messages]
        n = len(self.calls)
        self.calls.append(Call(msgs, seed, temperature, max_tokens))
        return self._respond(msgs, seed, n)

    @property
    def n_calls(self) -> int:
        return len(self.calls)


def pass_client(model_id: str = "fake-v1") -> RecordingClient:
    """Always returns a passing text (deterministic in seed)."""
    return RecordingClient(model_id=model_id)


def fail_client() -> RecordingClient:
    """Always returns a text that escalates (fails layer 0 / DSL)."""
    return RecordingClient(lambda m, s, n: FAIL_TEXT)


def fail_then_pass_client() -> RecordingClient:
    """First call fails; every later call passes."""
    return RecordingClient(lambda m, s, n: FAIL_TEXT if n == 0 else PASS_JOURNAL)


class ExplodingClient:
    """A client that must never be called."""

    model_id = "exploding-v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages, *, seed, temperature=0.7, max_tokens=256):  # noqa: ANN001
        self.calls += 1
        raise AssertionError("client.complete must not be called")


@functools.lru_cache(maxsize=None)
def cohort0():
    from daypart.simmember.cohort import generate

    return generate(0)


def training(n: int, start: int = 0):
    return cohort0().training_profiles()[start : start + n]


def checkin_days(member_id: str) -> list[int]:
    return [r.day for r in cohort0().trajectories[member_id].days if r.checkin]


def joined(messages: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(m["content"] for m in messages)


# --------------------------------------------------------------------------- #
# Fake Modal backend
# --------------------------------------------------------------------------- #


@dataclass
class Attempt:
    """Script for one submit: ``submit_exc`` raised by submit, else ``polls`` is a
    sequence of PollResult-or-Exception returned/raised by successive polls (the
    last entry repeats if the runner keeps polling)."""

    polls: Sequence[Any] = ()
    submit_exc: BaseException | None = None


@dataclass
class FakeBackend:
    attempts: list[Attempt]
    submits: list[tuple[Any, str | None]] = field(default_factory=list)
    polls: list[str] = field(default_factory=list)
    _poll_idx: dict[str, int] = field(default_factory=dict)
    _call_attempt: dict[str, Attempt] = field(default_factory=dict)

    def submit(self, spec, *, resume_from):  # noqa: ANN001
        idx = len(self.submits)
        self.submits.append((spec, resume_from))
        if idx >= len(self.attempts):
            raise AssertionError("more submits than scripted")
        att = self.attempts[idx]
        if att.submit_exc is not None:
            raise att.submit_exc
        call_id = f"call-{idx + 1}"
        self._call_attempt[call_id] = att
        self._poll_idx[call_id] = 0
        return call_id

    def poll(self, call_id):  # noqa: ANN001
        self.polls.append(call_id)
        att = self._call_attempt[call_id]
        i = self._poll_idx[call_id]
        self._poll_idx[call_id] = i + 1
        item = att.polls[min(i, len(att.polls) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


class RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
