"""F1-F3 — latent-conditioned narrative prompts, text clients, and the narrative cache.

A :class:`TextClient` turns chat messages into text. :class:`MuseClient` is the
only network client in the engine (stdlib ``urllib``, OpenAI-compatible chat
completions); :class:`TemplateClient` is the deterministic offline stand-in
used by the I1 gate and tests.

:func:`realize` walks a member's check-in days, asks the client for the
member's journal and the coach's reply, runs every draft through the
membrane's :func:`~daypart.membrane.post.post` gate and caches the outcome in
an append-only JSONL :class:`NarrativeCache`, so reruns never re-spend.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import pydantic
from pydantic import ConfigDict, model_validator

from daypart.membrane import dsl
from daypart.membrane.post import post
from daypart.simmember.cohort import CURATED_PROFILES, Profile
from daypart.simmember.manifest import CuratedLeakError, canonical_json
from daypart.simmember.trajectory import DayRecord, Trajectory

NARRATIVE_VERSION = "narrative-v1"
KINDS = ("journal", "dialogue")

# --------------------------------------------------------------------------- #
# F1 — text clients
# --------------------------------------------------------------------------- #


@runtime_checkable
class TextClient(Protocol):
    model_id: str

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str: ...


class MissingCredentialsError(RuntimeError):
    """A required Muse environment variable is missing or empty (names only, never values)."""


class TruncatedResponseError(ValueError):
    """The reply ran out of ``max_tokens`` before any content (reasoning used them all). Retry with more."""

    def __init__(self, max_tokens: int) -> None:
        super().__init__(f"Muse response truncated at max_tokens={max_tokens} with no content")
        self.max_tokens = max_tokens


#: Muse reasoning models spend ``max_tokens`` on reasoning first; "minimal" leaves room for the reply.
DEFAULT_REASONING_EFFORT = "minimal"
_UNSET: Any = object()


class MuseClient:
    """OpenAI-compatible chat-completions client for Muse. Construction does no I/O.

    ``reasoning_effort`` (default ``MUSE_REASONING_EFFORT`` or ``"minimal"``; ``None`` omits it)
    is sent with every request. ``last_usage`` holds the last response's token counts, if reported.
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        env: Mapping[str, str] | None = None,
        reasoning_effort: str | None = _UNSET,
    ) -> None:
        source = os.environ if env is None else env
        required = ["MUSE_API_KEY", "MUSE_BASE_URL"]
        if model_id is None:
            required.append("MUSE_SPARK_MODEL")
        missing = [name for name in required if not source.get(name)]
        if missing:
            raise MissingCredentialsError(f"missing Muse credentials: {', '.join(missing)}")
        self._api_key = str(source["MUSE_API_KEY"])
        self.base_url = str(source["MUSE_BASE_URL"]).rstrip("/")
        self.model_id = model_id if model_id is not None else str(source["MUSE_SPARK_MODEL"])
        self.timeout_s = 60.0
        if reasoning_effort is _UNSET:
            reasoning_effort = source.get("MUSE_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT
        self.reasoning_effort: str | None = reasoning_effort
        self.last_usage: dict[str, int] | None = None

    def __repr__(self) -> str:
        return f"MuseClient(model_id={self.model_id!r}, base_url={self.base_url!r})"

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "seed": int(seed),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        body = json.dumps(request).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - fixed https base
            payload = json.loads(resp.read().decode("utf-8"))
        self.last_usage = _usage(payload)
        try:
            choice = payload["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Muse response has no choices[0].message.content") from exc
        if not content and choice.get("finish_reason") == "length":
            raise TruncatedResponseError(int(max_tokens))
        if not isinstance(content, str):
            raise ValueError("Muse response content is not a string")
        return content


def _usage(payload: Any) -> dict[str, int] | None:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None
    try:
        return {"prompt_tokens": int(usage["prompt_tokens"]), "completion_tokens": int(usage["completion_tokens"])}
    except (KeyError, TypeError, ValueError):
        return None


#: Member-voice check-in journals (all pass ``post``).
TEMPLATE_JOURNALS: tuple[str, ...] = (
    "Busy day at work, but I kept to my evening routine and logged it.",
    "Felt a bit flat today. I remembered in the morning and then the day got away from me.",
    "Good morning walk, and I stuck to my plan before breakfast.",
    "Honestly I forgot until late. The reminder came while I was out.",
    "Weekend plans threw me off a little, but I am back on track tonight.",
    "Tired today. Not much energy, though I still checked in.",
    "Things felt easier this week. The routine is starting to feel normal.",
    "Rough couple of days. I want to get back to my usual rhythm.",
    "I set my phone alarm earlier and it worked out well.",
    "Family dinner ran long and I almost skipped my check-in.",
)

#: Coach-voice replies (all pass ``post``; some carry no question).
TEMPLATE_DIALOGUES: tuple[str, ...] = (
    "Thanks for checking in. What part of the evening routine felt easiest today?",
    "That sounds like a full day. Would an earlier reminder fit your schedule better?",
    "Nice work keeping your plan going. Want to keep the same time tomorrow?",
    "It is okay that the day got away from you. What usually helps you get back on track?",
    "Thanks for being honest about a tough stretch. Small steps count, and tomorrow is a fresh start.",
    "Good to hear the alarm helped. Keep using what works for you.",
    "Low energy days happen. Would a shorter routine feel more doable this week?",
    "Busy weekends are common. Shall we plan one simple step for Saturday?",
)

#: Refusal-with-redirection replies (all pass ``post``; mention a clinician, ask a question).
TEMPLATE_REFUSALS: tuple[str, ...] = (
    "That is a question for your clinician, who knows your full history. "
    "Would it help to write it down for your next visit?",
    "Amounts and changes are outside what I can help with. Your clinician or pharmacist is the "
    "right person to ask. Want a reminder to contact them this week?",
    "I am not able to answer that one, but your clinician can. "
    "Would you like help preparing a short list of questions for them?",
    "That decision belongs with your clinician. In the meantime, would it help to keep notes "
    "on how you are feeling each day?",
    "I can help with routines and reminders, but questions like this need your clinician. "
    "Should we set a reminder to call their office?",
)

_JOURNAL_MARKER = "You write the member's own short daily check-in journal"


def _digest_int(*parts: Any) -> int:
    """Deterministic non-negative int from canonical JSON (no hash-seed dependence)."""
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


class TemplateClient:
    """Deterministic offline :class:`TextClient`: output depends only on ``(messages, seed)``."""

    def __init__(self, model_id: str = "template-v1") -> None:
        self.model_id = model_id

    def __repr__(self) -> str:
        return f"TemplateClient(model_id={self.model_id!r})"

    @staticmethod
    def _pool(messages: Sequence[Mapping[str, str]]) -> tuple[str, ...]:
        system = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
        if _JOURNAL_MARKER in system:
            return TEMPLATE_JOURNALS
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if users and not dsl.check(users[-1]).passed:
            return TEMPLATE_REFUSALS
        return TEMPLATE_DIALOGUES

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        seed: int,
        temperature: float = 0.7,
        max_tokens: int = 256,
    ) -> str:
        pool = self._pool(messages)
        msgs = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))} for m in messages]
        return pool[_digest_int(msgs, int(seed)) % len(pool)]


# --------------------------------------------------------------------------- #
# F2 — prompts
# --------------------------------------------------------------------------- #

_CURATED_LATENTS = frozenset(p.latents for p in CURATED_PROFILES)
_CURATED_IDS = frozenset(p.member_id for p in CURATED_PROFILES)


def is_curated_profile(profile: Profile) -> bool:
    return (
        profile.cohort == "curated"
        or str(profile.member_id).startswith("cur-")
        or profile.member_id in _CURATED_IDS
        or profile.latents in _CURATED_LATENTS
    )


def _assert_not_curated(profile: Profile) -> None:
    if is_curated_profile(profile):
        raise CuratedLeakError("curated (held-out) profile must not be realized for training (I1)")


def _level(value: str) -> str:
    return value.replace("_", " ")


def _context_lines(profile: Profile, record: DayRecord) -> str:
    lat = profile.latents
    return (
        f"Member background: motivation {_level(lat.motivation)}; "
        f"main barrier {_level(lat.barrier_type)}; "
        f"health literacy {_level(lat.literacy)}; "
        f"engagement pattern {_level(lat.engagement_pattern)}.\n"
        f"Day {record.day} of the program. Current state: {record.state}. "
        f"Kept to the routine today: {'yes' if record.adhered else 'no'}."
    )


def prompt_hash_for(kind: str, messages: Sequence[Mapping[str, str]]) -> str:
    payload = {"version": NARRATIVE_VERSION, "kind": kind, "messages": [dict(m) for m in messages]}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class NarrativePrompt(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    member_id: str
    day: int
    kind: Literal["journal", "dialogue"]
    messages: tuple[dict[str, str], ...]
    prompt_hash: str


def build_prompt(
    profile: Profile, record: DayRecord, kind: str, *, journal: str | None = None
) -> NarrativePrompt:
    if kind not in KINDS:
        raise ValueError(f"unknown narrative kind {kind!r}; expected one of {KINDS}")
    _assert_not_curated(profile)
    context = _context_lines(profile, record)
    if kind == "journal":
        if journal is not None:
            raise ValueError("journal must be None for kind='journal'")
        messages = (
            {
                "role": "system",
                "content": (
                    f"{_JOURNAL_MARKER} for a simulated adherence-coaching program. "
                    "Write in the first person, one to three plain sentences, about the day's "
                    "routine and how it went. No names, no health conditions, no medical details."
                ),
            },
            {"role": "user", "content": f"{context}\nWrite today's check-in journal."},
        )
    else:
        if not isinstance(journal, str) or not journal.strip():
            raise ValueError("kind='dialogue' requires a non-empty journal")
        messages = (
            {
                "role": "system",
                "content": (
                    "You are a supportive adherence coach. Reply to the member's check-in with "
                    "one short, warm message. Stay within routines, reminders and motivation; "
                    "never give medical advice."
                ),
            },
            {
                "role": "user",
                "content": f"{context}\nMember check-in journal:\n{journal}\nWrite the coach's single reply.",
            },
        )
    return NarrativePrompt(
        member_id=profile.member_id,
        day=record.day,
        kind=kind,  # type: ignore[arg-type]
        messages=messages,
        prompt_hash=prompt_hash_for(kind, messages),
    )


# --------------------------------------------------------------------------- #
# F3 — cache + realization
# --------------------------------------------------------------------------- #


class NarrativeRow(pydantic.BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    member_id: str
    day: int
    kind: Literal["journal", "dialogue"]
    prompt_hash: str
    model_id: str
    decision: Literal["pass", "escalate"]
    text: str | None
    trace: dict

    @model_validator(mode="after")
    def _text_iff_pass(self) -> "NarrativeRow":
        if self.decision == "pass" and not isinstance(self.text, str):
            raise ValueError("decision 'pass' requires text")
        if self.decision == "escalate" and self.text is not None:
            raise ValueError("decision 'escalate' requires text=None")
        return self


class NarrativeCache:
    """Append-only JSONL cache keyed by ``(member_id, day, prompt_hash)``; first write wins."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self._rows: list[NarrativeRow] = []
        self._index: dict[tuple[str, int, str], NarrativeRow] = {}
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    row = NarrativeRow.model_validate(json.loads(line))
                except (json.JSONDecodeError, pydantic.ValidationError, TypeError) as exc:
                    raise ValueError(f"{self.path}:{lineno}: malformed narrative cache line") from exc
                key = (row.member_id, row.day, row.prompt_hash)
                if key not in self._index:
                    self._index[key] = row
                    self._rows.append(row)

    def get(self, member_id: str, day: int, prompt_hash: str) -> NarrativeRow | None:
        return self._index.get((member_id, day, prompt_hash))

    def put(self, row: NarrativeRow) -> None:
        key = (row.member_id, row.day, row.prompt_hash)
        if key in self._index:
            return
        line = canonical_json(row.model_dump(mode="json")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line)
        self._index[key] = row
        self._rows.append(row)

    def rows(self) -> list[NarrativeRow]:
        return list(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


def narrative_seed(member_id: str, day: int, kind: str) -> int:
    """Deterministic client seed from ``(member_id, day, kind)`` only (31-bit)."""
    return _digest_int("narrative-seed", member_id, int(day), kind) % (2**31 - 2)


def _realize_one(prompt: NarrativePrompt, client: TextClient, cache: NarrativeCache) -> NarrativeRow:
    cached = cache.get(prompt.member_id, prompt.day, prompt.prompt_hash)
    if cached is not None:
        return cached
    messages = [dict(m) for m in prompt.messages]
    s = narrative_seed(prompt.member_id, prompt.day, prompt.kind)
    draft = client.complete(messages, seed=s)
    result = post(draft, regenerate=lambda: client.complete(messages, seed=s + 1))
    row = NarrativeRow(
        member_id=prompt.member_id,
        day=prompt.day,
        kind=prompt.kind,
        prompt_hash=prompt.prompt_hash,
        model_id=client.model_id,
        decision=result.decision,
        text=result.text,
        trace=result.trace,
    )
    # Store/return the JSON-normalized form so fresh rows equal reloaded rows.
    row = NarrativeRow.model_validate(json.loads(canonical_json(row.model_dump(mode="json"))))
    cache.put(row)
    return row


def realize(
    profile: Profile,
    trajectory: Trajectory,
    client: TextClient,
    cache: NarrativeCache,
    *,
    days: Iterable[int] | None = None,
) -> list[NarrativeRow]:
    _assert_not_curated(profile)
    if trajectory.member_id != profile.member_id:
        raise ValueError(
            f"trajectory member {trajectory.member_id!r} does not match profile {profile.member_id!r}"
        )
    wanted = None if days is None else {int(d) for d in days}
    out: list[NarrativeRow] = []
    for record in sorted(trajectory.days, key=lambda r: r.day):
        if not record.checkin or (wanted is not None and record.day not in wanted):
            continue
        journal_row = _realize_one(build_prompt(profile, record, "journal"), client, cache)
        out.append(journal_row)
        if journal_row.decision != "pass" or not journal_row.text or not journal_row.text.strip():
            continue
        dialogue = build_prompt(profile, record, "dialogue", journal=journal_row.text)
        out.append(_realize_one(dialogue, client, cache))
    return out


__all__ = [
    "KINDS",
    "NARRATIVE_VERSION",
    "TEMPLATE_DIALOGUES",
    "TEMPLATE_JOURNALS",
    "TEMPLATE_REFUSALS",
    "MissingCredentialsError",
    "MuseClient",
    "NarrativeCache",
    "NarrativePrompt",
    "NarrativeRow",
    "TemplateClient",
    "TextClient",
    "build_prompt",
    "is_curated_profile",
    "narrative_seed",
    "prompt_hash_for",
    "realize",
]
