"""Run phase for slice F: realize narratives on Muse, then build SFT + DPO data.

``python -m daypart.train.run --out DIR`` does, in order:

1. realize journals + coach replies for the training cohort into a JSONL
   narrative cache (``DIR/narrative-v1.jsonl``; reruns are cache hits);
2. ``build_sft`` -> ``DIR/sft.jsonl``;
3. ``make_contexts`` + ``build_pairs`` -> ``DIR/dpo.jsonl``;
4. ``DIR/summary.json`` with counts, file hashes and the spend estimate.

Credentials come from the environment only. ``MUSE_API_KEY`` falls back to
``META_MODEL_API_KEY``; ``MUSE_BASE_URL`` defaults to the Meta model API.
Spend is capped twice: a hard call budget and a USD estimate (chars/4 token
approximation at the model's list price). Either cap stops the run cleanly;
what was cached so far is kept. ``--offline`` swaps in ``TemplateClient``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_MODEL = "muse-spark-1.3"
#: USD per 1M tokens (input, output), from the Meta model API price list (2026-11-04).
PRICES: dict[str, tuple[float, float]] = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.2": (1.25, 4.25),
    "muse-spark-1.1": (1.25, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.20),
    "muse-wonder-1.0": (4.00, 20.00),
}
_RETRYABLE_HTTP = {408, 409, 425, 429, 500, 502, 503, 504}


class BudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the call or spend cap."""


class BudgetedClient:
    """Wraps a ``TextClient``: counts calls, estimates spend, retries transient errors."""

    def __init__(
        self,
        inner: Any,
        *,
        max_calls: int,
        max_usd: float,
        price: tuple[float, float],
        retries: int = 4,
        backoff_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.price = price
        self.retries = retries
        self.backoff_s = backoff_s
        self.sleep = sleep
        self.calls = 0
        self.in_chars = 0
        self.out_chars = 0

    @property
    def est_usd(self) -> float:
        p_in, p_out = self.price
        return (self.in_chars / 4) * p_in / 1e6 + (self.out_chars / 4) * p_out / 1e6

    def complete(self, messages: Sequence[Mapping[str, str]], *, seed: int, **kw: Any) -> str:
        if self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget reached ({self.max_calls})")
        if self.est_usd >= self.max_usd:
            raise BudgetExceeded(f"spend cap reached (est ${self.est_usd:.2f} >= ${self.max_usd:.2f})")
        self.calls += 1
        self.in_chars += sum(len(m["content"]) for m in messages)
        for attempt in range(self.retries + 1):
            try:
                text = self.inner.complete(messages, seed=seed, **kw)
            except urllib.error.HTTPError as exc:
                if exc.code not in _RETRYABLE_HTTP or attempt == self.retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.retries:
                    raise
            else:
                self.out_chars += len(text)
                return text
            self.sleep(self.backoff_s * (2**attempt))
        raise AssertionError("unreachable")


def muse_env(env: Mapping[str, str]) -> dict[str, str]:
    out = dict(env)
    if not out.get("MUSE_API_KEY") and out.get("META_MODEL_API_KEY"):
        out["MUSE_API_KEY"] = out["META_MODEL_API_KEY"]
    out.setdefault("MUSE_BASE_URL", DEFAULT_BASE_URL)
    if not out["MUSE_BASE_URL"]:
        out["MUSE_BASE_URL"] = DEFAULT_BASE_URL
    return out


def _client(args: argparse.Namespace, env: Mapping[str, str]) -> Any:
    from daypart.simmember.narrative import MuseClient, TemplateClient

    if args.offline:
        return TemplateClient()
    return MuseClient(model_id=args.model, env=muse_env(env))


def run(args: argparse.Namespace, env: Mapping[str, str] | None = None, *, sleep=time.sleep) -> dict:
    from daypart.simmember.cohort import generate
    from daypart.simmember.narrative import NarrativeCache, realize
    from daypart.train.dpo_pairs import build_pairs, make_contexts
    from daypart.train.sft_dataset import build_sft, write_jsonl

    env = os.environ if env is None else env
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inner = _client(args, env)
    price = PRICES.get(inner.model_id, PRICES["muse-wonder-1.0"])  # unknown model: assume the dearest
    client = BudgetedClient(inner, max_calls=args.max_calls, max_usd=args.max_usd, price=price, sleep=sleep)

    cohort = generate(args.seed)
    members = cohort.training_profiles()
    if args.members is not None:
        members = members[: args.members]
    cache = NarrativeCache(out / "narrative-v1.jsonl")

    stopped: str | None = None
    failures: list[str] = []
    consecutive = 0
    for profile in members:
        try:
            realize(profile, cohort.trajectories[profile.member_id], client, cache)
            consecutive = 0
        except BudgetExceeded as exc:
            stopped = str(exc)
            break
        except Exception as exc:  # network/API error: skip member, stop on a streak
            failures.append(f"{profile.member_id}: {type(exc).__name__}: {exc}")
            consecutive += 1
            if consecutive >= args.max_consecutive_failures:
                stopped = f"{consecutive} consecutive member failures"
                break
        print(
            f"[narrative] {profile.member_id} calls={client.calls} est=${client.est_usd:.2f} rows={len(cache)}",
            file=sys.stderr,
        )

    sft_rows = build_sft(cohort, cache, refusal_ratio=args.refusal_ratio, seed=args.seed)
    sft_sha = write_jsonl(sft_rows, out / "sft.jsonl")

    pairs: list[dict] = []
    dpo_error: str | None = None
    if stopped is None or not stopped.startswith(("call budget", "spend cap")):
        contexts = make_contexts(cohort, cache, per_member=args.per_member, max_members=args.dpo_members)
        try:
            pairs = build_pairs(
                contexts,
                client,
                n_candidates=args.n_candidates,
                seed=args.seed,
                require_contrast=getattr(args, "require_contrast", True),
            )
        except BudgetExceeded as exc:
            dpo_error = str(exc)
        except Exception as exc:
            dpo_error = f"{type(exc).__name__}: {exc}"
    dpo_sha = write_jsonl(pairs, out / "dpo.jsonl")

    rows = cache.rows()
    summary = {
        "model_id": client.model_id,
        "offline": bool(args.offline),
        "members_requested": len(members),
        "narrative_rows": len(rows),
        "narrative_escalated": sum(r.decision == "escalate" for r in rows),
        "sft_rows": len(sft_rows),
        "sft_refusal_rows": sum(r["meta"]["source"] == "refusal" for r in sft_rows),
        "sft_sha256": sft_sha,
        "dpo_pairs": len(pairs),
        "dpo_sha256": dpo_sha,
        "calls": client.calls,
        "est_usd": round(client.est_usd, 4),
        "stopped": stopped,
        "dpo_error": dpo_error,
        "member_failures": failures[:20],
        "member_failure_count": len(failures),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m daypart.train.run", description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--members", type=int, default=None, help="first N training members (default: all)")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-calls", type=int, default=12000)
    p.add_argument("--max-usd", type=float, default=40.0)
    p.add_argument("--refusal-ratio", type=float, default=0.2)
    p.add_argument("--per-member", type=int, default=2)
    p.add_argument("--dpo-members", type=int, default=None)
    p.add_argument("--n-candidates", type=int, default=4)
    p.add_argument("--max-consecutive-failures", type=int, default=5)
    p.add_argument(
        "--require-contrast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DPO: keep only pairs whose rejected fails the DSL or rubric (default on)",
    )
    p.add_argument("--offline", action="store_true", help="TemplateClient instead of Muse (no network)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run(args)
    except Exception as exc:  # credentials missing etc.: name it, never print values
        print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["stopped"] is None and summary["dpo_error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
