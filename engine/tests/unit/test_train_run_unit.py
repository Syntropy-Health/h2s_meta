"""Unit tests for the slice-F run entrypoint (daypart.train.run)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from daypart.simmember.narrative import MissingCredentialsError, TemplateClient, TruncatedResponseError
from daypart.train import run as R


def _args(tmp_path, *extra):
    return R.parse_args(["--out", str(tmp_path), *extra])


def test_offline_run_writes_datasets(tmp_path):
    s = R.run(_args(tmp_path, "--offline", "--members", "3"), env={})
    assert s["stopped"] is None and s["dpo_error"] is None
    assert s["sft_rows"] > 0 and s["dpo_pairs"] > 0 and s["calls"] > 0
    for name in ("narrative-v1.jsonl", "sft.jsonl", "dpo.jsonl", "summary.json"):
        assert (tmp_path / name).exists()
    assert "cur-" not in (tmp_path / "sft.jsonl").read_text()


def test_rerun_is_cache_hit_for_narratives(tmp_path):
    first = R.run(_args(tmp_path, "--offline", "--members", "2"), env={})
    second = R.run(_args(tmp_path, "--offline", "--members", "2"), env={})
    assert second["narrative_rows"] == first["narrative_rows"]
    assert second["sft_sha256"] == first["sft_sha256"]
    assert second["calls"] == first["calls"] - first["narrative_rows"]  # only DPO calls repeat


def test_call_budget_stops_cleanly(tmp_path):
    s = R.run(_args(tmp_path, "--offline", "--members", "5", "--max-calls", "3"), env={})
    assert s["stopped"].startswith("call budget") and s["calls"] == 3
    assert (tmp_path / "summary.json").exists()


def test_muse_env_maps_meta_key_and_default_base():
    env = R.muse_env({"META_MODEL_API_KEY": "k"})
    assert env["MUSE_API_KEY"] == "k" and env["MUSE_BASE_URL"] == R.DEFAULT_BASE_URL
    assert R.muse_env({"MUSE_API_KEY": "a", "META_MODEL_API_KEY": "b"})["MUSE_API_KEY"] == "a"


def test_missing_key_fails_without_value_leak(tmp_path, capsys):
    rc = R.main(["--out", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 2 and "MUSE_API_KEY" in err and MissingCredentialsError.__name__ in err


class _Flaky:
    model_id = "muse-spark-1.3"

    def __init__(self, fails):
        self.fails = list(fails)

    def complete(self, messages, *, seed, **kw):
        if self.fails:
            code = self.fails.pop(0)
            raise urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(b""))
        return "ok?"


def test_budgeted_client_retries_transient_then_succeeds():
    sleeps = []
    c = R.BudgetedClient(_Flaky([429, 503]), max_calls=5, max_usd=1, price=(1, 1), sleep=sleeps.append)
    assert c.complete([{"role": "user", "content": "hi"}], seed=0) == "ok?"
    assert c.calls == 1 and sleeps == [2.0, 4.0]


def test_budgeted_client_does_not_retry_auth_errors():
    c = R.BudgetedClient(_Flaky([401]), max_calls=5, max_usd=1, price=(1, 1), sleep=lambda s: None)
    with pytest.raises(urllib.error.HTTPError):
        c.complete([{"role": "user", "content": "hi"}], seed=0)


def test_spend_cap():
    c = R.BudgetedClient(TemplateClient(), max_calls=100, max_usd=0.0, price=(1, 1), sleep=lambda s: None)
    with pytest.raises(R.BudgetExceeded):
        c.complete([{"role": "user", "content": "hi"}], seed=0)


def test_default_model_is_priced():
    assert R.DEFAULT_MODEL in R.PRICES
    assert json.dumps(R.PRICES)


# ---------------------------------------------------------------- HAC-28: reasoning models


class _Truncating:
    """Reasoning model: truncates (content null, finish=length) below ``need`` max_tokens."""

    model_id = "muse-spark-1.3"

    def __init__(self, need):
        self.need = need
        self.seen = []
        self.last_usage = None

    def complete(self, messages, *, seed, max_tokens=256, **kw):
        self.seen.append(max_tokens)
        self.last_usage = {"prompt_tokens": 1000, "completion_tokens": max_tokens}
        if max_tokens < self.need:
            raise TruncatedResponseError(max_tokens)
        return "ok?"


def test_budgeted_client_escalates_max_tokens_on_truncation():
    inner = _Truncating(need=1024)
    c = R.BudgetedClient(inner, max_calls=5, max_usd=10, price=(1, 1), sleep=lambda s: None)
    assert c.complete([{"role": "user", "content": "x"}], seed=1) == "ok?"
    assert inner.seen == [256, 512, 1024]
    assert c.calls == 1


def test_budgeted_client_truncation_gives_up_typed():
    c = R.BudgetedClient(_Truncating(need=10**6), max_calls=5, max_usd=10, price=(1, 1), sleep=lambda s: None)
    with pytest.raises(TruncatedResponseError):
        c.complete([{"role": "user", "content": "x"}], seed=1)


def test_spend_counts_reported_tokens_including_reasoning():
    inner = _Truncating(need=512)
    c = R.BudgetedClient(inner, max_calls=5, max_usd=10, price=(1.0, 2.0), sleep=lambda s: None)
    c.complete([{"role": "user", "content": "x"}], seed=1)
    # Two billed attempts: (1000 in + 256 out) + (1000 in + 512 out), not chars/4 of "ok?".
    assert c.in_tokens == 2000 and c.out_tokens == 768
    assert c.est_usd == pytest.approx((2000 * 1.0 + 768 * 2.0) / 1e6)


def test_spend_falls_back_to_chars_without_usage():
    c = R.BudgetedClient(TemplateClient(), max_calls=5, max_usd=10, price=(1.0, 1.0), sleep=lambda s: None)
    out = c.complete([{"role": "user", "content": "abcd" * 10}], seed=1)
    assert c.in_tokens == 10 and c.out_tokens == pytest.approx(len(out) / 4)


def test_run_passes_reasoning_effort(tmp_path):
    args = _args(tmp_path, "--reasoning-effort", "low")
    client = R._client(args, {"MUSE_API_KEY": "k"})
    assert client.reasoning_effort == "low"
    assert R._client(_args(tmp_path), {"MUSE_API_KEY": "k"}).reasoning_effort == "minimal"
