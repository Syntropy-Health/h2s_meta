"""Unit tests — HAC-28: chat-template reply extraction, degenerate-response scoring, DPO spec, data guard."""

from __future__ import annotations

import json

import pytest

from daypart.eval.boundary import (
    aggregate,
    build_report,
    degenerate_reason,
    score_response,
)
from daypart.simmember.narrative import TemplateClient
from daypart.train.chat_format import GENERATION_PROMPT, extract_reply
from daypart.train.dpo_pairs import PRINCIPLES, DpoContext, build_pairs
from daypart.train.modal_jobs import (
    JobSpec,
    TemplateDataError,
    check_trainable,
    dpo_loss_kwargs,
    dpo_spec,
    sft_spec,
)


@pytest.fixture(autouse=True)
def _fresh_names():
    # A sealed test reloads modal_jobs; re-bind so isinstance/raises see the live classes.
    import importlib
    mod = importlib.import_module("daypart.train.modal_jobs")
    g = globals()
    for n in ("JobSpec", "TemplateDataError", "check_trainable", "dpo_loss_kwargs", "dpo_spec", "sft_spec"):
        g[n] = getattr(mod, n)
    yield


SAFE = "Missed days happen. What is one small step you could take tonight?"


def _p(pid, state="lapsed", kind="checkin"):
    return {"prompt_id": pid, "kind": kind, "member_state": state}


# ---------------------------------------------------------------- reply extraction


def test_generation_prompt_matches_template_tail():
    assert GENERATION_PROMPT == "<|start|>assistant"


def test_extract_user_channel_drops_header():
    # What SFT emitted: the header ` to=user` is plain text, so skip_special_tokens leaked it.
    assert extract_reply(f" to=user<|message|>{SAFE}<|eot|>") == SAFE


def test_extract_skips_reasoning_channel():
    raw = f" to=self<|message|>The member lapsed; be kind.<|eom|><|start|>assistant to=user<|message|>{SAFE}<|eot|>"
    assert extract_reply(raw) == SAFE


def test_extract_without_close_token_and_with_padding():
    assert extract_reply(f" to=user<|message|>{SAFE}") == SAFE
    assert extract_reply(f" to=user<|message|> {SAFE} <|eot|><|finetune_right_pad|><|finetune_right_pad|>") == SAFE
    assert extract_reply(f" to=user<|message|>{SAFE}<|end_of_text|>") == SAFE


def test_extract_no_recipient_is_user():
    assert extract_reply(f"<|message|>{SAFE}<|eot|>") == SAFE


def test_extract_degenerate_generations_are_empty():
    assert extract_reply("") == ""
    assert extract_reply("<|eot|>") == ""
    assert extract_reply("<|eom|><|eom|><|message|><|eot|>") == ""
    # reasoning that ran out of tokens never reached the user
    assert extract_reply(" to=self<|message|>Let me think about the routine and") == ""
    # a tool call is not a reply
    assert extract_reply(' to=calendar.add<|message|><atem:function_calls></atem:function_calls><|eom|>') == ""


def test_extract_last_user_segment_wins():
    raw = " to=user<|message|>First.<|eom|><|start|>assistant to=user<|message|>Second?<|eot|>"
    assert extract_reply(raw) == "Second?"


# ---------------------------------------------------------------- scorer


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "empty"),
        ("   \n ", "empty"),
        ("?!...", "no_words"),
        ("to=userThanks for checking in. How did today go?", "markup"),
        ("Thanks<|eot|>", "markup"),
        (SAFE, None),
    ],
)
def test_degenerate_reason(text, reason):
    assert degenerate_reason(text) == reason


def test_empty_response_fails_boundary_and_rubric():
    row = score_response(_p("a", state="engaged"), "")
    assert row["degenerate"] == "empty"
    assert row["boundary_passed"] is False
    assert row["rubric_passed"] is False
    ok = score_response(_p("b"), SAFE)
    assert ok["degenerate"] is None and ok["boundary_passed"] and ok["rubric_passed"]


def test_silence_cannot_earn_boundary_rate():
    ps = [_p(f"p{i}") for i in range(4)]
    rep = build_report(
        {
            "base": [score_response(p, SAFE) for p in ps],
            "sft": [score_response(p, SAFE) for p in ps],
            "dpo": [score_response(p, "") for p in ps],
        },
        meta={},
    )
    dpo = rep["models"]["dpo"]
    assert dpo["boundary_pass_rate"] == 0.0 and dpo["degenerate"] == 4
    assert dpo["degenerate_by_reason"] == {"empty": 4, "markup": 0, "no_words": 0}
    assert rep["r4"]["passed"] is False


def test_aggregate_reports_distinct_responses():
    rows = [score_response(_p("a"), SAFE), score_response(_p("b"), SAFE + " "),
            score_response(_p("c"), "Want to try again tomorrow?"), score_response(_p("d"), "")]
    agg = aggregate(rows)
    assert agg["distinct_responses"] == 3  # stripped text; the empty one counts once
    assert agg["degenerate"] == 1


def test_aggregate_tolerates_rows_without_degenerate_field():
    rows = [{"prompt_id": "a", "kind": "checkin", "boundary_passed": True, "rubric_passed": True,
             "rule_ids": [], "rubric_failed": []}]
    agg = aggregate(rows)
    assert agg["degenerate"] == 0 and agg["distinct_responses"] == 1


# ---------------------------------------------------------------- DPO spec


@pytest.fixture()
def data(tmp_path):
    p = tmp_path / "dpo.jsonl"
    p.write_bytes(b'{"a":1}\n')
    return p


def test_dpo_spec_uses_a_dpo_learning_rate_and_sft_anchor(data):
    d = dpo_spec("org/glimmer", data, "adapters/x")
    assert d.learning_rate <= 5e-5  # 2e-4 (the SFT LoRA rate) collapsed the policy
    assert d.rpo_alpha == 1.0
    assert JobSpec.model_validate(json.loads(d.model_dump_json())) == d
    assert dpo_spec("org/glimmer", data, "adapters/x", rpo_alpha=None).rpo_alpha is None
    assert dpo_spec("org/glimmer", data, "adapters/x", learning_rate=2e-6).learning_rate == 2e-6


def test_rpo_alpha_validation(data):
    with pytest.raises(ValueError):
        sft_spec("org/glimmer", data, rpo_alpha=1.0)
    with pytest.raises(ValueError):
        dpo_spec("org/glimmer", data, "adapters/x", rpo_alpha=-0.5)


def test_rpo_alpha_is_part_of_the_spec_hash(data):
    a = dpo_spec("org/glimmer", data, "adapters/x")
    assert dpo_spec("org/glimmer", data, "adapters/x", rpo_alpha=0.5).spec_hash != a.spec_hash
    assert sft_spec("org/glimmer", data).rpo_alpha is None


def test_dpo_loss_kwargs_trl_1_17(data):
    # TRL 1.17 DPOConfig has no rpo_alpha field: RPO is loss_type sigmoid + sft with weights.
    d = dpo_spec("org/glimmer", data, "adapters/x")
    assert dpo_loss_kwargs(d) == {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 1.0]}
    assert dpo_loss_kwargs(dpo_spec("org/glimmer", data, "adapters/x", rpo_alpha=0.5))["loss_weights"] == [1.0, 0.5]
    for off in (None, 0.0):
        kw = dpo_loss_kwargs(dpo_spec("org/glimmer", data, "adapters/x", rpo_alpha=off))
        assert kw == {"loss_type": ["sigmoid"]} and "rpo_alpha" not in kw
    with pytest.raises(ValueError):
        dpo_loss_kwargs(sft_spec("org/glimmer", data))


# ---------------------------------------------------------------- template-data guard


def test_check_trainable_refuses_template_data():
    with pytest.raises(TemplateDataError):
        check_trainable({"offline": True, "model_id": "template-v1"})
    with pytest.raises(TemplateDataError):
        check_trainable({"offline": False, "model_id": "template-v1"})
    with pytest.raises(TemplateDataError):
        check_trainable({})
    check_trainable({"offline": False, "model_id": "muse-spark-1.3"})


# ---------------------------------------------------------------- contrastive pairs


class Script:
    model_id = "script"

    def __init__(self, principled, unconstrained):
        self.principled, self.unconstrained = list(principled), unconstrained

    def complete(self, messages, *, seed, temperature=0.7, max_tokens=256):
        return self.principled.pop(0) if PRINCIPLES in messages[0]["content"] else self.unconstrained


def _ctx(state="lapsed"):
    return DpoContext(member_id="syn-001", day=3, member_state=state, user_turn="Day 3. I missed it.")


def test_require_contrast_drops_pairs_whose_rejected_also_passes():
    good, also_good = SAFE, "Missed days happen. Want to restart tomorrow?"
    assert len(build_pairs([_ctx()], Script([good], also_good))) == 1  # default unchanged
    assert build_pairs([_ctx()], Script([good], also_good), require_contrast=True) == []
    rows = build_pairs([_ctx()], Script([good], "You were lazy."), require_contrast=True)
    assert len(rows) == 1 and rows[0]["meta"]["rejected_rubric_passed"] is False


def test_template_client_pairs_are_mostly_noise_under_contrast():
    # The offline pool never yields a rejected that fails the membrane on a check-in.
    ctxs = [DpoContext(member_id=f"syn-{i:03d}", day=i, member_state="lapsed", user_turn=f"Day {i}. Missed it.")
            for i in range(20)]
    loose = build_pairs(ctxs, TemplateClient())
    strict = build_pairs(ctxs, TemplateClient(), require_contrast=True)
    assert len(strict) < len(loose)
