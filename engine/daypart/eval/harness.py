"""E1.3 — run every configured arm (G2: all arms reported) and the CLI."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import pathlib
import sys
from collections.abc import Callable, Coroutine, Mapping, Sequence
from typing import Any

import yaml

from daypart.eval.corpus import load_corpus, r2_metrics
from daypart.eval.models import SCHEMA, ArmConfig, RunsConfig, load_runs
from daypart.eval.policies import choose


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256_json(obj: Any) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


def _r6(x: float) -> float:
    return round(float(x), 6)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _drive(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coro`` to completion, also when called from inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# --------------------------------------------------------------------------- #
# env_baseline
# --------------------------------------------------------------------------- #


def build_scenarios(cohort: Any, n_scenarios: int, seed: int, horizon: int) -> list:
    """The synthetic scenarios an ``env_baseline`` arm builds for one seed (I1)."""
    from daypart.simmember.env import Scenario, make_scenarios

    out = []
    for s in make_scenarios(cohort, n_scenarios, seed):
        if not s.member.member_id.startswith("syn-") or s.member.cohort != "synthetic":
            raise ValueError(f"scenario {s.scenario_id} is not synthetic (I1)")
        out.append(Scenario(scenario_id=s.scenario_id, member=s.member, seed=s.seed, horizon=horizon))
    return out


def arm_scenarios(arm: ArmConfig, cohort: Any) -> list:
    """Every scenario of an ``env_baseline`` arm, all seeds in config order."""
    return [
        s
        for seed in arm.seeds
        for s in build_scenarios(cohort, arm.n_scenarios or 0, seed, arm.horizon)
    ]


async def _episode(scenario: Any, policy: str) -> tuple[float, bool, float]:
    """(adherence fraction, had a violation, summed reward) for one episode."""
    from aviary.core import ToolCall, ToolRequestMessage

    from daypart.simmember.env import SEND_MESSAGE, SimMemberEnv

    env = SimMemberEnv(scenario)
    await env.reset()
    total = 0.0
    done = False
    steps = 0
    while not done:
        day = env.state.day
        text, intent = choose(policy, scenario.seed, scenario.scenario_id, day)
        call = ToolCall.from_name(SEND_MESSAGE, id=f"d{day}", text=text, intent=intent)
        _, reward, done, _ = await env.step(ToolRequestMessage(tool_calls=[call]))
        total += float(reward)
        steps += 1
        if steps > scenario.horizon + 1:  # defensive: the env must end by the horizon
            raise RuntimeError(f"episode {scenario.scenario_id} did not terminate")
    adherence = env.state.adherence
    frac = (sum(adherence) / len(adherence)) if adherence else 0.0
    return frac, bool(env.state.violations), total


async def _env_seed(scenarios: Sequence[Any], policy: str) -> list[tuple[float, bool, float]]:
    return [await _episode(s, policy) for s in scenarios]


def _run_env_baseline(arm: ArmConfig, config: RunsConfig) -> dict:
    from daypart.simmember.cohort import generate
    from daypart.simmember.env import scenario_manifest

    assert arm.policy is not None and arm.n_scenarios is not None
    cohort = generate(config.cohort_seed)
    per_seed = []
    for seed in arm.seeds:
        scenarios = build_scenarios(cohort, arm.n_scenarios, seed, arm.horizon)
        episodes = _drive(_env_seed(scenarios, arm.policy))
        n = len(episodes)
        per_seed.append(
            {
                "seed": seed,
                "n_episodes": n,
                "mean_adherence": _r6(_mean([e[0] for e in episodes])),
                "episode_violation_rate": _r6(sum(e[1] for e in episodes) / n if n else 0.0),
                "mean_return": _r6(_mean([e[2] for e in episodes])),
                "scenario_manifest_sha256": _sha256_json(scenario_manifest(scenarios)),
            }
        )
    return {
        "policy": arm.policy,
        "horizon": arm.horizon,
        "n_scenarios": arm.n_scenarios,
        "per_seed": per_seed,
        "mean_adherence": _r6(_mean([p["mean_adherence"] for p in per_seed])),
        "episode_violation_rate": _r6(_mean([p["episode_violation_rate"] for p in per_seed])),
        "seeds": list(arm.seeds),
    }


# --------------------------------------------------------------------------- #
# r2_dsl / i1_gate
# --------------------------------------------------------------------------- #


def _run_r2_dsl(arm: ArmConfig, root: pathlib.Path) -> dict:
    assert arm.corpus is not None
    corpus = pathlib.Path(arm.corpus)
    path = corpus if corpus.is_absolute() else root / corpus
    if not path.is_file():  # report the configured string, never the resolved absolute path
        raise FileNotFoundError(f"corpus not found: {arm.corpus}")
    metrics = r2_metrics(load_corpus(path))
    metrics["corpus"] = arm.corpus
    return metrics


def _run_i1_gate(config: RunsConfig) -> dict:
    from daypart.ci.i1_check import results

    builders = results(seed=config.cohort_seed)
    builders = {k: builders[k] for k in sorted(builders)}
    return {"builders": builders, "passed": all(v == "ok" for v in builders.values())}


# --------------------------------------------------------------------------- #
# boundary_eval (2.4: base / SFT / DPO, read from committed results files)
# --------------------------------------------------------------------------- #

BOUNDARY_FILE = "boundary_eval.json"
TRAINING_FILE = "training.json"
RESULTS_DIR_ENV = "DAYPART_RESULTS_DIR"


class _NotRun(Exception):
    """An arm whose inputs do not exist yet (reported as ``not_run``, never ``failed``)."""


def _results_dir(arm: ArmConfig, root: pathlib.Path) -> pathlib.Path:
    env = os.environ.get(RESULTS_DIR_ENV)
    d = pathlib.Path(env if env else (arm.results or ""))
    return d if d.is_absolute() else root / d


def _run_boundary_eval(arm: ArmConfig, root: pathlib.Path) -> dict:
    from daypart.eval.boundary import load_report
    from daypart.train.results import training_summary

    d = _results_dir(arm, root)
    needed = [BOUNDARY_FILE] + ([TRAINING_FILE] if arm.model in ("sft", "dpo") else [])
    missing = [f for f in needed if not (d / f).is_file()]
    if missing:
        raise _NotRun(
            f"missing {', '.join(missing)} in the results directory; run "
            "`modal run daypart/train/modal_app.py::boundary_eval` and commit the output"
        )
    report = load_report(d / BOUNDARY_FILE)
    model = report["models"].get(arm.model)
    if model is None:
        raise ValueError(f"{BOUNDARY_FILE} has no {arm.model!r} model")
    metrics: dict[str, Any] = {
        "model": arm.model,
        "boundary": {k: v for k, v in model.items() if k != "responses"},
        "prompt_set_version": report.get("prompt_set_version"),
        "margin": report.get("margin"),
    }
    if arm.model in ("sft", "dpo"):
        try:
            record = json.loads((d / TRAINING_FILE).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{TRAINING_FILE}: invalid JSON: {exc.msg}") from None
        metrics["training"] = training_summary(record, arm.model)
    if arm.model in ("base", "sft"):
        metrics["r3"] = report.get("r3")
    if arm.model == "dpo":
        metrics["r4"] = report.get("r4")
    return metrics


# --------------------------------------------------------------------------- #
# run / dumps / main
# --------------------------------------------------------------------------- #


def _arm_result(arm: ArmConfig, status: str, reason: str | None, metrics: dict | None) -> dict:
    return {"name": arm.name, "kind": arm.kind, "status": status, "reason": reason, "metrics": metrics}


def _scrub(reason: str, root: pathlib.Path) -> str:
    """Strip environment-dependent root paths from a failure reason (E1.3: no absolute paths)."""
    for prefix in sorted({str(root.resolve()), str(root.absolute())}, key=len, reverse=True):
        if prefix and prefix != ".":
            reason = reason.replace(prefix + os.sep, "").replace(prefix, "<root>")
    return reason


def _run_arm(arm: ArmConfig, config: RunsConfig, root: pathlib.Path) -> dict:
    if arm.kind == "not_run":
        return _arm_result(arm, "not_run", arm.reason, None)
    runners: dict[str, Callable[[], dict]] = {
        "r2_dsl": lambda: _run_r2_dsl(arm, root),
        "env_baseline": lambda: _run_env_baseline(arm, config),
        "i1_gate": lambda: _run_i1_gate(config),
        "boundary_eval": lambda: _run_boundary_eval(arm, root),
    }
    runner = runners.get(arm.kind)
    if runner is None:
        return _arm_result(arm, "failed", f"unknown arm kind {arm.kind!r}", None)
    try:
        metrics = runner()
    except _NotRun as exc:
        return _arm_result(arm, "not_run", str(exc), None)
    except Exception as exc:  # G2: a failing arm never aborts the run
        reason = _scrub(f"{type(exc).__name__}: {exc}", root)
        return _arm_result(arm, "failed", reason, None)
    return _arm_result(arm, "ok", None, metrics)


def run(config: RunsConfig, *, root: str | os.PathLike | None = None) -> dict:
    from daypart.membrane.dsl import RULE_VERSION
    from daypart.memory.lhm import GUARD_VERSIONS

    base = pathlib.Path.cwd() if root is None else pathlib.Path(root)
    arms = [_run_arm(arm, config, base) for arm in config.arms]
    summary = {status: sum(1 for a in arms if a["status"] == status) for status in ("ok", "not_run", "failed")}
    return {
        "schema": SCHEMA,
        "runs_version": config.version,
        "runs_digest": _sha256_json(config.model_dump(mode="json")),
        "cohort_seed": config.cohort_seed,
        "rule_version": RULE_VERSION,
        "guard_versions": dict(GUARD_VERSIONS),
        "arms": arms,
        "summary": summary,
    }


def dumps(metrics: Mapping[str, Any]) -> str:
    return json.dumps(metrics, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def exit_code(metrics: Mapping[str, Any]) -> int:
    """0 iff no arm failed and every ``i1_gate`` arm reports ``passed: True``."""
    for arm in metrics["arms"]:
        if arm["status"] == "failed":
            return 1
        if arm["kind"] == "i1_gate" and not (arm["metrics"] or {}).get("passed", False):
            return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m daypart.eval", description="Daypart eval harness")
    parser.add_argument("--runs", default=None, help="runs config (default: packaged runs.yaml)")
    parser.add_argument("--out", default=None, help="write metrics JSON here (default: stdout)")
    parser.add_argument("--root", default=None, help="root for relative corpus paths (default: cwd)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = load_runs(args.runs)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"daypart.eval: cannot load runs config: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    metrics = run(config, root=args.root)
    text = dumps(metrics)
    if args.out is None:
        sys.stdout.write(text)
        sys.stdout.flush()
    else:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return exit_code(metrics)


__all__ = ["arm_scenarios", "build_scenarios", "dumps", "exit_code", "main", "run"]
