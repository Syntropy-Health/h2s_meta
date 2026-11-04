"""HAC-20 — FastAPI playground routes (TestClient). Synthetic members only (I1)."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from daypart.demo.app import create_app  # noqa: E402
from daypart.demo.safety import WITHHELD_DRAFT  # noqa: E402
from daypart.eval import POLICY_TEXT  # noqa: E402
from daypart.passport import PASSPORT_TITLE, WITHHELD_MESSAGE  # noqa: E402
from daypart.simmember.dynamics import INTENTS  # noqa: E402

UNSAFE = "Take 400 mg of magnesium now."


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(cohort_seed=0)) as c:
        yield c


def _episode(client, member_id="syn-001", horizon=3, seed=0):
    r = client.post("/api/episodes", json={"member_id": member_id, "horizon": horizon, "seed": seed})
    assert r.status_code == 201, r.text
    return r.json()


def test_health_and_index(client):
    assert client.get("/api/health").json() == {"status": "ok", "cohort_seed": 0}
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Daypart playground" in r.text


def test_intents_lists_every_intent_with_safe_default_text(client):
    body = client.get("/api/intents").json()
    assert [i["intent"] for i in body] == list(INTENTS)
    assert all(i["default_text"] == POLICY_TEXT[i["intent"]] for i in body)


def test_members_are_generated_only(client):
    members = client.get("/api/members").json()
    assert len(members) > 0
    assert all(m["member_id"].startswith("syn-") for m in members)
    assert {"motivation", "barrier_type", "literacy", "engagement_pattern"} <= set(members[0])


def test_curated_member_is_rejected_everywhere(client):
    r = client.post("/api/episodes", json={"member_id": "cur-001"})
    assert r.status_code == 404
    assert client.get("/api/members/cur-001/passport").status_code == 404


def test_unknown_or_malformed_member_404(client):
    assert client.post("/api/episodes", json={"member_id": "syn-999"}).status_code == 404
    assert client.get("/api/members/someone@example.com/passport").status_code == 404


def test_episode_reset_returns_synthetic_observation(client):
    ep = _episode(client)
    assert ep["member_id"] == "syn-001"
    assert "syn-001" in ep["observation"]
    assert ep["state"] == {"day": 0, "member_state": ep["state"]["member_state"], "adherence": [],
                           "done": False, "violations": []}
    assert ep["horizon"] == 3


def test_step_default_text_passes_membrane_and_rewards(client):
    ep = _episode(client)
    r = client.post(f"/api/episodes/{ep['episode_id']}/step", json={"intent": "barrier_problem_solving"})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["draft"] == POLICY_TEXT["barrier_problem_solving"]
    assert s["intent"] == "barrier_problem_solving"
    assert s["membrane"]["decision"] == "pass"
    assert s["membrane"]["layer0"]["outcome"] == "pass"
    assert s["membrane"]["dsl"]["violations"] == []
    assert s["reward"] in (0.0, 1.0)
    assert s["state"]["day"] == 1 and not s["done"]


def test_step_violation_is_withheld_and_ends_episode(client):
    ep = _episode(client)
    s = client.post(f"/api/episodes/{ep['episode_id']}/step", json={"intent": "reminder", "text": UNSAFE}).json()
    assert s["membrane"]["decision"] == "escalate"
    assert s["membrane"]["dsl"]["violations"]  # DSL rule hits shown
    assert s["draft"] == WITHHELD_DRAFT  # the unsafe text never echoes back
    assert UNSAFE not in str(s)
    assert s["reward"] == -1.0 and s["done"] is True
    assert s["state"]["violations"]
    # stepping a finished episode is a client error, not a crash
    again = client.post(f"/api/episodes/{ep['episode_id']}/step", json={"intent": "none"})
    assert again.status_code == 409


def test_episode_runs_to_horizon(client):
    ep = _episode(client, horizon=2)
    eid = ep["episode_id"]
    first = client.post(f"/api/episodes/{eid}/step", json={"intent": "none"}).json()
    last = client.post(f"/api/episodes/{eid}/step", json={"intent": "none"}).json()
    assert not first["done"] and last["done"] and last["truncated"]
    assert len(last["state"]["adherence"]) == 2


def test_step_validation_errors(client):
    ep = _episode(client)
    eid = ep["episode_id"]
    assert client.post(f"/api/episodes/{eid}/step", json={"intent": "bribe"}).status_code == 422
    assert client.post("/api/episodes/nope/step", json={"intent": "none"}).status_code == 404
    assert client.post("/api/episodes", json={"member_id": "syn-001", "horizon": 0}).status_code == 422


def test_step_is_deterministic_for_same_scenario(client):
    rewards = []
    for _ in range(2):
        eid = _episode(client, horizon=4, seed=7)["episode_id"]
        rewards.append([client.post(f"/api/episodes/{eid}/step", json={"intent": "encouragement"}).json()["reward"]
                        for _ in range(4)])
    assert rewards[0] == rewards[1]


def test_passport_withheld_early_and_ready_later(client):
    early = client.get("/api/members/syn-001/passport", params={"day": 4}).json()
    assert early["title"] == PASSPORT_TITLE
    adh = {f["code"]: f for f in early["flags"]}["dp:adherence.protocol"]
    assert adh["status"] == "withheld" and adh["value"] is None and adh["message"] == WITHHELD_MESSAGE
    late = client.get("/api/members/syn-001/passport", params={"day": 41}).json()
    assert len(late["flags"]) == 4
    for f in late["flags"]:
        if f["status"] == "withheld":
            assert f["value"] is None and f["payload"] is None
        else:
            assert f["value"] is not None and f["caveats"]


def test_passport_day_bounds(client):
    assert client.get("/api/members/syn-001/passport", params={"day": -1}).status_code == 422
    assert client.get("/api/members/syn-001/passport", params={"day": 10_000}).status_code == 422
