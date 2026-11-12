from fastapi.testclient import TestClient

from triage import api


def client(small_triage):
    api.STATE["triage"] = small_triage
    return TestClient(api.app)


def test_health(small_triage):
    with client(small_triage) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["llm_drafting"] is False


def test_routine_question_is_auto_answered_with_citation(small_triage):
    with client(small_triage) as c:
        r = c.post("/triage", json={"text": "How long will it take for my new card to arrive?"}).json()
    assert r["intent"] in {"card_arrival", "card_delivery_estimate"}
    assert r["decision"] == "auto_answer"
    assert r["citations"] == ["KB-01"] and r["reply"]


def test_stolen_card_always_goes_to_a_person(small_triage):
    with client(small_triage) as c:
        r = c.post("/triage", json={"text": "Someone stole my card and wallet last night"}).json()
    assert r["intent"] == "lost_or_stolen_card"
    assert r["decision"] == "human_review" and r["reply"] is None


def test_validation_rejects_empty_and_tiny_batches(small_triage):
    with client(small_triage) as c:
        assert c.post("/triage", json={"text": ""}).status_code == 422
        assert c.post("/drift", json={"texts": ["a"] * 5}).status_code == 422


def test_drift_endpoint(small_triage, splits):
    _, _, test = splits
    with client(small_triage) as c:
        r = c.post("/drift", json={"texts": test.texts[:200]}).json()
    assert {"psi", "status", "alerts", "biggest_moves"} <= set(r)
