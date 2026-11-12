import numpy as np

from triage import drift
from triage.gate import choose_threshold, route


def test_threshold_meets_target_and_maximises_coverage():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 5000)
    correct = rng.uniform(0, 1, 5000) < conf  # higher confidence -> more often right
    g = choose_threshold(conf, correct, target_accuracy=0.9)
    assert g.auto_accuracy >= 0.9
    lower = conf >= g.threshold - 0.05
    assert correct[lower].mean() < 0.9 or lower.sum() == (conf >= g.threshold).sum()


def test_no_safe_threshold_sends_everything_to_a_person():
    conf = np.linspace(0, 1, 200)
    correct = np.zeros(200, dtype=bool)
    g = choose_threshold(conf, correct, target_accuracy=0.9)
    assert g.coverage == 0.0 and g.threshold > 1


def test_fraud_intent_bypasses_gate_even_at_full_confidence():
    decision, reason = route("compromised_card", 0.999, 0.5, {"compromised_card"})
    assert decision == "human_review" and "fraud" in reason


def test_fraud_probability_safety_net():
    # top intent looks harmless and confident, but 8% of the mass is on a fraud intent
    decision, reason = route("card_arrival", 0.9, 0.5, {"compromised_card"}, risk_mass=0.08, risk_cutoff=0.05)
    assert decision == "human_review" and "fraud" in reason
    assert route("card_arrival", 0.9, 0.5, {"compromised_card"}, risk_mass=0.01, risk_cutoff=0.05)[0] == "auto_answer"


def test_risk_cutoff_chosen_to_catch_validation_fraud():
    from triage.gate import choose_risk_cutoff
    risk = np.array([0.04, 0.2, 0.01])
    cut = choose_risk_cutoff(risk, [False, False, False], [True, False, False], np.array([0.9, 0.9, 0.9]), 0.5)
    assert cut <= 0.04


def test_low_confidence_goes_to_person():
    assert route("card_arrival", 0.4, 0.5, set())[0] == "human_review"
    assert route("card_arrival", 0.6, 0.5, set())[0] == "auto_answer"


def test_psi_zero_for_identical_and_alerts_on_shift():
    classes = ["a", "b", "c", "d"]
    ref = drift.reference_stats(["a", "b", "c", "d"] * 50, np.full(200, 0.9), classes, 0.5)
    same = drift.report(ref, ["a", "b", "c", "d"] * 50, np.full(200, 0.9), classes, 0.5)
    assert same["psi"] < 1e-6 and same["status"] == "ok"
    shifted = drift.report(ref, ["a"] * 170 + ["b"] * 30, np.full(200, 0.9), classes, 0.5)
    assert shifted["status"] == "alert" and shifted["psi"] > drift.PSI_ALERT


def test_no_false_alarm_on_unchanged_traffic_with_many_intents():
    """77 intents, ~750 messages: pure sampling noise must not raise an alert."""
    rng = np.random.default_rng(1)
    classes = [f"i{k}" for k in range(77)]
    p = rng.dirichlet(np.ones(77) * 5)
    a = list(rng.choice(classes, 750, p=p))
    b = list(rng.choice(classes, 750, p=p))
    ref = drift.reference_stats(a, np.full(750, 0.9), classes, 0.5)
    rep = drift.report(ref, b, np.full(750, 0.9), classes, 0.5)
    assert rep["status"] == "ok", rep


def test_confidence_collapse_alerts():
    classes = ["a", "b"]
    ref = drift.reference_stats(["a", "b"] * 50, np.full(100, 0.9), classes, 0.5)
    rep = drift.report(ref, ["a", "b"] * 50, np.full(100, 0.3), classes, 0.5)
    assert rep["status"] == "alert"
