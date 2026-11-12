"""The grounding check must fail closed: a bad draft never reaches a customer."""
import json
from types import SimpleNamespace

from triage.answer import draft_reply
from triage.kb import Article, Hit

HITS = [
    Hit(Article("KB-01", "Card delivery", ["card_arrival"], "Cards arrive in 3 to 5 working days."), 0.9),
    Hit(Article("KB-03", "Spare cards", ["getting_spare_card"], "You can hold up to 3 cards."), 0.4),
]


class FakeClient:
    def __init__(self, payload=None, stop="end_turn", exc=None):
        self.payload, self.stop, self.exc = payload, stop, exc
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        if self.exc:
            raise self.exc
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        return SimpleNamespace(
            stop_reason=self.stop,
            content=[SimpleNamespace(type="text", text=json.dumps(self.payload))],
        )


def test_good_llm_draft_passes():
    d = draft_reply("where's my card", HITS, FakeClient({"answer": "3-5 days.", "citations": ["KB-01"], "handoff": False}))
    assert not d.problems and d.citations == ["KB-01"] and d.mode == "llm"


def test_citing_unretrieved_article_is_blocked():
    d = draft_reply("q", HITS, FakeClient({"answer": "x", "citations": ["KB-99"], "handoff": False}))
    assert any("not retrieved" in p for p in d.problems)


def test_uncited_answer_is_blocked():
    d = draft_reply("q", HITS, FakeClient({"answer": "x", "citations": [], "handoff": False}))
    assert "no citations" in d.problems


def test_model_handoff_is_respected():
    d = draft_reply("q", HITS, FakeClient({"answer": "", "citations": [], "handoff": True}))
    assert d.problems


def test_refusal_or_truncation_fails_closed():
    d = draft_reply("q", HITS, FakeClient({"answer": "x", "citations": ["KB-01"], "handoff": False}, stop="refusal"))
    assert d.handoff and d.problems


def test_api_error_fails_closed():
    d = draft_reply("q", HITS, FakeClient(exc=RuntimeError("boom")))
    assert d.handoff and d.problems


def test_extractive_mode_without_llm():
    d = draft_reply("q", HITS, None)
    assert d.mode == "extractive" and d.citations == ["KB-01"] and not d.problems
