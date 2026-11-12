"""Draft a reply that is grounded in retrieved help articles - or refuse to.

Two modes:
- 'llm'        : Claude writes the reply from the retrieved articles only, returning
                 structured JSON with the article IDs it used.
- 'extractive' : no API key / LLM disabled - return the best article verbatim.

Either way the draft goes through `grounding_check`. A reply that cites nothing, cites
an article that wasn't retrieved, or asks for a handoff is NOT sent: the message is
routed to a person instead. The system fails closed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .kb import Hit

MODEL = os.environ.get("TRIAGE_LLM_MODEL", "claude-opus-5-5")

SYSTEM = (
    "You draft replies for a bank's customer support team. Answer the customer's message "
    "using ONLY the help articles provided. Keep it under 90 words, plain and friendly, no "
    "markdown. Put every article ID you relied on in `citations`. If the articles don't "
    "answer the question, or the customer may be at risk (fraud, distress, a complaint), "
    "set `handoff` to true and leave `answer` empty. Never invent fees, limits or timings."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "handoff": {"type": "boolean"},
    },
    "required": ["answer", "citations", "handoff"],
    "additionalProperties": False,
}


@dataclass
class Draft:
    answer: str
    citations: list[str]
    mode: str
    handoff: bool = False
    problems: list[str] = field(default_factory=list)


def extractive(hits: list[Hit]) -> Draft:
    top = hits[0].article
    return Draft(top.body, [top.id], "extractive")


def llm_draft(question: str, hits: list[Hit], client) -> Draft:
    articles = "\n\n".join(
        f'<article id="{h.article.id}" title="{h.article.title}">\n{h.article.body}\n</article>' for h in hits
    )
    resp = client.beta.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": f"{articles}\n\n<customer_message>\n{question}\n</customer_message>"}],
    )
    if resp.stop_reason != "end_turn":
        return Draft("", [], "llm", handoff=True, problems=[f"model stopped with '{resp.stop_reason}'"])
    text = "".join(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    return Draft(data["answer"].strip(), list(data["citations"]), "llm", handoff=bool(data["handoff"]))


def grounding_check(draft: Draft, hits: list[Hit]) -> Draft:
    allowed = {h.article.id for h in hits}
    if draft.handoff:
        draft.problems.append("drafter asked for a human")
    if not draft.answer:
        draft.problems.append("empty answer")
    if not draft.citations:
        draft.problems.append("no citations")
    bad = [c for c in draft.citations if c not in allowed]
    if bad:
        draft.problems.append(f"cited articles that were not retrieved: {bad}")
    return draft


def make_client():
    """Return an Anthropic client if credentials are configured, else None."""
    if os.environ.get("TRIAGE_DISABLE_LLM") == "1":
        return None
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return None
    import anthropic

    return anthropic.Anthropic()


def draft_reply(question: str, hits: list[Hit], client=None) -> Draft:
    if client is None:
        return grounding_check(extractive(hits), hits)
    try:
        return grounding_check(llm_draft(question, hits, client), hits)
    except Exception as e:  # any failure -> a person handles it, never a silent bad reply
        return Draft("", [], "llm", handoff=True, problems=[f"drafting failed: {type(e).__name__}"])
