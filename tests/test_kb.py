from collections import Counter

from triage.kb import load_kb


def test_every_intent_has_exactly_one_article(splits):
    articles, _ = load_kb()
    mapped = Counter(i for a in articles for i in a.intents)
    dataset_intents = set(splits[0].labels)
    assert set(mapped) == dataset_intents, set(mapped) ^ dataset_intents
    assert all(n == 1 for n in mapped.values()), [i for i, n in mapped.items() if n > 1]


def test_always_human_intents_are_real_intents(splits):
    _, always_human = load_kb()
    assert always_human <= set(splits[0].labels)
    assert {"compromised_card", "lost_or_stolen_card", "card_payment_not_recognised"} <= always_human


def test_article_ids_unique():
    articles, _ = load_kb()
    ids = [a.id for a in articles]
    assert len(ids) == len(set(ids))
