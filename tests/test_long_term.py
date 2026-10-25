"""Long-term memory: update instead of duplicate, forgetting, recall ranking, rules."""

from __future__ import annotations

import pytest
from langgraph.store.memory import InMemoryStore

from memory_agent.embeddings import HashingEmbeddings
from memory_agent.long_term import forget_fact
from memory_agent.long_term import get_rules
from memory_agent.long_term import get_watermark
from memory_agent.long_term import list_facts
from memory_agent.long_term import looks_secret
from memory_agent.long_term import recall
from memory_agent.long_term import register_thread
from memory_agent.long_term import save_rules
from memory_agent.long_term import set_watermark
from memory_agent.long_term import upsert_fact
from memory_agent.long_term import wipe_user
from memory_agent.namespaces import namespace


def _put(store: InMemoryStore, user: str, content: str, slot: str | None = None) -> str:
    return upsert_fact(store, user, content, slot, source="test", dedup_threshold=0.9).action


def test_a_slot_update_replaces_the_value_and_keeps_history(store: InMemoryStore) -> None:
    assert _put(store, "u", "Lives in Lisbon.", "home_city") == "added"
    assert _put(store, "u", "Lives in Porto.", "home_city") == "updated"
    assert _put(store, "u", "lives in porto.", "Home City") == "unchanged"
    facts = list_facts(store, "u")
    assert len(facts) == 1
    value = facts[0].value
    assert (value["content"], value["version"], value["history"]) == (
        "Lives in Porto.",
        2,
        ["Lives in Lisbon."],
    )


def test_history_is_capped(store: InMemoryStore) -> None:
    for i in range(12):
        _put(store, "u", f"Works as engineer number {i}.", "occupation")
    value = list_facts(store, "u")[0].value
    assert value["version"] == 12 and len(value["history"]) == 5


def test_a_slotless_near_duplicate_updates_the_similar_fact(store: InMemoryStore) -> None:
    assert _put(store, "u", "Prefers dark mode in every editor.") == "added"
    assert _put(store, "u", "Prefers dark mode in every editor!") == "unchanged"
    assert _put(store, "u", "prefers DARK mode in every editor and terminal.") == "updated"
    assert _put(store, "u", "Plays the cello on weekends.") == "added"
    contents = sorted(i.value["content"] for i in list_facts(store, "u"))
    assert contents == [
        "Plays the cello on weekends.",
        "prefers DARK mode in every editor and terminal.",
    ]


def test_secrets_and_empty_facts_are_never_stored(store: InMemoryStore) -> None:
    for text in (
        "My password is hunter2.",
        "Card 4111 1111 1111 1111.",
        "API key sk-abcdefghijklmnop.",
        "SSN 123-45-6789.",
        "   ",
    ):
        assert _put(store, "u", text) == "refused"
    assert list_facts(store, "u") == []
    assert looks_secret("my pin is 1234") and not looks_secret("Lives in Lisbon.")


def test_an_invalid_slot_falls_back_to_similarity(store: InMemoryStore) -> None:
    assert _put(store, "u", "Likes jazz.", "../../etc") == "added"
    assert _put(store, "u", "Likes jazz.", "$$$") == "unchanged"
    assert len(list_facts(store, "u")) == 1


def test_forget_by_slot_or_by_similarity(store: InMemoryStore) -> None:
    _put(store, "u", "Is vegetarian.", "diet")
    _put(store, "u", "Lives in Lisbon.", "home_city")
    _put(store, "u", "Plays the cello on weekends.")
    assert forget_fact(store, "u", "diet") == ["diet"]
    assert forget_fact(store, "u", "that I play the cello") != []
    assert forget_fact(store, "u", "quantum chromodynamics") == []
    assert [i.key for i in list_facts(store, "u")] == ["home_city"]


def test_recall_ranks_by_similarity_and_respects_k(store: InMemoryStore) -> None:
    _put(store, "u", "Is vegetarian and never eats meat.", "diet")
    _put(store, "u", "Lives in Lisbon, Portugal.", "home_city")
    _put(store, "u", "Favourite programming language is Rust.", "favorite_language")
    _put(store, "u", "Has a dog named Pixel.", "pet")
    found = recall(store, "u", "Which city do I live in?", facts=3, episodes=0)
    assert found.facts[0].key == "home_city"
    assert len(found.facts) == 3
    scores = [f.score or 0.0 for f in found.facts]
    assert scores == sorted(scores, reverse=True)
    assert recall(store, "u", "what should I cook, I never eat meat", facts=1, episodes=0).facts[
        0
    ].key == ("diet")
    assert recall(store, "u", "my programming language", facts=1, episodes=0).facts[0].key == (
        "favorite_language"
    )
    assert recall(store, "u", "anything", facts=0, episodes=0).facts == []


def test_recall_only_searches_the_users_namespace(store: InMemoryStore) -> None:
    _put(store, "alice", "Lives in Lisbon.", "home_city")
    _put(store, "bob", "Lives in Paris.", "home_city")
    found = recall(store, "bob", "Where do I live?", facts=10, episodes=10)
    assert [f.content for f in found.facts] == ["Lives in Paris."]


def test_rules_are_deduplicated_capped_and_secret_free(store: InMemoryStore) -> None:
    rules = [f"Rule {i}." for i in range(15)] + ["Rule 14", "Never share my password abc."]
    saved = save_rules(store, "u", rules)
    assert len(saved) == 10 and saved[-1] == "Rule 14."
    assert get_rules(store, "u") == saved
    assert store.get(namespace("u", "instructions"), "style").value["version"] == 1
    save_rules(store, "u", ["Be brief."])
    assert store.get(namespace("u", "instructions"), "style").value["version"] == 2
    assert get_rules(store, "nobody") == []


def test_watermark_round_trip(store: InMemoryStore) -> None:
    assert get_watermark(store, "u", "t") == set()
    register_thread(store, "u", "t")
    set_watermark(store, "u", "t", ["m1", "m2"])
    assert get_watermark(store, "u", "t") == {"m1", "m2"}
    assert "created" in store.get(namespace("u", "threads"), "t").value


def test_wipe_user_deletes_every_namespace_and_only_that_user(store: InMemoryStore) -> None:
    for user in ("alice", "alice2"):
        _put(store, user, "Lives in Lisbon.", "home_city")
        save_rules(store, user, ["Be brief."])
        register_thread(store, user, "t")
    for i in range(130):  # more than one page
        _put(store, "alice", f"Collects stamp series {i} from year {1900 + i}.", f"stamp_{i}")
    assert wipe_user(store, "alice") == 133
    assert store.search(("memories", "alice")) == []
    assert len(store.search(("memories", "alice2"))) == 3


def test_hashing_embeddings_are_deterministic_and_normalised() -> None:
    emb = HashingEmbeddings(64)
    a, b = emb.embed_documents(["Lives in Lisbon", "Lives in Lisbon"])
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-9
    assert emb.embed_query("the a of") == [0.0] * 64
    near = sum(x * y for x, y in zip(emb.embed_query("where do I live"), a, strict=True))
    far = sum(x * y for x, y in zip(emb.embed_query("favourite food"), a, strict=True))
    assert near > far


@pytest.mark.parametrize("dims", [8, 256, 1024])
def test_hashing_embeddings_have_the_configured_size(dims: int) -> None:
    assert len(HashingEmbeddings(dims).embed_query("hello world")) == dims


class _LexicalStore(InMemoryStore):
    """Like a BM25 store: a query with no shared words finds nothing."""

    def search(self, namespace_prefix, /, *, query=None, **kwargs):
        return [] if query else super().search(namespace_prefix, **kwargs)


def test_recall_tops_up_with_recent_facts_when_search_finds_too_few() -> None:
    store = _LexicalStore()
    upsert_fact(store, "u", "Lives in Lisbon.", "home_city", source="t", dedup_threshold=0.9)
    upsert_fact(store, "u", "Is vegetarian.", "diet", source="t", dedup_threshold=0.9)
    found = recall(store, "u", "What do you know about me?", facts=5, episodes=1)
    assert sorted(f.key for f in found.facts) == ["diet", "home_city"]
    assert all(f.score is None for f in found.facts)
