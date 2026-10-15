"""Offline tests for core.py. No API keys or network needed: the model calls are stubbed."""

import pytest
from pydantic import ValidationError

import core
from core import GatherReply, Intent, Recipe, State

CLIENT = object()  # never used: every function that would call the API is stubbed

CURRY = Recipe(
    name="Chicken curry",
    ingredients=["chicken", "onion", "garlic", "ginger", "curry powder"],
    prep=["Chop the onion.", "Slice the chicken."],
    cooking=["Fry the onion.", "Add the chicken and spices."],
    keyterms=["garam masala"],
)


@pytest.fixture
def stubbed(monkeypatch):
    """Stub out every model call so the phase machine can be walked offline."""
    monkeypatch.setattr(core, "triage_request", lambda client, text: None)
    monkeypatch.setattr(core, "find_recipe", lambda client, text: CURRY)
    monkeypatch.setattr(core.CookingSession, "_interpret", lambda self, text: Intent("next"))


def test_chunk_evenly():
    assert core.chunk_evenly([], 4) == []
    assert core.chunk_evenly(list("abcd"), 4) == [list("abcd")]
    assert [len(c) for c in core.chunk_evenly(list("abcde"), 4)] == [3, 2]  # not 4 + 1
    assert [len(c) for c in core.chunk_evenly(list("abcdefghi"), 4)] == [3, 3, 3]


def test_spoken_list():
    assert core.spoken_list(["a"]) == "a"
    assert core.spoken_list(["a", "b"]) == "a and b"
    assert core.spoken_list(["a", "b", "c"]) == "a, b, and c"


def test_greeting_and_quit_detection():
    assert core.is_greeting("Hello there!")
    assert core.is_greeting("what's up")
    assert not core.is_greeting("hey, chicken curry")
    assert core.is_quit("okay, goodbye!")
    assert not core.is_quit("please stop cooking the garlic for me now")  # too long to be a quit


def test_parse_recipe_cleans_keyterms():
    raw = (
        '{"name": " Dal ", "ingredients": ["lentils"], "prep": ["Rinse the lentils."], "cooking": [],'
        ' "keyterms": ["dal", "dal", " ", "x' + "y" * 60 + '", "tadka"]}'
    )
    recipe = core.parse_recipe(raw)
    assert recipe is not None
    assert recipe.name == "Dal"
    assert recipe.keyterms == ["dal", "tadka"]  # duplicate, blank and over-long terms dropped


def test_parse_recipe_rejects_bad_input():
    assert core.parse_recipe("not json") is None
    assert core.parse_recipe('{"name": "", "ingredients": [], "prep": ["a"], "cooking": []}') is None
    no_steps = '{"name": "x", "ingredients": [], "prep": [], "cooking": [], "keyterms": []}'
    assert core.parse_recipe(no_steps) is None
    too_long = '{"name": "x", "ingredients": [], "prep": [%s], "cooking": [], "keyterms": []}' % (
        ",".join(['"step"'] * (core.MAX_PLAN_STEPS + 1))
    )
    assert core.parse_recipe(too_long) is None


def test_full_flow(stubbed, monkeypatch):
    state = core.new_session().state
    assert state.phase == "choose"

    # Ask for a dish: the chef announces the lookup and asks the browser to continue.
    r = core.handle_turn(CLIENT, state, "chicken curry")
    assert r.say == [core.LOOKUP_LINE] and r.auto_continue
    assert r.state.pending_lookup == "chicken curry"

    # The lookup runs and the chef asks for confirmation.
    r = core.handle_continue(CLIENT, r.state)
    assert r.state.phase == "confirm" and "Chicken curry" in r.say[0]

    # Confirm: ingredients are read in two groups (3 + 2).
    r = core.handle_turn(CLIENT, r.state, "yes please")
    assert r.state.phase == "gather"
    assert r.say[0].startswith("Let's gather your ingredients. First, you'll need chicken, onion, and garlic")

    # Missing an ingredient: the chef answers, stays on the same group, and notes it.
    monkeypatch.setattr(
        core, "interpret_gather",
        lambda *a: GatherReply("answer", "Use ground ginger instead. Let me know when you're ready.", "No fresh ginger"),
    )
    r = core.handle_turn(CLIENT, r.state, "I don't have ginger")
    assert r.state.chunk == 0 and r.state.notes == ["No fresh ginger"]
    assert r.say == ["Use ground ginger instead. Let me know when you're ready."]
    assert len(r.state.exchange) == 2

    # "Got them" moves on without an API call, then the last group, then cooking starts.
    r = core.handle_turn(CLIENT, r.state, "got them")
    assert r.state.chunk == 1 and r.say[0].startswith("And finally, you'll need ginger and curry powder")
    r = core.handle_turn(CLIENT, r.state, "okay")
    assert r.state.phase == "cook"
    assert r.say[0] == "Let's make Chicken curry. We'll start with prep. Prep 1: Chop the onion."

    # A cooking turn: "next" advances, and the position and history live in the state.
    r = core.handle_turn(CLIENT, r.state, "done")
    assert r.say == ["Prep 2: Slice the chicken."]
    assert r.state.current == 1 and len(r.state.history) == 2

    r = core.handle_turn(CLIENT, r.state, "goodbye")
    assert r.done and r.say == [core.GOODBYE] and r.state.phase == "done"


def test_turn_does_not_mutate_input_state(stubbed):
    state = core.new_session().state
    before = state.model_dump()
    core.handle_turn(CLIENT, state, "chicken curry")
    assert state.model_dump() == before


def test_greeting_does_not_call_the_api(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("greetings must not reach the API")

    monkeypatch.setattr(core, "triage_request", boom)
    r = core.handle_turn(CLIENT, State(), "hi")
    assert r.say == [core.DISH_PROMPT] and not r.auto_continue


def test_non_food_request_is_declined(monkeypatch):
    monkeypatch.setattr(core, "triage_request", lambda c, t: "I can only help with cooking.")
    r = core.handle_turn(CLIENT, State(), "what's the weather")
    assert r.say == ["I can only help with cooking."]
    assert not r.auto_continue and r.state.pending_lookup is None


def test_declining_the_dish_goes_back_to_choosing(stubbed):
    state = State(phase="confirm", recipe=CURRY)
    r = core.handle_turn(CLIENT, state, "no, something else")
    assert r.state.phase == "choose" and r.state.recipe is None


def test_three_failed_lookups_fall_back_to_default(monkeypatch):
    monkeypatch.setattr(core, "find_recipe", lambda c, t: None)
    state = State(pending_lookup="x")
    for expected_failures in (1, 2):
        r = core.handle_continue(CLIENT, state)
        assert r.say == [core.LOOKUP_FAILED] and r.state.failures == expected_failures
        state = r.state.model_copy(update={"pending_lookup": "x"})
    r = core.handle_continue(CLIENT, state)
    assert r.say[0] == core.FALLBACK_LINE
    assert r.state.recipe == core.DEFAULT_RECIPE and r.state.phase == "gather"


def test_non_recipe_request_during_lookup(monkeypatch):
    def not_food(client, text):
        raise core.NotARecipeRequest

    monkeypatch.setattr(core, "find_recipe", not_food)
    r = core.handle_continue(CLIENT, State(pending_lookup="hello?"))
    assert r.say == [core.DISH_PROMPT] and r.state.failures == 0


def test_continue_without_pending_lookup_is_a_noop():
    r = core.handle_continue(CLIENT, State())
    assert r.say == [] and not r.auto_continue


def test_state_without_a_recipe_resets(stubbed):
    r = core.handle_turn(CLIENT, State(phase="cook"), "next")
    assert r.state.phase == "choose"


def test_state_size_limits():
    with pytest.raises(ValidationError):
        State(chunk=10_000)
    with pytest.raises(ValidationError):
        State(notes=["x"] * (core.MAX_NOTES + 1))


def test_cooking_position_is_clamped():
    session = core.CookingSession("x", ["a"], ["b"], client=CLIENT, current=99)
    assert session.current == 1  # last item of a two-step plan
