"""Offline tests for server.py. Speech and Claude calls are stubbed; no keys or network needed."""

import json

import pytest
from fastapi.testclient import TestClient

import core
import server


@pytest.fixture
def api(monkeypatch):
    monkeypatch.delenv("ACCESS_CODE", raising=False)
    monkeypatch.setattr(server, "speak", lambda text: "QUJD")  # pretend every line was spoken
    monkeypatch.setattr(server, "get_client", lambda: object())
    server._hits.clear()
    return TestClient(server.app)


def turn(api, state, **fields):
    return api.post("/api/turn", data={"state": json.dumps(state), **fields})


def test_page_is_served(api):
    r = api.get("/")
    assert r.status_code == 200 and "Start cooking" in r.text


def test_healthz(api):
    assert api.get("/healthz").json() == {"ok": True}


def test_start(api):
    r = api.post("/api/start")
    assert r.status_code == 200
    body = r.json()
    assert body["say"] == [core.ASK_DISH]
    assert body["audio"] == ["QUJD"]
    assert body["state"]["phase"] == "choose"


def test_typed_greeting_needs_no_model(api):
    state = api.post("/api/start").json()["state"]
    body = turn(api, state, text="hello").json()
    assert body["transcript"] == "hello"
    assert body["say"] == [core.DISH_PROMPT]


def test_audio_is_transcribed_with_recipe_keyterms(api, monkeypatch):
    seen = {}

    def fake_transcribe(data, content_type, keyterms):
        seen.update(content_type=content_type, keyterms=keyterms)
        return "hello"

    monkeypatch.setattr(server, "transcribe", fake_transcribe)
    state = core.State(phase="confirm", recipe=core.DEFAULT_RECIPE).model_dump(mode="json")
    r = api.post(
        "/api/turn",
        data={"state": json.dumps(state)},
        files={"audio": ("speech.webm", b"x" * 3000, "audio/webm;codecs=opus")},
    )
    assert r.status_code == 200 and r.json()["transcript"] == "hello"
    assert seen == {"content_type": "audio/webm;codecs=opus", "keyterms": ["simmer"]}


def test_tiny_recording_is_treated_as_silence(api):
    state = api.post("/api/start").json()["state"]
    r = api.post(
        "/api/turn",
        data={"state": json.dumps(state)},
        files={"audio": ("speech.webm", b"x" * 10, "audio/webm")},
    )
    assert "didn't catch that" in r.json()["say"][0]


def test_oversized_recording_is_rejected(api):
    state = api.post("/api/start").json()["state"]
    r = api.post(
        "/api/turn",
        data={"state": json.dumps(state)},
        files={"audio": ("speech.webm", b"x" * (server.MAX_AUDIO_BYTES + 10), "audio/webm")},
    )
    assert r.status_code == 413


def test_turn_needs_audio_or_text(api):
    state = api.post("/api/start").json()["state"]
    assert turn(api, state).status_code == 400


def test_invalid_state_is_rejected(api):
    assert turn(api, {"phase": "nonsense"}, text="hi").status_code == 422
    assert api.post("/api/turn", data={"state": "not json", "text": "hi"}).status_code == 422


def test_continue_runs_the_queued_lookup(api, monkeypatch):
    monkeypatch.setattr(core, "find_recipe", lambda client, text: core.DEFAULT_RECIPE)
    state = core.State(pending_lookup="pasta").model_dump(mode="json")
    body = api.post("/api/continue", json={"state": state}).json()
    assert body["state"]["phase"] == "confirm" and "Simple tomato pasta" in body["say"][0]


def test_access_code(api, monkeypatch):
    monkeypatch.setenv("ACCESS_CODE", "secret")
    assert api.post("/api/start").status_code == 401
    assert api.post("/api/start", headers={"X-Access-Code": "wrong"}).status_code == 401
    assert api.post("/api/start", headers={"X-Access-Code": "secret"}).status_code == 200


def test_rate_limit(api, monkeypatch):
    monkeypatch.setattr(server, "RATE_LIMIT_PER_MIN", 2)
    assert api.post("/api/start").status_code == 200
    assert api.post("/api/start").status_code == 200
    assert api.post("/api/start").status_code == 429


def test_failed_speech_still_returns_the_text(api, monkeypatch):
    monkeypatch.setattr(server, "speak", lambda text: None)  # ElevenLabs down
    body = api.post("/api/start").json()
    assert body["say"] == [core.ASK_DISH] and body["audio"] == [None]
