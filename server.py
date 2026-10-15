"""FastAPI server for the cooking assistant web app.

Run it with:
    uvicorn server:app --reload

The browser does the listening and the speaking. This server:
  * transcribes the audio it receives (ElevenLabs Scribe, with keyterm hints),
  * runs one step of the assistant (core.py),
  * turns each line the chef says into speech and sends the audio back with the reply.

It keeps no session data. The browser sends the session state with every request and
gets the updated state back, so any number of people can use it at once and a
restart loses nothing.

API keys stay on the server. Anyone who can reach this app can spend your credits,
so set ACCESS_CODE before putting it on the internet.
"""

import base64
import hmac
import io
import logging
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

import core

load_dotenv()  # reads .env into environment variables
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chef.server")

# ---------- config ----------

BASE_DIR = Path(__file__).parent

# Text-to-speech / speech-to-text
VOICE_ID = "1SM7GgM6IMuvQlz2BwM3"  # "Mark", https://elevenlabs.io/voices/1SM7GgM6IMuvQlz2BwM3
TTS_MODEL = "eleven_flash_v2_5"    # lowest-latency ElevenLabs model
STT_MODEL = "scribe_v2"

MAX_AUDIO_BYTES = 5 * 1024 * 1024  # a 25 second recording is well under 1 MB
MIN_AUDIO_BYTES = 800              # smaller than this is a mis-tap, not speech
MAX_TEXT_CHARS = 1000
TTS_CACHE_SIZE = 256               # step lines get repeated a lot; don't pay for them twice
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))

app = FastAPI(title="Cooking assistant")


# ---------- clients (created on first use so the app can start without keys) ----------

@lru_cache
def get_client() -> anthropic.Anthropic:
    return core.make_client()


@lru_cache
def get_eleven() -> ElevenLabs:
    return ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))


# ---------- access control ----------

_hits: dict[str, deque[float]] = defaultdict(deque)


def guard(request: Request, x_access_code: str | None = Header(default=None)) -> None:
    """Require the access code (if one is set) and apply a simple per-IP rate limit."""
    code = os.getenv("ACCESS_CODE")
    if code and not hmac.compare_digest((x_access_code or "").encode(), code.encode()):
        raise HTTPException(401, "Access code required.")

    # Behind a proxy every request may share one address; configure your host to
    # pass the real client IP through if you need this to be per-person.
    ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _hits[ip]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "Too many requests. Wait a moment and try again.")
    window.append(now)


# ---------- audio ----------

def transcribe(data: bytes, content_type: str, keyterms: list[str]) -> str:
    """Speech to text. `keyterms` bias Scribe toward words we expect (Scribe v2, costs extra)."""
    ext = (
        "webm" if "webm" in content_type
        else "mp4" if "mp4" in content_type or "m4a" in content_type
        else "ogg" if "ogg" in content_type
        else "wav"
    )
    buf = io.BytesIO(data)
    buf.name = f"speech.{ext}"  # the extension helps the service identify the format
    extra = {"keyterms": keyterms} if keyterms else {}  # only send the field when we have terms
    try:
        result = get_eleven().speech_to_text.convert(
            file=buf,
            model_id=STT_MODEL,
            language_code="eng",     # set to None to auto-detect the language
            tag_audio_events=False,  # stops things like "(laughter)" appearing in text
            **extra,
        )
    except Exception as e:
        log.warning("STT failed: %s: %s", type(e).__name__, e)
        raise HTTPException(502, "Couldn't transcribe that. Check the ElevenLabs key and try again.")
    return result.text.strip()


@lru_cache(maxsize=TTS_CACHE_SIZE)
def _speech_bytes(text: str) -> bytes:
    chunks = get_eleven().text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=TTS_MODEL,
        output_format="mp3_44100_128",
    )
    return b"".join(chunks)  # the SDK returns an iterator of bytes


def speak(text: str) -> str | None:
    """Base64 MP3 of `text`, or None if speech failed (the page then shows or reads the text)."""
    try:
        return base64.b64encode(_speech_bytes(text)).decode("ascii")
    except Exception as e:
        log.warning("TTS failed: %s: %s", type(e).__name__, e)
        return None


_pool = ThreadPoolExecutor(max_workers=4)  # speak a reply's lines in parallel


# ---------- API ----------

class TurnResponse(BaseModel):
    transcript: str = ""
    say: list[str]
    audio: list[str | None]  # one base64 MP3 per line in `say`
    state: core.State
    auto_continue: bool = False
    done: bool = False


class ContinueBody(BaseModel):
    state: core.State


def respond(transcript: str, result: core.TurnResult) -> TurnResponse:
    return TurnResponse(
        transcript=transcript,
        say=result.say,
        audio=list(_pool.map(speak, result.say)),
        state=result.state,
        auto_continue=result.auto_continue,
        done=result.done,
    )


def parse_state(raw: str) -> core.State:
    try:
        return core.State.model_validate_json(raw)
    except ValidationError:
        raise HTTPException(422, "That session can't be continued. Reload the page to start again.")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/api/start", response_model=TurnResponse, dependencies=[Depends(guard)])
def start() -> TurnResponse:
    """Begin a session: the chef asks what to cook."""
    return respond("", core.new_session())


@app.post("/api/turn", response_model=TurnResponse, dependencies=[Depends(guard)])
def turn(
    state: str = Form(...),
    text: str | None = Form(default=None),
    audio: UploadFile | None = File(default=None),
) -> TurnResponse:
    """One thing the user said, as recorded audio or typed text."""
    current = parse_state(state)

    if audio is not None:
        data = audio.file.read(MAX_AUDIO_BYTES + 1)
        if len(data) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "That recording is too long. Keep it under 25 seconds.")
        keyterms = current.recipe.keyterms if current.recipe else []
        transcript = (
            transcribe(data, audio.content_type or "", keyterms) if len(data) >= MIN_AUDIO_BYTES else ""
        )
    elif text is not None:
        transcript = text.strip()[:MAX_TEXT_CHARS]
    else:
        raise HTTPException(400, "Send audio or text.")

    if not transcript:
        return respond("", core.TurnResult(
            say=["Sorry, I didn't catch that. Please try again."], state=current
        ))

    return respond(transcript, core.handle_turn(get_client(), current, transcript))


@app.post("/api/continue", response_model=TurnResponse, dependencies=[Depends(guard)])
def continue_turn(body: ContinueBody) -> TurnResponse:
    """Run the slow recipe lookup that the previous turn queued up."""
    return respond("", core.handle_continue(get_client(), body.state))
