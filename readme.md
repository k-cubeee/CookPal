# Chef: a talking cooking assistant for the web

Chef asks what you want to cook, finds a recipe on the web, reads out the ingredients a few at a
time, then talks you through each step. It listens for your answer on its own (or you can type).

## Run it on your machine

```
pip install -r requirements.txt
cp .env.example .env        # then add your two API keys
uvicorn server:app --reload
```

Open http://localhost:8000. Browsers allow the microphone on `localhost` without HTTPS.

Web search must be enabled for your organisation in the Claude Console, or the recipe lookup fails
and Chef falls back to the built-in tomato pasta.

## Files

| File | What it does |
| --- | --- |
| `core.py` | The assistant's logic: recipe lookup, ingredient gathering, cooking steps. No audio. |
| `server.py` | FastAPI app: transcribes speech, runs a turn, speaks the reply, serves the page. |
| `index.html` | The page: hands-free listening, big current line, progress, typed fallback. |
| `test_core.py`, `test_server.py` | Offline tests. They need no API keys: `pytest` |
| `cooking_assistant.py` | The original command-line version. It still works on its own. |

## How a turn works

1. After the chef finishes speaking, the page starts listening by itself. It begins recording when you
   talk and stops when you pause, then sends the recording and the session state.
2. The server transcribes it (Scribe v2, with keyterms from the recipe) and runs one step of `core.py`.
3. The reply comes back as text plus spoken audio, and the new session state.
4. For the slow recipe search, Chef says "Let me look that up" first. The page then calls
   `/api/continue` to run the search while that line plays.

The server stores nothing between requests. The session state lives in the browser, so the server can
restart, or run as several copies, without losing anyone's place.

## Put it online

Use a host that runs a normal always-on process (Render, Fly.io or Railway). The recipe search can take
up to 90 seconds, which serverless platforms with short request limits will cut off.

```
uvicorn server:app --host 0.0.0.0 --port $PORT
```

- **HTTPS is required.** Browsers only allow the microphone on secure pages. The hosts above provide it.
- **Set `ACCESS_CODE`.** Without it, anyone who finds the URL can spend your Anthropic and ElevenLabs credits.
- **Rate limiting is per IP and kept in memory.** It resets on restart and isn't shared between copies. If
  your host puts a proxy in front, make sure the real client IP is passed through.
- Keep your API keys in the host's environment settings, not in the repository.

## Known limits

- Hands-free listening is on by default. It listens to the room first and treats anything clearly louder
  as speech, so a very noisy kitchen can fool it. Untick "Listen automatically" to use the mic button
  instead: tap it to start talking and tap again to send. Tapping also works in hands-free mode if the
  page misses you. Wear headphones if the chef's voice keeps interrupting you.
- The page never listens while the chef is speaking, so you can't talk over it. Tap the mic to interrupt.
- Steps are read out exactly as written, so a substitution you mention is remembered for questions but
  not applied to the step text.
- Scribe's keyterm hints cost extra per request.

**DISCLAIMER: I made this in an afternoon using claude and elevenlabs for a hackathon called Accel Innovate**
