"""LLM judge (stage 2). Model-agnostic: one function `chat(system, user) -> str`
per provider, all plain HTTP via the standard library. Swap with --llm/--model
or env vars; nothing else in the pipeline knows which model is used.

Providers
  gemini      GEMINI_API_KEY              (free tier, default model gemini-3-flash-preview)
  openai      OPENAI_API_KEY + OPENAI_BASE_URL (any OpenAI-compatible API:
              Groq, OpenRouter, Together, Ollama, vLLM, OpenAI, an in-house gateway)
  anthropic   ANTHROPIC_API_KEY
  none        skip the LLM, rules only
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .finder import Candidate, MIN_LEN, MAX_LEN, _safe_cut, _window, hook_score
from .parser import Episode, fmt

DEFAULT_MODELS = {"gemini": "gemini-3-flash-preview", "openai": "llama-3.3-70b-versatile", "anthropic": "claude-haiku-4-5"}

SYSTEM = """You are a senior performance-marketing editor for vertical microdrama apps (ReelShort, DramaBox style).
You pick 20-60 second promo clips that make strangers stop scrolling on TikTok/Reels and install the app.
You judge ONLY from the timestamped screenplay excerpt you are given. Never invent lines or timestamps.
Return strict JSON, no markdown."""

PROMPT = """Show premise (for your context only; the viewer of the ad does NOT see this):
{premise}

Candidate promo ({kind}, {duration:.1f}s total). Segments are played in order:
{transcript}

Score this candidate 1-10 on each criterion:
- hook: do the FIRST 3 SECONDS stop a scrolling viewer (slap, reveal, shocking line, danger)? A camera establishing shot is weak.
- cliffhanger: does it END on an unresolved moment so the viewer must know what happens next? Ending on resolution, comfort or a calm beat is weak.
- standalone: would someone who has never seen the show understand who wants what and why it matters, without backstory?
- emotion: intensity of the conflict (betrayal, humiliation, revenge, forbidden romance, power reversal).
- overall: would you spend ad budget on it?

Also:
- hook_quote: copy the exact line or action text from the first 3 seconds that works as the hook.
- cliffhanger_quote: copy the exact final line/action that leaves it unresolved.
- why: ONE sentence on why it works (or why not), written for a marketing manager.
- trim: optionally tighten the clip. Give new start/end times in seconds for each segment, only moving start LATER or end EARLIER, or null for no change.
- risks: short list of problems (needs backstory, too many characters, resolved ending, slow start, spoiler of a major twist, etc.).

JSON schema:
{{"hook": int, "cliffhanger": int, "standalone": int, "emotion": int, "overall": int,
  "hook_quote": str, "cliffhanger_quote": str, "why": str,
  "trim": [{{"start": float, "end": float}}] | null, "risks": [str]}}"""


class LLMError(RuntimeError):
    pass


def _post(url: str, payload: dict, headers: dict, retries: int = 4) -> dict:
    data = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            full = e.read().decode(errors="replace")
            body = full[:300]
            if e.code == 429 and re.search(r"per ?day|PerDay", full):
                raise LLMError(f"DAILY QUOTA: {body[:160]}") from e   # retrying won't help today
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(2 ** attempt * 4)   # free tiers rate-limit hard
                continue
            raise LLMError(f"HTTP {e.code}: {body}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise LLMError(str(e)) from e
    raise LLMError("unreachable")


class Judge:
    def __init__(self, provider: str = "none", model: str | None = None, cache_dir: str | Path = Path(__file__).resolve().parent.parent / ".cache"):
        self.provider = provider
        self.model = model or os.getenv("PROMO_MODEL") or DEFAULT_MODELS.get(provider, "")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def enabled(self) -> bool:
        return self.provider != "none"

    # -- provider adapters ---------------------------------------------------
    def chat(self, system: str, user: str) -> str:
        p = self.provider
        if p == "gemini":
            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise LLMError("set GEMINI_API_KEY (free at aistudio.google.com)")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            r = _post(url, {"systemInstruction": {"parts": [{"text": system}]},
                            "contents": [{"role": "user", "parts": [{"text": user}]}],
                            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}},
                      {"x-goog-api-key": key})
            u = r.get("usageMetadata", {})
            self.tokens_in += u.get("promptTokenCount", 0)
            self.tokens_out += u.get("candidatesTokenCount", 0)
            return r["candidates"][0]["content"]["parts"][0]["text"]
        if p == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            base = os.environ.get("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
            r = _post(f"{base}/chat/completions",
                      {"model": self.model, "temperature": 0.2, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                      {"Authorization": f"Bearer {key}"} if key else {})
            u = r.get("usage", {})
            self.tokens_in += u.get("prompt_tokens", 0)
            self.tokens_out += u.get("completion_tokens", 0)
            return r["choices"][0]["message"]["content"]
        if p == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("set ANTHROPIC_API_KEY")
            r = _post("https://api.anthropic.com/v1/messages",
                      {"model": self.model, "max_tokens": 800, "temperature": 0.2, "system": system,
                       "messages": [{"role": "user", "content": user}]},
                      {"x-api-key": key, "anthropic-version": "2023-06-01"})
            u = r.get("usage", {})
            self.tokens_in += u.get("input_tokens", 0)
            self.tokens_out += u.get("output_tokens", 0)
            return "".join(b.get("text", "") for b in r["content"])
        raise LLMError(f"unknown provider {p}")

    def _cache_path(self, system: str, user: str) -> Path:
        h = hashlib.sha256(f"{self.provider}|{self.model}|{system}|{user}".encode()).hexdigest()[:24]
        return self.cache / f"{h}.json"

    def _cached(self, system: str, user: str) -> str:
        f = self._cache_path(system, user)
        if f.exists():
            return f.read_text(encoding="utf-8")
        out = self.chat(system, user)
        self.calls += 1
        f.write_text(out, encoding="utf-8")
        return out

    # -- judging -----------------------------------------------------------------
    def judge(self, c: Candidate, premise: str, episodes: dict[int, Episode]) -> None:
        user = PROMPT.format(premise=premise, kind=c.kind, duration=c.duration, transcript=c.transcript())
        raw = self._cached(SYSTEM, user)
        try:
            data = parse_json(raw)
        except (json.JSONDecodeError, LLMError):
            # malformed JSON: ask once more, uncached, with a stricter nudge
            raw = self.chat(SYSTEM, user + "\n\nReturn ONLY valid JSON. Escape quotes inside strings.")
            self.calls += 1
            data = parse_json(raw)
            self._cache_path(SYSTEM, user).write_text(raw, encoding="utf-8")
        c.llm = validate(data, c, episodes)


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise LLMError(f"no JSON in model output: {raw[:200]}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        # Seen in practice: the model writes 01:51.1 as the number 0151.1 (invalid JSON,
        # wrong unit). Convert MMSS.s with a leading zero back into seconds.
        fixed = re.sub(r"(?<=[:\[,\s])0(\d)(\d{2}(?:\.\d+)?)(?=\s*[,}\]])",
                       lambda g: str(round(int(g.group(1)) * 60 + float(g.group(2)), 2)), m.group(0))
        return json.loads(fixed)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


LABEL_RE = re.compile(r"(\[[^\]]*\]\s*)?\b([A-Z][A-Z0-9'.\-]+(?: [A-Z0-9'.\-]+)*(\s*\([^)]*\))*)\s*:\s*")


def quote_in_script(val: str, norm_text: str) -> bool:
    """True if every sentence of the quote is in the script.
    Speaker/SFX labels ('CATHERINE (Venomous):', 'SFX:') anywhere in the quote are
    ignored, one-word lines ('Her?!') count, invented words fail."""
    if not val.strip():
        return False
    stripped = LABEL_RE.sub(" | ", val)
    pieces = [_norm(p) for p in re.split(r"[.!?…|\n]+", stripped)]
    pieces = [p for p in pieces if p]
    padded = f" {norm_text} "
    return bool(pieces) and all((f" {p} " if len(p) <= 60 else f" {p[:60]}") in padded for p in pieces)


def validate(d: dict, c: Candidate, episodes: dict[int, Episode]) -> dict:
    """Never trust the model blindly: clamp scores, check quotes exist in the
    script, and only accept trims that land on real, safe cut points."""
    keys = ("hook", "cliffhanger", "standalone", "emotion", "overall")
    out, missing = {}, []
    for k in keys:
        try:
            out[k] = max(1, min(10, int(round(float(d[k])))))
        except (KeyError, TypeError, ValueError):
            missing.append(k)
    # a skipped score is filled from the others, never a silent constant
    fill = round(sum(out.values()) / len(out)) if out else 5
    for k in missing:
        out[k] = fill
    out["missing_scores"] = missing
    text = _norm(" ".join(b.text for b in c.beats))
    for q in ("hook_quote", "cliffhanger_quote"):
        val = str(d.get(q, "") or "")
        out[q] = val
        out[q + "_verified"] = quote_in_script(val, text)
    out["why"] = str(d.get("why", ""))[:300]
    out["risks"] = [str(r)[:120] for r in (d.get("risks") or [])][:5]
    out["trim_applied"] = False
    trim = d.get("trim")
    if isinstance(trim, list) and len(trim) == len(c.segments):
        new = []
        for seg, t in zip(c.segments, trim):
            if not isinstance(t, dict):
                new.append((seg.start, seg.end)); continue
            ep = episodes[seg.episode]
            s = float(t.get("start", seg.start) or seg.start)
            e = float(t.get("end", seg.end) or seg.end)
            s = max(seg.start, min(s, seg.end))
            e = min(seg.end, max(e, seg.start))
            # snap to the nearest real beat boundary that does not cut dialogue
            starts = [b.start for b in ep.beats if seg.start <= b.start <= seg.end and _safe_cut(ep, b.start)]
            ends = [b.end for b in ep.beats if seg.start <= b.end <= seg.end and _safe_cut(ep, b.end)]
            s = min(starts, key=lambda x: abs(x - s), default=seg.start)
            e = min(ends, key=lambda x: abs(x - e), default=seg.end)
            new.append((s, e) if e - s > 3 else (seg.start, seg.end))
        total = sum(e - s for s, e in new)
        # never accept a trim that weakens the opening hook (the model once cut
        # "You should turn around, Juliana" off the front of the Ep 1 altar clip)
        first_ep = episodes[c.segments[0].episode]
        old_hook = hook_score(first_ep, c.segments[0].start)[0]
        new_hook = hook_score(first_ep, new[0][0])[0]
        out["trim_rejected_weaker_hook"] = new[0][0] != c.segments[0].start and new_hook < 0.9 * old_hook
        if MIN_LEN <= total <= MAX_LEN and not out["trim_rejected_weaker_hook"] and any(abs(a - seg.start) + abs(b - seg.end) > 0.1 for (a, b), seg in zip(new, c.segments)):
            out["trim_from"] = [f"E{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in c.segments]
            for seg, (s, e) in zip(c.segments, new):
                seg.start, seg.end = s, e
                seg.beats = _window(episodes[seg.episode], s, e)
            out["trim_applied"] = True
    return out
