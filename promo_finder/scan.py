"""Stage 1b: the LLM reads each whole episode and proposes its own promo moments.

Why: the rule stage is cheap but keyword-blind. On the first show it scored the
best clip in the pack (E41, cathedral explosion) 0.6/10, and rules and LLM ranked
the shortlist very differently (Spearman 0.21). Anything the rules miss never
reaches the judge. One extra call per episode (~1.5k tokens) fixes that recall
gap; proposals are validated exactly like rule candidates and then judged by the
same judge, so they compete on equal terms.
"""
from __future__ import annotations

from .finder import Candidate, MIN_LEN, MAX_LEN, _safe_cut, _score, _seg
from .llm import Judge, parse_json, quote_in_script, _norm
from .parser import Episode

SCAN_SYSTEM = """You are a senior performance-marketing editor for vertical microdrama apps (ReelShort, DramaBox style).
You read one episode's timestamped screenplay and find the moments that would make the best 20-60 second promo ads.
Use ONLY lines and times that appear in the script. Times are in SECONDS. Return strict JSON, no markdown."""

SCAN_PROMPT = """Show premise (context only, the ad viewer does not see it):
{premise}

Episode {num} ({duration:.0f}s). Logline: {logline}
Script (one beat per line, [start-end] in seconds; camera directions removed):
{lines}

Propose up to {k} promo clips from THIS episode. A great promo:
- opens on a hook in its first 3 seconds (slap, reveal, shocking or threatening line, danger) - not a calm establishing beat
- ends on a cliffhanger: unresolved, the viewer must know what happens next - never on comfort or resolution
- makes sense to someone who has never seen the show
- is 20-60 seconds total (25-45 ideal). Usually one continuous stretch; optionally a 3-10s hook moment followed by a later stretch from the same episode.

Start each segment exactly at the start time of a beat and end it exactly at the end time of a beat.
Prefer distinct, non-overlapping moments. Fewer, better proposals beat filler.

JSON: {{"promos": [{{"segments": [{{"start": float, "end": float}}], "hook_quote": str, "cliffhanger_quote": str, "reason": str}}]}}"""


def episode_lines(ep: Episode) -> str:
    out = []
    for b in ep.beats:
        if b.kind == "camera":
            continue
        who = f"{b.speaker} ({b.tone})" if b.kind == "dialogue" else b.kind.upper()
        out.append(f"[{b.start:.1f}-{b.end:.1f}] {who}: {b.text}")
    return "\n".join(out)


def _snap_start(ep: Episode, t: float) -> float | None:
    opts = [b.start for b in ep.beats if b.kind != "camera" and _safe_cut(ep, b.start)]
    return min(opts, key=lambda x: abs(x - t)) if opts else None


def _snap_end(ep: Episode, t: float) -> float | None:
    opts = [b.end for b in ep.beats if b.kind != "camera" and _safe_cut(ep, b.end)]
    return min(opts, key=lambda x: abs(x - t)) if opts else None


def scan_episode(judge: Judge, ep: Episode, premise: str, k: int = 3) -> tuple[list[Candidate], list[str]]:
    """Returns valid candidates and a list of rejection reasons (kept for the QA report)."""
    user = SCAN_PROMPT.format(premise=premise, num=ep.number, duration=ep.duration, logline=ep.logline,
                              lines=episode_lines(ep), k=k)
    data = parse_json(judge._cached(SCAN_SYSTEM, user))
    cands, rejects = [], []
    for p in (data.get("promos") or [])[:k]:
        try:
            segs = []
            for s in p.get("segments") or []:
                a, b = _snap_start(ep, float(s["start"])), _snap_end(ep, float(s["end"]))
                drift = abs(a - float(s["start"])) + abs(b - float(s["end"]))
                if a is None or b is None or b - a < 2:
                    raise ValueError("segment shorter than 2s after snapping")
                if drift > 6:
                    raise ValueError(f"times not on real beats (drift {drift:.1f}s)")
                segs.append((a, b))
            segs.sort()
            if not segs or len(segs) > 3:
                raise ValueError("needs 1-3 segments")
            if any(nxt[0] < prev[1] for prev, nxt in zip(segs, segs[1:])):
                raise ValueError("segments overlap")
            total = sum(b - a for a, b in segs)
            extended = False
            if total < MIN_LEN:
                # Slightly short: pull the last segment's start earlier (the ending,
                # i.e. the cliffhanger, never moves) until the promo reaches 20s.
                need = MIN_LEN - total
                floor = segs[-2][1] if len(segs) > 1 else 0.0
                earlier = sorted((b.start for b in ep.beats if b.kind != "camera" and _safe_cut(ep, b.start)
                                  and floor <= b.start <= segs[-1][0] - need), reverse=True)
                if not earlier or need > 12:
                    raise ValueError(f"length {total:.1f}s under 20s and cannot be extended")
                segs[-1] = (earlier[0], segs[-1][1])
                total = sum(b - a for a, b in segs)
                extended = True
            if not (MIN_LEN <= total <= MAX_LEN):
                raise ValueError(f"length {total:.1f}s outside 20-60s")
            c = Candidate("single" if len(segs) == 1 else "stitch", [_seg(ep, a, b) for a, b in segs])
            _score(c, ep, ep)   # rule features, computed the same way as for rule candidates
            c.features["scan_extended"] = extended
            text = _norm(" ".join(x.text for x in c.beats))
            c.features["scan_hook_quote_verified"] = quote_in_script(str(p.get("hook_quote", "")), text)
            c.source = "ai_scan"
            c.features["scan_reason"] = str(p.get("reason", ""))[:200]
            cands.append(c)
        except (KeyError, TypeError, ValueError) as e:
            rejects.append(f"E{ep.number}: {e}")
    return cands, rejects
