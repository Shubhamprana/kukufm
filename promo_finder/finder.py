"""Rule-based promo candidate generation and scoring.

Stage 1 of the pipeline. It is cheap, deterministic and runs on every beat of
every episode, so the (paid) LLM only ever sees a short list of good windows.

Promo shape we search for:
  * single:  one continuous stretch, 20-60 s, inside one episode
  * stitch:  a short "cold open" hook clip (3-10 s) + a continuous stretch that
             ends on a cliffhanger. The hook comes from the same episode or the
             previous one. This is the classic ReelShort/DramaBox ad format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import Beat, Episode, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0
SWEET_LO, SWEET_HI = 25.0, 45.0
HOOK_WINDOW = 3.0
CLIFF_WINDOW = 6.0

# --- lexicons (kept small and readable on purpose; tune per genre) -------------
VIOLENCE = r"slap|strike|struck|strikes|hit|punch|shove|shoves|grab|seize|drag|chain|blood|bleed|wound|knife|blade|sword|dagger|crop|whip|iron|burn|fire|flame|ablaze|smash|shatter|crash|slam|choke|strangle|poison|convuls|collapse|faint|kill|dead|death|execut|hang|gallows|arrest|guards|thug|attack|fight|lunge|explosion|blast"
SHOCK = r"shock|stun|gasp|horror|horrif|disbelief|freez|realiz|reveal|unmask|recogni|frozen|jaw drops|eyes widen|eyes snap|trembl|scream|sob|tears|betray|humiliat|kneel|bow|salute|crest|signet|duke|duchess|heir|witch"
ROMANCE = r"kiss|embrace|naked|shirtless|lips|passion|pull(s|ed)? her|press(es|ed)? against"
CLIFF_TAG = r"cliffhanger|cut to black|smash cut|cuts? to black|ominous|sting|abrupt|freeze frame|hard cut"
RESOLVE = r"smile|smiling|relief|relieved|peace|peaceful|warm|gentle|soft(ly)?|fade(s)? out|laugh|content|sleep|calm"
STRONG_TONE = r"furious|rage|venom|vicious|cold|scream|shout|desperate|cruel|scorn|snide|mock|threat|menac|sinister|shocked|horrified|stunned|commanding|icy|deadly|hiss|sneer|accus|enraged|panic|terrified|defiant"
THREAT = r"\b(will|shall|tomorrow|tonight|dawn|before|unless|or else|never|die|death|burn|hang|execute|kill|destroy|ruin|pay|regret|mine|yours)\b"
NAME_CALL = r"\b(you|your)\b"
REACTION_TONE = r"gasp|devastat|heartbroken|incredulous|stunned|shock|horrif|breathless|whisper|disbelief|unbeliev|aghast|broken|faint|trembl"

_rx = {k: re.compile(v if k in ("threat", "you") else r"\b(?:" + v + ")", re.I) for k, v in dict(
    violence=VIOLENCE, shock=SHOCK, romance=ROMANCE, cliff=CLIFF_TAG, resolve=RESOLVE,
    tone=STRONG_TONE, threat=THREAT, you=NAME_CALL, react=REACTION_TONE).items()}


def _hits(key: str, text: str) -> int:
    return len(_rx[key].findall(text))


def beat_intensity(b: Beat) -> float:
    """How likely this single beat is to stop a scroll (0..~10)."""
    t = b.text
    s = 1.6 * min(_hits("violence", t), 3) + 1.2 * min(_hits("shock", t), 3) + 1.0 * min(_hits("romance", t), 2)
    if b.kind == "dialogue":
        words = len(t.split())
        s += 1.5 * min(_hits("tone", b.tone), 2)
        if words <= 12:
            s += 1.0
        if "!" in t:
            s += 1.0
        if "?" in t:
            s += 0.7
        if _hits("you", t):
            s += 0.6
        s += 0.5 * min(_hits("threat", t), 2)
    elif b.kind == "sfx":
        s += 0.8 if _hits("violence", t) or "slam" in t.lower() or "crack" in t.lower() else 0
    elif b.kind == "camera":
        s *= 0.5  # camera notes describe framing; trust them less
    return s


@dataclass
class Segment:
    episode: int
    start: float
    end: float
    beats: list[Beat] = field(repr=False, default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self):
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_sec": round(self.start, 2), "end_sec": round(self.end, 2),
                "duration": round(self.duration, 2)}


@dataclass
class Candidate:
    kind: str                     # single | stitch
    segments: list[Segment]
    features: dict = field(default_factory=dict)
    score: float = 0.0
    hook_text: str = ""
    cliff_text: str = ""
    why: str = ""
    llm: dict | None = None
    final_score: float = 0.0
    source: str = "rules"         # rules | ai_scan | both

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def main_episode(self) -> int:
        return self.segments[-1].episode

    @property
    def beats(self) -> list[Beat]:
        return [b for s in self.segments for b in s.beats]

    def transcript(self, with_camera: bool = True) -> str:
        out = []
        for k, s in enumerate(self.segments, 1):
            out.append(f"--- SEGMENT {k}: Episode {s.episode} {fmt(s.start)} -> {fmt(s.end)} ({s.duration:.1f}s) ---")
            out += [b.line() for b in s.beats if with_camera or b.kind != "camera"]
        return "\n".join(out)

    def key(self) -> str:
        return "|".join(f"{s.episode}:{s.start:.2f}-{s.end:.2f}" for s in self.segments)


# --- boundary helpers -------------------------------------------------------------

def _safe_cut(ep: Episode, t: float) -> bool:
    """A cut at time t must not chop a line of dialogue in half."""
    return not any(b.kind == "dialogue" and b.start + 0.05 < t < b.end - 0.05 for b in ep.beats)


def _window(ep: Episode, s: float, e: float) -> list[Beat]:
    return [b for b in ep.beats if b.start >= s - 1e-6 and b.end <= e + 1e-6]


def _beats_in(ep: Episode, s: float, e: float) -> list[Beat]:
    return [b for b in ep.beats if b.start < e and b.end > s]


def hook_score(ep: Episode, start: float) -> tuple[float, Beat | None]:
    """Strength of the first 3 seconds, and the beat that carries it."""
    early = [b for b in ep.beats if start - 1e-6 <= b.start < start + HOOK_WINDOW]
    if not early:
        return 0.0, None
    best = max(early, key=beat_intensity)
    s = beat_intensity(best)
    # something must actually happen or be said on screen, not just framing
    if not any(b.kind in ("dialogue", "action", "sfx") for b in early):
        s *= 0.4
    return s, best


def cliff_score(ep: Episode, end: float) -> tuple[float, Beat | None, dict]:
    tail = [b for b in ep.beats if end - CLIFF_WINDOW <= b.end <= end + 1e-6]
    if not tail:
        return 0.0, None, {}
    text = " ".join(b.text for b in tail)
    last_line = next((b for b in reversed(tail) if b.kind == "dialogue"), None)
    f = {"tag": _hits("cliff", text) > 0,
         "episode_end": end >= ep.duration - 1.5,
         "question": bool(last_line and "?" in last_line.text),
         "trail": bool(last_line and re.search(r"(\.\.\.|…|—|--)$", last_line.text.strip())),
         "threat": bool(last_line and _hits("threat", last_line.text)),
         "resolve": _hits("resolve", text),
         # a stunned one-to-four-word reaction ("No...", "Her?!") right after a reveal
         "reaction": bool(last_line and len(last_line.text.split()) <= 4 and _hits("react", last_line.tone)),
         "reveal": any(b.kind == "dialogue" and _hits("shock", b.text + " " + b.tone) for b in tail)}
    s = (3.0 * f["tag"] + 2.0 * f["episode_end"] + 1.5 * f["question"] + 1.5 * f["trail"] + 1.0 * f["threat"]
         + 1.5 * f["reaction"] + 0.8 * f["reveal"]
         + 0.6 * sum(min(beat_intensity(b), 6) for b in tail) / max(len(tail), 1)
         - 1.2 * min(f["resolve"], 3))
    anchor = last_line or max(tail, key=beat_intensity)
    return s, anchor, f


def standalone_score(beats: list[Beat], ep_number: int, groups: list[list[Beat]] | None = None) -> tuple[float, dict]:
    """Proxy for 'makes sense to a stranger': few characters, real conflict,
    enough dialogue to carry the story, no long dead air."""
    speakers = {b.speaker for b in beats if b.kind == "dialogue"}
    dialog = [b for b in beats if b.kind == "dialogue"]
    talk_time = sum(b.end - b.start for b in dialog)
    groups = groups or [beats]
    span = sum((max(b.end for b in g) - min(b.start for b in g)) for g in groups if g) or 1
    gaps = [0.0]
    for g in groups:
        ev = sorted(b.start for b in g if b.kind in ("dialogue", "action", "sfx"))
        gaps += [b - a for a, b in zip(ev, ev[1:])]
    f = {"speakers": len(speakers), "talk_ratio": round(talk_time / span, 2), "max_gap": round(max(gaps), 1),
         "conflict": sum(1 for b in dialog if _hits("tone", b.tone)), "lines": len(dialog)}
    s = 0.0
    s += 2.0 if 2 <= f["speakers"] <= 3 else (0.8 if f["speakers"] in (1, 4) else -1.0)
    s += 2.0 * min(f["talk_ratio"], 0.6) / 0.6
    s += min(f["conflict"], 4) * 0.5
    s -= 1.5 if f["max_gap"] > 10 else 0
    s -= 1.0 if f["lines"] < 3 else 0
    # assumption: early episodes carry the premise and need less backstory
    s += 1.0 if ep_number <= 5 else (0.4 if ep_number <= 15 else 0)
    return s, f


def length_score(d: float) -> float:
    if SWEET_LO <= d <= SWEET_HI:
        return 1.0
    return 1.0 - min(abs(d - (SWEET_LO if d < SWEET_LO else SWEET_HI)) / 15.0, 1.0)


def _score(c: Candidate, ep: Episode, hook_ep: Episode) -> None:
    first, last = c.segments[0], c.segments[-1]
    h, hb = hook_score(hook_ep, first.start)
    cl, cb, cf = cliff_score(ep, last.end)
    sa, sf = standalone_score(c.beats, ep.number, [x.beats for x in c.segments])
    ln = length_score(c.duration)
    peak = max((beat_intensity(b) for b in c.beats), default=0)
    c.score = round(1.0 * h + 1.1 * cl + 0.8 * sa + 2.0 * ln + 0.25 * peak - (1.8 if c.kind == "stitch" else 0), 3)
    c.features = {"hook": round(h, 2), "cliff": round(cl, 2), "standalone": round(sa, 2), "length": round(ln, 2),
                  "peak": round(peak, 2), **{f"cliff_{k}": v for k, v in cf.items()}, **sf}
    c.hook_text = hb.line() if hb else ""
    c.cliff_text = cb.line() if cb else ""


# --- generation ---------------------------------------------------------------------

def _starts(ep: Episode, top: int = 14) -> list[float]:
    cands = {0.0}
    for b in ep.beats:
        if b.kind != "camera" and _safe_cut(ep, b.start):
            cands.add(round(b.start, 2))
    ranked = sorted(cands, key=lambda t: -hook_score(ep, t)[0])
    return ranked[:top]


def _ends(ep: Episode, top: int = 16) -> list[float]:
    cands = {round(ep.duration, 2)}
    for b in ep.beats:
        if b.kind in ("dialogue", "action", "sfx") and _safe_cut(ep, b.end):
            cands.add(round(b.end, 2))
    ranked = sorted(cands, key=lambda t: -cliff_score(ep, t)[0])
    return ranked[:top]


def _seg(ep: Episode, s: float, e: float) -> Segment:
    return Segment(ep.number, s, e, _window(ep, s, e))


def _connected(hook: Segment, main: Segment) -> bool:
    """A stitched hook must be about the same people as the clip it leads into,
    otherwise a stranger sees two unrelated scenes."""
    hs = {b.speaker for b in hook.beats if b.kind == "dialogue"}
    ms = {b.speaker for b in main.beats if b.kind == "dialogue"}
    if hs & ms:
        return True
    main_text = " ".join(b.text for b in main.beats).upper()
    hook_text = " ".join(b.text for b in hook.beats).upper()
    return any(n.split()[0] in main_text for n in hs if n) or any(n.split()[0] in hook_text for n in ms if n)


def generate(episodes: list[Episode]) -> list[Candidate]:
    by_num = {e.number: e for e in episodes}
    out: list[Candidate] = []
    for ep in episodes:
        starts, ends = _starts(ep), _ends(ep)
        # 1) single continuous stretch
        for s in starts:
            for e in ends:
                if MIN_LEN <= e - s <= MAX_LEN:
                    c = Candidate("single", [_seg(ep, s, e)])
                    _score(c, ep, ep)
                    out.append(c)
        # 2) stitch: short hook clip + main stretch ending on a cliffhanger
        hook_sources = [ep] + ([by_num[ep.number - 1]] if ep.number - 1 in by_num else [])
        for hep in hook_sources:
            hooks = []
            for b in hep.beats:
                if b.kind == "camera" or not _safe_cut(hep, b.start):
                    continue
                he = next((x.end for x in hep.beats if x.end >= b.start + 3 and x.end <= b.start + 10
                           and _safe_cut(hep, x.end)), None)
                if he:
                    hooks.append((beat_intensity(b), b.start, he))
            for _, hs, he in sorted(hooks, reverse=True)[:3]:
                for e in ends[:5]:
                    for s in sorted(set(starts) | {0.0}):
                        if hep is ep and s < he + 2:     # hook must come from before the main stretch
                            continue
                        total = (he - hs) + (e - s)
                        if MIN_LEN <= total <= MAX_LEN and e - s >= 12:
                            hook_seg, main_seg = _seg(hep, hs, he), _seg(ep, s, e)
                            if not _connected(hook_seg, main_seg):
                                continue
                            c = Candidate("stitch", [hook_seg, main_seg])
                            _score(c, ep, hep)
                            out.append(c)
    return out


def _overlap(a: Candidate, b: Candidate) -> float:
    """Worst-case overlap between any two segments, as a share of the shorter one.
    Stops the same hook clip or the same ending being reused across promos."""
    worst = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                inter = max(0.0, min(x.end, y.end) - max(x.start, y.start))
                worst = max(worst, inter / max(min(x.duration, y.duration), 1e-6))
    return worst


def select(cands: list[Candidate], per_episode: int = 3, max_overlap: float = 0.35, key=lambda c: c.score) -> list[Candidate]:
    """Greedy non-max suppression: best first, drop near-duplicates, cap per episode."""
    chosen: list[Candidate] = []
    per_ep: dict[int, int] = {}
    for c in sorted(cands, key=key, reverse=True):
        if per_ep.get(c.main_episode, 0) >= per_episode:
            continue
        if any(_overlap(c, k) > max_overlap for k in chosen):
            continue
        chosen.append(c)
        per_ep[c.main_episode] = per_ep.get(c.main_episode, 0) + 1
    return chosen
