"""Parse Kuku-style timestamped screenplays into a flat list of beats.

A "beat" is one CAMERA / ACTION / SOUND FX / dialogue block with a start and
end time in seconds. Everything downstream works on beats, so a new show only
needs to parse into this shape.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"\((\d+):(\d+(?:\.\d+)?)\)"
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
TIME_RE = re.compile(rf"^Start:\s*{TS}\s*\|\s*End:\s*{TS}")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)", re.I)
HEADING_RE = re.compile(r"^Scene heading:\s*(.+)", re.I)
EP_RE = re.compile(r"^EPISODE NUMBER:\s*Episode\s*(\d+)", re.I)
NON_DIALOGUE = {"CAMERA", "ACTION", "SOUND FX", "SOUND", "SFX", "MUSIC", "VO", "TEXT", "SUPER"}


def to_sec(m: str, s: str) -> float:
    return int(m) * 60 + float(s)


def fmt(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    episode: int
    idx: int
    scene: int
    heading: str
    kind: str            # camera | action | sfx | dialogue
    speaker: str         # character name for dialogue, else ""
    tone: str            # parenthetical like "Cold, Commanding"
    text: str
    start: float
    end: float

    def line(self) -> str:
        t = f"[{fmt(self.start)}-{fmt(self.end)}]"
        if self.kind == "dialogue":
            tone = f" ({self.tone})" if self.tone else ""
            return f"{t} {self.speaker}{tone}: {self.text}"
        return f"{t} {self.kind.upper()}: {self.text}"


@dataclass
class Episode:
    number: int
    path: str
    logline: str = ""
    continuity: str = ""
    beats: list[Beat] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)

    def to_dict(self):
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _label(raw: str) -> tuple[str, str, str]:
    """'LADY JULIANA (O.S.) (Cold, Firm)' -> (kind, speaker, tone)."""
    name = re.split(r"\s*\(", raw, maxsplit=1)[0].strip()
    parens = re.findall(r"\(([^)]*)\)", raw)
    tone = next((p for p in reversed(parens) if p.upper() not in {"O.S.", "V.O.", "OFF-SCREEN", "CONT'D"}), "")
    up = name.upper()
    if up == "CAMERA":
        return "camera", "", ""
    if up == "ACTION":
        return "action", "", ""
    if up in NON_DIALOGUE or up.startswith("SOUND"):
        return "sfx", "", ""
    # Normalise a couple of obvious aliases so the same person counts once.
    speaker = re.sub(r"^(LADY|LORD|THE)\s+", "", up) if up not in {"LORD"} else up
    return "dialogue", speaker, tone


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    num_match = re.search(r"episode-(\d+)", path.name)
    ep = Episode(number=int(num_match.group(1)) if num_match else 0, path=str(path))

    scene, heading = 0, ""
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if (m := EP_RE.match(line)):
            ep.number = int(m.group(1))
        elif line.startswith("EPISODE LOGLINE:"):
            ep.logline = line.split(":", 1)[1].strip()
        elif line.startswith("CONTINUITY NOTES:"):
            ep.continuity = line.split(":", 1)[1].strip()
        elif (m := SCENE_RE.match(line)):
            scene = int(m.group(1))
        elif (m := HEADING_RE.match(line)):
            heading = m.group(1).strip()
        elif (m := BLOCK_RE.match(line)) and i + 1 < n and (t := TIME_RE.match(lines[i + 1].strip())):
            kind, speaker, tone = _label(m.group(1))
            j, body = i + 2, []
            while j < n and lines[j].strip() and not BLOCK_RE.match(lines[j].strip()) and not lines[j].startswith("---"):
                body.append(lines[j].strip())
                j += 1
            text = " ".join(body).strip().strip('"').strip("“”")
            ep.beats.append(Beat(ep.number, len(ep.beats), scene, heading, kind, speaker, tone, text,
                                 to_sec(t.group(1), t.group(2)), to_sec(t.group(3), t.group(4))))
            i = j
            continue
        i += 1
    ep.beats.sort(key=lambda b: (b.start, b.idx))
    for k, b in enumerate(ep.beats):
        b.idx = k
    return ep


def load_show(folder: str | Path) -> list[Episode]:
    files = sorted(Path(folder).glob("*.txt"))
    eps = [parse_episode(f) for f in files]
    return sorted([e for e in eps if e.beats], key=lambda e: e.number)
