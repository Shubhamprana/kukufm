"""Unit tests for the parts that caught real bugs. Run: python -m pytest -q
Uses a tiny synthetic screenplay, so no confidential input is needed."""
import json
import textwrap
from pathlib import Path

import pytest

from promo_finder import finder, llm, scan
from promo_finder.finder import Candidate, _hits, _safe_cut, _seg, beat_intensity, generate, select
from promo_finder.llm import parse_json, quote_in_script, validate, _norm
from promo_finder.parser import parse_episode


def block(label, start, end, text):
    m1, s1 = divmod(start, 60)
    m2, s2 = divmod(end, 60)
    return f"**{label}:**\nStart: ({int(m1):02d}:{s1:05.2f}) | End: ({int(m2):02d}:{s2:05.2f})\n{text}\n"


@pytest.fixture
def ep(tmp_path: Path):
    beats = [block("CAMERA", 0, 3, "Wide shot of a hall."),
             block("ACTION", 1, 3, "Mara freezes in shock."),
             block("VILLAIN (Cold, Menacing)", 3, 6, '"You should not have come back."'),
             block("MARA (Defiant)", 6.5, 9, '"I came for what you stole."'),
             block("VILLAIN (Furious, Shouting)", 9.5, 12, '"Guards! Seize her!"'),
             block("ACTION", 12, 15, "Guards grab Mara and drag her toward the door."),
             block("MARA (Desperate)", 15.5, 20, '"You will pay for this. Every one of you..."'),
             block("CAMERA", 20, 24, "Close-up on the villain smiling."),
             block("VILLAIN (Whispering, Threatening)", 24, 29, '"Tomorrow, at dawn, she hangs."'),
             block("SOUND FX", 29, 32, "Ominous drone swells into a cliffhanger sting."),
             block("CAMERA", 32, 33, "Cut to black.")]
    txt = "EPISODE NUMBER: Episode 1\nEPISODE LOGLINE: Mara returns.\nCONTINUITY NOTES: First.\n\n--- SCRIPT START ---\n\nSCENE 01\nScene heading: INT. HALL - NIGHT\n\n" + "\n".join(beats) + "\n--- SCRIPT END ---"
    p = tmp_path / "episode-001-test.txt"
    p.write_text(txt, encoding="utf-8")
    return parse_episode(p)


def test_parser_reads_every_block(ep):
    assert ep.number == 1 and len(ep.beats) == 11
    v = [b for b in ep.beats if b.speaker == "VILLAIN"]
    assert v[0].tone == "Cold, Menacing" and v[0].text == "You should not have come back."
    assert ep.duration == 33


def test_keywords_match_whole_word_stems_only():
    assert _hits("violence", "a white environment, elbow deep") == 0   # bug fixed: 'hit' in 'white'
    assert _hits("violence", "she hits him with an iron") == 2


def test_cuts_never_land_mid_dialogue(ep):
    assert not _safe_cut(ep, 4.0)          # inside "You should not have come back."
    assert _safe_cut(ep, 6.0) and _safe_cut(ep, 32.0)
    for c in generate([ep]):
        for s in c.segments:
            assert _safe_cut(ep, s.start) and _safe_cut(ep, s.end)
            assert finder.MIN_LEN <= c.duration <= finder.MAX_LEN


def test_select_drops_overlaps(ep):
    a = Candidate("single", [_seg(ep, 3, 29)]); a.score = 2
    b = Candidate("single", [_seg(ep, 6.5, 29)]); b.score = 1
    assert select([a, b], per_episode=5) == [a]


@pytest.mark.parametrize("quote,ok", [
    ("VILLAIN (Cold, Menacing): You should not have come back.", True),   # speaker label allowed
    ("Guards! Seize her! ... Tomorrow, at dawn, she hangs.", True),         # two lines joined
    ("Seize her!", True),
    ("Guards! I will burn you alive!", False),                              # invented line
    ("Tomo", False),                                                        # word prefix is not a match
])
def test_quote_check(ep, quote, ok):
    text = _norm(" ".join(b.text for b in ep.beats))
    assert quote_in_script(quote, text) is ok


def test_json_repair_mmss_numbers():
    # Gemini once wrote 01:51.1 as the bare number 0151.1
    assert parse_json('{"trim": [{"start": 0151.1, "end": 0216.7}]}')["trim"][0] == {"start": 111.1, "end": 136.7}


def test_validate_fills_missing_score_and_rejects_hook_cutting_trim(ep):
    c = Candidate("single", [_seg(ep, 3, 29)])
    finder._score(c, ep, ep)
    d = {"hook": 9, "cliffhanger": 9, "standalone": 7, "emotion": 8,           # 'overall' missing
         "hook_quote": "You should not have come back.", "cliffhanger_quote": "Tomorrow, at dawn, she hangs.",
         "why": "x", "trim": [{"start": 6.5, "end": 29}], "risks": []}
    out = validate(d, c, {1: ep})
    assert out["missing_scores"] == ["overall"] and out["overall"] == 8
    assert out["hook_quote_verified"] and out["cliffhanger_quote_verified"]
    assert out["trim_rejected_weaker_hook"] and c.segments[0].start == 3   # cutting the hook line is refused


def test_scan_snaps_extends_and_rejects(ep, monkeypatch):
    reply = {"promos": [
        {"segments": [{"start": 12.1, "end": 29.0}], "hook_quote": "Guards grab Mara", "cliffhanger_quote": "", "reason": "17s, extendable"},
        {"segments": [{"start": 3, "end": 6}, {"start": 9.5, "end": 29}], "hook_quote": "Seize her!", "cliffhanger_quote": "", "reason": "stitch"},
        {"segments": [{"start": 300, "end": 330}], "hook_quote": "", "cliffhanger_quote": "", "reason": "made-up times"},
        {"segments": [{"start": 3.0, "end": 20.0}], "hook_quote": "", "cliffhanger_quote": "", "reason": "17s, nothing earlier to extend into"},
    ]}
    j = llm.Judge("gemini", "x", cache_dir=Path(ep.path).parent / "c")
    monkeypatch.setattr(j, "_cached", lambda s, u: json.dumps(reply))
    cands, rejects = scan.scan_episode(j, ep, "premise", k=4)
    assert len(cands) == 2 and len(rejects) == 2
    assert "beats" in rejects[0] and "cannot be extended" in rejects[1]
    first = cands[0]
    # snapped to the real beat start 12.0, then pulled back to 6.5 so it reaches 20s; the ending never moves
    assert first.segments[0].start == 6.5 and first.segments[0].end == 29.0 and first.features["scan_extended"]
    assert cands[1].kind == "stitch" and cands[1].source == "ai_scan"
