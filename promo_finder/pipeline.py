"""End-to-end: screenplay folder -> ranked promos + quality report."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from .finder import Candidate, MIN_LEN, MAX_LEN, _overlap, _safe_cut, beat_intensity, generate, select
from .llm import Judge, LLMError
from .parser import Episode, fmt, load_show
from .scan import scan_episode

LLM_WEIGHTS = {"hook": 2.0, "cliffhanger": 2.0, "standalone": 1.5, "emotion": 1.0, "overall": 2.5}


def _rule_why(c: Candidate) -> str:
    f = c.features
    hb = c.hook_text.split("] ", 1)[-1]
    opener = "a line of dialogue" if ":" in hb and not hb.startswith(("ACTION", "SFX", "CAMERA")) else "on-screen action"
    ends = []
    if f.get("cliff_question"): ends.append("an unanswered question")
    if f.get("cliff_trail"): ends.append("a line cut off mid-sentence")
    if f.get("cliff_threat"): ends.append("a threat still hanging")
    if f.get("cliff_tag"): ends.append("the script's own cliffhanger beat")
    if not ends: ends.append("a tense, unresolved beat")
    return (f"Opens with {opener} at peak intensity, ends on {' and '.join(ends[:2])}; "
            f"{f.get('speakers', '?')} speaking characters and {int(100 * f.get('talk_ratio', 0))}% dialogue keep it easy to follow cold.")


def _norm(vals: list[float]) -> list[float]:
    lo, hi = min(vals), max(vals)
    return [10 * (v - lo) / (hi - lo) if hi > lo else 5.0 for v in vals]


def _llm_score(l: dict) -> float:
    return sum(l[k] * w for k, w in LLM_WEIGHTS.items()) / sum(LLM_WEIGHTS.values())


def run(folder: str, out_dir: str = "outputs", provider: str = "none", model: str | None = None,
        shortlist: int = 40, top: int = 20, per_episode: int = 2, log=print,
        scan: bool = True, scan_k: int = 2, rule_weight: float = 0.35) -> dict:
    t0 = time.time()
    episodes = load_show(folder)
    if not episodes:
        raise SystemExit(f"No parsable episodes in {folder}")
    by_num = {e.number: e for e in episodes}
    premise = " ".join(f"Ep{e.number}: {e.logline}" for e in episodes[:3])
    log(f"Parsed {len(episodes)} episodes, {sum(len(e.beats) for e in episodes)} beats")

    cands = generate(episodes)
    log(f"Stage 1 (rules): {len(cands)} raw windows")
    pool = select(cands, per_episode=per_episode + 1, max_overlap=0.35)[:shortlist]
    log(f"Shortlisted {len(pool)} non-overlapping candidates")

    judge = Judge(provider, model)
    errors = []
    scan_stats = {"proposed": 0, "rejected": [], "same_as_rule_candidate": 0, "new": 0}
    if judge.enabled and scan:
        # Stage 1b: the LLM reads every episode and proposes moments the rules may have missed
        streak = 0
        for ep in episodes:
            try:
                found, rejects = scan_episode(judge, ep, premise, k=scan_k)
                streak = 0
            except Exception as e:
                streak += 1
                errors.append(f"scan E{ep.number}: {str(e)[:200]}")
                log(f"  scan E{ep.number} failed: {str(e)[:120]}")
                if "DAILY QUOTA" in str(e) or streak >= 3:
                    log("  LLM unavailable; skipping the rest of the scan")
                    break
                continue
            scan_stats["proposed"] += len(found) + len(rejects)
            scan_stats["rejected"] += rejects
            for c in found:
                twin = next((p for p in pool if _overlap(c, p) >= 0.6), None)
                if twin:
                    twin.source = "both"
                    scan_stats["same_as_rule_candidate"] += 1
                else:
                    pool.append(c)
                    scan_stats["new"] += 1
            log(f"  scan E{ep.number}: {len(found)} valid, {len(rejects)} rejected")
        log(f"Stage 1b (AI scan): {scan_stats['proposed']} proposals, {scan_stats['new']} new, "
            f"{scan_stats['same_as_rule_candidate']} already found by rules, {len(scan_stats['rejected'])} rejected")

    if judge.enabled:
        streak = 0
        for i, c in enumerate(pool, 1):
            try:
                judge.judge(c, premise, by_num)
                streak = 0
                log(f"  LLM {i}/{len(pool)} {c.key()} -> overall {c.llm['overall']}")
            except Exception as e:  # one bad call must never sink the whole run
                streak += 1
                e = e if isinstance(e, LLMError) else f"{type(e).__name__}: {e}"
                errors.append(f"{c.key()}: {str(e)[:200]}")
                log(f"  LLM {i}/{len(pool)} failed: {str(e)[:120]}")
                if "DAILY QUOTA" in str(e) or streak >= 3:
                    log("  LLM unavailable; remaining candidates keep rule scores (cached answers are kept for a re-run)")
                    break

    rule_n = _norm([c.score for c in pool])
    for c, r in zip(pool, rule_n):
        c.features["rule_score_0_10"] = round(r, 2)
        if c.llm:
            c.final_score = round(rule_weight * r + (1 - rule_weight) * _llm_score(c.llm), 2)
            c.why = c.llm["why"] or _rule_why(c)
        else:
            c.final_score = round(r, 2)
            c.why = _rule_why(c)

    ranked = select(pool, per_episode=per_episode, max_overlap=0.35, key=lambda c: c.final_score)[:top]
    # Editorial rule: new viewers land on Episode 1, so the best premise promo
    # (episodes 1-3) always sits in the top 3.
    premise_c = next((c for c in ranked if c.main_episode <= 3), None)
    if premise_c and ranked.index(premise_c) > 2:
        ranked.remove(premise_c)
        ranked.insert(2, premise_c)
        premise_c.features["premise_slot"] = True
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_row(i, c) for i, c in enumerate(ranked, 1)]
    meta = {"show_folder": str(folder), "show_name": Path(folder).name, "episodes": len(episodes),
            "episode_durations": {str(e.number): round(e.duration, 2) for e in episodes}, "raw_windows": len(cands), "shortlist": len(pool),
            "provider": provider, "model": judge.model if judge.enabled else None, "llm_calls": judge.calls,
            "llm_tokens_in": judge.tokens_in, "llm_tokens_out": judge.tokens_out, "llm_errors": errors,
            "scan": scan_stats if judge.enabled and scan else None, "rule_weight": rule_weight,
            "runtime_sec": round(time.time() - t0, 1)}
    qa = quality(ranked, pool, by_num)
    (out / "promos.json").write_text(json.dumps({"meta": meta, "quality": qa, "promos": rows}, indent=2), encoding="utf-8")
    _write_csv(out / "promos.csv", rows)
    _write_csv(out / "shortlist_all_scores.csv", [_row(i, c) for i, c in enumerate(sorted(pool, key=lambda c: -c.final_score), 1)])
    (out / "promos.md").write_text(_report(rows, ranked, meta, qa), encoding="utf-8")
    log(f"Wrote {len(rows)} ranked promos to {out.resolve()} in {meta['runtime_sec']}s")
    return {"meta": meta, "quality": qa, "promos": rows, "candidates": ranked}


def _row(rank: int, c: Candidate) -> dict:
    l = c.llm or {}
    return {
        "rank": rank, "tier": "A - post first" if rank <= 5 else ("B - test next" if rank <= 12 else "C - backup"),
        "type": c.kind, "source": c.source, "episodes": sorted({s.episode for s in c.segments}), "duration_sec": round(c.duration, 1),
        "segments": [s.to_dict() for s in c.segments],
        "segments_str": " + ".join(f"E{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in c.segments),
        "hook": l.get("hook_quote") or c.hook_text.split("] ", 1)[-1],
        "cliffhanger": l.get("cliffhanger_quote") or c.cliff_text.split("] ", 1)[-1],
        "why": c.why, "final_score": c.final_score, "rule_score": c.features.get("rule_score_0_10"),
        "llm_scores": {k: l[k] for k in LLM_WEIGHTS if k in l} or None,
        "llm_risks": l.get("risks"), "llm_trimmed": l.get("trim_applied"),
        "quotes_verified": (l.get("hook_quote_verified"), l.get("cliffhanger_quote_verified")) if l else None,
        "features": c.features,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    cols = ["rank", "tier", "type", "source", "segments_str", "duration_sec", "hook", "cliffhanger", "why", "final_score", "rule_score", "llm_scores", "llm_risks"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], (dict, list)) else r[k]) for k in cols})


# --- automatic quality checks ------------------------------------------------------

def quality(ranked: list[Candidate], pool: list[Candidate], eps: dict[int, Episode]) -> dict:
    flags = []
    for i, c in enumerate(ranked, 1):
        f = []
        if not (MIN_LEN <= c.duration <= MAX_LEN):
            f.append("length out of 20-60s")
        for s in c.segments:
            ep = eps[s.episode]
            if not (_safe_cut(ep, s.start) and _safe_cut(ep, s.end)):
                f.append("cuts a line of dialogue")
        first = [b for b in c.segments[0].beats if b.start < c.segments[0].start + 3]
        if not any(b.kind in ("dialogue", "action", "sfx") for b in first):
            f.append("first 3s is camera-only (weak hook)")
        if c.features.get("cliff_resolve", 0) >= 2:
            f.append("ending reads as resolved/soft")
        if c.features.get("speakers", 0) >= 4:
            f.append("4+ speaking characters (hard to follow cold)")
        if c.features.get("max_gap", 0) > 8:
            f.append(f"dead air {c.features['max_gap']}s")
        if c.main_episode > 10 and c.features.get("standalone", 0) < 4:
            f.append("late episode, may need backstory")
        if c.llm and not all((c.llm.get("hook_quote_verified"), c.llm.get("cliffhanger_quote_verified"))):
            f.append("LLM quote not found in script")
        if f:
            flags.append({"rank": i, "segments": " + ".join(f"E{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in c.segments), "flags": f})
    d = [c.duration for c in ranked]
    stats = {
        "promos": len(ranked),
        "duration_min_mean_max": [round(min(d), 1), round(sum(d) / len(d), 1), round(max(d), 1)] if d else None,
        "episodes_covered": sorted({c.main_episode for c in ranked}),
        "share_ending_at_episode_end": round(sum(bool(c.features.get("cliff_episode_end")) for c in ranked) / max(len(ranked), 1), 2),
        "share_stitched": round(sum(c.kind == "stitch" for c in ranked) / max(len(ranked), 1), 2),
        "source_counts": {s: sum(c.source == s for c in ranked) for s in ("rules", "ai_scan", "both")},
        "flagged": flags,
    }
    judged = [c for c in pool if c.llm]
    if len(judged) >= 5:
        a = _ranks([c.features["rule_score_0_10"] for c in judged])
        b = _ranks([_llm_score(c.llm) for c in judged])
        n = len(judged)
        stats["rule_vs_llm_spearman"] = round(1 - 6 * sum((x - y) ** 2 for x, y in zip(a, b)) / (n * (n * n - 1)), 2)
        stats["llm_quote_verified_rate"] = round(sum(bool(c.llm.get("hook_quote_verified") and c.llm.get("cliffhanger_quote_verified")) for c in judged) / n, 2)
        stats["biggest_disagreements"] = [
            {"segments": c.key(), "rule": c.features["rule_score_0_10"], "llm": round(_llm_score(c.llm), 1), "llm_risks": c.llm.get("risks")}
            for c in sorted(judged, key=lambda c: -abs(c.features["rule_score_0_10"] - _llm_score(c.llm)))[:5]]
    return stats


def _ranks(v: list[float]) -> list[int]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0] * len(v)
    for k, i in enumerate(order):
        r[i] = k
    return r


def _report(rows: list[dict], ranked: list[Candidate], meta: dict, qa: dict) -> str:
    L = ["# Promo picks", "",
         f"Show folder: `{meta['show_folder']}` | episodes: {meta['episodes']} | raw windows: {meta['raw_windows']} | "
         f"shortlist: {meta['shortlist']} | judge: {meta['provider']} {meta['model'] or ''} | runtime: {meta['runtime_sec']}s", "",
         "Tier A = post first, B = test next, C = backups. Timestamps are mm:ss.ss from each episode's screenplay.", ""]
    for r, c in zip(rows, ranked):
        L += [f"## #{r['rank']} [{r['tier']}] {r['segments_str']} ({r['duration_sec']}s, {r['type']})", "",
              f"- **Hook:** {r['hook']}", f"- **Cliffhanger:** {r['cliffhanger']}", f"- **Why it works:** {r['why']}",
              f"- **Score:** {r['final_score']} (rules {r['rule_score']}" + (f", LLM {r['llm_scores']}" if r['llm_scores'] else "") + ")"]
        if r.get("llm_risks"):
            L.append(f"- **Risks:** {'; '.join(r['llm_risks'])}")
        L += ["", "<details><summary>Script excerpt</summary>", "", "```", c.transcript(with_camera=False), "```", "", "</details>", ""]
    L += ["# Automatic quality checks", "", "```json", json.dumps(qa, indent=2), "```"]
    return "\n".join(L)
