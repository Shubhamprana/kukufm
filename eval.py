"""Blind human evaluation of promo picks.

  python eval.py sheet  <show_folder> <outputs/run1> <outputs/run2> ...
      Pools every promo any run put in its top N, merges duplicates (same moment),
      shuffles them and hides which system picked what. Writes:
        eval/review_sheet.md   script excerpt for each clip, to read
        eval/labels.csv        fill the `label` column: Y = would post, M = maybe, N = no
        eval/key.json          (hidden) which run ranked each clip where

  python eval.py score
      Editor acceptance per run: precision@5/10/20 (Y=1, M=0.5, N=0), mean rank of
      Y clips, and which Y clips each run missed (recall of the pooled good clips).
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

from promo_finder.parser import fmt, load_show

EVAL = Path(__file__).with_name("eval")
VALUE = {"Y": 1.0, "M": 0.5, "N": 0.0}


def _segs(p: dict) -> list[tuple[int, float, float]]:
    return [(s["episode"], s["start_sec"], s["end_sec"]) for s in p["segments"]]


def _same(a, b, thr=0.5) -> bool:
    """Same moment: the main (last) segments overlap by >= 50% of the shorter one."""
    (ea, sa, xa), (eb, sb, xb) = a[-1], b[-1]
    if ea != eb:
        return False
    inter = max(0.0, min(xa, xb) - max(sa, sb))
    return inter >= thr * min(xa - sa, xb - sb)


def sheet(show: str, runs: list[str], top: int = 20, seed: int = 7) -> None:
    eps = {e.number: e for e in load_show(show)}
    clips: list[dict] = []
    for run in runs:
        rows = json.loads((Path(run) / "promos.json").read_text(encoding="utf-8"))["promos"][:top]
        for r in rows:
            segs = _segs(r)
            hit = next((c for c in clips if _same(c["segs"], segs)), None)
            if hit is None:
                hit = {"segs": segs, "ranks": {}}
                clips.append(hit)
            hit["ranks"][Path(run).name] = r["rank"]
    random.Random(seed).shuffle(clips)
    EVAL.mkdir(exist_ok=True)
    md = ["# Promo review sheet (blind)", "",
          "For each clip ask: *would I post this as a TikTok/Reels ad to pull in strangers?*",
          "Check: does the first 3s stop the scroll, does the end leave you needing more, does it make sense cold?",
          "Write Y / M / N in `eval/labels.csv`. Camera directions are hidden, as an ad viewer only sees and hears the scene.", ""]
    key, rows = {}, []
    for i, c in enumerate(clips, 1):
        cid = f"C{i:02d}"
        key[cid] = c["ranks"]
        seg_str = " + ".join(f"E{e} {fmt(s)}-{fmt(x)}" for e, s, x in c["segs"])
        dur = sum(x - s for _, s, x in c["segs"])
        md += [f"## {cid} · {seg_str} · {dur:.1f}s", "", "```"]
        for e, s, x in c["segs"]:
            md += [b.line() for b in eps[e].beats if b.kind != "camera" and b.start >= s - 1e-6 and b.end <= x + 1e-6]
            md.append("   ---")
        md += ["```", ""]
        rows.append({"id": cid, "segments": seg_str, "duration": round(dur, 1), "label": "", "note": ""})
    (EVAL / "review_sheet.md").write_text("\n".join(md), encoding="utf-8")
    (EVAL / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    with (EVAL / "labels.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "segments", "duration", "label", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(clips)} unique clips from {len(runs)} runs -> {EVAL / 'review_sheet.md'} and labels.csv")


def score() -> dict:
    key = json.loads((EVAL / "key.json").read_text(encoding="utf-8"))
    labels = {}
    with (EVAL / "labels.csv").open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            v = r["label"].strip().upper()[:1]
            if v in VALUE:
                labels[r["id"]] = v
    if not labels:
        sys.exit("No labels yet: fill the label column in eval/labels.csv with Y / M / N")
    runs = sorted({run for ranks in key.values() for run in ranks})
    good = {cid for cid, v in labels.items() if v == "Y"}
    report = {"labelled": len(labels), "of": len(key), "Y": len(good),
              "M": sum(v == "M" for v in labels.values()), "N": sum(v == "N" for v in labels.values()), "runs": {}}
    for run in runs:
        ranked = sorted(((ranks[run], cid) for cid, ranks in key.items() if run in ranks and cid in labels))
        res = {}
        for k in (5, 10, 20):
            top = [cid for rank, cid in ranked if rank <= k]
            res[f"precision@{k}"] = round(sum(VALUE[labels[c]] for c in top) / len(top), 2) if top else None
        ys = [rank for rank, cid in ranked if cid in good]
        res["Y_found"] = f"{len(ys)}/{len(good)}"
        res["mean_rank_of_Y"] = round(sum(ys) / len(ys), 1) if ys else None
        res["missed_Y"] = sorted(good - {cid for _, cid in ranked})
        report["runs"][run] = res
    (EVAL / "score.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "sheet":
        sheet(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 2 and sys.argv[1] == "score":
        score()
    else:
        print(__doc__)
