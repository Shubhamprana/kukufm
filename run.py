"""CLI: python run.py <screenplay_folder> [--llm gemini|openai|anthropic|none] [--model NAME]"""
import argparse
import os
from pathlib import Path

from promo_finder.pipeline import run


def load_env(path: Path = Path(__file__).with_name(".env")) -> None:
    """Read KEY=VALUE lines from .env (gitignored) without overriding real env vars."""
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_env()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Find and rank promo clips in a microdrama screenplay folder.")
    ap.add_argument("folder", help="folder of episode-*.txt screenplays")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--llm", default="none", choices=["none", "gemini", "openai", "anthropic"], help="LLM judge provider")
    ap.add_argument("--model", default=None, help="model name (defaults per provider)")
    ap.add_argument("--shortlist", type=int, default=40, help="how many rule-stage candidates the LLM judges")
    ap.add_argument("--top", type=int, default=20, help="how many ranked promos to output")
    ap.add_argument("--per-episode", type=int, default=2, help="max promos from one episode")
    ap.add_argument("--no-scan", action="store_true", help="skip the per-episode LLM scan (rules-only candidates)")
    ap.add_argument("--scan-k", type=int, default=2, help="promos the LLM may propose per episode")
    ap.add_argument("--rule-weight", type=float, default=0.35, help="weight of the rule score in the final rank (rest = LLM)")
    a = ap.parse_args()
    run(a.folder, a.out, a.llm, a.model, a.shortlist, a.top, a.per_episode,
        scan=not a.no_scan, scan_k=a.scan_k, rule_weight=a.rule_weight)
