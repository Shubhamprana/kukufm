# Microdrama Promo Finder

Reads a folder of timestamped microdrama screenplays and returns ranked 20–60 second promo clips. Each promo comes with exact segment timestamps, its hook, its cliffhanger and one line on why it works.

```
                    ┌─► rules: ~5,400 windows scored ──► top 40 ─┐
screenplays ─► parser┤                                           ├─► merge ─► LLM judge ─► validate ─► rank + QA ─► promos.csv / .md / .json
                    └─► LLM reads each full episode, proposes 2 ─┘   (swappable model)
```

## Quick start

```bash
pip install -r requirements.txt

# rules only: free, offline, seconds for 50 episodes
python run.py "path/to/show_folder" --out outputs/rules_only

# full pipeline with Gemini (free tier). Put GEMINI_API_KEY=... in a .env file next to run.py (gitignored)
python run.py "path/to/show_folder" --llm gemini --model gemini-3.1-flash-lite --out outputs/v2

# any OpenAI-compatible API: Groq, OpenRouter, Ollama, OpenAI, an in-house gateway
# (.env: OPENAI_API_KEY=...  OPENAI_BASE_URL=https://api.groq.com/openai/v1)
python run.py "path/to/show_folder" --llm openai --model llama-3.3-70b-versatile

# UI
streamlit run app.py

# tests (synthetic screenplay, no confidential input needed)
python -m pytest -q
```

Swapping the model is one flag: `--llm {gemini|openai|anthropic|none} --model <name>`. No SDKs are needed, only plain HTTP from the standard library. Every response is cached in `.cache/`, so re-runs and re-weighting cost nothing.

Useful flags:
* `--no-scan`: skip the full-episode AI scan
* `--scan-k 2`: proposals per episode
* `--rule-weight 0.35`: share of the rule score in the final rank (0 = LLM only)
* `--shortlist 40`, `--top 20`, `--per-episode 2`

## Submitted raw outputs
| folder | what |
|---|---|
| `outputs/rules_only/` | rules only, no LLM |
| `outputs/gemini_3.1_flash_lite/` | v1: 40 rule candidates judged by Gemini 3.1 Flash Lite |
| `outputs/v2_scan_lite/` | **v2 (main):** rule candidates + AI full-episode scan, judged, rule weight 0.35 |
| `outputs/v2_scan_lite_llm_only/` | v2 ranked by the LLM judge alone (rule weight 0) |

Each folder holds:
* `promos.csv`: the editor's sheet
* `promos.md`: readable report with the script excerpt of every promo
* `promos.json`: all scores, features, run metadata and QA
* `shortlist_all_scores.csv`: every judged candidate, including the ones that were dropped

## How it works
1. **Parse** (`parser.py`): every CAMERA / ACTION / SFX / dialogue block becomes a *beat* with a start and end time in seconds.
2. **Rule candidates** (`finder.py`): start points are the most intense beats, and end points are the strongest cliffhanger beats.
   * Shapes: a single 20–60s stretch, or a stitch (a 3–10s cold-open hook that shares a character with the main clip, plus a stretch ending on a cliffhanger).
   * A cut never lands mid-line.
   * Scored on **hook** (first 3s), **cliffhanger** (script tags, episode end, open questions, cut-off lines, threats, stunned reactions; resolved endings penalised), **standalone** (2–3 speakers, enough dialogue, conflict, no dead air) and **length** (25–45s ideal).
   * Non-max suppression drops overlaps and reused hooks, then the top 40 go forward.
3. **AI scan** (`scan.py`): the LLM reads each whole episode (~1.5k tokens) and proposes up to 2 promos of 1–3 segments.
   * Proposals are snapped to real beats and must not drift more than 6s.
   * A proposal slightly under 20s is extended backwards, without moving the cliffhanger ending.
   * Proposals that match a rule candidate are merged and marked `both`.
4. **LLM judge** (`llm.py`): 1–10 for hook, cliffhanger, standalone, emotion and overall, plus quoted hook and cliffhanger, a one-line "why", risks and an optional tighter cut. Guardrails:
   * quotes must exist in the script
   * trims can only shrink, must snap to real beats and are rejected if they weaken the hook
   * scores are clamped, and a missing score is derived from the others and flagged
   * malformed JSON is repaired (e.g. `0151.1` → 111.1s) or retried
   * a daily-quota error stops the LLM stage at once, and everything else falls back to rules
5. **Rank**: final = `rule_weight` × rules + (1 − `rule_weight`) × LLM.
   * Overlaps are dropped and each episode is capped at 2 promos.
   * A premise slot keeps the best Episode 1–3 promo in the top 3.
   * Tiers: A = post first (1–5), B = test next (6–12), C = backup.
6. **QA** (`pipeline.quality`): flags length, camera-only hooks, soft endings, 4+ characters, dead air, late episodes and unverified quotes. It also reports rule-vs-LLM rank agreement and where each promo came from (rules / ai_scan / both).

## Evaluating against a human (`eval.py`)
```bash
python eval.py sheet "path/to/show_folder" outputs/rules_only outputs/v2_scan_lite ...   # blind, shuffled review sheet
# fill eval/labels.csv: Y = would post, M = maybe, N = no
python eval.py score        # precision@5/10/20 and recall of the good clips, per run
```

## Input format
The parser expects the input-pack format (`**LABEL (tone):**`, then `Start: (mm:ss.ss) | End: (mm:ss.ss)`, then text). Another show only needs to parse into the same `Beat` shape.

## Confidentiality
The screenplays are Kuku's. `.gitignore` excludes all `.txt` files and `.env`. The `promos.md` reports and `eval/review_sheet.md` contain script excerpts, so share them privately (e.g. a Drive link), not in a public repo.
