"""Streamlit UI: upload a show's screenplays, get ranked promos.

Run locally:  streamlit run app.py
"""
import base64
import html
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.llm import DEFAULT_MODELS
from promo_finder.pipeline import run
from run import load_env

load_env()  # picks up GEMINI_API_KEY etc. from .env when running locally

# Community Cloud stores deployment secrets in st.secrets rather than a local
# .env file. The pipeline reads standard environment variables, so mirror only
# the provider keys into the running process when they have been configured.
for _secret_name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    try:
        _secret_value = st.secrets.get(_secret_name)
    except Exception:
        _secret_value = None
    if _secret_value:
        os.environ.setdefault(_secret_name, str(_secret_value))

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data
def backdrop_uri() -> str:
    """Original, generated night-lights photo (assets/backdrop.jpg), embedded so the app needs no static server."""
    p = HERE / "assets" / "backdrop.jpg"
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


BG = backdrop_uri()

# Frosted glass over an out-of-focus night photo. One warm accent; mono for every timecode.
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{--g:rgba(16,15,20,.46);--g2:rgba(255,255,255,.055);--g3:rgba(255,255,255,.09);--bd:rgba(255,255,255,.10);--bd2:rgba(255,255,255,.18);
--tx:#f4f1ec;--mu:rgba(244,241,236,.68);--fa:rgba(244,241,236,.46);--ac:#ffb15c;--acs:rgba(255,177,92,.18);--ok:#7ee2a8;--warn:#ffd27a;
--hook:#ff6b6b;--cliff:#c4a1ff;--blur:blur(22px) saturate(150%);}
html,body,[class*="css"],p,li,label,input,textarea,button{font-family:'Manrope',system-ui,sans-serif;}
.mono,code{font-family:'JetBrains Mono',ui-monospace,Consolas,monospace !important;}
[data-testid="stApp"]{background:linear-gradient(180deg,rgba(8,7,12,.35) 0%,rgba(8,7,12,.62) 55%,rgba(8,7,12,.86) 100%),
url('__BG__') center/cover fixed no-repeat,#0b0a0f;}
.block-container{padding-top:1.4rem;padding-bottom:4rem;max-width:1180px;}
header[data-testid="stHeader"]{background:transparent;}
.stAppDeployButton,[data-testid="stMainMenu"]{display:none;}
[data-testid="stSidebar"]{background:rgba(14,13,18,.62);backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);border-right:1px solid var(--bd);}
[data-testid="stSidebar"] h3{font-size:.7rem;text-transform:uppercase;letter-spacing:.12em;color:var(--fa);font-weight:600;margin:1.1rem 0 .15rem;}
.glass{background:var(--g);backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);border:1px solid var(--bd);
border-radius:20px;box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 20px 50px rgba(0,0,0,.25);}
div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]),
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"][class*="border"]{background:var(--g);
backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);border:1px solid var(--bd) !important;border-radius:20px !important;}
/* nav pill */
.nav{display:flex;align-items:center;gap:18px;padding:9px 12px 9px 16px;border-radius:999px;margin:0 auto 26px;width:fit-content;max-width:100%;}
.logo{display:flex;align-items:center;gap:9px;font-family:'Sora',sans-serif;font-weight:600;color:var(--tx);font-size:1rem;white-space:nowrap;}
.logo i{width:22px;height:22px;border-radius:7px;background:var(--ac);display:grid;place-items:center;}
.logo i svg{width:12px;height:12px;}
.nav a{color:var(--mu) !important;text-decoration:none !important;font-size:.88rem;white-space:nowrap;}
.nav a:hover{color:var(--tx) !important;}
.st8{font-size:.76rem;color:var(--mu);background:var(--g2);border:1px solid var(--bd);border-radius:999px;padding:4px 11px;white-space:nowrap;}
.st8 b{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--ok);margin-right:7px;vertical-align:1px;}
.st8 b.off{background:var(--fa);}
/* hero */
.hero{display:grid;grid-template-columns:1.25fr 1fr;gap:18px;margin-bottom:18px;}
.hl{padding:34px 34px 30px;}
.eb{display:inline-flex;align-items:center;gap:8px;font-size:.74rem;letter-spacing:.14em;text-transform:uppercase;color:var(--tx);
border:1px solid var(--bd2);border-radius:999px;padding:5px 12px;background:var(--g2);}
.eb i{width:7px;height:7px;border-radius:50%;background:var(--ac);}
.hl h1{font-family:'Sora',sans-serif;font-weight:500;font-size:3.05rem;line-height:1.06;letter-spacing:-.03em;color:var(--tx);margin:20px 0 14px;padding:0;}
.hl h1 em{font-style:normal;color:var(--ac);}
.hl p.sub{color:var(--mu);font-size:1.02rem;line-height:1.6;margin:0 0 24px;max-width:540px;}
.ctas{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:28px;}
.btn{display:inline-block;border-radius:999px;padding:11px 20px;font-weight:600;font-size:.92rem;text-decoration:none !important;}
.btn.w{background:#f7f3ec;color:#121016 !important;}
.btn.w:hover{background:#fff;}
.btn.o{border:1px solid var(--bd2);color:var(--tx) !important;background:var(--g2);}
.facts{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;border-top:1px solid var(--bd);padding-top:18px;}
.facts b{display:block;font-family:'Sora',sans-serif;font-size:1.4rem;font-weight:500;color:var(--tx);}
.facts span{font-size:.8rem;color:var(--fa);line-height:1.35;display:block;}
.hr{padding:22px 22px 18px;display:flex;flex-direction:column;}
.hr .cap{display:flex;justify-content:space-between;align-items:center;font-size:.76rem;color:var(--fa);margin-bottom:14px;}
.hr .cap b{color:var(--tx);font-weight:600;font-size:.82rem;}
.hr .cap span{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:48%;}
.hr .cap b{white-space:nowrap;}
.pick-t{font-family:'Sora',sans-serif;font-size:1.25rem;color:var(--tx);margin:2px 0 4px;}
.pick-s{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--mu);margin-bottom:12px;}
.pq{border-left:2px solid var(--hook);padding:2px 0 2px 12px;margin:10px 0;}
.pq.c{border-left-color:var(--cliff);}
.pq small{display:block;font-size:.68rem;letter-spacing:.1em;text-transform:uppercase;color:var(--fa);margin-bottom:3px;}
.pq div{color:var(--tx);font-size:.93rem;line-height:1.45;}
.hr .foot{margin-top:auto;padding-top:12px;font-size:.78rem;color:var(--fa);}
/* method strip */
.strip{display:grid;grid-template-columns:repeat(6,1fr);margin:0 0 30px;overflow:hidden;}
.stp{padding:14px 16px;border-right:1px solid var(--bd);}
.stp:last-child{border-right:0;}
.stp b{display:block;font-family:'Sora',sans-serif;font-size:1.25rem;font-weight:500;color:var(--tx);}
.stp span{font-size:.76rem;color:var(--fa);}
.stp em{font-style:normal;display:block;font-size:.7rem;color:var(--ac);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;}
.sec{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:8px 0 10px;flex-wrap:wrap;}
.sec .h{font-family:'Sora',sans-serif;font-weight:500;font-size:1.5rem;letter-spacing:-.02em;color:var(--tx);line-height:1.2;}
.sec span{font-size:.84rem;color:var(--fa);}
.ptitle{font-family:'Sora',sans-serif;font-size:.95rem;color:var(--tx);margin-bottom:2px;}
.psub{font-size:.82rem;color:var(--fa);margin-bottom:8px;}
.cfg{display:grid;grid-template-columns:auto 1fr;gap:7px 14px;font-size:.84rem;margin:10px 0 14px;}
.cfg dt{color:var(--fa);} .cfg dd{margin:0;color:var(--tx);} .cfg dd.mono{font-size:.78rem;}
.stButton>button[kind="primary"]{background:#f7f3ec;border:0;color:#121016;border-radius:999px;font-weight:700;padding:.62rem 1.1rem;width:100%;}
.stButton>button[kind="primary"]:hover{background:#fff;color:#121016;}
.stButton>button[kind="primary"]:disabled{background:var(--g2);color:var(--fa);border:1px solid var(--bd);}
.stButton>button[kind="secondary"]{border-radius:999px;background:var(--g2);border:1px solid var(--bd2);}
/* results */
.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:4px 0 18px;}
.kpi{padding:14px 16px;border-radius:16px;}
.kpi .l{font-size:.76rem;color:var(--fa);margin-bottom:4px;}
.kpi .v{font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:500;color:var(--tx);}
.kpi .s{font-size:.74rem;color:var(--mu);margin-top:2px;}
.pc{padding:16px 18px 14px;margin-top:12px;border-radius:18px;}
.pc-h{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.rk{font-family:'JetBrains Mono',monospace;font-size:.8rem;color:var(--tx);background:var(--g3);border-radius:8px;padding:3px 8px;}
.tg{font-size:.72rem;font-weight:600;border-radius:999px;padding:3px 10px;white-space:nowrap;}
.tg.A{color:#0f2a1b;background:var(--ok);} .tg.B{color:#2c1f00;background:var(--warn);}
.tg.C{color:var(--tx);background:var(--g3);}
.tg.n{color:var(--mu);border:1px solid var(--bd);font-weight:500;}
.pc-t{font-family:'Sora',sans-serif;font-size:1.02rem;color:var(--tx);margin-right:4px;}
.pc-sc{margin-left:auto;text-align:right;white-space:nowrap;}
.pc-sc b{font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:500;color:var(--tx);}
.pc-sc small{color:var(--fa);font-size:.74rem;}
.tracks{margin:14px 0 4px;display:grid;gap:7px;}
.trk{display:grid;grid-template-columns:50px 1fr 66px;align-items:center;gap:10px;}
.trk .e{font-family:'JetBrains Mono',monospace;font-size:.74rem;color:var(--mu);}
.trk .d{font-family:'JetBrains Mono',monospace;font-size:.72rem;color:var(--fa);text-align:right;}
.bar{position:relative;height:16px;background:rgba(255,255,255,.06);border-radius:999px;overflow:hidden;}
.bar .tick{position:absolute;top:4px;bottom:4px;width:1px;background:rgba(255,255,255,.14);}
.bar .cut{position:absolute;top:0;bottom:0;background:var(--acs);border:1px solid var(--ac);border-radius:999px;}
.bar .hk{position:absolute;top:2px;bottom:2px;width:4px;border-radius:2px;background:var(--hook);}
.bar .cl{position:absolute;top:2px;bottom:2px;width:4px;border-radius:2px;background:var(--cliff);}
.tcs{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 12px;}
.tc{font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--tx);background:var(--g2);border:1px solid var(--bd);border-radius:8px;padding:3px 9px;}
.hc{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.q{background:var(--g2);border:1px solid var(--bd);border-radius:14px;padding:11px 13px;}
.q .k{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px;}
.q.h .k{color:var(--hook);} .q.c .k{color:var(--cliff);}
.who{font-size:.74rem;color:var(--mu);margin-bottom:2px;}
.line{font-size:.94rem;color:var(--tx);line-height:1.45;}
.line.act{color:rgba(244,241,236,.82);font-style:italic;}
.why{font-size:.88rem;color:var(--mu);line-height:1.55;margin:12px 0 10px;}
.mt{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px;}
.m .ml{display:flex;justify-content:space-between;font-size:.72rem;color:var(--fa);margin-bottom:4px;}
.m .ml b{color:var(--tx);font-weight:500;font-family:'JetBrains Mono',monospace;}
.m .mb{height:4px;background:rgba(255,255,255,.08);border-radius:2px;overflow:hidden;}
.m .mb div{height:100%;background:var(--ac);border-radius:2px;}
.fl{margin-top:12px;font-size:.8rem;color:var(--warn);background:rgba(255,210,122,.08);border:1px solid rgba(255,210,122,.25);
border-radius:10px;padding:7px 11px;}
.emp{padding:34px 20px;text-align:center;margin-top:6px;border-radius:18px;}
.emp b{display:block;font-family:'Sora',sans-serif;font-weight:500;color:var(--tx);margin-bottom:4px;}
.emp span{color:var(--fa);font-size:.9rem;}
.legend{display:flex;gap:16px;font-size:.76rem;color:var(--fa);margin:4px 0 2px;flex-wrap:wrap;}
.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;vertical-align:-1px;}
.foot{color:var(--fa);font-size:.78rem;text-align:center;margin-top:36px;}
[data-testid="stExpander"] details{background:rgba(16,15,20,.38);backdrop-filter:var(--blur);-webkit-backdrop-filter:var(--blur);
border:1px solid var(--bd) !important;border-radius:14px !important;}
[data-testid="stExpander"] summary{color:var(--mu);}
@media (max-width:900px){.hero{grid-template-columns:1fr;}.hl{padding:26px 22px;}.hl h1{font-size:2.2rem;}
.strip{grid-template-columns:repeat(3,1fr);}.stp:nth-child(3){border-right:0;}.stp:nth-child(-n+3){border-bottom:1px solid var(--bd);}
.kpis{grid-template-columns:repeat(2,minmax(0,1fr));}.hc{grid-template-columns:1fr;}.mt{grid-template-columns:repeat(3,minmax(0,1fr));}
.nav a{display:none;}.trk{grid-template-columns:42px 1fr;}.trk .d{display:none;}.facts b{font-size:1.15rem;}}
</style>""".replace("__BG__", BG), unsafe_allow_html=True)

esc = html.escape
LABEL = re.compile(r"^\s*([A-Z][A-Z0-9 '.\-]+?)((?:\s*\([^)]*\))*)\s*:\s*(.+)$", re.S)
KINDS = {"ACTION": "Action", "SFX": "Sound", "SOUND FX": "Sound", "CAMERA": "Camera"}
TIER = {"A": "Post first", "B": "Test next", "C": "Backup"}
SOURCE = {"both": "Rules + episode read", "rules": "Rules", "ai_scan": "Episode read"}
KEY_ENV = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
LOGO = ('<svg viewBox="0 0 16 16" fill="none"><rect x="1.5" y="3" width="13" height="10" rx="2" stroke="#1a1208" stroke-width="1.8"/>'
        '<path d="M6.5 6v4l3.5-2-3.5-2z" fill="#1a1208"/></svg>')


def split_line(text: str) -> tuple[str, str, bool]:
    """'CATHERINE (Venomous): line' -> ('Catherine (Venomous)', 'line', is_action)."""
    m = LABEL.match(text or "")
    if not m:
        return "", text or "", False
    who, tone, line = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
    if who in KINDS:
        return KINDS[who], line, True
    return f"{who.title()} {tone}".strip(), line, False


def line_html(text: str) -> str:
    who, line, act = split_line(text)
    return (f'<div class="who">{esc(who)}</div>' if who else "") + f'<div class="line{" act" if act else ""}">{esc(line)}</div>'


def mmss(sec: float) -> str:
    m, s = divmod(max(sec, 0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def tracks_html(r: dict, durs: dict) -> str:
    """One track per episode showing where the cut sits; red = hook start, violet = cliffhanger end."""
    by_ep: dict[int, list] = {}
    for s in r["segments"]:
        by_ep.setdefault(s["episode"], []).append(s)
    first, last = r["segments"][0], r["segments"][-1]
    out = ['<div class="tracks">']
    for ep, segs in by_ep.items():
        dur = float(durs.get(str(ep)) or max(s["end_sec"] for s in segs) * 1.05)
        ticks = "".join(f'<div class="tick" style="left:{t / dur * 100:.2f}%"></div>' for t in range(30, int(dur), 30))
        cuts = ""
        for s in segs:
            left = s["start_sec"] / dur * 100
            width = max((s["end_sec"] - s["start_sec"]) / dur * 100, 1.2)
            cuts += f'<div class="cut" style="left:{left:.2f}%;width:{width:.2f}%"></div>'
            if s is first:
                cuts += f'<div class="hk" style="left:{left:.2f}%"></div>'
            if s is last:
                cuts += f'<div class="cl" style="left:calc({left + width:.2f}% - 4px)"></div>'
        out.append(f'<div class="trk"><span class="e">EP {ep:02d}</span><div class="bar">{ticks}{cuts}</div>'
                   f'<span class="d">/ {mmss(dur)}</span></div>')
    out.append("</div>")
    return "".join(out)


def load_saved(p: Path) -> dict:
    data = json.loads((p / "promos.json").read_text(encoding="utf-8"))
    return {"meta": data["meta"], "quality": data["quality"], "promos": data["promos"], "candidates": None,
            "label": f"Saved run · {p.name}"}


# ---------------------------------------------------------------- sidebar (settings)
with st.sidebar:
    st.markdown("### Model")
    provider = st.selectbox("Provider", ["none", "gemini", "openai", "anthropic"], index=1,
                            help="'none' = rules only (free, instant). 'openai' = any OpenAI-compatible API (Groq, OpenRouter, Ollama…).")
    model = st.text_input("Model", value="gemini-3.1-flash-lite" if provider == "gemini" else DEFAULT_MODELS.get(provider, ""),
                          disabled=provider == "none")
    key = st.text_input("API key", type="password", disabled=provider == "none",
                        help="Kept only for this session. Leave blank to use the .env file.")
    base_url = st.text_input("Base URL", value="https://api.groq.com/openai/v1") if provider == "openai" else None
    st.markdown("### Search")
    scan = st.toggle("Read each full episode", value=True, disabled=provider == "none",
                     help="The model also reads every episode and proposes moments the keyword rules may miss. +1 call per episode.")
    scan_k = st.slider("Proposals per episode", 1, 4, 2, disabled=provider == "none" or not scan)
    shortlist = st.slider("Rule candidates to judge", 10, 80, 40, 5)
    st.markdown("### Ranking")
    rule_weight = st.slider("Rule score weight", 0.0, 1.0, 0.35, 0.05, disabled=provider == "none",
                            help="0 = rank only by the model's judgement, 1 = rules only.")
    top = st.slider("Promos to return", 5, 40, 20)
    per_ep = st.slider("Max per episode", 1, 4, 2)

saved_runs = sorted(p.parent for p in (HERE / "outputs").glob("*/promos.json")) if (HERE / "outputs").exists() else []
main_saved = next((p for p in saved_runs if p.name == "v2_scan_lite"), saved_runs[0] if saved_runs else None)
showcase = load_saved(main_saved) if main_saved else None
res = st.session_state.get("res")
has_key = provider == "none" or bool(key) or bool(os.environ.get(KEY_ENV.get(provider, ""), ""))
status_txt = "Rules only" if provider == "none" else (esc(model) if has_key else f"{esc(model)} · no API key")

# ---------------------------------------------------------------- nav + hero
st.markdown(
    f'<div class="nav glass"><span class="logo"><i>{LOGO}</i>Promo Finder</span>'
    f'<a href="#workspace" target="_self">Workspace</a><a href="#results" target="_self">Results</a>'
    f'<a href="#method" target="_self">Method</a>'
    f'<span class="st8"><b class="{"" if has_key else "off"}"></b>{status_txt}</span></div>', unsafe_allow_html=True)

if showcase and showcase["promos"]:
    sm, sp = showcase["meta"], showcase["promos"][0]
    hook_who, hook_line, _ = split_line(sp["hook"])
    cliff_who, cliff_line, _ = split_line(sp["cliffhanger"])
    right = (f'<div class="hr glass"><div class="cap"><b>Top pick from the sample run</b><span>{esc(sm.get("show_name", ""))}</span></div>'
             f'<div class="pick-t">Episode {sp["segments"][-1]["episode"]} · {sp["duration_sec"]}s</div>'
             f'<div class="pick-s">{esc(sp["segments_str"])}</div>{tracks_html(sp, sm.get("episode_durations") or {})}'
             f'<div class="pq"><small>Opens on</small><div>{esc(hook_line)}</div></div>'
             f'<div class="pq c"><small>Cuts on</small><div>{esc(cliff_line)}</div></div>'
             f'<div class="foot">Score {sp["final_score"]:.1f} / 10 · {esc(TIER[sp["tier"][0]])} · quotes checked against the script</div></div>')
    facts = (f'<div class="facts"><div><b>{sm["episodes"]} → {len(showcase["promos"])}</b><span>episodes in, ranked promos out</span></div>'
             f'<div><b>{sm["raw_windows"]:,}</b><span>possible cuts scored per show</span></div>'
             f'<div><b>{int(100 * (showcase["quality"].get("llm_quote_verified_rate") or 0))}%</b><span>quotes found word for word in the script</span></div></div>')
else:
    right = ('<div class="hr glass"><div class="cap"><b>What you get</b></div>'
             '<div class="pq"><small>Opens on</small><div>The line that stops the scroll in the first 3 seconds.</div></div>'
             '<div class="pq c"><small>Cuts on</small><div>The unresolved moment that makes them tap through.</div></div></div>')
    facts = ""

st.markdown(
    '<div class="hero"><div class="hl glass"><span class="eb"><i></i>Promo engine for microdramas</span>'
    '<h1>Cut the moment they <em>can\'t scroll past.</em></h1>'
    '<p class="sub">Give it a show\'s screenplays. It reads every episode, finds the 20–60 second stretches with a real hook '
    'and an open ending, checks every line against the script, and hands the editor timecodes to cut.</p>'
    '<div class="ctas"><a class="btn w" href="#workspace" target="_self">Start a run</a>'
    '<a class="btn o" href="#method" target="_self">How it ranks</a></div>'
    f'{facts}</div>{right}</div>', unsafe_allow_html=True)

if showcase:
    sm, sq = showcase["meta"], showcase["quality"]
    scan_stats = sm.get("scan") or {}
    st.markdown(
        '<div id="method"></div><div class="strip glass">'
        f'<div class="stp"><em>Parse</em><b>{sm["episodes"]}</b><span>episodes into timed beats</span></div>'
        f'<div class="stp"><em>Score</em><b>{sm["raw_windows"]:,}</b><span>windows by hook, ending, length</span></div>'
        f'<div class="stp"><em>Read</em><b>{scan_stats.get("proposed", "–")}</b><span>moments proposed by the model</span></div>'
        f'<div class="stp"><em>Judge</em><b>{sm.get("shortlist", "–")}</b><span>candidates scored 1–10</span></div>'
        f'<div class="stp"><em>Verify</em><b>{int(100 * (sq.get("llm_quote_verified_rate") or 0))}%</b><span>quotes in script, 0 mid-line cuts</span></div>'
        f'<div class="stp"><em>Rank</em><b>{len(showcase["promos"])}</b><span>promos in tiers A, B, C</span></div></div>',
        unsafe_allow_html=True)

# ---------------------------------------------------------------- workspace
st.markdown('<div id="workspace"></div><div class="sec"><div class="h">Workspace</div><span>Model and ranking settings are in the sidebar (top left)</span></div>',
            unsafe_allow_html=True)
left, rightc = st.columns([1.7, 1], gap="medium")
with left:
    with st.container(border=True):
        st.markdown('<div class="ptitle">Source</div><div class="psub">One .txt screenplay per episode, in the input-pack format.</div>',
                    unsafe_allow_html=True)
        tabs = st.tabs(["Upload", "Folder"] + (["Saved runs"] if saved_runs else []))
        with tabs[0]:
            files = st.file_uploader("Episode screenplays", type="txt", accept_multiple_files=True, label_visibility="collapsed")
        with tabs[1]:
            local = st.text_input("Folder path", value="", placeholder="../I Married a Broken Soldier. He was a Duke",
                                  label_visibility="collapsed")
        if saved_runs:
            with tabs[2]:
                c1, c2 = st.columns([3, 1])
                pick = c1.selectbox("Saved run", saved_runs, index=saved_runs.index(main_saved), format_func=lambda p: p.name,
                                    label_visibility="collapsed")
                if c2.button("Open", use_container_width=True):
                    st.session_state["res"] = load_saved(pick)
                    st.rerun()
with rightc:
    with st.container(border=True):
        st.markdown('<div class="ptitle">Run</div>'
                    f'<dl class="cfg"><dt>Judge</dt><dd class="mono">{"rules only" if provider == "none" else esc(model)}</dd>'
                    f'<dt>Episode read</dt><dd>{"On, " + str(scan_k) + " per episode" if scan and provider != "none" else "Off"}</dd>'
                    f'<dt>Rule weight</dt><dd>{rule_weight:.2f}</dd>'
                    f'<dt>Output</dt><dd>Top {top}, max {per_ep} per episode</dd></dl>', unsafe_allow_html=True)
        go = st.button("Find promos", type="primary", disabled=not (files or local))

if go:
    env = KEY_ENV.get(provider)
    if env and key:
        os.environ[env] = key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    tmp = tempfile.mkdtemp()
    folder = local
    if files:
        folder = tmp
        for f in files:
            (Path(tmp) / f.name).write_bytes(f.getvalue())
    lines = []
    with st.status("Finding promos…", expanded=True) as status:
        box = st.empty()

        def log(msg):
            lines.append(str(msg))
            box.code("\n".join(lines[-10:]), language=None)

        try:
            out = run(folder, out_dir=str(Path(tmp) / "out"), provider=provider, model=model or None,
                      shortlist=shortlist, top=top, per_episode=per_ep, log=log,
                      scan=scan, scan_k=scan_k, rule_weight=rule_weight)
            out["label"] = "Latest run"
            st.session_state["res"] = out
            status.update(label=f"Done · {len(out['promos'])} promos in {out['meta']['runtime_sec']}s", state="complete",
                          expanded=False)
            st.rerun()
        except SystemExit as e:
            status.update(label=str(e), state="error")


# ---------------------------------------------------------------- results
def card_html(r: dict, flags: list[str], durs: dict) -> str:
    t = r["tier"][0]
    eps = sorted({s["episode"] for s in r["segments"]})
    title = f"Episode {eps[0]}" if len(eps) == 1 else "Episodes " + " + ".join(map(str, eps))
    tags = (f'<span class="tg {t}">{t} · {TIER[t]}</span>'
            f'<span class="tg n">{"Stitched" if r["type"] == "stitch" else "Single take"}</span>'
            f'<span class="tg n">{SOURCE.get(r.get("source", "rules"), "")}</span>'
            + ('<span class="tg n" title="New viewers start at Episode 1, so the best premise promo is kept in the top 3">'
               'Premise slot</span>' if (r.get("features") or {}).get("premise_slot") else ""))
    tcs = "".join(f'<span class="tc">EP{s["episode"]:02d} {s["start"]} → {s["end"]}</span>' for s in r["segments"])
    s = r.get("llm_scores") or {}
    meters = "".join(
        f'<div class="m"><div class="ml"><span>{n}</span><b>{s[k]}</b></div><div class="mb"><div style="width:{s[k] * 10}%"></div></div></div>'
        for k, n in [("hook", "Hook"), ("cliffhanger", "Cliffhanger"), ("standalone", "Clarity"), ("emotion", "Emotion"),
                     ("overall", "Overall")] if k in s)
    flag = f'<div class="fl">Review before posting: {esc("; ".join(flags))}</div>' if flags else ""
    return (f'<div class="pc glass"><div class="pc-h"><span class="rk">#{r["rank"]:02d}</span><span class="pc-t">{title}</span>'
            f'{tags}<div class="pc-sc"><b>{r["final_score"]:.1f}</b><small> / 10 · {r["duration_sec"]}s</small></div></div>'
            f'{tracks_html(r, durs)}<div class="tcs">{tcs}</div>'
            f'<div class="hc"><div class="q h"><div class="k">Hook · first 3s</div>{line_html(r["hook"])}</div>'
            f'<div class="q c"><div class="k">Cliffhanger</div>{line_html(r["cliffhanger"])}</div></div>'
            f'<div class="why">{esc(r["why"])}</div>'
            + (f'<div class="mt">{meters}</div>' if meters else "") + f'{flag}</div>')


def edl(r: dict) -> str:
    return "\n".join(f'{r["rank"]:03d}  EP{s["episode"]:02d}  {s["start"]}  {s["end"]}  ({s["duration"]:.2f}s)'
                     for s in r["segments"])


def kpis_html(items) -> str:
    return '<div class="kpis">' + "".join(
        f'<div class="kpi glass"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}</div></div>'
        for l, v, s in items) + "</div>"


st.markdown('<div id="results"></div>', unsafe_allow_html=True)
res = st.session_state.get("res")
if not res:
    st.markdown('<div class="sec" style="margin-top:28px"><div class="h">Results</div></div>'
                '<div class="emp glass"><b>No results yet</b><span>Upload screenplays or point to a folder, then run. '
                + ('Or open a saved run under Workspace → Saved runs.' if saved_runs else '') + '</span></div>',
                unsafe_allow_html=True)
else:
    meta, qa, rows, cands = res["meta"], res["quality"], res["promos"], res.get("candidates")
    durs = meta.get("episode_durations") or {}
    tiers = {k: sum(r["tier"][0] == k for r in rows) for k in "ABC"}
    st.markdown(f'<div class="sec" style="margin-top:28px"><div class="h">Results</div><span>{esc(meta.get("show_name", ""))} · '
                f'{esc(res.get("label", ""))} · judge {esc(str(meta.get("model") or "rules only"))}</span></div>',
                unsafe_allow_html=True)
    vr = qa.get("llm_quote_verified_rate")
    st.markdown(kpis_html([
        ("Episodes read", meta["episodes"], f'{meta["raw_windows"]:,} windows scored'),
        ("Candidates judged", meta.get("shortlist", "–"), f'{meta.get("llm_calls", 0)} new model calls'),
        ("Promos", len(rows), f'{tiers["A"]} post first · {tiers["B"]} test next'),
        ("Quotes verified", f"{int(100 * vr)}%" if vr is not None else "–", "found word for word in the script"),
        ("Run time", f'{meta["runtime_sec"]}s', "cached calls cost nothing")]), unsafe_allow_html=True)
    if meta.get("llm_errors"):
        st.warning(f"{len(meta['llm_errors'])} model calls failed; those candidates use rule scores only. "
                   f"First: {meta['llm_errors'][0][:160]}")

    flagged = {f["rank"]: f["flags"] for f in qa.get("flagged", [])}
    t1, t2, t3 = st.tabs(["Promos", "Table & export", "Quality"])
    with t1:
        f1, f2, f3 = st.columns([1.6, 1, 1.4])
        tier_pick = f1.segmented_control("Tier", ["All", "A", "B", "C"], default="All", label_visibility="collapsed",
                                         format_func=lambda x: f"All {len(rows)}" if x == "All" else f"Tier {x} · {tiers[x]}",
                                         help="A = post first, B = test next, C = backup")
        sort = f2.selectbox("Sort", ["Sort: rank", "Sort: score", "Sort: episode", "Sort: shortest", "Sort: longest"],
                            label_visibility="collapsed")
        q = f3.text_input("Search", placeholder="Search lines, e.g. poison", label_visibility="collapsed").strip().lower()
        st.markdown('<div class="legend"><span><i style="background:var(--acs);border:1px solid var(--ac)"></i>Cut</span>'
                    '<span><i style="background:var(--hook)"></i>Hook start</span>'
                    '<span><i style="background:var(--cliff)"></i>Cliffhanger end</span><span>Ticks every 30s</span></div>',
                    unsafe_allow_html=True)
        keyf = {"Sort: rank": lambda i: rows[i]["rank"], "Sort: score": lambda i: -rows[i]["final_score"],
                "Sort: episode": lambda i: (rows[i]["segments"][-1]["episode"], rows[i]["segments"][-1]["start_sec"]),
                "Sort: shortest": lambda i: rows[i]["duration_sec"], "Sort: longest": lambda i: -rows[i]["duration_sec"]}[sort]
        shown = [i for i in sorted(range(len(rows)), key=keyf) if tier_pick in (None, "All") or rows[i]["tier"][0] == tier_pick]
        if q:
            shown = [i for i in shown if q in " ".join([rows[i]["hook"], rows[i]["cliffhanger"], rows[i]["why"]]).lower()]
        if not shown:
            st.markdown('<div class="emp glass"><b>No promos match</b><span>Try another tier or search term.</span></div>',
                        unsafe_allow_html=True)
        for i in shown:
            r = rows[i]
            st.markdown(card_html(r, flagged.get(r["rank"], []), durs), unsafe_allow_html=True)
            with st.expander("Edit list and script"):
                st.code(edl(r), language=None)
                if cands:
                    st.code(cands[i].transcript(with_camera=False), language=None)
    with t2:
        df = pd.DataFrame([{"Rank": r["rank"], "Tier": r["tier"], "Found by": SOURCE.get(r.get("source", "rules"), ""),
                            "Segments": r["segments_str"], "Seconds": r["duration_sec"], "Hook": r["hook"],
                            "Cliffhanger": r["cliffhanger"], "Why": r["why"], "Score": r["final_score"]} for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=10, format="%.1f")})
        d1, d2, d3, _ = st.columns([1, 1, 1.2, 2])
        d1.download_button("Download CSV", df.to_csv(index=False), "promos.csv", use_container_width=True)
        d2.download_button("Download JSON", json.dumps({"meta": meta, "quality": qa, "promos": rows}, indent=2), "promos.json",
                           use_container_width=True)
        d3.download_button("Download edit list", "\n".join(edl(r) for r in rows), "promos_edl.txt", use_container_width=True)
    with t3:
        dmin, dmean, dmax = qa.get("duration_min_mean_max") or ("–", "–", "–")
        sc = qa.get("source_counts") or {}
        st.markdown(kpis_html([
            ("Length range", f"{dmin}–{dmax}s", f"mean {dmean}s, target 25–45s"),
            ("Episodes covered", len(qa.get("episodes_covered", [])), f"max {per_ep} promos per episode"),
            ("Stitched", f'{int(100 * qa.get("share_stitched", 0))}%', "multi-segment promos"),
            ("Found by both", sc.get("both", "–"), f'{sc.get("rules", 0)} rules only · {sc.get("ai_scan", 0)} read only'),
            ("Rule–judge agreement", qa.get("rule_vs_llm_spearman", "–"), "Spearman rank correlation")]), unsafe_allow_html=True)
        st.markdown("**Flagged for review**")
        if qa.get("flagged"):
            st.dataframe(pd.DataFrame([{"Rank": f["rank"], "Segments": f["segments"], "Why it's flagged": "; ".join(f["flags"])}
                                       for f in qa["flagged"]]), use_container_width=True, hide_index=True)
        else:
            st.caption("Nothing flagged.")

st.markdown('<div class="foot">Promo Finder · Kuku AI Generalist assignment, Problem 1 · Shubham Prajapati · '
            'backdrop is an original generated image</div>', unsafe_allow_html=True)
