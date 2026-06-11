"""
CropCredit: Agricultural Risk Engine — Streamlit Dashboard
===========================================================
Run:  streamlit run app.py
"""

import os
import pickle
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropCredit | Agricultural Risk Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://cropcredit.ai/docs",
        "About": "CropCredit — AI-Powered Agricultural Loan Engine v2.0",
    },
)

# ══════════════════════════════════════════════════════════════════════════════
#  DARK THEME CSS
#  Palette: Deep Obsidian · Electric Teal · Amber Gold
#  Fonts:   Outfit (display) · JetBrains Mono (data)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

/* ── Design Tokens ── */
:root {
  /* Backgrounds — deep obsidian layers */
  --bg:       #080c14;
  --bg-card:  #0d1422;
  --bg-sub:   #111827;
  --bg-inp:   #0f1720;
  --bg-hover: #141e30;

  /* Teal accent — primary interactive */
  --teal1: #0d9488; --teal2: #14b8a6; --teal3: #2dd4bf;
  --teal4: #5eead4; --teal5: #ccfbf1;
  --teal-gl: rgba(20,184,166,0.12);

  /* Amber — financial / warning */
  --amb1: #78350f; --amb2: #b45309; --amb3: #d97706;
  --amb4: #f59e0b; --amb5: #fef3c7;
  --amb-gl: rgba(245,158,11,0.12);

  /* Red — danger / rejected */
  --red1: #7f1d1d; --red2: #dc2626; --red3: #f87171;
  --red-gl: rgba(220,38,38,0.10);

  /* Blue — info */
  --blu1: #1e3a8a; --blu2: #3b82f6; --blu3: #93c5fd;
  --blu-gl: rgba(59,130,246,0.10);

  /* Text */
  --t1: #f0f6ff; --t2: #cbd5e1; --t3: #94a3b8;
  --t4: #64748b; --t5: #3d5068;

  /* Borders */
  --bd:      rgba(255,255,255,0.07);
  --bd-md:   rgba(255,255,255,0.11);
  --bd-teal: rgba(20,184,166,0.30);
  --bd-amb:  rgba(245,158,11,0.28);
  --bd-red:  rgba(220,38,38,0.28);
  --bd-blu:  rgba(59,130,246,0.25);

  /* Typography */
  --fd: 'Outfit', sans-serif;
  --fb: 'Outfit', sans-serif;
  --fm: 'JetBrains Mono', monospace;

  /* Shadows */
  --sh-xs: 0 1px 4px rgba(0,0,0,0.4);
  --sh-sm: 0 2px 10px rgba(0,0,0,0.45);
  --sh-md: 0 4px 20px rgba(0,0,0,0.50);
  --sh-teal: 0 0 24px rgba(20,184,166,0.12);
  --sh-amb:  0 0 24px rgba(245,158,11,0.10);
}

/* ── Global Reset ── */
*, *::before, *::after { box-sizing:border-box; }
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family:var(--fb) !important;
  background:var(--bg) !important;
  color:var(--t2) !important;
}

/* Animated dot grid texture */
[data-testid="stAppViewContainer"]::before {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:radial-gradient(circle, rgba(20,184,166,0.06) 1px, transparent 1px);
  background-size:28px 28px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg-sub);}
::-webkit-scrollbar-thumb{background:var(--teal1);border-radius:10px;}
::-webkit-scrollbar-thumb:hover{background:var(--teal2);}

/* ── Layout ── */
.main .block-container{
  padding:1.5rem 2.25rem 3rem !important;
  max-width:1500px !important;
  position:relative; z-index:1;
}

/* ══════════════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"]{
  background:var(--bg-sub) !important;
  border-right:1px solid var(--bd) !important;
}
[data-testid="stSidebar"]::after{
  content:''; position:absolute; top:0; right:0; width:1px; height:100%;
  background:linear-gradient(180deg,transparent,var(--teal2),var(--teal1),transparent);
  opacity:0.3;
}
[data-testid="stSidebar"] .block-container{padding:0 1rem 2rem !important;}

/* Brand block */
.sb-brand{padding:20px 4px 14px;border-bottom:1px solid var(--bd);margin-bottom:4px;}
.sb-logo-row{display:flex;align-items:center;gap:10px;margin-bottom:5px;}
.sb-icon{
  width:38px;height:38px;border-radius:10px;flex-shrink:0;
  background:linear-gradient(135deg,var(--teal1),var(--teal2));
  display:flex;align-items:center;justify-content:center;
  font-size:1.2rem;
  box-shadow:0 0 18px rgba(20,184,166,0.40);
}
.sb-name{font-family:var(--fd) !important;font-size:1.18rem;font-weight:800;
  letter-spacing:-0.02em;color:var(--t1) !important;}
.sb-sub{font-size:0.62rem;color:var(--t4);letter-spacing:0.09em;text-transform:uppercase;}
.sb-live{
  display:inline-flex;align-items:center;gap:5px;
  background:rgba(20,184,166,0.08);border:1px solid rgba(20,184,166,0.22);
  border-radius:20px;padding:3px 9px;margin-top:7px;
  font-size:0.61rem;font-weight:600;color:var(--teal3);
  letter-spacing:0.09em;text-transform:uppercase;
}
.sb-dot{width:6px;height:6px;background:var(--teal2);border-radius:50%;
  animation:pdot 2.2s ease-in-out infinite;}
@keyframes pdot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.35;transform:scale(.7);}}

/* Sidebar section headers */
.sb-sec{display:flex;align-items:center;gap:6px;margin:16px 0 8px;}
.sb-sec-icon{font-size:0.74rem;opacity:0.55;}
.sb-sec-lbl{font-family:var(--fd) !important;font-size:0.58rem;font-weight:700;
  letter-spacing:0.16em;text-transform:uppercase;color:var(--teal2) !important;white-space:nowrap;}
.sb-sec-line{flex:1;height:1px;background:linear-gradient(90deg,var(--bd-teal),transparent);}

/* Sidebar footer */
.sb-foot{border-top:1px solid var(--bd);padding-top:12px;margin-top:18px;
  font-size:0.61rem;color:var(--t5);text-align:center;line-height:2;}
.sb-tags{display:flex;justify-content:center;gap:5px;margin-top:5px;flex-wrap:wrap;}
.sb-tag{background:var(--bg-card);border:1px solid var(--bd);border-radius:4px;
  padding:2px 6px;font-size:0.56rem;color:var(--t4);font-family:var(--fm) !important;}

/* Category badge */
.cat-badge{display:inline-block;font-size:0.58rem;font-weight:600;
  padding:2px 9px;border-radius:12px;margin:2px 0 6px;
  letter-spacing:0.05em;text-transform:uppercase;}
.cat-grain{background:rgba(245,158,11,0.12);color:var(--amb3);border:1px solid var(--bd-amb);}
.cat-veg{background:rgba(20,184,166,0.10);color:var(--teal3);border:1px solid var(--bd-teal);}
.cat-cash{background:rgba(59,130,246,0.10);color:var(--blu3);border:1px solid var(--bd-blu);}

/* ══════════════════════════════════════════════════════════════════
   INPUTS
══════════════════════════════════════════════════════════════════ */
label,[data-testid="stSidebar"] label{
  color:var(--t3) !important;font-size:0.73rem !important;
  font-weight:600 !important;letter-spacing:0.01em !important;
  font-family:var(--fb) !important;
}
[data-testid="stSelectbox"] > div > div{
  background:var(--bg-inp) !important;border:1.5px solid var(--bd-md) !important;
  border-radius:8px !important;color:var(--t1) !important;font-size:0.82rem !important;
}
[data-testid="stSelectbox"] > div > div:focus-within{
  border-color:var(--teal2) !important;box-shadow:0 0 0 3px var(--teal-gl) !important;
}
[data-testid="stNumberInput"] input{
  background:var(--bg-inp) !important;border:1.5px solid var(--bd-md) !important;
  border-radius:8px !important;color:var(--t1) !important;
  font-family:var(--fm) !important;font-size:0.82rem !important;
}
[data-testid="stNumberInput"] input:focus{
  border-color:var(--teal2) !important;box-shadow:0 0 0 3px var(--teal-gl) !important;
}
[data-testid="stSlider"] > div > div > div{background:rgba(255,255,255,0.08) !important;}
[data-testid="stSlider"] > div > div > div > div{
  background:linear-gradient(90deg,var(--teal1),var(--teal2)) !important;
}

/* ══════════════════════════════════════════════════════════════════
   CTA BUTTON
══════════════════════════════════════════════════════════════════ */
.stButton > button{
  width:100% !important;
  background:linear-gradient(135deg,var(--teal1) 0%,var(--teal2) 100%) !important;
  color:#fff !important;font-family:var(--fd) !important;font-weight:700 !important;
  font-size:0.75rem !important;letter-spacing:0.13em !important;text-transform:uppercase !important;
  border:none !important;border-radius:9px !important;padding:13px 16px !important;
  transition:all .22s cubic-bezier(.4,0,.2,1) !important;
  box-shadow:0 4px 16px rgba(20,184,166,0.30),inset 0 1px 0 rgba(255,255,255,.10) !important;
  position:relative !important;overflow:hidden !important;
}
.stButton > button::after{
  content:'';position:absolute;top:-50%;left:-60%;
  width:50%;height:200%;background:rgba(255,255,255,0.12);
  transform:skewX(-20deg);transition:left .5s ease;
}
.stButton > button:hover::after{left:130%;}
.stButton > button:hover{
  transform:translateY(-2px) !important;
  box-shadow:0 8px 24px rgba(20,184,166,0.40),inset 0 1px 0 rgba(255,255,255,.14) !important;
}
.stButton > button:active{transform:translateY(0) !important;}

/* ══════════════════════════════════════════════════════════════════
   MAIN HEADER
══════════════════════════════════════════════════════════════════ */
.mh{
  position:relative;border-radius:16px;overflow:hidden;
  padding:30px 40px;margin-bottom:22px;
  background:linear-gradient(135deg,var(--bg-card) 0%,var(--bg-sub) 50%,var(--bg-card) 100%);
  border:1px solid var(--bd);box-shadow:var(--sh-md);
}
.mh::before{
  content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,var(--teal1),var(--teal3),var(--amb4),var(--teal3),var(--teal1));
  background-size:300% 100%;animation:hsh 6s ease-in-out infinite;
}
.mh::after{
  content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;
  background:radial-gradient(circle,rgba(20,184,166,0.08) 0%,transparent 70%);
  pointer-events:none;
}
@keyframes hsh{0%{background-position:0% center;}50%{background-position:100% center;}100%{background-position:0% center;}}
.mh-inner{display:flex;justify-content:space-between;align-items:flex-start;
  flex-wrap:wrap;gap:18px;position:relative;z-index:1;}
.mh-eyebrow{font-family:var(--fm) !important;font-size:0.58rem;font-weight:500;
  letter-spacing:0.22em;text-transform:uppercase;color:var(--teal2);margin-bottom:8px;}
.mh-title{font-family:var(--fd) !important;font-size:2.6rem;font-weight:800;
  letter-spacing:-0.035em;line-height:1;color:var(--t1) !important;margin:0 0 5px;}
.mh-title span{
  background:linear-gradient(135deg,var(--teal3) 0%,var(--teal2) 60%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.mh-sub{font-size:0.83rem;color:var(--t4);max-width:500px;line-height:1.65;margin-top:8px;}
.mh-right{display:flex;flex-direction:column;align-items:flex-end;gap:11px;}
.mh-stat-row{display:flex;gap:20px;align-items:center;}
.mh-stat{text-align:right;}
.mh-stat-val{font-family:var(--fm) !important;font-size:0.9rem;font-weight:500;color:var(--t1);}
.mh-stat-key{font-size:0.58rem;color:var(--t4);letter-spacing:0.08em;text-transform:uppercase;}
.mh-divider{width:1px;height:30px;background:var(--bd-md);}
.mh-tags{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;}
.htag{font-family:var(--fm) !important;font-size:0.58rem;font-weight:500;
  letter-spacing:0.06em;padding:3px 9px;border-radius:5px;text-transform:uppercase;}
.htag-a{background:var(--amb-gl);border:1px solid var(--bd-amb);color:var(--amb4);}
.htag-b{background:var(--blu-gl);border:1px solid var(--bd-blu);color:var(--blu3);}
.htag-g{background:var(--teal-gl);border:1px solid var(--bd-teal);color:var(--teal3);}

/* ══════════════════════════════════════════════════════════════════
   PILLAR CARDS
══════════════════════════════════════════════════════════════════ */
.pc{background:var(--bg-card);border:1px solid var(--bd);border-radius:13px;
  padding:20px 22px;height:100%;position:relative;overflow:hidden;
  box-shadow:var(--sh-xs);transition:border-color .22s,transform .22s,box-shadow .22s;}
.pc:hover{border-color:var(--bd-teal);transform:translateY(-2px);box-shadow:var(--sh-teal);}
.pc::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:13px 13px 0 0;}
.pc1::before{background:linear-gradient(90deg,var(--blu2),transparent);}
.pc2::before{background:linear-gradient(90deg,var(--amb3),transparent);}
.pc3::before{background:linear-gradient(90deg,var(--teal2),transparent);}
.pc-num{font-family:var(--fm) !important;font-size:0.56rem;font-weight:500;
  letter-spacing:0.15em;text-transform:uppercase;margin-bottom:9px;}
.pc1 .pc-num{color:var(--blu2);}  .pc2 .pc-num{color:var(--amb3);}  .pc3 .pc-num{color:var(--teal2);}
.pc-head{font-family:var(--fd) !important;font-size:0.96rem;font-weight:700;
  color:var(--t1);margin-bottom:8px;letter-spacing:-0.01em;}
.pc-chip{display:inline-block;font-family:var(--fm) !important;font-size:0.55rem;
  font-weight:500;padding:2px 7px;border-radius:4px;margin-bottom:9px;letter-spacing:0.05em;}
.pc1 .pc-chip{background:var(--blu-gl);color:var(--blu3);border:1px solid var(--bd-blu);}
.pc2 .pc-chip{background:var(--amb-gl);color:var(--amb4);border:1px solid var(--bd-amb);}
.pc3 .pc-chip{background:var(--teal-gl);color:var(--teal3);border:1px solid var(--bd-teal);}
.pc-desc{font-size:0.76rem;color:var(--t4);line-height:1.65;}

/* ══════════════════════════════════════════════════════════════════
   SECTION LABEL
══════════════════════════════════════════════════════════════════ */
.sl{display:flex;align-items:center;gap:9px;margin-bottom:13px;margin-top:4px;}
.sl-t{font-family:var(--fd) !important;font-size:0.61rem;font-weight:700;
  letter-spacing:0.17em;text-transform:uppercase;color:var(--teal2);white-space:nowrap;}
.sl-l{flex:1;height:1px;background:linear-gradient(90deg,var(--bd-teal),transparent);}

/* ══════════════════════════════════════════════════════════════════
   KPI METRICS
══════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"]{
  background:var(--bg-card) !important;border:1px solid var(--bd) !important;
  border-radius:12px !important;padding:18px 20px 14px !important;
  box-shadow:var(--sh-xs) !important;
  transition:border-color .2s,box-shadow .2s !important;
}
[data-testid="stMetric"]:hover{border-color:var(--bd-teal) !important;box-shadow:var(--sh-teal) !important;}
[data-testid="stMetricLabel"]{font-family:var(--fb) !important;font-size:0.64rem !important;
  font-weight:700 !important;letter-spacing:0.11em !important;text-transform:uppercase !important;
  color:var(--t4) !important;}
[data-testid="stMetricValue"]{font-family:var(--fm) !important;font-size:1.78rem !important;
  font-weight:500 !important;color:var(--t1) !important;
  letter-spacing:-0.025em !important;line-height:1.2 !important;}
[data-testid="stMetricDelta"]{font-size:0.7rem !important;font-weight:500 !important;
  font-family:var(--fb) !important;}
[data-testid="stMetricDelta"] svg{display:none !important;}

/* ══════════════════════════════════════════════════════════════════
   PARAMETER GRID
══════════════════════════════════════════════════════════════════ */
.fgrid{display:grid;grid-template-columns:1fr 1fr;gap:1px;
  background:var(--bd);border:1px solid var(--bd);border-radius:10px;
  overflow:hidden;box-shadow:var(--sh-xs);}
.fcell{background:var(--bg-card);padding:11px 15px;transition:background .15s;}
.fcell:hover{background:var(--bg-hover);}
.fkey{font-size:0.6rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;
  color:var(--t5);margin-bottom:3px;}
.fval{font-family:var(--fm) !important;font-size:0.84rem;font-weight:400;color:var(--t2);}

/* ══════════════════════════════════════════════════════════════════
   SPOILAGE GAUGE
══════════════════════════════════════════════════════════════════ */
.gw{background:var(--bg-card);border:1px solid var(--bd);border-radius:14px;
  padding:22px 20px 18px;box-shadow:var(--sh-xs);}
.gw-title{display:flex;align-items:center;gap:7px;font-size:0.62rem;font-weight:700;
  letter-spacing:0.13em;text-transform:uppercase;color:var(--t4);margin-bottom:14px;}
.gw-title::before{content:'';display:inline-block;width:7px;height:7px;
  background:var(--teal2);border-radius:2px;}
.gw-num{font-family:var(--fm) !important;font-size:4.2rem;font-weight:600;
  line-height:1;letter-spacing:-0.04em;margin-bottom:3px;}
.gw-badge{display:inline-block;font-family:var(--fd) !important;font-size:0.62rem;
  font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
  padding:3px 11px;border-radius:5px;margin-top:5px;margin-bottom:16px;}
.bh{background:var(--red-gl);border:1px solid var(--bd-red);color:var(--red3);}
.bm{background:var(--amb-gl);border:1px solid var(--bd-amb);color:var(--amb4);}
.bl{background:var(--teal-gl);border:1px solid var(--bd-teal);color:var(--teal3);}
.gtrack{background:rgba(255,255,255,0.07);border-radius:6px;overflow:hidden;
  height:8px;margin-bottom:5px;}
.gfill{height:100%;border-radius:6px;transition:width 1.2s cubic-bezier(.4,0,.2,1);}
.gticks{display:flex;justify-content:space-between;
  font-family:var(--fm) !important;font-size:0.57rem;color:var(--t5);margin-top:3px;}
.gdrivers{margin-top:16px;padding-top:13px;border-top:1px solid var(--bd);}
.gd-lbl{font-size:0.59rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
  color:var(--t5);margin-bottom:9px;}
.gd-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;}
.gd-name{font-size:0.71rem;color:var(--t3);display:flex;align-items:center;gap:5px;}
.gd-val{font-family:var(--fm) !important;font-size:0.71rem;font-weight:400;color:var(--t2);}
.gd-side{display:flex;align-items:center;gap:8px;}
.gd-track{width:55px;height:4px;background:rgba(255,255,255,0.08);border-radius:3px;overflow:hidden;}
.gd-fill{height:100%;border-radius:3px;}

/* ══════════════════════════════════════════════════════════════════
   FORECAST PANEL
══════════════════════════════════════════════════════════════════ */
.fpanel{background:var(--bg-card);border:1px solid var(--bd);border-radius:14px;
  padding:20px 26px 14px;box-shadow:var(--sh-xs);}
.fp-header{display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:14px;flex-wrap:wrap;gap:10px;}
.fp-title{font-family:var(--fd) !important;font-size:0.97rem;font-weight:700;
  color:var(--t1);letter-spacing:-0.01em;}
.fp-sub{font-size:0.71rem;color:var(--t4);margin-top:3px;}
.fp-stats{display:flex;gap:8px;flex-wrap:wrap;}
.fps{text-align:center;padding:7px 12px;background:var(--bg-sub);
  border:1px solid var(--bd);border-radius:7px;min-width:72px;}
.fps-val{font-family:var(--fm) !important;font-size:0.85rem;font-weight:400;color:var(--t1);}
.fps-key{font-size:0.56rem;color:var(--t5);letter-spacing:0.08em;text-transform:uppercase;margin-top:1px;}
.chart-leg{display:flex;gap:16px;justify-content:center;margin-top:7px;}
.cl-item{display:flex;align-items:center;gap:5px;font-size:0.67rem;color:var(--t4);}
.cl-dot{width:8px;height:8px;border-radius:2px;}

/* ══════════════════════════════════════════════════════════════════
   DECISION BANNER
══════════════════════════════════════════════════════════════════ */
.db-wrap{border-radius:14px;padding:26px 32px;
  position:relative;overflow:hidden;margin-top:6px;box-shadow:var(--sh-md);}
.db-wrap.app{
  background:linear-gradient(135deg,rgba(13,148,136,0.08) 0%,rgba(13,20,34,0.95) 100%);
  border:1px solid var(--bd-teal);}
.db-wrap.rej{
  background:linear-gradient(135deg,rgba(220,38,38,0.08) 0%,rgba(13,20,34,0.95) 100%);
  border:1px solid var(--bd-red);}
.db-wrap::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;}
.db-wrap.app::before{background:linear-gradient(90deg,transparent,var(--teal2),transparent);}
.db-wrap.rej::before{background:linear-gradient(90deg,transparent,var(--red3),transparent);}
.db-inner{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:22px;}
.db-tag{font-family:var(--fm) !important;font-size:0.58rem;font-weight:500;
  letter-spacing:0.18em;text-transform:uppercase;margin-bottom:6px;}
.db-wrap.app .db-tag{color:var(--teal3);}  .db-wrap.rej .db-tag{color:var(--red3);}
.db-amt{font-family:var(--fd) !important;font-size:2.9rem;font-weight:800;
  letter-spacing:-0.04em;line-height:1;margin-bottom:7px;}
.db-wrap.app .db-amt{color:var(--teal3);}  .db-wrap.rej .db-amt{color:var(--red3);}
.db-detail{font-size:0.78rem;color:var(--t3);line-height:1.65;max-width:420px;}
.db-kv{text-align:right;}
.db-kv-val{font-family:var(--fm) !important;font-size:1.05rem;font-weight:500;color:var(--t1);}
.db-kv-key{font-size:0.58rem;color:var(--t4);letter-spacing:0.08em;text-transform:uppercase;}
.db-kvs{display:flex;flex-direction:column;align-items:flex-end;gap:9px;}

/* ══════════════════════════════════════════════════════════════════
   ANALYTICS CHART PANELS
══════════════════════════════════════════════════════════════════ */
.cpanel{background:var(--bg-card);border:1px solid var(--bd);border-radius:14px;
  padding:18px 20px;box-shadow:var(--sh-xs);}
.cpanel-title{font-family:var(--fd) !important;font-size:0.83rem;font-weight:700;
  color:var(--t1);margin-bottom:2px;}
.cpanel-sub{font-size:0.69rem;color:var(--t4);margin-bottom:12px;}

/* ══════════════════════════════════════════════════════════════════
   COLLATERAL CALCULATOR BARS
══════════════════════════════════════════════════════════════════ */
.calc-card{background:var(--bg-card);border:1px solid var(--bd-amb);
  border-radius:12px;padding:18px 22px;box-shadow:var(--sh-xs);}
.calc-card.green{border-color:var(--bd-teal);}
.calc-bar-track{background:rgba(255,255,255,0.07);border-radius:6px;height:10px;overflow:hidden;}

/* ══════════════════════════════════════════════════════════════════
   DB INFO CARD
══════════════════════════════════════════════════════════════════ */
.db-info{background:var(--bg-sub);border:1px solid var(--bd);border-radius:8px;
  padding:13px 17px;font-family:var(--fm) !important;
  font-size:0.7rem;line-height:2;color:var(--t4);}
.db-info strong{color:var(--t2);font-weight:500;}

/* ══════════════════════════════════════════════════════════════════
   ALERTS (override Streamlit defaults)
══════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"]{border-radius:10px !important;border-left:none !important;padding:14px 18px !important;}

/* ══════════════════════════════════════════════════════════════════
   PLACEHOLDER
══════════════════════════════════════════════════════════════════ */
.ph-wrap{padding:52px 20px;text-align:center;}
.ph-icon{font-size:3rem;margin-bottom:16px;}
.ph-title{font-family:var(--fd) !important;font-size:1.35rem;font-weight:700;
  color:var(--t2);margin-bottom:9px;letter-spacing:-0.02em;}
.ph-sub{font-size:0.81rem;color:var(--t4);max-width:440px;margin:0 auto;line-height:1.75;}
.ph-cta{display:inline-block;margin-top:20px;background:var(--teal-gl);
  border:1px solid var(--bd-teal);border-radius:8px;padding:9px 18px;
  font-family:var(--fm) !important;font-size:0.69rem;color:var(--teal3);letter-spacing:0.06em;}

/* ══════════════════════════════════════════════════════════════════
   MISC
══════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden;}
#MainMenu,footer,header{visibility:hidden;}

@keyframes fade-up{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:translateY(0);}}
.a1{animation:fade-up .35s ease both;}
.a2{animation:fade-up .35s .07s ease both;}
.a3{animation:fade-up .35s .14s ease both;}
.a4{animation:fade-up .35s .21s ease both;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGING & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("CropCredit.App")

MODELS_DIR = "models"

# Restricted to 5 commodities with real data
COMMODITIES = [
    "Wheat","Rice","Potato","Onion","Tomato",
]

COMMODITY_CATEGORY = {
    "Wheat":"Grain","Rice":"Grain",
    "Potato":"Vegetable","Onion":"Vegetable","Tomato":"Vegetable",
}

BASE_PRICES = {
    "Wheat":2275,"Rice":2183,
    "Potato":1200,"Onion":1800,"Tomato":2200,
}

# Higher spoilage sensitivity for perishable vegetables
SPOILAGE_SENSITIVITY = {
    "Wheat":0.7,"Rice":0.9,
    "Potato":1.3,"Onion":1.0,"Tomato":2.0,
}



# ══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_commodity_models(commodity):
    commodity_lower = commodity.lower()
    fp = os.path.join(MODELS_DIR, f"financialriskmodel_{commodity_lower}.pkl")
    pp = os.path.join(MODELS_DIR, f"physicalriskmodel_{commodity_lower}.pkl")
    ep = os.path.join(MODELS_DIR, "label_encoder.pkl")
    for p in [fp, pp, ep]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Not found: {p}\nRun: python train_models.py")
    with open(fp,"rb") as f: fm = pickle.load(f)
    with open(pp,"rb") as f: pm = pickle.load(f)
    with open(ep,"rb") as f: le = pickle.load(f)
    logger.info(f"Models for {commodity} loaded.")
    return fm, pm, le

# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def run_inference(fm, pm, le, commodity, tonnage, market_arrivals, current_price,
                  rainfall_deficit, warehouse_temp, humidity, moisture_content):
    # Fall back to a known training commodity for encoding
    known = list(le.classes_)
    enc_commodity = commodity if commodity in known else known[0]
    enc = le.transform([enc_commodity])[0]

    X = pd.DataFrame([{
        "commodity_encoded": enc,       "tonnage": tonnage,
        "market_arrivals": market_arrivals, "current_price": current_price,
        "rainfall_deficit": rainfall_deficit, "warehouse_temp": warehouse_temp,
        "humidity": humidity,           "moisture_content": moisture_content,
    }])

    predicted_price = float(fm.predict(X)[0])

    # Adjust spoilage for vegetable perishability
    proba_all = pm.predict_proba(X)
    if proba_all.shape[1] == 1:
        base_spoi = 1.0 if pm.classes_[0] == 1 else 0.0
    else:
        base_spoi = float(proba_all[0, 1])
    sensitivity = SPOILAGE_SENSITIVITY.get(commodity, 1.0)
    spoilage_pct = round(min(1.0, base_spoi * sensitivity) * 100, 1)

    # Dynamic LTV
    if spoilage_pct > 70:
        ltv, decision, reason = 0.0, "REJECTED", f"Spoilage risk ({spoilage_pct}%) exceeds 70% threshold."
    elif spoilage_pct < 30:
        ltv, decision, reason = 85.0, "APPROVED", None
    else:
        ltv, decision, reason = 60.0, "APPROVED", None

    quintals  = tonnage * 10
    col_val   = quintals * predicted_price
    sanc_amt  = col_val * (ltv / 100)
    curve     = _price_curve(current_price, predicted_price)
    opt_day   = int(curve["Price (₹/Qtl)"].argmax()) + 1

    return dict(
        commodity=commodity, tonnage=tonnage, market_arrivals=market_arrivals,
        current_price=current_price, rainfall_deficit=rainfall_deficit,
        warehouse_temp=warehouse_temp, humidity=humidity, moisture_content=moisture_content,
        spoilage_pct=spoilage_pct, predicted_price=round(predicted_price, 2),
        optimal_sell_day=opt_day, approved_ltv=ltv,
        collateral_value=round(col_val, 2), sanctioned_amount=round(sanc_amt, 2),
        loan_decision=decision, rejection_reason=reason, price_curve=curve,
        requested_loan=0.0,   # overwritten in main() after sidebar input
    )

def _price_curve(s, e, days=90):
    np.random.seed(42)
    t  = np.linspace(0, 1, days)
    p  = s + (e-s)*t + 0.04*s*np.sin(4*np.pi*t + np.pi/6) + np.random.normal(0, 0.012*s, days)
    p  = np.maximum(p, s * 0.7)
    dts = [(datetime.today()+timedelta(days=i)).strftime("%b %d") for i in range(days)]
    df  = pd.DataFrame({"Date": dts, "Price (₹/Qtl)": np.round(p, 2)}).set_index("Date")
    df["7-Day MA"] = df["Price (₹/Qtl)"].rolling(7, min_periods=1).mean().round(2)
    return df



# ══════════════════════════════════════════════════════════════════════════════
#  UI — HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_header():
    now = datetime.now()
    st.markdown(f"""
    <div class="mh a1">
      <div class="mh-inner">
        <div>
          <div class="mh-eyebrow">Agricultural Credit Intelligence · B2B Risk Engine</div>
          <h1 class="mh-title">Crop<span>Credit</span></h1>
          <p class="mh-sub">3-Pillar AI system for dynamic Loan-to-Value assessment on stored
            agricultural commodities — grains, pulses, cash crops &amp; vegetables.</p>
        </div>
        <div class="mh-right">
          <div class="mh-stat-row">
            <div class="mh-stat"><div class="mh-stat-val">XGBoost</div><div class="mh-stat-key">Price Oracle</div></div>
            <div class="mh-divider"></div>
            <div class="mh-stat"><div class="mh-stat-val">Random.Forest</div><div class="mh-stat-key">Spoilage Engine</div></div>
          </div>
          <div class="mh-tags">
            <span class="htag htag-a">AGMARKNET Data</span>
            <span class="htag htag-b">OpenWeather IoT</span>
            <span class="htag htag-g">{now.strftime("%d %b %Y  ·  %H:%M")}</span>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UI — SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def render_sidebar(fm, pm, le):
    st.sidebar.markdown("""
    <div class="sb-brand">
      <div class="sb-logo-row">
        <div class="sb-icon">🌾</div>
        <div><div class="sb-name">CropCredit</div></div>
      </div>
      <div class="sb-sub">Loan Evaluation Terminal</div>
      <div class="sb-live"><div class="sb-dot"></div>Engine Online</div>
    </div>""", unsafe_allow_html=True)

    # ── Commodity ────────────────────────────────────────────────────────────
    st.sidebar.markdown("""<div class="sb-sec"><span class="sb-sec-icon">📦</span>
    <span class="sb-sec-lbl">Commodity</span><span class="sb-sec-line"></span></div>
    """, unsafe_allow_html=True)

    commodity = st.sidebar.selectbox("Type", COMMODITIES, label_visibility="collapsed")
    cat       = COMMODITY_CATEGORY.get(commodity, "Other")
    cat_cls   = {"Grain":"cat-grain","Oilseed":"cat-grain","Pulse":"cat-grain",
                 "Cash Crop":"cat-cash","Vegetable":"cat-veg"}.get(cat, "cat-grain")
    st.sidebar.markdown(f'<span class="cat-badge {cat_cls}">{cat}</span>', unsafe_allow_html=True)

    tonnage = st.sidebar.number_input("Tonnage (MT)", min_value=1.0,
                                       max_value=50_000.0, value=500.0, step=50.0)

    # ── Loan Request ─────────────────────────────────────────────────────────
    st.sidebar.markdown("""<div class="sb-sec"><span class="sb-sec-icon">💸</span>
    <span class="sb-sec-lbl">Loan Request</span><span class="sb-sec-line"></span></div>
    """, unsafe_allow_html=True)
    requested_loan = st.sidebar.number_input(
        "Requested Loan Amount (₹)",
        min_value=10_000.0, max_value=500_000_000.0,
        value=500_000.0, step=10_000.0,
        help="Farmer's requested loan — drives the Reverse Collateral Calculator.",
    )

    # ── Market Intelligence ──────────────────────────────────────────────────
    st.sidebar.markdown("""<div class="sb-sec"><span class="sb-sec-icon">📈</span>
    <span class="sb-sec-lbl">Market Intelligence</span><span class="sb-sec-line"></span></div>
    """, unsafe_allow_html=True)
    default_price   = float(BASE_PRICES.get(commodity, 2000))
    market_arrivals = st.sidebar.number_input("Market Arrivals (Qtl/day)",
                                               min_value=100.0, max_value=500_000.0,
                                               value=15_000.0, step=1_000.0)
    current_price   = st.sidebar.number_input("Mandi Price (₹/Qtl)",
                                               min_value=100.0, max_value=100_000.0,
                                               value=default_price, step=50.0)
    rainfall_deficit= st.sidebar.number_input("Rainfall Deficit (mm)",
                                               min_value=-200.0, max_value=500.0,
                                               value=15.0, step=5.0)

    # ── Warehouse Sensors ────────────────────────────────────────────────────
    st.sidebar.markdown("""<div class="sb-sec"><span class="sb-sec-icon">🌡️</span>
    <span class="sb-sec-lbl">Warehouse Sensors</span><span class="sb-sec-line"></span></div>
    """, unsafe_allow_html=True)
    warehouse_temp   = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 25.0, 0.5)
    humidity         = st.sidebar.slider("Humidity (%)",      10.0, 98.0, 60.0, 1.0)
    moisture_content = st.sidebar.slider("Moisture Content (%)", 5.0, 30.0, 13.0, 0.5)

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    evaluate = st.sidebar.button("Evaluate Risk & Sanction Loan", type="primary")

    st.sidebar.markdown("""
    <div class="sb-foot">
      CropCredit AI Risk Engine &nbsp;·&nbsp; v2.0
      <div class="sb-tags">
        <span class="sb-tag">XGBoost</span><span class="sb-tag">RandomForest</span>
        <span class="sb-tag">Streamlit</span>
      </div>
      <div style="margin-top:8px;">© 2025 CropCredit Technologies</div>
    </div>""", unsafe_allow_html=True)

    return dict(
        commodity=commodity,         tonnage=tonnage,
        requested_loan=requested_loan,
        market_arrivals=market_arrivals, current_price=current_price,
        rainfall_deficit=rainfall_deficit, warehouse_temp=warehouse_temp,
        humidity=humidity,           moisture_content=moisture_content,
        evaluate=evaluate,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  UI — PILLAR CARDS
# ══════════════════════════════════════════════════════════════════════════════
def render_pillars():
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="pc pc1 a2">
          <div class="pc-num">Pillar 01 · Financial Risk</div>
          <div class="pc-head">90-Day Price Oracle</div>
          <span class="pc-chip">XGBoost Regressor</span>
          <p class="pc-desc">Forecasts commodity price 90 days forward using AGMARKNET
          market arrivals, seasonal trends, and rainfall supply shocks.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="pc pc2 a3">
          <div class="pc-num">Pillar 02 · Physical Risk</div>
          <div class="pc-head">Spoilage Classifier</div>
          <span class="pc-chip">Random Forest · Balanced</span>
          <p class="pc-desc">Classifies HIGH / LOW spoilage risk from IoT warehouse sensors:
          temperature, humidity, and grain moisture content.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="pc pc3 a4">
          <div class="pc-num">Pillar 03 · Yield Optimizer</div>
          <div class="pc-head">Optimal Sell Date</div>
          <span class="pc-chip">Time-Series Simulation</span>
          <p class="pc-desc">Identifies peak liquidation window in the 90-day price curve
          to maximise collateral recovery for the lending bank.</p>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UI — KPI METRICS ROW
# ══════════════════════════════════════════════════════════════════════════════
def render_kpis(r):
    st.markdown("""<div class="sl"><span class="sl-t">⚡ AI Risk Assessment — Core KPIs</span>
    <span class="sl-l"></span></div>""", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    risk = r["spoilage_pct"]
    ra   = "▲ HIGH RISK" if risk > 70 else ("◈ MODERATE" if risk > 30 else "▼ LOW RISK")
    pct  = (r["predicted_price"] - r["current_price"]) / r["current_price"] * 100
    c1.metric("🦠  Spoilage Risk",    f"{risk}%",                     ra,
              delta_color="inverse" if risk > 70 else "normal")
    c2.metric("📈  90-Day Forecast",  f"₹{r['predicted_price']:,.0f}", f"{pct:+.1f}% vs today")
    c3.metric("🏦  Approved LTV",     f"{r['approved_ltv']:.0f}%",    r["loan_decision"],
              delta_color="normal" if r["loan_decision"]=="APPROVED" else "inverse")
    c4.metric("📅  Optimal Sell Day", f"Day {r['optimal_sell_day']}",  f"+{r['optimal_sell_day']} days")

# ══════════════════════════════════════════════════════════════════════════════
#  UI — FINANCIALS + SPOILAGE GAUGE
# ══════════════════════════════════════════════════════════════════════════════
def render_financials_and_gauge(r):
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="sl"><span class="sl-t">💰 Loan Financials</span>
        <span class="sl-l"></span></div>""", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        q = r["tonnage"] * 10
        fc1.metric("Tonnage",           f"{r['tonnage']:,.0f} MT")
        fc2.metric("Collateral Value",  f"₹{r['collateral_value']/1e5:,.1f}L", f"{q:,.0f} qtl")
        fc3.metric("Sanctioned Amount", f"₹{r['sanctioned_amount']/1e5:,.1f}L",
                   f"LTV {r['approved_ltv']:.0f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="sl"><span class="sl-t">📋 Application Parameters</span>
        <span class="sl-l"></span></div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="fgrid">
          <div class="fcell"><div class="fkey">Commodity</div><div class="fval">{r['commodity']}</div></div>
          <div class="fcell"><div class="fkey">Category</div><div class="fval">{COMMODITY_CATEGORY.get(r['commodity'],'—')}</div></div>
          <div class="fcell"><div class="fkey">Tonnage</div><div class="fval">{r['tonnage']:,.0f} MT</div></div>
          <div class="fcell"><div class="fkey">Market Arrivals</div><div class="fval">{r['market_arrivals']:,.0f} Qtl/day</div></div>
          <div class="fcell"><div class="fkey">Mandi Price</div><div class="fval">₹{r['current_price']:,.0f} / Qtl</div></div>
          <div class="fcell"><div class="fkey">Rainfall Deficit</div><div class="fval">{r['rainfall_deficit']:+.1f} mm</div></div>
          <div class="fcell"><div class="fkey">Warehouse Temp</div><div class="fval">{r['warehouse_temp']:.1f} °C</div></div>
          <div class="fcell"><div class="fkey">Relative Humidity</div><div class="fval">{r['humidity']:.1f} %</div></div>
          <div class="fcell"><div class="fkey">Moisture Content</div><div class="fval">{r['moisture_content']:.1f} %</div></div>
          <div class="fcell"><div class="fkey">Spoilage Sensitivity</div><div class="fval">{SPOILAGE_SENSITIVITY.get(r['commodity'],1.0):.1f}×</div></div>
        </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""<div class="sl"><span class="sl-t">🌡️ Spoilage Risk Gauge</span>
        <span class="sl-l"></span></div>""", unsafe_allow_html=True)
        rp = r["spoilage_pct"]
        if rp > 70:   gc, bc, bt = "#f87171", "bh", "HIGH RISK"
        elif rp > 30: gc, bc, bt = "#f59e0b", "bm", "MODERATE"
        else:         gc, bc, bt = "#2dd4bf", "bl", "LOW RISK"
        tp  = min(100, max(0, (r['warehouse_temp']-10)/40*100))
        rhp = min(100, r['humidity'])
        mp  = min(100, r['moisture_content']/30*100)
        tc  = "#f87171" if r['warehouse_temp']>30 else "#f59e0b" if r['warehouse_temp']>25 else "#2dd4bf"
        rc  = "#f87171" if r['humidity']>75      else "#f59e0b" if r['humidity']>60      else "#2dd4bf"
        mc  = "#f87171" if r['moisture_content']>18 else "#f59e0b" if r['moisture_content']>14 else "#2dd4bf"

        st.markdown(f"""
        <div class="gw">
          <div class="gw-title">Spoilage Probability — Physical Risk</div>
          <div class="gw-num" style="color:{gc};">{rp:.1f}%</div>
          <div><span class="gw-badge {bc}">{bt}</span></div>
          <div class="gtrack"><div class="gfill"
            style="width:{rp}%;background:linear-gradient(90deg,{gc}88,{gc});"></div></div>
          <div class="gticks">
            <span>0%</span><span style="color:#f59e0b;">30%</span>
            <span style="color:#f87171;">70%</span><span>100%</span>
          </div>
          <div class="gdrivers">
            <div class="gd-lbl">Key Risk Drivers</div>
            <div class="gd-row">
              <div class="gd-name">🌡️ Temperature</div>
              <div class="gd-side">
                <div class="gd-track"><div class="gd-fill" style="width:{tp:.0f}%;background:{tc};"></div></div>
                <div class="gd-val">{r['warehouse_temp']:.1f}°C</div>
              </div>
            </div>
            <div class="gd-row">
              <div class="gd-name">💧 Humidity</div>
              <div class="gd-side">
                <div class="gd-track"><div class="gd-fill" style="width:{rhp:.0f}%;background:{rc};"></div></div>
                <div class="gd-val">{r['humidity']:.1f}%</div>
              </div>
            </div>
            <div class="gd-row">
              <div class="gd-name">🌾 Moisture</div>
              <div class="gd-side">
                <div class="gd-track"><div class="gd-fill" style="width:{mp:.0f}%;background:{mc};"></div></div>
                <div class="gd-val">{r['moisture_content']:.1f}%</div>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UI — ANALYTICS CHARTS (Plotly)
# ══════════════════════════════════════════════════════════════════════════════
def render_analytics_charts(r):
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Run `pip install plotly` to enable analytics charts.")
        return

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="sl"><span class="sl-t">📈 Portfolio Analytics Dashboard</span>
    <span class="sl-l"></span></div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    PAPER = "rgba(0,0,0,0)"
    FONT  = dict(family="JetBrains Mono", color="#94a3b8")

    # ── Chart 1: Donut — Loan Utilisation ────────────────────────────────────
    with col1:
        st.markdown("""<div class="cpanel">
          <div class="cpanel-title">Loan Utilisation</div>
          <div class="cpanel-sub">Sanctioned vs available collateral buffer</div>""",
          unsafe_allow_html=True)
        used = min(r["sanctioned_amount"], r["requested_loan"]) if r["requested_loan"] > 0 else r["sanctioned_amount"]
        free = max(0, r["collateral_value"] - used)
        fig1 = go.Figure(go.Pie(
            labels=["Sanctioned Loan", "Collateral Buffer"],
            values=[used, free], hole=0.62,
            marker=dict(colors=["#0d9488","#1a2535"],
                        line=dict(color="#0d1422", width=3)),
            textinfo="percent",
            textfont=dict(family="JetBrains Mono", size=11, color="#cbd5e1"),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
        ))
        fig1.add_annotation(
            text=f"₹{used/1e5:,.1f}L<br><span style='font-size:10px;color:#64748b'>Sanctioned</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="JetBrains Mono", size=13, color="#f0f6ff"), align="center",
        )
        fig1.update_layout(showlegend=True, margin=dict(t=10,b=10,l=10,r=10), height=230,
            paper_bgcolor=PAPER, plot_bgcolor=PAPER,
            legend=dict(font=dict(**FONT, size=9), orientation="h",
                        yanchor="bottom", y=-0.22, xanchor="center", x=0.5))
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Chart 2: Bar — Risk Factor Breakdown ─────────────────────────────────
    with col2:
        st.markdown("""<div class="cpanel">
          <div class="cpanel-title">Risk Factor Breakdown</div>
          <div class="cpanel-sub">Normalised score of each physical risk driver</div>""",
          unsafe_allow_html=True)
        scores  = [
            min(100, max(0, (r['warehouse_temp']-10)/40*100)),
            min(100, r['humidity']),
            min(100, r['moisture_content']/30*100),
            r['spoilage_pct'],
        ]
        labels  = ["Temp","Humidity","Moisture","Spoilage"]
        colors  = ["#f87171" if s>70 else "#f59e0b" if s>40 else "#2dd4bf" for s in scores]
        fig2 = go.Figure(go.Bar(
            x=labels, y=scores,
            marker=dict(color=colors, line=dict(color="#0d1422",width=1)),
            text=[f"{s:.0f}%" for s in scores], textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#cbd5e1"),
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>",
        ))
        fig2.add_shape(type="line", x0=-0.5, x1=3.5, y0=70, y1=70,
                       line=dict(color="#f87171", width=1.5, dash="dot"))
        fig2.add_annotation(x=3.4, y=76, text="Danger 70%", showarrow=False,
                            font=dict(size=9, color="#f87171", family="JetBrains Mono"))
        fig2.update_layout(
            yaxis=dict(range=[0,115], ticksuffix="%",
                       tickfont=FONT, gridcolor="rgba(255,255,255,0.05)", zeroline=False),
            xaxis=dict(tickfont=dict(family="JetBrains Mono", color="#94a3b8", size=11)),
            margin=dict(t=20,b=10,l=0,r=0), height=230, bargap=0.3,
            paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Chart 3: Horizontal Bar — LTV Tier Comparison ───────────────────────
    with col3:
        st.markdown("""<div class="cpanel">
          <div class="cpanel-title">LTV Tier Comparison</div>
          <div class="cpanel-sub">Approved LTV vs policy tiers by risk band</div>""",
          unsafe_allow_html=True)
        tiers  = ["High Risk (>70%)","Moderate (30-70%)","Low Risk (<30%)"]
        ltvs   = [0, 60, 85]
        bcolors= ["rgba(248,113,113,0.15)","rgba(245,158,11,0.15)","rgba(45,212,191,0.15)"]
        blines = ["#f87171","#f59e0b","#2dd4bf"]
        fig3 = go.Figure()
        for tier, ltv, bc, bl in zip(tiers, ltvs, bcolors, blines):
            fig3.add_trace(go.Bar(
                y=[tier], x=[ltv], orientation="h",
                marker=dict(color=bc, line=dict(color=bl, width=1.5)),
                text=[f"  {ltv}%"] if ltv > 0 else ["  0% — Rejected"],
                textposition="inside" if ltv > 15 else "outside",
                textfont=dict(family="JetBrains Mono", size=11, color=bl),
                hovertemplate=f"<b>{tier}</b><br>Max LTV: {ltv}%<extra></extra>",
                showlegend=False,
            ))
        cur = r["approved_ltv"]
        lc  = "#2dd4bf" if cur > 0 else "#f87171"
        fig3.add_shape(type="line", x0=cur, x1=cur, y0=-0.5, y1=2.5,
                       line=dict(color=lc, width=2.5, dash="dash"))
        fig3.add_annotation(x=cur, y=2.7, text=f"▼ This App: {cur:.0f}%",
                            showarrow=False, font=dict(size=9, color=lc, family="JetBrains Mono"))
        fig3.update_layout(
            xaxis=dict(range=[0,100], ticksuffix="%", tickfont=FONT,
                       gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(tickfont=dict(family="JetBrains Mono", color="#94a3b8", size=9)),
            margin=dict(t=28,b=10,l=0,r=10), height=230, barmode="overlay",
            paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UI — FORECAST CHART
# ══════════════════════════════════════════════════════════════════════════════
def render_forecast(r):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="sl">
    <span class="sl-t">📊 Pillar 3 — 90-Day Price Forecast Curve · Yield Optimizer</span>
    <span class="sl-l"></span></div>""", unsafe_allow_html=True)
    pct = (r['predicted_price'] - r['current_price']) / r['current_price'] * 100
    pc  = "#2dd4bf" if pct >= 0 else "#f87171"
    st.markdown(f"""
    <div class="fpanel">
      <div class="fp-header">
        <div>
          <div class="fp-title">{r['commodity']} · Price Trajectory Simulation</div>
          <div class="fp-sub">Trend + seasonal oscillation · 7-day smoothing overlay</div>
        </div>
        <div class="fp-stats">
          <div class="fps"><div class="fps-val">₹{r['current_price']:,.0f}</div><div class="fps-key">Today</div></div>
          <div class="fps"><div class="fps-val" style="color:#2dd4bf;">₹{r['predicted_price']:,.0f}</div><div class="fps-key">Day 90</div></div>
          <div class="fps"><div class="fps-val" style="color:{pc};">{pct:+.1f}%</div><div class="fps-key">Expected Δ</div></div>
          <div class="fps"><div class="fps-val" style="color:#f59e0b;">Day {r['optimal_sell_day']}</div><div class="fps-key">Optimal Exit</div></div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.line_chart(r["price_curve"], use_container_width=True, height=260,
                  color=["#14b8a6","#f59e0b"])
    st.markdown("""<div class="chart-leg">
      <div class="cl-item"><div class="cl-dot" style="background:#14b8a6;"></div>Forecasted Price</div>
      <div class="cl-item"><div class="cl-dot" style="background:#f59e0b;"></div>7-Day Moving Average</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  UI — DECISION BANNER
# ══════════════════════════════════════════════════════════════════════════════
def render_decision(r):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="sl"><span class="sl-t">🏛️ Final Loan Sanction Decision</span>
    <span class="sl-l"></span></div>""", unsafe_allow_html=True)
    if r["loan_decision"] == "APPROVED":
        st.markdown(f"""
        <div class="db-wrap app">
          <div class="db-inner">
            <div>
              <div class="db-tag">✦ Sanctioned — Credit Approved</div>
              <div class="db-amt">₹{r['sanctioned_amount']/1e5:,.2f}L</div>
              <div class="db-detail">
                Loan sanctioned against <strong style="color:#f0f6ff;">{r['commodity']}</strong> collateral.
                Optimal liquidation window: <strong style="color:#f0f6ff;">Day {r['optimal_sell_day']}</strong>.
                90-day forecast: <strong style="color:#f0f6ff;">₹{r['predicted_price']:,.0f}/Qtl</strong>.
                Spoilage risk within threshold at <strong style="color:#f0f6ff;">{r['spoilage_pct']}%</strong>.
              </div>
            </div>
            <div class="db-kvs">
              <div class="db-kv"><div class="db-kv-val">{r['approved_ltv']:.0f}%</div><div class="db-kv-key">LTV Ratio</div></div>
              <div class="db-kv"><div class="db-kv-val">₹{r['collateral_value']/1e5:,.1f}L</div><div class="db-kv-key">Collateral</div></div>
              <div class="db-kv"><div class="db-kv-val">{r['spoilage_pct']}%</div><div class="db-kv-key">Spoilage Risk</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.success(
            f"✅ **SANCTIONED:** ₹{r['sanctioned_amount']/1e5:,.2f}L at {r['approved_ltv']:.0f}% LTV "
            f"· Optimal sell Day {r['optimal_sell_day']} "
            f"· Forecast ₹{r['predicted_price']:,.0f}/Qtl"
        )
    else:
        st.markdown(f"""
        <div class="db-wrap rej">
          <div class="db-inner">
            <div>
              <div class="db-tag">✕ Rejected — Credit Declined</div>
              <div class="db-amt">₹0</div>
              <div class="db-detail">
                {r['rejection_reason']}<br>
                CropCredit policy requires spoilage risk below <strong style="color:#f0f6ff;">70%</strong>.
                Reduce warehouse humidity and temperature before reapplication.
              </div>
            </div>
            <div class="db-kvs">
              <div class="db-kv"><div class="db-kv-val" style="color:#f87171;">{r['spoilage_pct']}%</div><div class="db-kv-key">Spoilage Risk</div></div>
              <div class="db-kv"><div class="db-kv-val">0%</div><div class="db-kv-key">LTV Ratio</div></div>
              <div class="db-kv"><div class="db-kv-val">₹0</div><div class="db-kv-key">Sanctioned</div></div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.error(f"❌ **REJECTED:** {r['rejection_reason']} Fix warehouse conditions to qualify.")

# ══════════════════════════════════════════════════════════════════════════════
#  UI — REVERSE COLLATERAL CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def render_collateral_calculator(r):
    if r["loan_decision"] != "APPROVED":
        return
    requested  = r["requested_loan"]
    max_sanc   = r["sanctioned_amount"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="sl"><span class="sl-t">🔁 Reverse Collateral Calculator</span>
    <span class="sl-l"></span></div>""", unsafe_allow_html=True)

    if requested > max_sanc:
        shortfall      = requested - max_sanc
        extra_quintals = shortfall / (r["predicted_price"] * (r["approved_ltv"] / 100))
        extra_tonnes   = extra_quintals / 10
        cov_pct        = min(100.0, (max_sanc / requested) * 100)

        st.warning(
            f"⚠️ **PARTIAL APPROVAL** — You requested **₹{requested:,.0f}**, but current "
            f"collateral (**{r['tonnage']:,.0f} MT** of {r['commodity']}) only covers "
            f"**₹{max_sanc:,.0f}** at {r['approved_ltv']:.0f}% LTV."
        )
        st.info(
            f"📦 **Collateral Gap:** Deposit an additional **{extra_tonnes:,.2f} MT** "
            f"({extra_quintals:,.1f} qtl) of {r['commodity']} to bridge the ₹{shortfall:,.0f} gap."
        )
        st.markdown(f"""
        <div class="calc-card">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:0.68rem;font-weight:700;letter-spacing:.09em;
              text-transform:uppercase;color:var(--t4);">Collateral Coverage</span>
            <span style="font-family:'JetBrains Mono';font-size:0.8rem;color:var(--amb4);">{cov_pct:.1f}% covered</span>
          </div>
          <div class="calc-bar-track">
            <div style="width:{cov_pct:.1f}%;background:linear-gradient(90deg,var(--amb3),var(--amb4));
              height:100%;border-radius:6px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:10px;">
            <div><div style="font-family:'JetBrains Mono';font-size:0.87rem;color:var(--teal3);">₹{max_sanc:,.0f}</div>
              <div style="font-size:0.6rem;color:var(--t5);text-transform:uppercase;letter-spacing:.07em;">Max Covered</div></div>
            <div style="text-align:right;"><div style="font-family:'JetBrains Mono';font-size:0.87rem;color:var(--red3);">₹{shortfall:,.0f}</div>
              <div style="font-size:0.6rem;color:var(--t5);text-transform:uppercase;letter-spacing:.07em;">Shortfall</div></div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        req_qtl    = requested / (r["predicted_price"] * (r["approved_ltv"] / 100))
        req_tonnes = req_qtl / 10
        free_tonnes= r["tonnage"] - req_tonnes
        pledge_pct = (req_tonnes / r["tonnage"]) * 100

        st.success(
            f"✅ **FULLY APPROVED** — Requested **₹{requested:,.0f}** is within the "
            f"maximum sanctionable limit of **₹{max_sanc:,.0f}**."
        )
        st.info(
            f"🧮 **Smart Pledge** — Pledge only **{req_tonnes:,.2f} MT** ({req_qtl:,.1f} qtl) "
            f"of your **{r['tonnage']:,.0f} MT** inventory.\n\n"
            f"🆓 **Free to sell {free_tonnes:,.2f} MT** — aim for Day {r['optimal_sell_day']} "
            f"at peak ₹{r['predicted_price']:,.0f}/Qtl."
        )
        st.markdown(f"""
        <div class="calc-card green">
          <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="font-size:0.68rem;font-weight:700;letter-spacing:.09em;
              text-transform:uppercase;color:var(--t4);">Inventory Allocation</span>
            <span style="font-family:'JetBrains Mono';font-size:0.8rem;color:var(--teal3);">Only {pledge_pct:.1f}% pledged</span>
          </div>
          <div class="calc-bar-track" style="display:flex;">
            <div style="width:{pledge_pct:.1f}%;background:linear-gradient(90deg,var(--amb3),var(--amb4));height:100%;"></div>
            <div style="flex:1;background:linear-gradient(90deg,var(--teal1),var(--teal2));height:100%;border-radius:0 6px 6px 0;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:10px;">
            <div><div style="font-family:'JetBrains Mono';font-size:0.87rem;color:var(--amb4);">{req_tonnes:,.2f} MT</div>
              <div style="font-size:0.6rem;color:var(--t5);text-transform:uppercase;letter-spacing:.07em;">Pledged</div></div>
            <div style="text-align:right;"><div style="font-family:'JetBrains Mono';font-size:0.87rem;color:var(--teal3);">{free_tonnes:,.2f} MT</div>
              <div style="font-size:0.6rem;color:var(--t5);text-transform:uppercase;letter-spacing:.07em;">Free to Sell</div></div>
          </div>
        </div>""", unsafe_allow_html=True)



# ══════════════════════════════════════════════════════════════════════════════
#  UI — PLACEHOLDER (before first evaluation)
# ══════════════════════════════════════════════════════════════════════════════
def render_placeholder():
    render_pillars()
    st.markdown("""
    <div class="ph-wrap">
      <div class="ph-icon">⚡</div>
      <div class="ph-title">Ready for Evaluation</div>
      <div class="ph-sub">Configure crop &amp; warehouse parameters in the left panel,
        then click <strong>Evaluate Risk &amp; Sanction Loan</strong>
        to run the full 3-Pillar AI engine.</div>
      <div class="ph-cta">← Configure inputs · Click CTA to evaluate</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def render_results(r):
    render_kpis(r)
    render_financials_and_gauge(r)
    render_analytics_charts(r)
    render_forecast(r)
    render_decision(r)
    render_collateral_calculator(r)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    render_header()
    try:
        ep = os.path.join(MODELS_DIR, "label_encoder.pkl")
        if not os.path.exists(ep):
            raise FileNotFoundError(f"Not found: {ep}\nRun: python train_models.py")
        with open(ep, "rb") as f: le = pickle.load(f)
    except FileNotFoundError as e:
        st.error(f"**Models/Encoder not found.**\n\n```\n{e}\n```\nRun: `python train_models.py`")
        st.stop()
    except Exception as e:
        st.error(f"**Model loading error:** {e}"); st.stop()

    inp = render_sidebar(None, None, le)

    if inp["evaluate"]:
        with st.spinner("⚙️ Running 3-Pillar AI Engine …"):
            try:
                fm, pm, le = load_commodity_models(inp["commodity"])
                result = run_inference(
                    fm, pm, le,
                    inp["commodity"],      inp["tonnage"],
                    inp["market_arrivals"],inp["current_price"],
                    inp["rainfall_deficit"],inp["warehouse_temp"],
                    inp["humidity"],       inp["moisture_content"],
                )
                result["requested_loan"] = inp["requested_loan"]
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"**Inference error:** {e}")
                logger.error("Inference failed", exc_info=True); st.stop()
        render_results(result)

    elif "last_result" in st.session_state:
        render_results(st.session_state["last_result"])
    else:
        render_placeholder()

if __name__ == "__main__":
    main()
