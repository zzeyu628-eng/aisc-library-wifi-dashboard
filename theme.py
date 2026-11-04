"""Visual system for the Library Wi-Fi dashboard (taste-skill + impeccable pass, 2026-11-03).

Style dealt by design-taste-frontend `draw.mjs --app`: *achromatic-ledger* (seed 260721935).
The move: no colour in the chrome; figures set in ink; one chart hue (#1d5fbf) marks the main
data element; the only other hue is a small change signal.
Face chosen: a blue-grey "meter" ground (#f2f4f7) instead of #f8f8f8; cards stay white.
Type: the sheet dealt Geist; impeccable's detector flags Geist as an overused face, so IBM Plex Sans
(tabular figures, institutional/instrument feel) replaces it. Recorded in docs/TASTE-DASHBOARD-CHANGELOG.md.

Everything visual lives here so app.py keeps its data logic untouched.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------- tokens
GROUND = "#f2f4f7"
SURFACE = "#ffffff"
BORDER = "#e3e6eb"
RULE = "#eef0f3"
CONTROL = "#d4d4d4"
INK = "#0a0a0a"
MUTED = "#5f6368"
FAINT = "#8a8f98"
HUE = "#1d5fbf"          # the one chart hue: the main data element
HUE_TINT = "#e8effa"
POSITIVE = "#2e7d32"
NEGATIVE = "#c4242a"
NEUTRAL_BAR = "#c9ced6"  # non-selected bars on the predictive weekday chart

FONT = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif'

# Categorical hues follow the ENTITY, never its rank (dataviz reference palette, fixed order).
GROUP_COLORS = {
    "UG": "#2a78d6",
    "TPG": "#eb6834",
    "RPG": "#1baf7a",
    "Staff": "#eda100",
    "Alumni": "#e87ba4",
    "Non-HKUST": "#008300",
    "Others(HKUST)": "#6250d6",
}
BUCKET_COLORS = {"UG": GROUP_COLORS["UG"], "PG (RPG+TPG)": GROUP_COLORS["TPG"], "Staff": GROUP_COLORS["Staff"]}

# Sequential blue ramp (light -> dark) for magnitude.
SEQ_BLUE = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# ------------------------------------------------------- bilingual glossary
GROUP_ZH = {
    "UG": "本科生",
    "TPG": "授课型研究生",
    "RPG": "研究型研究生",
    "Staff": "教职员",
    "Alumni": "校友",
    "Non-HKUST": "校外访客",
    "Others(HKUST)": "其他校内人员",
    "PG (RPG+TPG)": "研究生",
}
DOW_EN = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DOW_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
MONTH_EN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def group_label(g: str) -> str:
    zh = GROUP_ZH.get(g)
    return f"{g} · {zh}" if zh else str(g)


def dow_label(i: int) -> str:
    return f"{DOW_EN[i]} {DOW_ZH[i]}"


def month_label(m: int) -> str:
    return f"{MONTH_EN[m - 1]} {m}月"


# ---------------------------------------------------------------- plotly
_axis = dict(
    showgrid=False,
    zeroline=False,
    showline=True,
    linecolor=CONTROL,
    linewidth=1,
    ticks="",
    tickfont=dict(size=11, color=MUTED),
    title=dict(font=dict(size=12, color=MUTED), standoff=10),
    automargin=True,
)

pio.templates["ledger"] = go.layout.Template(
    layout=dict(
        font=dict(family=FONT, size=12, color=INK),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        colorway=list(GROUP_COLORS.values()),
        margin=dict(l=10, r=14, t=52, b=40),
        title=dict(x=0, xanchor="left", xref="container", y=1, yanchor="top", yref="container",
                   pad=dict(t=14, l=10), font=dict(size=15, color=INK)),
        xaxis=_axis,
        yaxis={**_axis, "showgrid": True, "gridcolor": RULE, "gridwidth": 1, "showline": False, "nticks": 6},
        legend=dict(
            orientation="h", x=0, xanchor="left", y=0, yanchor="bottom", yref="container",
            font=dict(size=12, color=INK), title=dict(text=""), bgcolor="rgba(0,0,0,0)",
            itemsizing="constant",
        ),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=CONTROL, font=dict(family=FONT, size=12, color=INK)),
        hovermode="closest",
        bargap=0.28,
        bargroupgap=0.08,
        coloraxis=dict(colorbar=dict(outlinewidth=0, thickness=10, tickfont=dict(size=11, color=MUTED))),
    )
)
TEMPLATE = "ledger"
PLOTLY_CONFIG = {"displayModeBar": False, "displaylogo": False}


def finish(fig: go.Figure, title: str | None = None, subtitle: str | None = None, height: int = 360) -> go.Figure:
    """Apply the ledger template, a bilingual title (+ grey sub-line) and a fixed height."""
    fig.update_layout(template=TEMPLATE, height=height)
    if title:
        text = f"<b style='font-weight:500'>{title}</b>"
        if subtitle:
            text += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
        fig.update_layout(title=dict(text=text))
        fig.update_layout(margin=dict(t=76 if subtitle else 52))
    return fig


# ---------------------------------------------------------------- CSS
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {{
  --ground:{GROUND}; --surface:{SURFACE}; --border:{BORDER}; --rule:{RULE};
  --ink:{INK}; --muted:{MUTED}; --faint:{FAINT}; --hue:{HUE};
  --positive:{POSITIVE}; --negative:{NEGATIVE};
  --radius-card:8px; --radius-control:6px;
}}

html, body, .stApp, [data-testid="stAppViewContainer"], .stMarkdown, .stCaption,
[data-testid="stSidebar"], button, input, textarea, select {{
  font-family: {FONT} !important;
}}
.stApp, [data-testid="stAppViewContainer"] {{ background: var(--ground); color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container, [data-testid="stMainBlockContainer"] {{ padding-top: 2.25rem; padding-bottom: 4rem; max-width: 1360px; }}

/* sidebar: on the ground, one hairline, no colour */
[data-testid="stSidebar"] {{ background: var(--ground); border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
  font-size: 13px !important; font-weight: 500 !important; color: var(--muted) !important;
  letter-spacing: 0; margin: 1.25rem 0 .25rem 0; padding: 0;
}}
[data-testid="stSidebar"] label p {{ font-size: 13px; color: var(--ink); }}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {{ background: var(--surface); border-color: var(--border); border-radius: var(--radius-control); }}
[data-testid="stSidebar"] [data-baseweb="tag"] {{ background: #e6e9ee !important; color: var(--ink) !important; border-radius: 4px; }}
[data-testid="stSidebar"] [data-baseweb="tag"] span {{ color: var(--ink) !important; }}

/* type */
h1 {{ font-size: 26px !important; font-weight: 600 !important; letter-spacing: -0.02em; line-height: 1.2 !important; padding: 0 !important; color: var(--ink); }}
h2, h3 {{ font-size: 17px !important; font-weight: 500 !important; letter-spacing: -0.01em; color: var(--ink); padding-top: .25rem !important; }}
p, li {{ font-size: 14px; line-height: 1.55; }}
[data-testid="stCaptionContainer"], .stCaption p {{ color: var(--muted) !important; font-size: 13px !important; line-height: 1.5; }}
[data-testid="stHeadingWithActionElements"] a, h1 a, h2 a, h3 a {{ display: none !important; }}

/* page lead */
.lede {{ font-size: 15px; color: var(--muted); margin: .35rem 0 0 0; max-width: 78ch; }}
.finding {{ font-size: 15px; color: var(--ink); margin: .9rem 0 0 0; max-width: 92ch; line-height: 1.55; }}
.finding b {{ font-weight: 600; }}
.section-gap {{ height: 1.75rem; }}

/* KPI strip: type, not tiles */
.kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px 24px; margin: 1.4rem 0 .4rem 0; }}
.kpi .l {{ font-size: 12px; color: var(--muted); }}
.kpi .v {{ font-size: 26px; font-weight: 500; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; margin-top: 2px; line-height: 1.15; }}
.kpi .s {{ font-size: 12px; color: var(--faint); font-variant-numeric: tabular-nums; }}
@media (max-width: 900px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}

/* cards = keyed bordered containers */
[class*="st-key-card"] {{
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius-card) !important; box-shadow: none !important;
}}
[class*="st-key-card"] [data-testid="stVerticalBlockBorderWrapper"] {{ border: 0 !important; }}

/* insight cards */
.ins-fig {{ font-size: 30px; font-weight: 500; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1.1; color: var(--ink); }}
.ins-figcap {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
.ins-title {{ font-size: 15px; font-weight: 500; margin: .9rem 0 .35rem 0; line-height: 1.35; }}
.ins-title .zh {{ color: var(--muted); font-weight: 400; }}
.ins-en {{ font-size: 13.5px; line-height: 1.55; color: var(--ink); margin: 0 0 .5rem 0; }}
.ins-zh {{ font-size: 13.5px; line-height: 1.65; color: #33363b; margin: 0; }}

.dq {{ font-size: 12.5px; color: var(--muted); line-height: 1.55; margin: .5rem 0 0 0; max-width: 110ch; }}

/* predictive panel */
.pred-scope {{ font-size: 15px; margin: .1rem 0 .9rem 0; color: var(--ink); }}
.pred-scope b {{ font-weight: 600; }}
.pred-scope .sep {{ color: var(--faint); padding: 0 .35rem; }}
[data-testid="stMetric"] {{ padding: 0; }}
[data-testid="stMetricLabel"] p {{ font-size: 12px !important; color: var(--muted) !important; }}
[data-testid="stMetricValue"] {{ font-size: 30px !important; font-weight: 500 !important; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }}
.st-key-kpi_pred [data-testid="stMetricValue"] {{ color: var(--hue) !important; }}
[data-testid="stMetricDelta"] {{ font-size: 12.5px !important; font-variant-numeric: tabular-nums; }}
.honest {{ font-size: 13px; color: #33363b; line-height: 1.6; margin: .75rem 0 0 0; max-width: 120ch; }}
.honest b {{ font-weight: 600; }}

.foot {{ font-size: 12px; color: var(--muted); margin-top: 2.5rem; }}
.foot a {{ color: var(--ink); }}

/* focus stays visible */
:focus-visible {{ outline: 2px solid var(--hue) !important; outline-offset: 2px; }}
</style>
"""
