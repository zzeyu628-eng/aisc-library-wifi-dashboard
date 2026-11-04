"""AISC1000B GenZ glasses — HKUST Library Wi-Fi Predictive Advanced dashboard."""
from __future__ import annotations

from pathlib import Path

import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `theme` imports under export_metrics too
import theme as T  # noqa: E402  (visual system only: tokens, CSS, Plotly template, bilingual labels)

ROOT = Path(__file__).resolve().parent  # click-open deploy pack: data/ beside app.py
DATA_PATH = ROOT / "data" / "wifi_clean.parquet"
DOW_LABELS = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
DOI = "doi:10.14711/dataset/RQXRQB"
SOURCE_URL = "https://dataspace.hkust.edu.hk/dataset.xhtml?persistentId=doi:10.14711/dataset/RQXRQB"

try:
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAS_HGB = True
except Exception:  # pragma: no cover
    from sklearn.linear_model import Ridge

    HAS_HGB = False


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error(f"Missing {DATA_PATH}. Run: python scripts/merge_wifi.py · 缺少数据文件，请先运行 python scripts/merge_wifi.py")
        st.stop()
    df = pd.read_parquet(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def filter_df(df: pd.DataFrame, date_range, floors, groups) -> pd.DataFrame:
    out = df
    if date_range and len(date_range) == 2:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        out = out[(out["Date"] >= start) & (out["Date"] <= end)]
    if floors:
        out = out[out["Floor"].isin(floors)]
    if groups:
        out = out[out["User-Group"].isin(groups)]
    return out


def _encode_features(df: pd.DataFrame, floor_cats, group_cats) -> pd.DataFrame:
    """Build model matrix: floor/group one-hots + dow, month, is_weekend."""
    x = pd.DataFrame(index=df.index)
    for f in floor_cats:
        x[f"floor_{f}"] = (df["Floor"] == f).astype(np.float64)
    for g in group_cats:
        safe = str(g).replace(" ", "_").replace("(", "").replace(")", "")
        x[f"group_{safe}"] = (df["User-Group"] == g).astype(np.float64)
    x["dow"] = df["dow"].astype(np.float64)
    x["month"] = df["month"].astype(np.float64)
    x["is_weekend"] = df["is_weekend"].astype(np.float64)
    return x


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Time-aware 80/20 split; HGB (or Ridge) + same-dow baseline MAE."""
    work = df.sort_values("Date").reset_index(drop=True)
    dates = sorted(work["Date"].unique())
    cut = dates[int(len(dates) * 0.8)]
    train = work[work["Date"] < cut]
    hold = work[work["Date"] >= cut]

    # one-hot schema from the TRAIN window only (review 2026-11-03: was full timeline;
    # numerically identical here because every floor/group appears daily, but cleaner)
    floor_cats = sorted(train["Floor"].unique().tolist())
    group_cats = sorted(train["User-Group"].unique().tolist())

    X_train = _encode_features(train, floor_cats, group_cats)
    y_train = train["User-Count"].astype(np.float64)
    X_hold = _encode_features(hold, floor_cats, group_cats)
    y_hold = hold["User-Count"].astype(np.float64)

    if HAS_HGB:
        model = HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.08,
            max_iter=200,
            random_state=42,
        )
        model_name = "HistGradientBoostingRegressor"
    else:
        model = Ridge(alpha=1.0)
        model_name = "Ridge"

    model.fit(X_train, y_train)
    yhat = model.predict(X_hold)
    mae_model = float(np.mean(np.abs(y_hold - yhat)))

    # same-dow baseline keyed by Floor × User-Group × dow
    means = (
        train.groupby(["Floor", "User-Group", "dow"])["User-Count"]
        .mean()
        .rename("yhat_base")
    )
    hold_b = hold.merge(means, on=["Floor", "User-Group", "dow"], how="left")
    # fallback: Floor×Group mean, then global train mean
    fg = train.groupby(["Floor", "User-Group"])["User-Count"].mean().rename("yhat_fg")
    hold_b = hold_b.merge(fg, on=["Floor", "User-Group"], how="left")
    hold_b["yhat_base"] = hold_b["yhat_base"].fillna(hold_b["yhat_fg"]).fillna(y_train.mean())
    mae_base = float(np.mean(np.abs(hold_b["User-Count"] - hold_b["yhat_base"])))

    # Opus final review: two STRONGER simple baselines, both built only from data before `cut`.
    # (a) recency: Floor×Group×dow mean over the last 28 days of train with non-zero building
    #     total (train ends after an all-zero gap 2021-02-12..03-02); (b) seasonal: 2019
    #     Floor×Group×month×dow mean (last full pre-pandemic year).
    def _keyed_mae(src: pd.DataFrame, keys) -> float:
        m = src.groupby(keys)["User-Count"].mean().rename("b")
        hb = hold.merge(m, on=keys, how="left").merge(fg, on=["Floor", "User-Group"], how="left")
        pred = hb["b"].fillna(hb["yhat_fg"]).fillna(y_train.mean())
        return float(np.mean(np.abs(hb["User-Count"] - pred)))

    day_tot = train.groupby("Date")["User-Count"].transform("sum")
    nz = train[day_tot > 0]
    recent = nz[nz["Date"] >= nz["Date"].max() - pd.Timedelta(days=27)]
    mae_recent = _keyed_mae(recent, ["Floor", "User-Group", "dow"])
    mae_season = _keyed_mae(train[train["Date"].dt.year == 2019], ["Floor", "User-Group", "month", "dow"])

    meta = {
        "cut": str(pd.Timestamp(cut).date()),
        "n_train": int(len(train)),
        "n_hold": int(len(hold)),
        "mae_model": mae_model,
        "mae_baseline": mae_base,
        "mae_rel_improvement_pct": 100.0 * (mae_base - mae_model) / mae_base,
        "mae_baseline_recent28d_same_dow": mae_recent,
        "mae_baseline_2019_same_month_dow": mae_season,
        "train_mean": float(y_train.mean()),
        "hold_mean": float(y_hold.mean()),
        "model_name": model_name,
        "floor_cats": floor_cats,
        "group_cats": group_cats,
    }
    return model, meta, means


def predict_user_count(model, meta, floor: str, group: str, dow: int, month: int) -> float:
    row = pd.DataFrame(
        [
            {
                "Floor": floor,
                "User-Group": group,
                "dow": int(dow),
                "month": int(month),
                "is_weekend": bool(int(dow) in (5, 6)),
            }
        ]
    )
    X = _encode_features(row, meta["floor_cats"], meta["group_cats"])
    return float(model.predict(X)[0])


def compute_insights(df: pd.DataFrame) -> list[dict]:
    """≥3 auto-calculated non-trivial findings from the cleaned table.

    Each dict carries `values` (raw numbers) so metrics.json can be regenerated
    from exactly the same code that renders the dashboard text.
    """
    insights: list[dict] = []
    U = "User-Count"

    # 1) Teaching-term vs summer-break weekday load (Opus final review 2026-11-03).
    #    Data-quality fix: on Mon 2018-11-05 the daily building total doubles overnight
    #    (~5.3k -> ~10.8k weekday users) while WiFi-Conn per user jumps 3.0 -> 5.3 and
    #    Duration per user halves: a counting-definition change, not behaviour. Nov-Dec 2019
    #    is the campus-disruption period. So each year is compared WITHIN one counting
    #    regime: Feb-May + Sep-Oct weekdays vs Jul-Aug weekdays of the SAME year
    #    (2018: all before 2018-11-05; 2019: all after it and before the disruption).
    #    Month sets are calendar PROXIES, not the official HKUST calendar.
    wd_all = df[~df["is_weekend"]]
    term_m, summer_m = [2, 3, 4, 5, 9, 10], [7, 8]

    def _ratio(sub: pd.DataFrame, a, b) -> float:
        num = sub.loc[sub["month"].isin(a), U].mean()
        den = sub.loc[sub["month"].isin(b), U].mean()
        return float(num / den) if den and den > 0 else float("nan")

    w18 = wd_all[wd_all["year"] == 2018]
    w19 = wd_all[wd_all["year"] == 2019]
    if len(w18) and len(w19):
        r18 = _ratio(w18, term_m, summer_m)
        r19 = _ratio(w19, term_m, summer_m)
        by_floor = {
            fl: _ratio(w19[w19["Floor"] == fl], term_m, summer_m) for fl in sorted(df["Floor"].unique())
        }
        top = max(by_floor, key=by_floor.get)
        lo = min(by_floor, key=by_floor.get)
        may_vs_marapr = _ratio(w19, [5], [3, 4])
        insights.append(
            {
                "title_en": "Teaching-term vs summer-break weekday load",
                "title_zh": "学期 vs 暑假：工作日人数倍率",
                "body_en": (
                    f"Weekday mean User-Count, teaching months (Feb–May, Sep–Oct) vs Jul–Aug of the same year: "
                    f"{r18:.2f}× in 2018 and {r19:.2f}× in 2019. By floor (2019) it ranges from "
                    f"{by_floor[lo]:.2f}× on {lo} to {by_floor[top]:.2f}× on {top}. "
                    f"No exam-month surge: May 2019 weekdays were {may_vs_marapr:.2f}× Mar–Apr 2019; the busiest "
                    f"weeks fall inside teaching weeks, not exam periods. Years are compared separately because the counting "
                    f"definition changes on 2018-11-05; month sets are proxies, not the official calendar."
                ),
                "body_zh": (
                    f"工作日均值，同年教学月（2–5、9–10 月）对 7–8 月：2018 年 {r18:.2f} 倍，2019 年 {r19:.2f} 倍；"
                    f"按楼层（2019）由 {lo} 层 {by_floor[lo]:.2f}× 到 {top} 层 {by_floor[top]:.2f}×。"
                    f"无考试月激增：2019 年 5 月工作日仅为 3–4 月的 {may_vs_marapr:.2f} 倍，最忙的周落在授课周而非考试期。"
                    f"因 2018-11-05 起计数口径变化，各年分开比较；月份仅为代理，非官方校历。"
                ),
                "values": {
                    "term_vs_summer_weekday_2018": round(r18, 3),
                    "term_vs_summer_weekday_2019": round(r19, 3),
                    "term_vs_summer_weekday_by_floor_2019": {k: round(v, 3) for k, v in by_floor.items()},
                    "term_vs_summer_top_floor": top,
                    "term_vs_summer_low_floor": lo,
                    "exam_proxy_may_vs_marapr_weekday_2019": round(may_vs_marapr, 3),
                },
            }
        )

    # 2) 2020 vs 2018–19 same calendar months
    base = (
        df[df["year"].isin([2018, 2019])]
        .groupby(df.loc[df["year"].isin([2018, 2019]), "month"])[U]
        .mean()
    )
    y20 = df[df["year"] == 2020].groupby("month")[U].mean()
    common = sorted(set(base.index) & set(y20.index))
    if common:
        drop = (y20.loc[common] - base.loc[common]) / base.loc[common] * 100
        worst_m = int(drop.idxmin())
        tot20 = df.loc[df["year"] == 2020, U].sum()
        tot_base = df.loc[df["year"].isin([2018, 2019]), U].sum() / 2.0
        total_drop = float((tot20 / tot_base - 1) * 100) if tot_base > 0 else float("nan")
        insights.append(
            {
                "title_en": "2020 vs 2018–19 same-month drop",
                "title_zh": "2020 相对 2018–19 同月跌幅",
                "body_en": (
                    f"Average of the 12 same-calendar-month changes in mean User-Count: "
                    f"{drop.mean():.1f}% (total 2020 volume {total_drop:.1f}% vs the 2018–19 yearly average; "
                    f"worst month {worst_m}: {drop.loc[worst_m]:.1f}%; median month {drop.median():.1f}%; "
                    f"{int((drop < -50).sum())} of {len(drop)} months below −50%). "
                    f"Timing is consistent with the campus-closure / remote-teaching period; "
                    f"the 2018–19 baseline also includes late-2019 disruption and a counting change on "
                    f"2018-11-05, so the size is indicative (conservative), not exact."
                ),
                "body_zh": (
                    f"12 个同公历月份变化的平均值：{drop.mean():.1f}%"
                    f"（2020 全年总量较 2018–19 年均 {total_drop:.1f}%；最差月份 {worst_m} 月：{drop.loc[worst_m]:.1f}%；"
                    f"月度中位数 {drop.median():.1f}%；{int((drop < -50).sum())}/{len(drop)} 个月跌超 50%）。"
                    f"时间上与校园停摆/远程教学时段一致；2018–19 基线含 2019 年末扰动及 2018-11-05 计数口径变化，幅度仅供参考（偏保守）。"
                ),
                "values": {
                    "y2020_vs_1819_mean_drop_pct": round(float(drop.mean()), 2),
                    "y2020_vs_1819_total_drop_pct": round(total_drop, 2),
                    "y2020_worst_month": worst_m,
                    "y2020_worst_month_drop_pct": round(float(drop.loc[worst_m]), 2),
                    "y2020_vs_1819_median_drop_pct": round(float(drop.median()), 2),
                    "y2020_months_below_minus50": int((drop < -50).sum()),
                },
            }
        )

    # 3) UG vs RPG: weekend retention (stable) and peak floor (period-dependent)
    def peak_floor(g: str, years=None):
        sub = df[df["User-Group"] == g]
        if years is not None:
            sub = sub[sub["year"].isin(years)]
        s = sub.groupby("Floor")[U].mean()
        return (s.idxmax(), float(s.max())) if len(s) else (None, float("nan"))

    def we_ratio(g: str, year=None) -> float:
        sub = df[df["User-Group"] == g]
        if year is not None:
            sub = sub[sub["year"] == year]
        wd = sub.loc[~sub["is_weekend"], U].mean()
        we = sub.loc[sub["is_weekend"], U].mean()
        return float(we / wd) if wd and wd > 0 else float("nan")

    ug_peak, ug_peak_v = peak_floor("UG")
    rpg_peak, rpg_peak_v = peak_floor("RPG")
    if ug_peak and rpg_peak:
        ug_peak_pre, _ = peak_floor("UG", [2018, 2019])
        rpg_peak_pre, _ = peak_floor("RPG", [2018, 2019])
        ug_we, rpg_we = we_ratio("UG"), we_ratio("RPG")
        yrs = sorted(df["year"].unique())
        ug_y = {int(y): we_ratio("UG", y) for y in yrs}
        rpg_y = {int(y): we_ratio("RPG", y) for y in yrs}
        rpg_above_every_year = all(rpg_y[y] > ug_y[y] for y in ug_y)
        insights.append(
            {
                "title_en": "UG vs RPG: weekend retention gap (and peak floors)",
                "title_zh": "本科 vs 研究型研究生：周末留存差异（及峰值楼层）",
                "body_en": (
                    f"Weekend/weekday ratio over 2018–21: UG {ug_we:.1%} vs RPG {rpg_we:.1%}"
                    + (" — RPG is higher in every year" if rpg_above_every_year else "")
                    + f" (UG {min(ug_y.values()):.0%}–{max(ug_y.values()):.0%}, "
                    f"RPG {min(rpg_y.values()):.0%}–{max(rpg_y.values()):.0%} across years). "
                    f"Highest-mean floor over all years: UG {ug_peak} ({ug_peak_v:.1f}), "
                    f"RPG {rpg_peak} ({rpg_peak_v:.1f}); but in 2018–19 alone both peak on "
                    f"{ug_peak_pre}/{rpg_peak_pre}, so the floor gap is period-dependent."
                ),
                "body_zh": (
                    f"2018–21 周末/工作日比：UG {ug_we:.1%} vs RPG {rpg_we:.1%}"
                    + ("（逐年 RPG 都更高）" if rpg_above_every_year else "")
                    + f"。全期均值最高楼层：UG {ug_peak}（{ug_peak_v:.1f}）、RPG {rpg_peak}（{rpg_peak_v:.1f}）；"
                    f"但仅看 2018–19，两者峰值分别为 {ug_peak_pre}/{rpg_peak_pre}，楼层差异随时期而变。"
                ),
                "values": {
                    "ug_peak_floor": ug_peak,
                    "rpg_peak_floor": rpg_peak,
                    "ug_peak_floor_1819": ug_peak_pre,
                    "rpg_peak_floor_1819": rpg_peak_pre,
                    "ug_weekend_retention": round(ug_we, 4),
                    "rpg_weekend_retention": round(rpg_we, 4),
                    "ug_weekend_retention_by_year": {str(k): round(v, 3) for k, v in ug_y.items()},
                    "rpg_weekend_retention_by_year": {str(k): round(v, 3) for k, v in rpg_y.items()},
                },
            }
        )

    # 4) Guest (Non-HKUST) share: rises when members leave (summer), annual share is volatile.
    def _nh_share(sub: pd.DataFrame) -> float:
        tot = sub[U].sum()
        return float(100.0 * sub.loc[sub["User-Group"] == "Non-HKUST", U].sum() / tot) if tot > 0 else float("nan")

    shares = {int(y): _nh_share(df[df["year"] == int(y)]) for y in sorted(df["year"].unique())}
    shares = {k: v for k, v in shares.items() if pd.notna(v)}
    season = {}
    for y in (2018, 2019):
        s = wd_all[wd_all["year"] == y]
        if len(s):
            season[y] = (_nh_share(s[s["month"].isin(term_m)]), _nh_share(s[s["month"].isin(summer_m)]))
    if len(shares) >= 2 and len(season) == 2:
        y1 = max(shares)
        last = df[df["year"] == y1]
        nh_m = last[last["User-Group"] == "Non-HKUST"].groupby("month")[U].sum()
        pk_m = int(nh_m.idxmax())
        pk_frac = float(nh_m.max() / nh_m.sum() * 100) if nh_m.sum() > 0 else float("nan")
        share_ex = _nh_share(last[last["month"] != pk_m])
        share_q4 = _nh_share(last[last["month"] >= 10])
        (t18, s18), (t19, s19) = season[2018], season[2019]
        insights.append(
            {
                "title_en": "Guests (Non-HKUST) fill a bigger share when students are away",
                "title_zh": "学生离校时访客（Non-HKUST）占比上升",
                "body_en": (
                    f"Weekday Non-HKUST share of User-Count, teaching months vs Jul–Aug: "
                    f"{t18:.2f}% → {s18:.2f}% (2018) and {t19:.2f}% → {s19:.2f}% (2019). Guest use also falls in "
                    f"summer, but far less than member use, so the library's summer user mix is more external. "
                    f"Annual shares (" + ", ".join(f"{y}={v:.2f}%" for y, v in shares.items()) + ") are NOT a trend: "
                    f"month {pk_m} of {y1} alone holds {pk_frac:.0f}% of that year's guest volume (excl. it "
                    f"{share_ex:.2f}%; Oct–Dec {y1} {share_q4:.2f}%), in a period when counting definitions shift."
                ),
                "body_zh": (
                    f"工作日 Non-HKUST 占比，教学月 → 7–8 月：2018 年 {t18:.2f}% → {s18:.2f}%，"
                    f"2019 年 {t19:.2f}% → {s19:.2f}%。暑假访客也减少，但远少于校内用户，所以暑期用户构成更“外部”。"
                    f"年度占比（" + "，".join(f"{y}={v:.2f}%" for y, v in shares.items()) + "）不是趋势："
                    f"{y1} 年 {pk_m} 月一个月占全年访客量 {pk_frac:.0f}%（剔除后 {share_ex:.2f}%；10–12 月 {share_q4:.2f}%），"
                    f"且该时期计数口径在变。"
                ),
                "values": {
                    "non_hkust_weekday_share_term_vs_summer_2018": [round(t18, 3), round(s18, 3)],
                    "non_hkust_weekday_share_term_vs_summer_2019": [round(t19, 3), round(s19, 3)],
                    "non_hkust_share_by_year": {str(k): round(v, 3) for k, v in shares.items()},
                    "non_hkust_last_year_peak_month": pk_m,
                    "non_hkust_last_year_peak_month_volume_pct": round(pk_frac, 1),
                    "non_hkust_last_year_share_excl_peak_month": round(share_ex, 3),
                    "non_hkust_last_year_q4_share": round(share_q4, 3),
                },
            }
        )

    return insights


# ============================================================================
# Presentation layer (taste-skill achromatic-ledger + impeccable polish, 2026-11-03).
# Everything above this line is the reviewed data/model logic and is unchanged;
# everything below only arranges and draws it.
# ============================================================================
import html as _html_mod  # noqa: E402


def predict_grid(model, meta, floor: str, group: str, dows, months) -> np.ndarray:
    """Vectorised `predict_user_count` for many (dow, month) cells of one Floor × User-Group.

    Builds exactly the same feature rows as `predict_user_count`, so each value equals the
    single-cell prediction (asserted in the harness check, docs/taste-pass/).
    """
    rows = pd.DataFrame(
        {
            "Floor": floor,
            "User-Group": group,
            "dow": [int(d) for d in dows],
            "month": [int(m) for m in months],
        }
    )
    rows["is_weekend"] = rows["dow"].isin((5, 6))
    X = _encode_features(rows, meta["floor_cats"], meta["group_cats"])
    return model.predict(X).astype(float)


@st.cache_data
def seasonal_2019_means(df: pd.DataFrame, cut: str) -> pd.Series:
    """Observed 2019 Floor × Group × month × dow mean, train window only (= the seasonal baseline)."""
    tr = df[(df["Date"] < pd.Timestamp(cut)) & (df["Date"].dt.year == 2019)]
    return tr.groupby(["Floor", "User-Group", "month", "dow"])["User-Count"].mean()


def _md(body: str) -> None:
    st.markdown(body, unsafe_allow_html=True)


def _esc(s: str) -> str:
    return _html_mod.escape(str(s), quote=False)


def insight_headline(ins: dict) -> tuple[str, str]:
    """Large figure + its caption for an insight card, read from the insight's own `values`."""
    v = ins.get("values", {})
    if "term_vs_summer_weekday_2018" in v:
        return (
            f"{v['term_vs_summer_weekday_2018']:.2f}× / {v['term_vs_summer_weekday_2019']:.2f}×",
            "Term ÷ summer weekday load, 2018 / 2019 · 学期 ÷ 暑假 工作日人数（2018 / 2019）",
        )
    if "y2020_vs_1819_mean_drop_pct" in v:
        return (
            f"{v['y2020_vs_1819_mean_drop_pct']:.1f}%",
            "2020 vs 2018–19, mean same-month change · 2020 相对 2018–19 同月平均变化",
        )
    if "ug_weekend_retention" in v:
        return (
            f"{v['ug_weekend_retention']:.0%} vs {v['rpg_weekend_retention']:.0%}",
            "Weekend ÷ weekday, UG vs RPG · 周末 ÷ 工作日：本科生 vs 研究型研究生",
        )
    if "non_hkust_weekday_share_term_vs_summer_2019" in v:
        t, s = v["non_hkust_weekday_share_term_vs_summer_2019"]
        return f"{t:.2f}% → {s:.2f}%", "Non-HKUST guest weekday share, term → summer 2019 · 2019 校外访客工作日占比：学期 → 暑假"
    return "", ""


def _kpi_strip(items: list[tuple[str, str, str]]) -> None:
    cells = "".join(
        f"<div class='kpi'><div class='l'>{_esc(l)}</div><div class='v'>{_esc(v)}</div>"
        f"<div class='s'>{_esc(s)}</div></div>"
        for l, v, s in items
    )
    _md(f"<div class='kpis'>{cells}</div>")


def _plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=T.PLOTLY_CONFIG, key=key)


# ---------------------------------------------------------------- charts
def fig_heatmap(fdf: pd.DataFrame) -> go.Figure:
    heat = fdf.groupby(["Floor", "dow"], as_index=False)["User-Count"].mean()
    pivot = heat.pivot(index="Floor", columns="dow", values="User-Count").reindex(columns=range(7))
    pivot = pivot.dropna(axis=1, how="all")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[T.dow_label(int(d)).replace(" ", "<br>") for d in pivot.columns],
            y=[f"{f}" for f in pivot.index],
            colorscale=[[i / (len(T.SEQ_BLUE) - 1), c] for i, c in enumerate(T.SEQ_BLUE)],
            texttemplate="%{z:.0f}",
            textfont=dict(size=12),
            xgap=2,
            ygap=2,
            colorbar=dict(title=dict(text="Mean users<br>平均人数", font=dict(size=11, color=T.MUTED)), thickness=10),
            hovertemplate="Floor 楼层 %{y} · %{x}<br>Mean User-Count 平均人数 %{z:.1f}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", showline=False, title_text="Floor · 楼层", showgrid=False)
    fig.update_xaxes(showline=False)
    return T.finish(fig, "(a) Floor × weekday heatmap (mean User-Count)", "楼层 × 星期 热力图（平均人数）", height=380)


def fig_buckets(fdf: pd.DataFrame) -> go.Figure | None:
    focus = {"UG", "RPG", "TPG", "Staff"}
    bar_df = fdf[fdf["User-Group"].isin(focus)].copy()
    if bar_df.empty:
        return None
    bar_df["bucket"] = bar_df["User-Group"].map(
        lambda g: "UG" if g == "UG" else ("Staff" if g == "Staff" else "PG (RPG+TPG)")
    )
    bar = bar_df.groupby(["Floor", "bucket"], as_index=False)["User-Count"].mean()
    fig = go.Figure()
    for b in ["UG", "PG (RPG+TPG)", "Staff"]:
        sub = bar[bar["bucket"] == b]
        if sub.empty:
            continue
        fig.add_bar(
            x=sub["Floor"], y=sub["User-Count"], name=T.group_label(b), marker_color=T.BUCKET_COLORS[b],
            hovertemplate=f"{T.group_label(b)} · Floor 楼层 %{{x}}<br>Mean User-Count 平均人数 %{{y:.1f}}<extra></extra>",
        )
    fig.update_layout(barmode="group", barcornerradius=3)
    fig.update_xaxes(title_text="Floor · 楼层")
    fig.update_yaxes(title_text="Mean User-Count · 平均人数", rangemode="tozero")
    return T.finish(fig, "(c) UG vs PG/Staff mean User-Count by Floor", "各楼层：本科生 vs 研究生 / 教职员 平均人数", height=380)


def fig_series(fdf: pd.DataFrame, smooth: bool) -> go.Figure:
    ts = fdf.groupby(["Date", "User-Group"], as_index=False)["User-Count"].sum()
    if smooth:
        ts = ts.sort_values("Date")
        ts["User-Count"] = ts.groupby("User-Group")["User-Count"].transform(
            lambda s: s.rolling(7, min_periods=1).mean()
        )
    fig = go.Figure()
    order = [g for g in T.GROUP_COLORS if g in set(ts["User-Group"])] + sorted(
        set(ts["User-Group"]) - set(T.GROUP_COLORS)
    )
    for g in order:
        sub = ts[ts["User-Group"] == g]
        fig.add_scatter(
            x=sub["Date"], y=sub["User-Count"], mode="lines", name=T.group_label(g),
            line=dict(width=1.4, color=T.GROUP_COLORS.get(g, T.FAINT)),
            hovertemplate=f"{T.group_label(g)}: %{{y:,.0f}}<extra></extra>",
        )
    # Evidence beside the data: the counting-definition change noted in the insights caption.
    change = pd.Timestamp("2018-11-05")
    if len(ts) and ts["Date"].min() <= change <= ts["Date"].max():
        fig.add_shape(type="line", x0=change, x1=change, y0=0, y1=1, yref="paper",
                      line=dict(color=T.FAINT, width=1, dash="dot"))
        fig.add_annotation(x=change, y=1, yref="paper", text="counting change · 计数口径变化",
                           showarrow=False, xanchor="left", yanchor="top", xshift=4,
                           font=dict(size=11, color=T.MUTED))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text="User-Count · 人数", rangemode="tozero", tickformat=",")
    fig.update_xaxes(title_text="")
    sub = "各用户群每日总人数" + ("（7 日滚动均值）" if smooth else "")
    title = "(b) Daily total User-Count by User-Group" + (" — 7-day rolling mean" if smooth else "")
    return T.finish(fig, title, sub, height=420)


def fig_pred_months(yhat_m, obs_m, base_pred, p_month, scope: str) -> go.Figure:
    months = list(range(1, 13))
    fig = go.Figure()
    fig.add_shape(type="rect", x0=p_month - 0.5, x1=p_month + 0.5, y0=0, y1=1, yref="paper",
                  fillcolor=T.HUE_TINT, line=dict(width=0), layer="below")
    fig.add_scatter(
        x=months, y=[base_pred] * 12, mode="lines", name="Same-dow baseline · 同星期基线",
        line=dict(color=T.FAINT, width=1.5, dash="dot"),
        hovertemplate="Same-dow baseline 同星期基线: %{y:.1f}<extra></extra>",
    )
    fig.add_scatter(
        x=months, y=obs_m, mode="markers", name="2019 observed mean · 2019 实测均值",
        marker=dict(size=8, color=T.SURFACE, line=dict(color=T.MUTED, width=1.5)),
        hovertemplate="2019 observed mean 2019 实测均值: %{y:.1f}<extra></extra>",
    )
    fig.add_scatter(
        x=months, y=yhat_m, mode="lines+markers", name="Model forecast · 模型预测",
        line=dict(color=T.HUE, width=2.5), marker=dict(size=7, color=T.HUE, line=dict(color=T.SURFACE, width=1.5)),
        hovertemplate="Model forecast 模型预测: %{y:.1f}<extra></extra>",
    )
    sel = float(yhat_m[p_month - 1])
    fig.add_scatter(
        x=[p_month], y=[sel], mode="markers+text", showlegend=False, hoverinfo="skip",
        marker=dict(size=15, color=T.HUE, line=dict(color=T.SURFACE, width=2.5)),
        text=[f"<b>{sel:.0f}</b>"], textposition="top center", textfont=dict(size=13, color=T.INK),
    )
    fig.update_xaxes(tickvals=months, ticktext=[f"{T.MONTH_EN[m - 1]}<br>{m}月" for m in months],
                     range=[0.5, 12.5], title_text="Month · 月份")
    ymax = np.nanmax([np.nanmax(yhat_m), np.nanmax(obs_m) if np.isfinite(np.nanmax(obs_m)) else 0, base_pred])
    fig.update_yaxes(title_text="User-Count · 人数", range=[0, ymax * 1.18 + 1])
    fig.update_layout(hovermode="x unified")
    return T.finish(fig, "Forecast by month · 各月份预测",
                    f"{scope} · shaded = selected 阴影为选中月", height=360)


def fig_pred_weekdays(yhat_w, base_w, p_dow, p_month) -> go.Figure:
    labels = [T.dow_label(d).replace(" ", "<br>") for d in range(7)]
    sel = [y if d == p_dow else None for d, y in enumerate(yhat_w)]
    oth = [None if d == p_dow else y for d, y in enumerate(yhat_w)]
    fig = go.Figure()
    fig.add_bar(x=labels, y=oth, name="Other days · 其他日", marker_color=T.NEUTRAL_BAR,
                hovertemplate="Model forecast 模型预测: %{y:.1f}<extra></extra>")
    fig.add_bar(x=labels, y=sel, name="Selected day · 选中日", marker_color=T.HUE,
                text=[f"{v:.0f}" if v is not None else "" for v in sel], textposition="outside",
                textfont=dict(size=13, color=T.INK), cliponaxis=False,
                hovertemplate="Model forecast 模型预测: %{y:.1f}<extra></extra>")
    fig.add_scatter(x=labels, y=base_w, mode="markers", name="Same-dow baseline · 同星期基线",
                    marker=dict(symbol="line-ew-open", size=30, color=T.INK, line=dict(color=T.INK, width=2)),
                    hovertemplate="Same-dow baseline 同星期基线: %{y:.1f}<extra></extra>")
    fig.update_layout(barmode="overlay", barcornerradius=3, bargap=0.3)
    ymax = np.nanmax([np.nanmax(yhat_w), np.nanmax(base_w)])
    fig.update_yaxes(title_text="User-Count · 人数", range=[0, ymax * 1.2 + 1])
    fig.update_xaxes(title_text="")
    return T.finish(fig, "Forecast by weekday · 各星期预测",
                    f"{T.month_label(p_month)} · blue = selected 蓝色为选中日", height=360)


# ---------------------------------------------------------------- page
def main() -> None:
    st.set_page_config(page_title="Library Wi-Fi Dashboard · 图书馆 Wi-Fi 看板", layout="wide")
    st.html(T.CSS)

    df = load_data()
    floors_all = sorted(df["Floor"].unique().tolist())
    groups_all = sorted(df["User-Group"].unique().tolist())
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    n_months = df["Date"].dt.to_period("M").nunique()

    with st.sidebar:
        st.header("Filters · 筛选")
        date_range = st.date_input("Date range · 日期范围", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        floors = st.multiselect("Floor · 楼层", floors_all, default=floors_all)
        groups = st.multiselect("User-Group · 用户群", groups_all, default=groups_all, format_func=T.group_label)

        st.header("Predict controls · 预测设置")
        p_floor = st.selectbox("Floor (predict) · 楼层（预测）", floors_all, index=0)
        p_group = st.selectbox(
            "User-Group (predict) · 用户群（预测）",
            groups_all,
            index=groups_all.index("UG") if "UG" in groups_all else 0,
            format_func=T.group_label,
        )
        p_dow = st.selectbox(
            "Day of week · 星期",
            options=list(range(7)),
            format_func=T.dow_label,
            index=0,
        )
        p_month = st.slider("Month · 月份", min_value=1, max_value=12, value=10)

        st.markdown("---")
        st.caption(
            f"**Data source:** HKUST Library Wi-Fi Usage "
            f"([{DOI}]({SOURCE_URL})). "
            f"Cleaned rows: {len(df):,} · {dmin} → {dmax} · {n_months} months. "
            "Aggregates All-Floors / Total dropped.\n\n"
            f"**数据来源：** 香港科技大学图书馆 Wi-Fi 使用数据（[{DOI}]({SOURCE_URL})）。"
            f"清洗后 {len(df):,} 行 · {dmin} → {dmax} · 共 {n_months} 个月。"
            "已去除 All-Floors / Total 汇总行。"
        )

    fdf = filter_df(df, date_range, floors, groups)
    insights = compute_insights(df)
    ivals: dict = {}
    for ins in insights:
        ivals.update(ins["values"])

    # ---- title row: what this is + the one finding
    st.title("HKUST Library Wi-Fi Occupancy Explorer · 港科大图书馆 Wi-Fi 使用探索")
    _md(
        "<p class='lede'>AISC1000B · GenZ glasses · Predictive Advanced — interactive infographics, "
        "non-trivial insights and an sklearn User-Count forecast.<br>"
        "AISC1000B · GenZ glasses · 预测进阶 —— 互动信息图、非平凡洞察与 sklearn 人数预测。</p>"
    )
    if "term_vs_summer_weekday_2018" in ivals and "y2020_vs_1819_mean_drop_pct" in ivals:
        r18, r19 = ivals["term_vs_summer_weekday_2018"], ivals["term_vs_summer_weekday_2019"]
        d20 = ivals["y2020_vs_1819_mean_drop_pct"]
        _md(
            f"<p class='finding'>Teaching-term weekdays carry <b>{r18:.2f}×</b> (2018) and <b>{r19:.2f}×</b> (2019) "
            f"the summer-break load, and 2020 same-month use averaged <b>{d20:.1f}%</b> against 2018–19. "
            f"<br><span style='color:var(--muted)'>学期工作日人数为暑假的 {r18:.2f} 倍（2018）/ {r19:.2f} 倍（2019）；"
            f"2020 年同月平均变化 {d20:.1f}%。</span></p>"
        )

    # ---- KPI strip for the current filter view (type, not tiles)
    if len(fdf):
        daily = fdf.groupby("Date")["User-Count"].sum()
        peak_day, peak_v = daily.idxmax(), daily.max()
        f0, f1 = fdf["Date"].min().date(), fdf["Date"].max().date()
        kpis = [
            ("Rows in view · 当前行数", f"{len(fdf):,}", f"of {len(df):,} cleaned rows · 共 {len(df):,} 行清洗数据"),
            ("Dates in view · 日期", f"{(f1 - f0).days + 1:,} days · 天", f"{f0} → {f1}"),
            ("Months in full set · 全集月份", f"{n_months}", f"{dmin.year}–{dmax.year}"),
            ("Mean User-Count · 平均人数", f"{fdf['User-Count'].mean():.1f}", "per floor × group × day · 每楼层×用户群×日"),
            ("Busiest day in view · 最忙一天", f"{peak_v:,.0f}", f"{peak_day.date()} · {T.dow_label(peak_day.dayofweek)}"),
        ]
    else:
        kpis = [("Rows in view · 当前行数", "0", f"of {len(df):,} cleaned rows · 共 {len(df):,} 行清洗数据")] + [
            (l, "—", "no rows match the filters · 无匹配数据") for l in
            ("Dates in view · 日期", "Months in full set · 全集月份", "Mean User-Count · 平均人数", "Busiest day in view · 最忙一天")
        ]
    _kpi_strip(kpis)

    # ---- insights
    _md("<div class='section-gap'></div>")
    st.subheader("Non-Trivial Insights · 非平凡洞察")
    st.caption(
        "Auto-calculated from wifi_clean — not hand-picked anecdotes. Data-quality notes: counting definition "
        "changes on 2018-11-05 (daily total doubles overnight; WiFi-Conn per user 3.0→5.3) and again in 2021; "
        "29 dates have an all-zero total (2021-02-12→03-02, 2021-06-12→14, 2021-12-25→31: closure or logging gap).\n\n"
        "由 wifi_clean 自动计算，并非人工挑选的个例。数据质量说明：2018-11-05 起计数口径变化（每日总数一夜翻倍；"
        "每用户 WiFi-Conn 由 3.0 升至 5.3），2021 年再次变化；29 个日期总数全为 0（2021-02-12→03-02、"
        "2021-06-12→14、2021-12-25→31：闭馆或记录缺失）。"
    )
    for row_start in range(0, len(insights), 2):
        cols = st.columns(2, gap="small")
        for j, ins in enumerate(insights[row_start:row_start + 2]):
            i = row_start + j + 1
            fig_txt, fig_cap = insight_headline(ins)
            with cols[j].container(border=True, key=f"card_ins{i}"):
                _md(
                    (f"<div class='ins-fig'>{_esc(fig_txt)}</div><div class='ins-figcap'>{_esc(fig_cap)}</div>" if fig_txt else "")
                    + f"<div class='ins-title'>{i}. {_esc(ins['title_en'])} <span class='zh'>/ {_esc(ins['title_zh'])}</span></div>"
                    + f"<p class='ins-en'>{_esc(ins['body_en'])}</p>"
                    + f"<p class='ins-zh'>{_esc(ins['body_zh'])}</p>"
                )

    # ---- patterns
    _md("<div class='section-gap'></div>")
    st.subheader("Patterns · 使用规律")
    st.caption("These three charts follow the sidebar filters (date range, Floor, User-Group).\n\n以下三张图随侧栏筛选（日期范围、楼层、用户群）变化。")
    if not len(fdf):
        st.info("No data for the current filters — widen the date range or add a Floor / User-Group.\n\n当前筛选没有数据——请放宽日期范围，或加入楼层 / 用户群。")
    else:
        c_left, c_right = st.columns([3, 2], gap="small")
        with c_left.container(border=True, key="card_heat"):
            _plot(fig_heatmap(fdf), "fig_heat")
        with c_right.container(border=True, key="card_bucket"):
            fb = fig_buckets(fdf)
            if fb is None:
                _md("<p class='dq'>(c) needs UG, RPG, TPG or Staff in the User-Group filter. 请在用户群筛选中加入 UG / RPG / TPG / Staff。</p>")
            else:
                _plot(fb, "fig_bucket")
        with st.container(border=True, key="card_series"):
            smooth = st.radio(
                "Resolution · 粒度", ["Daily · 每日", "7-day mean · 7 日均值"], index=0,
                horizontal=True, label_visibility="collapsed", key="series_res",
            ).startswith("7")
            _plot(fig_series(fdf, smooth), "fig_series")

    # ---- predictive panel
    _md("<div class='section-gap'></div>")
    st.subheader("Predictive panel — User-Count forecast · 人数预测")
    model, meta, _means = train_models(df)
    pred = predict_user_count(model, meta, p_floor, p_group, int(p_dow), int(p_month))
    # baseline for selected combo
    base_key = (p_floor, p_group, int(p_dow))
    train_cut = pd.Timestamp(meta["cut"])
    train_part = df[df["Date"] < train_cut]
    base_means = train_part.groupby(["Floor", "User-Group", "dow"])["User-Count"].mean()
    base_pred = float(base_means.get(base_key, train_part["User-Count"].mean()))

    st.caption(
        f"Model: **{meta['model_name']}** · features floor, user_group, dow, month, is_weekend · "
        f"time-aware split cut **{meta['cut']}** "
        f"(train {meta['n_train']:,} / holdout {meta['n_hold']:,}). "
        "MAE = mean |y − ŷ| on holdout rows (lower is better).\n\n"
        f"模型：**{meta['model_name']}** · 特征：楼层、用户群、星期、月份、是否周末 · "
        f"按时间切分，切点 **{meta['cut']}**（训练 {meta['n_train']:,} 行 / 留出 {meta['n_hold']:,} 行）。"
        "MAE（平均绝对误差）= 留出行上 |y − ŷ| 的平均值，越低越好。"
    )
    scope = f"Floor 楼层 {p_floor} · {T.group_label(p_group)} · {T.dow_label(int(p_dow))}"
    _md(
        f"<p class='pred-scope'>Forecast for · 预测条件：<b>Floor 楼层 {_esc(p_floor)}</b><span class='sep'>·</span>"
        f"<b>{_esc(T.group_label(p_group))}</b><span class='sep'>·</span><b>{_esc(T.dow_label(int(p_dow)))}</b>"
        f"<span class='sep'>·</span><b>{_esc(T.month_label(int(p_month)))}</b>"
        "<br><span style='color:var(--muted)'>Change any Predict control in the sidebar and the numbers and charts below update."
        "<br>调整侧栏任一预测设置，下方数字与图表即时更新。</span></p>"
    )

    with st.container(border=True, key="card_pred"):
        m1, m2, m3, m4 = st.columns(4)
        with m1.container(key="kpi_pred"):
            st.metric(
                "Predicted User-Count · 预测人数", f"{pred:.1f}",
                delta=f"{pred - base_pred:+.1f} vs same-dow baseline · 较同星期基线", delta_color="off",
            )
        m2.metric("Same-dow baseline (selected) · 同星期基线", f"{base_pred:.1f}")
        m3.metric("Holdout MAE (model) · 模型 MAE", f"{meta['mae_model']:.2f}")
        m4.metric(
            "Holdout MAE (same-dow baseline) · 基线 MAE",
            f"{meta['mae_baseline']:.2f}",
            delta=f"{meta['mae_baseline'] - meta['mae_model']:.2f} vs baseline · 模型误差更低",
            delta_color="normal",
        )

        # The dynamic chart: real model predictions for the selected Floor × User-Group.
        months = list(range(1, 13))
        yhat_m = predict_grid(model, meta, p_floor, p_group, [int(p_dow)] * 12, months)
        s19 = seasonal_2019_means(df, meta["cut"])
        obs_m = np.array([s19.get((p_floor, p_group, m, int(p_dow)), np.nan) for m in months], dtype=float)
        yhat_w = predict_grid(model, meta, p_floor, p_group, list(range(7)), [int(p_month)] * 7)
        base_w = np.array(
            [base_means.get((p_floor, p_group, d), train_part["User-Count"].mean()) for d in range(7)], dtype=float
        )
        g1, g2 = st.columns([3, 2], gap="small")
        with g1:
            _plot(fig_pred_months(yhat_m, obs_m, base_pred, int(p_month), scope), "fig_pred_months")
        with g2:
            _plot(fig_pred_weekdays(yhat_w, base_w, int(p_dow), int(p_month)), "fig_pred_weekdays")

    _md(
        f"<p class='honest'>Relative MAE reduction vs the long-run same-dow baseline: <b>{meta['mae_rel_improvement_pct']:.1f}%</b>. "
        f"But two simple baselines do slightly better than the model: last-28-days same-dow mean "
        f"MAE <b>{meta['mae_baseline_recent28d_same_dow']:.2f}</b> and 2019 same-month×dow mean MAE "
        f"<b>{meta['mae_baseline_2019_same_month_dow']:.2f}</b> — the model has no recency/level signal, so it "
        "under-predicts the 2021 rebound. "
        f"Absolute error stays large (holdout mean User-Count {meta['hold_mean']:.0f} vs train mean "
        f"{meta['train_mean']:.0f}: the holdout covers the 2021 rebound), so read the panel as a "
        "typical-level estimate for a floor × group × weekday × month cell, not a precise daily forecast.</p>"
        f"<p class='honest' style='color:var(--muted)'>相对长期同星期基线，MAE 降低 {meta['mae_rel_improvement_pct']:.1f}%；"
        f"但两个简单基线略优于模型（近 28 天同星期均值 MAE {meta['mae_baseline_recent28d_same_dow']:.2f}，"
        f"2019 同月×同星期均值 MAE {meta['mae_baseline_2019_same_month_dow']:.2f}）——模型没有近期水平信号，会低估 2021 年回升。"
        "请把本面板视为某楼层×用户群×星期×月份的典型水平估计，而非精确的逐日预测。</p>"
    )
    _md(
        f"<p class='foot'>Data · 数据：HKUST Library Wi-Fi Usage 港科大图书馆 Wi-Fi 使用数据 · <a href='{SOURCE_URL}' target='_blank'>{DOI}</a> · "
        f"{len(df):,} cleaned rows 清洗后行数 · {dmin} → {dmax}</p>"
    )


if __name__ == "__main__":
    main()
