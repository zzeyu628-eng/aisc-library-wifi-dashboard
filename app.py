"""AISC1000B GenZ glasses — HKUST Library Wi-Fi Predictive Advanced dashboard."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent  # Streamlit Cloud: app.py at repo root; data/ beside it
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
        st.error(f"Missing {DATA_PATH}. Run: python scripts/merge_wifi.py")
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


def main() -> None:
    st.set_page_config(page_title="Library Wi-Fi Predictive Advanced", layout="wide")
    st.title("HKUST Library Wi-Fi Occupancy Explorer")
    st.caption(
        "AISC1000B · GenZ glasses · Predictive Advanced — interactive infographics + "
        "non-trivial insights + sklearn forecast"
    )

    df = load_data()
    floors_all = sorted(df["Floor"].unique().tolist())
    groups_all = sorted(df["User-Group"].unique().tolist())
    dmin, dmax = df["Date"].min().date(), df["Date"].max().date()
    n_months = df["Date"].dt.to_period("M").nunique()

    with st.sidebar:
        st.header("Filters")
        date_range = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        floors = st.multiselect("Floor", floors_all, default=floors_all)
        groups = st.multiselect("User-Group", groups_all, default=groups_all)

        st.header("Predict controls")
        p_floor = st.selectbox("Floor (predict)", floors_all, index=0)
        p_group = st.selectbox(
            "User-Group (predict)",
            groups_all,
            index=groups_all.index("UG") if "UG" in groups_all else 0,
        )
        p_dow = st.selectbox(
            "Day of week",
            options=list(range(7)),
            format_func=lambda i: DOW_LABELS[i],
            index=0,
        )
        p_month = st.slider("Month", min_value=1, max_value=12, value=10)

        st.markdown("---")
        st.caption(
            f"**Data source:** HKUST Library Wi-Fi Usage "
            f"([{DOI}]({SOURCE_URL})). "
            f"Cleaned rows: {len(df):,} · {dmin} → {dmax} · {n_months} months. "
            "Aggregates All-Floors / Total dropped."
        )

    fdf = filter_df(df, date_range, floors, groups)
    st.write(
        f"**Rows:** {len(fdf):,} · **Dates:** "
        f"{fdf['Date'].min().date() if len(fdf) else '—'} → "
        f"{fdf['Date'].max().date() if len(fdf) else '—'} · "
        f"**Months in full set:** {n_months}"
    )

    # Insights
    st.subheader("Non-Trivial Insights（非平凡洞察）")
    st.caption(
        "Auto-calculated from wifi_clean — not hand-picked anecdotes. Data-quality notes: counting definition "
        "changes on 2018-11-05 (daily total doubles overnight; WiFi-Conn per user 3.0→5.3) and again in 2021; "
        "29 dates have an all-zero total (2021-02-12→03-02, 2021-06-12→14, 2021-12-25→31: closure or logging gap)."
    )
    for i, ins in enumerate(compute_insights(df), start=1):
        with st.container():
            st.markdown(f"**{i}. {ins['title_en']} / {ins['title_zh']}**")
            st.write(ins["body_en"])
            st.write(ins["body_zh"])

    # (a) Floor × dow heatmap
    st.subheader("(a) Floor × weekday heatmap (mean User-Count)")
    if len(fdf):
        heat = (
            fdf.groupby(["Floor", "dow"], as_index=False)["User-Count"]
            .mean()
            .assign(weekday=lambda x: x["dow"].map(DOW_LABELS))
        )
        pivot = heat.pivot(index="Floor", columns="weekday", values="User-Count")
        cols = [DOW_LABELS[i] for i in range(7) if DOW_LABELS[i] in pivot.columns]
        pivot = pivot.reindex(columns=cols)
        fig_h = px.imshow(
            pivot, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(color="User-Count")
        )
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("No data for current filters.")

    # (b) time series
    st.subheader("(b) Daily total User-Count by User-Group")
    if len(fdf):
        ts = fdf.groupby(["Date", "User-Group"], as_index=False)["User-Count"].sum()
        fig_ts = px.line(ts, x="Date", y="User-Count", color="User-Group")
        st.plotly_chart(fig_ts, use_container_width=True)

    # (c) UG vs PG/Staff bar
    st.subheader("(c) UG vs PG/Staff mean User-Count by Floor")
    if len(fdf):
        focus = {"UG", "RPG", "TPG", "Staff"}
        bar_df = fdf[fdf["User-Group"].isin(focus)].copy()
        bar_df["bucket"] = bar_df["User-Group"].map(
            lambda g: "UG" if g == "UG" else ("Staff" if g == "Staff" else "PG (RPG+TPG)")
        )
        bar = bar_df.groupby(["Floor", "bucket"], as_index=False)["User-Count"].mean()
        fig_b = px.bar(bar, x="Floor", y="User-Count", color="bucket", barmode="group")
        st.plotly_chart(fig_b, use_container_width=True)

    # Predictive panel
    st.subheader("Predictive panel — User-Count forecast")
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
        "MAE = mean |y − ŷ| on holdout rows."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted User-Count", f"{pred:.1f}")
    m2.metric("Same-dow baseline (selected)", f"{base_pred:.1f}")
    m3.metric("Holdout MAE (model)", f"{meta['mae_model']:.2f}")
    m4.metric(
        "Holdout MAE (same-dow baseline)",
        f"{meta['mae_baseline']:.2f}",
        delta=f"{meta['mae_baseline'] - meta['mae_model']:.2f} vs baseline",
        delta_color="normal",
    )
    st.caption(
        f"Relative MAE reduction vs the long-run same-dow baseline: **{meta['mae_rel_improvement_pct']:.1f}%**. "
        f"But two simple baselines do slightly better than the model: last-28-days same-dow mean "
        f"MAE **{meta['mae_baseline_recent28d_same_dow']:.2f}** and 2019 same-month×dow mean MAE "
        f"**{meta['mae_baseline_2019_same_month_dow']:.2f}** — the model has no recency/level signal, so it "
        "under-predicts the 2021 rebound. "
        f"Absolute error stays large (holdout mean User-Count {meta['hold_mean']:.0f} vs train mean "
        f"{meta['train_mean']:.0f}: the holdout covers the 2021 rebound), so read the panel as a "
        "typical-level estimate for a floor × group × weekday × month cell, not a precise daily forecast."
    )


if __name__ == "__main__":
    main()
