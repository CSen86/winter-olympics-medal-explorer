"""
Winter Olympics Medal Explorer
Run with:  streamlit run winter_olympics_app.py
CSVs must be in the same directory.
"""

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Winter Olympics Medal Explorer",
                   page_icon="🏅", layout="wide")

st.markdown("""<style>
  .stApp { background-color: #0d1b2a; color: #e8edf2; }
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, #1a2d44 0%, #0d1b2a 100%);
      border-right: 1px solid #2a4060; }
  .metric-card { background: linear-gradient(135deg,#1e3a5f,#162d48);
      border:1px solid #2a5080; border-radius:12px; padding:18px 20px;
      text-align:center; margin-bottom:6px; }
  .metric-card .val { font-size:2rem; font-weight:700; color:#7ec8e3; line-height:1; }
  .metric-card .lbl { font-size:.74rem; color:#8aafc8; margin-top:4px;
      text-transform:uppercase; letter-spacing:.08em; }
  .gold{color:#ffd700!important} .silver{color:#c0c0c0!important} .bronze{color:#cd7f32!important}
  h1{color:#e8edf2!important} h2,h3{color:#b8d4e8!important} label{color:#8aafc8!important}
  .stTabs [data-baseweb="tab"]{background:#1a2d44;border-radius:8px 8px 0 0;
      color:#8aafc8;padding:8px 18px;}
  .stTabs [aria-selected="true"]{background:#1e3a5f!important;color:#7ec8e3!important;}
</style>""", unsafe_allow_html=True)

# Medal colors (already high contrast)
MC = {"GOLD": "#ffd700", "SILVER": "#c0c0c0", "BRONZE": "#cd7f32"}

# High-contrast country palette for dark backgrounds
# (Bold is very readable; Dark24 gives more variety if many countries are shown)
COUNTRY_COLORS = px.colors.qualitative.Bold + px.colors.qualitative.Dark24

# Global layout
_L = dict(
    paper_bgcolor="#111e2d",
    plot_bgcolor="#0d1b2a",
    font=dict(color="#e8edf2", family="Inter,sans-serif"),
    margin=dict(t=40, b=40, l=60, r=20),
    legend=dict(
        title_text="",
        bgcolor="rgba(0,0,0,0.55)",           # <<< makes legend readable on dark bg
        bordercolor="rgba(255,255,255,0.25)",  # <<< subtle border
        borderwidth=1,
        font=dict(color="#e8edf2", size=12),   # <<< readable legend text
        itemsizing="constant",
    ),
)

_LX = dict(
    **_L,
    xaxis=dict(
        gridcolor="#1e3a5f",
        zerolinecolor="#1e3a5f",
        tickfont=dict(color="#e8edf2"),
        title=dict(font=dict(color="#e8edf2")),
    ),
    yaxis=dict(
        gridcolor="#1e3a5f",
        zerolinecolor="#1e3a5f",
        tickfont=dict(color="#e8edf2"),
        title=dict(font=dict(color="#e8edf2")),
    ),
)

def apply_legend_readability(fig, horizontal=False, showlegend=True):
    """Make legend readable on dark background (and consistent)."""
    fig.update_layout(showlegend=showlegend)
    if showlegend:
        if horizontal:
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0,
                    bgcolor="rgba(0,0,0,0.55)",
                    bordercolor="rgba(255,255,255,0.25)",
                    borderwidth=1,
                    font=dict(color="#e8edf2", size=12),
                )
            )
        else:
            fig.update_layout(
                legend=dict(
                    bgcolor="rgba(0,0,0,0.55)",
                    bordercolor="rgba(255,255,255,0.25)",
                    borderwidth=1,
                    font=dict(color="#e8edf2", size=12),
                )
            )
    return fig


@st.cache_data(show_spinner="Loading Olympic data…")
def load_data():
    hosts = pd.read_csv("olympic_hosts.csv")
    medals = pd.read_csv("olympic_medals.csv")
    athletes = pd.read_csv("olympic_athletes.csv")
    results = pd.read_csv("olympic_results.csv")

    # Rinomina game_slug → slug_game per uniformità
    hosts = hosts.rename(columns={"game_slug": "slug_game"})
    medals = medals.rename(columns={"game_slug": "slug_game"}) if "game_slug" in medals.columns else medals
    results = results.rename(columns={"game_slug": "slug_game"}) if "game_slug" in results.columns else results

    winter_slugs = hosts[hosts["game_season"] == "Winter"]["slug_game"].unique()
    hw = (hosts[hosts["game_season"] == "Winter"]
          .assign(game_year=lambda d: pd.to_numeric(d["game_year"], errors="coerce"))
          .sort_values("game_year"))

    mw = (medals[medals["slug_game"].isin(winter_slugs)]
          .merge(hw[["slug_game", "game_year", "game_location", "game_name"]], on="slug_game", how="left")
          .assign(
              game_year=lambda d: d["game_year"].astype("Int64"),
              medal_type=lambda d: d["medal_type"].astype(str).str.upper().str.strip(),
              is_team=lambda d: d["participant_type"].astype(str).str.lower().str.contains("team", na=False),
          )
          .merge(athletes[["athlete_full_name", "athlete_year_birth", "games_participations", "first_game"]],
                 on="athlete_full_name", how="left"))
    return hw, mw, athletes


hosts_w, medals_w, athletes = load_data()

# ── Sidebar ──────────────────────────────────
with st.sidebar:
    st.markdown("## 🏔️ Filters")
    yrs = sorted(medals_w["game_year"].dropna().unique().astype(int))
    yr = st.select_slider("Edition range", options=yrs, value=(yrs[0], yrs[-1]))
    sc = st.multiselect("Country", sorted(medals_w["country_name"].dropna().unique()), placeholder="All")
    sd = st.multiselect("Discipline", sorted(medals_w["discipline_title"].dropna().unique()), placeholder="All")
    sm = st.multiselect("Medal type", ["GOLD", "SILVER", "BRONZE"], default=["GOLD", "SILVER", "BRONZE"])
    to = st.radio("Athlete type", ["All", "Individual only", "Team only"])
    st.caption("Source: Kaggle · Olympic History Dataset")

# ── Filter ───────────────────────────────────
df = medals_w.copy()
df = df[(df["game_year"] >= yr[0]) & (df["game_year"] <= yr[1])]
if sc:
    df = df[df["country_name"].isin(sc)]
if sd:
    df = df[df["discipline_title"].isin(sd)]
if sm:
    df = df[df["medal_type"].isin(sm)]
if to == "Individual only":
    df = df[~df["is_team"]]
elif to == "Team only":
    df = df[df["is_team"]]

# ── Header ───────────────────────────────────
st.markdown("# 🏅 Winter Olympics Medal Explorer")
st.caption(f"**{len(df):,}** records · {df['game_year'].nunique()} editions · "
           f"{df['country_name'].nunique()} countries · {df['discipline_title'].nunique()} disciplines")

gn = (df["medal_type"] == "GOLD").sum()
sn = (df["medal_type"] == "SILVER").sum()
bn = (df["medal_type"] == "BRONZE").sum()
tc = df.groupby("country_name").size().idxmax() if len(df) > 0 else "—"
ta = df.groupby("athlete_full_name").size().idxmax() if len(df) > 0 else "—"

def kpi(col, v, l, c=""):
    col.markdown(f'<div class="metric-card"><div class="val {c}">{v}</div>'
                 f'<div class="lbl">{l}</div></div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, f"{gn:,}", "Gold", "gold")
kpi(c2, f"{sn:,}", "Silver", "silver")
kpi(c3, f"{bn:,}", "Bronze", "bronze")
kpi(c4, tc, "Top Country")
kpi(c5, ta.split()[-1] if ta != "—" else "—", "Top Athlete")
st.markdown("---")

# ── Tabs ─────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs(["📊 Country Rankings", "📅 Timeline",
                             "🗺️ Heatmap", "🏆 Athletes", "📋 Raw Data"])

with t1:
    L, R = st.columns([3, 2], gap="large")
    with L:
        st.markdown("### Stacked Medal Bar")
        n = st.slider("Top N", 5, 30, 15, key="tn")
        tl = (df.groupby(["country_name", "medal_type"]).size().reset_index(name="count"))
        tp = tl.groupby("country_name")["count"].sum().nlargest(n).index
        tl = tl[tl["country_name"].isin(tp)]
        ord_ = (tl[tl["medal_type"] == "GOLD"].groupby("country_name")["count"].sum()
                .sort_values().index.tolist())
        fig = px.bar(
            tl, x="count", y="country_name", color="medal_type",
            color_discrete_map=MC, orientation="h",
            category_orders={"country_name": ord_, "medal_type": ["BRONZE", "SILVER", "GOLD"]},
            labels={"count": "Medals", "country_name": "", "medal_type": ""}
        )
        fig.update_layout(**{k: v for k, v in _L.items()}, height=500)
        fig = apply_legend_readability(fig, horizontal=True, showlegend=True)  # <<< readable legend
        st.plotly_chart(fig, use_container_width=True)

    with R:
        st.markdown("### Medal Table")
        pv = (df.groupby(["country_name", "medal_type"]).size().unstack(fill_value=0)
              .reindex(columns=["GOLD", "SILVER", "BRONZE"], fill_value=0))
        pv["TOTAL"] = pv.sum(axis=1)
        pv = pv.sort_values(["GOLD", "SILVER", "BRONZE"], ascending=False)
        st.dataframe(
            pv.reset_index().rename(columns={
                "country_name": "Country", "GOLD": "🥇", "SILVER": "🥈", "BRONZE": "🥉", "TOTAL": "∑"
            }),
            hide_index=True, use_container_width=True, height=500
        )

    st.markdown("### Treemap — Medal Share")
    tm = df.groupby(["medal_type", "country_name"]).size().reset_index(name="count")
    ft = px.treemap(tm, path=["medal_type", "country_name"], values="count",
                    color="medal_type", color_discrete_map=MC)
    ft.update_layout(**_LX, height=360)
    # Make treemap labels readable on dark theme
    ft.update_traces(
        textfont_color="#e8edf2",
        hovertemplate="<b>%{label}</b><br>%{value} medals<extra></extra>"
    )
    st.plotly_chart(ft, use_container_width=True)

with t2:
    st.markdown("### Medal Count per Edition")
    tn2 = st.slider("Top N countries", 3, 15, 8, key="tn2")
    tp2 = df.groupby("country_name").size().nlargest(tn2).index
    tl2 = (df[df["country_name"].isin(tp2)].groupby(["game_year", "country_name"])
           .size().reset_index(name="medals"))

    # IMPORTANT: high-contrast colors for readability on dark background
    fl = px.line(
        tl2, x="game_year", y="medals", color="country_name", markers=True,
        color_discrete_sequence=COUNTRY_COLORS,
        labels={"game_year": "Year", "medals": "Medals", "country_name": "Country"}
    )
    fl.update_layout(**_LX, height=400)
    fl.update_traces(line=dict(width=2.4), marker=dict(size=7))
    fl = apply_legend_readability(fl, horizontal=False, showlegend=True)  # <<< readable legend box
    st.plotly_chart(fl, use_container_width=True)

    st.markdown("### Medals by Type per Edition")
    ya = df.groupby(["game_year", "medal_type"]).size().reset_index(name="count")
    fa = px.area(
        ya, x="game_year", y="count", color="medal_type", color_discrete_map=MC,
        category_orders={"medal_type": ["BRONZE", "SILVER", "GOLD"]},
        labels={"game_year": "Year", "count": "Medals", "medal_type": ""}
    )
    fa.update_layout(**_LX, height=280)
    fa = apply_legend_readability(fa, horizontal=True, showlegend=True)
    st.plotly_chart(fa, use_container_width=True)

    st.markdown("### 🎬 Animated Cumulative Race (Top 10)")
    t10 = df.groupby("country_name").size().nlargest(10).index.tolist()
    cum, run = [], {c: 0 for c in t10}
    for yr_ in sorted(df["game_year"].unique()):
        sub = df[(df["game_year"] == yr_) & (df["country_name"].isin(t10))]
        for c in t10:
            run[c] += (sub["country_name"] == c).sum()
        for c, v in run.items():
            cum.append({"year": yr_, "country": c, "cumulative": v})
    cdf = pd.DataFrame(cum)

    fr = px.bar(
        cdf, x="cumulative", y="country",
        animation_frame="year", orientation="h",
        color="country",
        color_discrete_sequence=COUNTRY_COLORS,
        range_x=[0, cdf["cumulative"].max() * 1.05],
        labels={"cumulative": "Cumulative Medals", "country": ""}
    )
    fr.update_layout(**_LX, height=420, showlegend=False)
    st.plotly_chart(fr, use_container_width=True)

with t3:
    st.markdown("### Country × Discipline Heatmap")
    hc = st.slider("Top countries", 5, 25, 15, key="hc")
    hd = st.slider("Top disciplines", 5, 30, 15, key="hd")
    tc_ = df.groupby("country_name").size().nlargest(hc).index
    td_ = df.groupby("discipline_title").size().nlargest(hd).index
    hm = (df[df["country_name"].isin(tc_) & df["discipline_title"].isin(td_)]
          .groupby(["country_name", "discipline_title"]).size().reset_index(name="medals"))
    hp = hm.pivot(index="country_name", columns="discipline_title", values="medals").fillna(0)

    fh = px.imshow(
        hp,
        color_continuous_scale=["#0d1b2a", "#1a5276", "#2980b9", "#ffd700"],
        labels=dict(color="Medals"),
        aspect="auto"
    )
    fh.update_layout(**_LX, height=480)
    fh.update_xaxes(tickangle=-40, showgrid=False)
    fh.update_yaxes(showgrid=False)
    fh.update_traces(hovertemplate="<b>%{y}</b> · %{x}<br>Medals: %{z}<extra></extra>")
    st.plotly_chart(fh, use_container_width=True)

    st.markdown("### Drill Down: Countries in One Discipline")
    dd_sel = st.selectbox("Discipline", sorted(df["discipline_title"].dropna().unique()), key="dds")
    ddf = df[df["discipline_title"] == dd_sel]
    ddt = ddf.groupby(["country_name", "medal_type"]).size().reset_index(name="count")
    ddo = ddt.groupby("country_name")["count"].sum().nlargest(12).index
    fdd = px.bar(
        ddt[ddt["country_name"].isin(ddo)], x="country_name", y="count",
        color="medal_type", color_discrete_map=MC, barmode="stack",
        category_orders={"country_name": ddo.tolist(), "medal_type": ["BRONZE", "SILVER", "GOLD"]},
        labels={"count": "Medals", "country_name": "", "medal_type": ""}
    )
    fdd.update_layout(**_LX, height=340)
    fdd = apply_legend_readability(fdd, horizontal=True, showlegend=True)
    st.plotly_chart(fdd, use_container_width=True)

with t4:
    ca, cb = st.columns([2, 3], gap="large")
    with ca:
        st.markdown("### Top Athletes")
        an = st.slider("Top N", 5, 50, 20, key="an")
        at = (df.groupby(["athlete_full_name", "country_name", "medal_type"]).size().reset_index(name="count"))
        atn = at.groupby("athlete_full_name")["count"].sum().nlargest(an)
        atf = at[at["athlete_full_name"].isin(atn.index)]
        oa = (atf.groupby("athlete_full_name")["count"].sum().sort_values().index.tolist())
        faa = px.bar(
            atf, x="count", y="athlete_full_name", color="medal_type",
            color_discrete_map=MC, orientation="h",
            category_orders={"athlete_full_name": oa, "medal_type": ["BRONZE", "SILVER", "GOLD"]},
            labels={"count": "Medals", "athlete_full_name": "", "medal_type": ""}
        )
        faa.update_layout(**_LX, height=560, showlegend=False)
        st.plotly_chart(faa, use_container_width=True)

    with cb:
        st.markdown("### Athlete Profile")
        srch = st.text_input("Search name", placeholder="e.g. Bjørndalen…")
        pool = (df[df["athlete_full_name"].str.contains(srch, case=False, na=False)]
                ["athlete_full_name"].unique() if srch else atn.index.tolist())
        if len(pool) == 0:
            st.info("No athletes found.")
        else:
            sa = st.selectbox("Athlete", pool, key="sa")
            adf = df[df["athlete_full_name"] == sa]
            am = athletes[athletes["athlete_full_name"] == sa]
            if len(am) > 0:
                m = am.iloc[0]
                m1, m2, m3 = st.columns(3)
                m1.metric("Born", int(m["athlete_year_birth"]) if pd.notna(m["athlete_year_birth"]) else "—")
                m2.metric("Games", int(m["games_participations"]) if pd.notna(m["games_participations"]) else "—")
                m3.metric("First", m["first_game"] if pd.notna(m["first_game"]) else "—")

            ap = adf.groupby(["game_year", "medal_type"]).size().reset_index(name="count")
            fp = px.bar(
                ap, x="game_year", y="count", color="medal_type", color_discrete_map=MC, barmode="stack",
                category_orders={"medal_type": ["BRONZE", "SILVER", "GOLD"]},
                labels={"game_year": "Year", "count": "Medals", "medal_type": ""}, title="By Edition"
            )
            fp.update_layout(**_LX, height=250)
            fp = apply_legend_readability(fp, horizontal=True, showlegend=True)
            st.plotly_chart(fp, use_container_width=True)

            dp2 = adf.groupby(["discipline_title", "medal_type"]).size().reset_index(name="count")
            fd = px.bar(
                dp2, x="discipline_title", y="count", color="medal_type",
                color_discrete_map=MC, barmode="stack", title="By Discipline",
                labels={"discipline_title": "", "count": "Medals", "medal_type": ""}
            )
            fd.update_layout(**_LX, height=240)
            fd = apply_legend_readability(fd, horizontal=True, showlegend=True)
            st.plotly_chart(fd, use_container_width=True)

            st.markdown("**All Medal Events**")
            st.dataframe(
                adf[["game_year", "game_location", "discipline_title", "event_title", "medal_type"]]
                .sort_values("game_year", ascending=False)
                .rename(columns={"game_year": "Year", "game_location": "Host",
                                 "discipline_title": "Discipline", "event_title": "Event",
                                 "medal_type": "Medal"}),
                hide_index=True, use_container_width=True, height=200
            )

with t5:
    st.markdown(f"### Filtered Records ({len(df):,} rows)")
    cols = ["game_year", "game_location", "discipline_title", "event_title",
            "medal_type", "athlete_full_name", "country_name", "participant_type"]
    cols = [c for c in cols if c in df.columns]
    vw = df[cols].sort_values(["game_year", "country_name"]).rename(columns={
        "game_year": "Year", "game_location": "Host", "discipline_title": "Discipline",
        "event_title": "Event", "medal_type": "Medal", "athlete_full_name": "Athlete",
        "country_name": "Country", "participant_type": "Type"
    })
    st.dataframe(vw, hide_index=True, use_container_width=True, height=500)
    st.download_button(
        "⬇️ Download as CSV",
        vw.to_csv(index=False).encode(),
        file_name="winter_olympics_filtered.csv",
        mime="text/csv"
    )