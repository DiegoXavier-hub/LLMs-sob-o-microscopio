from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from i18n import tr, figures_dir, tables_dir

sns.set_theme(style="whitegrid", context="talk")

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
TAB = tables_dir(ROOT)
FIG = figures_dir(ROOT)

LDB = pd.read_parquet(IN / "leaderboard_valid.parquet")
LDB["organizacao"] = LDB["fullname"].astype(str).str.split("/").str[0]
MOE = pd.read_csv(TAB / "04_margem_erro_teorica.csv")
PUD = pd.read_csv(TAB / "04_pares_indistinguiveis_top100.csv")
BT = pd.read_csv(TAB / "09_bradley_terry.csv")
GLOBAL_WR = pd.read_csv(TAB / "09_winrate_global.csv")
POWER = pd.read_csv(TAB / "06_analise_poder.csv")
PAIRS_RES = pd.read_csv(TAB / "06_pares_top50_resumo.csv")
CLUSTERS = pd.read_csv(TAB / "07_perfil_clusters.csv", index_col=0)
RANKING_BOOT = pd.read_csv(TAB / "08_ranking_bootstrap_top50.csv")
CROSS = pd.read_csv(TAB / "09_cross_validation.csv")
KTAU = pd.read_csv(TAB / "08_kendall_tau_benchmarks.csv", index_col=0)

pairs_res = PAIRS_RES.rename(columns={PAIRS_RES.columns[0]: "benchmark"})

fig = make_subplots(
    rows=6, cols=2,
    subplot_titles=[
        tr("1. Margem de erro teorica por benchmark", "1. Theoretical margin of error by benchmark"),
        tr("2. % pares top-100 indistinguiveis", "2. % top-100 pairs indistinguishable"),
        tr("3. Distribuicao do average", "3. Average distribution"),
        tr("4. % de pares significativos (brut vs FDR)", "4. % significant pairs (raw vs FDR)"),
        tr("5. Kendall tau entre benchmarks", "5. Kendall tau between benchmarks"),
        tr("6. Perfil dos clusters", "6. Cluster profiles"),
        tr("7. Ranking bootstrap top-30", "7. Top-30 bootstrap ranking"),
        tr("8. Arena - winrate global com IC Wilson", "8. Arena - global win rate with Wilson CI"),
        tr("9. Arena - Bradley-Terry", "9. Arena - Bradley-Terry"),
        tr("10. Cross-validation Arena vs Leaderboard", "10. Cross-validation Arena vs Leaderboard"),
        tr("11. n necessario - top1 vs top10", "11. Required n - top1 vs top10"),
        tr("12. Top-15 organizacoes", "12. Top-15 organizations"),
    ],
    vertical_spacing=0.055, horizontal_spacing=0.13,
    specs=[[{"type": "bar"}, {"type": "bar"}],
           [{"type": "histogram"}, {"type": "bar"}],
           [{"type": "heatmap"}, {"type": "heatmap"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}],
           [{"type": "bar"}, {"type": "bar"}]],
)

fig.add_trace(go.Bar(x=MOE["benchmark"], y=MOE["MOE_p50"], marker_color="#1f77b4",
                     text=[f"{v:.1f}pp" for v in MOE["MOE_p50"]], textposition="outside"),
              row=1, col=1)
fig.add_trace(go.Bar(x=PUD["benchmark"], y=PUD["pct"], marker_color="#9467bd",
                     text=[f"{v}%" for v in PUD["pct"]], textposition="outside"),
              row=1, col=2)
fig.add_trace(go.Histogram(x=LDB["average"], nbinsx=60, marker_color="#2ca02c"),
              row=2, col=1)
fig.add_trace(go.Bar(x=pairs_res["benchmark"], y=pairs_res["pct_sig_brut"],
                     name=tr("brut", "raw"), marker_color="#1f77b4"), row=2, col=2)
fig.add_trace(go.Bar(x=pairs_res["benchmark"], y=pairs_res["pct_sig_fdr"],
                     name="FDR", marker_color="#d62728"), row=2, col=2)
fig.add_trace(go.Heatmap(z=KTAU.values, x=KTAU.columns, y=KTAU.index,
                         zmin=0, zmax=1, colorscale="RdYlBu_r", showscale=False,
                         text=KTAU.round(2).values, texttemplate="%{text}"),
              row=3, col=1)
fig.add_trace(go.Heatmap(z=CLUSTERS.values, x=CLUSTERS.columns,
                         y=[f"C{c}" for c in CLUSTERS.index],
                         colorscale="YlGnBu", showscale=False,
                         text=CLUSTERS.round(1).values, texttemplate="%{text}"),
              row=3, col=2)

rb30 = RANKING_BOOT.head(30)
fig.add_trace(go.Scatter(
    x=rb30["rank_mediano_boot"], y=rb30["fullname"].str.slice(0, 30),
    error_x=dict(type="data", symmetric=False,
                 array=rb30["rank_hi95"] - rb30["rank_mediano_boot"],
                 arrayminus=rb30["rank_mediano_boot"] - rb30["rank_lo95"]),
    mode="markers", marker=dict(color="#1f77b4", size=7),
), row=4, col=1)

fig.add_trace(go.Scatter(
    x=GLOBAL_WR["win_rate"] * 100, y=GLOBAL_WR["model"],
    error_x=dict(type="data", symmetric=False,
                 array=(GLOBAL_WR["wilson_hi"] - GLOBAL_WR["win_rate"]) * 100,
                 arrayminus=(GLOBAL_WR["win_rate"] - GLOBAL_WR["wilson_lo"]) * 100),
    mode="markers", marker=dict(color="#d62728", size=7),
), row=4, col=2)

fig.add_trace(go.Scatter(
    x=BT["bt_rating_elo"], y=BT["model"],
    error_x=dict(type="data", symmetric=False,
                 array=BT["bt_hi95"] - BT["bt_rating_elo"],
                 arrayminus=BT["bt_rating_elo"] - BT["bt_lo95"]),
    mode="markers", marker=dict(color="#ff7f0e", size=7),
), row=5, col=1)

fig.add_trace(go.Scatter(
    x=CROSS["arena_wr"] * 100, y=CROSS["ldb_average"],
    mode="markers+text", text=CROSS["arena_model"],
    textposition="top center",
    marker=dict(size=15, color="#9467bd"),
), row=5, col=2)

fig.add_trace(go.Bar(x=POWER["benchmark"], y=POWER["n_necessario"],
                     marker_color="#8c564b",
                     text=POWER["n_necessario"], textposition="outside"),
              row=6, col=1)

org_top = LDB.groupby("organizacao").filter(lambda g: len(g) >= 5)
org_top = org_top.groupby("organizacao")["average"].median().sort_values(ascending=False).head(15)
fig.add_trace(go.Bar(x=org_top.values, y=org_top.index, orientation="h",
                     marker_color="#17becf"),
              row=6, col=2)

fig.update_layout(
    height=2200, width=1500,
    title_text=tr("<b>LLMs sob o microscopio</b> - dashboard",
                  "<b>LLMs under the microscope</b> - dashboard"),
    title_font_size=22,
    showlegend=False,
    paper_bgcolor="white",
    plot_bgcolor="#fafafa",
)
fig.update_yaxes(autorange="reversed", row=4, col=1)
fig.update_yaxes(autorange="reversed", row=4, col=2)
fig.update_yaxes(autorange="reversed", row=5, col=1)
fig.update_yaxes(type="log", row=6, col=1)

fig.write_html(FIG / "10_dashboard.html", include_plotlyjs="cdn", full_html=True)

fig2, axes = plt.subplots(4, 3, figsize=(20, 22))

ax = axes[0, 0]
ax.bar(MOE["benchmark"], MOE["MOE_p50"], color="#1f77b4")
ax.set_title(tr("Margem de erro teorica", "Theoretical margin of error"))
ax.set_ylabel(tr("pontos percentuais", "percentage points"))
ax.tick_params(axis="x", rotation=20)

ax = axes[0, 1]
ax.barh(PUD["benchmark"], PUD["pct"], color="#9467bd")
for i, v in enumerate(PUD["pct"]):
    ax.text(v + 0.5, i, f"{v}%", va="center")
ax.set_title(tr("% pares top-100 indistinguiveis", "% top-100 pairs indistinguishable"))

ax = axes[0, 2]
x = np.arange(len(pairs_res))
w = 0.35
ax.bar(x - w / 2, pairs_res["pct_sig_brut"], w, label="brut", color="#1f77b4")
ax.bar(x + w / 2, pairs_res["pct_sig_fdr"], w, label="FDR", color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(pairs_res["benchmark"], rotation=20)
ax.set_title(tr("% pares sig. (top-50)", "% significant pairs (top-50)"))
ax.legend()

ax = axes[1, 0]
ax.hist(LDB["average"], bins=60, color="#2ca02c")
ax.set_title(tr(f"Distribuicao do average (n={len(LDB)})",
                f"Average distribution (n={len(LDB)})"))

ax = axes[1, 1]
sns.heatmap(KTAU, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            vmin=0, vmax=1, ax=ax, square=True, cbar=False)
ax.set_title(tr("Kendall tau entre benchmarks", "Kendall tau between benchmarks"))

ax = axes[1, 2]
sns.heatmap(CLUSTERS, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar=False)
ax.set_title(tr(f"Perfil dos {len(CLUSTERS)} clusters",
                f"Profile of {len(CLUSTERS)} clusters"))

ax = axes[2, 0]
y = np.arange(len(GLOBAL_WR))
ax.errorbar(GLOBAL_WR["win_rate"] * 100, y,
            xerr=[(GLOBAL_WR["win_rate"] - GLOBAL_WR["wilson_lo"]) * 100,
                  (GLOBAL_WR["wilson_hi"] - GLOBAL_WR["win_rate"]) * 100],
            fmt="o", color="#d62728", capsize=3)
ax.set_yticks(y)
ax.set_yticklabels(GLOBAL_WR["model"], fontsize=8)
ax.axvline(50, ls="--", color="gray")
ax.set_title(tr("Arena - winrate global", "Arena - global win rate"))
ax.invert_yaxis()

ax = axes[2, 1]
y = np.arange(len(BT))
ax.errorbar(BT["bt_rating_elo"], y,
            xerr=[BT["bt_rating_elo"] - BT["bt_lo95"], BT["bt_hi95"] - BT["bt_rating_elo"]],
            fmt="o", color="#ff7f0e", capsize=3)
ax.set_yticks(y)
ax.set_yticklabels(BT["model"], fontsize=8)
ax.set_title(tr("Arena - Bradley-Terry", "Arena - Bradley-Terry"))
ax.invert_yaxis()

ax = axes[2, 2]
ax.scatter(CROSS["arena_wr"] * 100, CROSS["ldb_average"], s=180, color="#9467bd")
for _, r in CROSS.iterrows():
    ax.annotate(r["arena_model"], (r["arena_wr"] * 100, r["ldb_average"]),
                xytext=(5, 5), textcoords="offset points", fontsize=9)
ax.set_xlabel("Arena winrate (%)")
ax.set_ylabel("Open LLM average (%)")
ax.set_title("Cross-validation")

ax = axes[3, 0]
ax.bar(POWER["benchmark"], POWER["n_necessario"], color="#8c564b")
ax.set_yscale("log")
ax.set_title(tr("n necessario (poder 80%)", "required n (80% power)"))
ax.tick_params(axis="x", rotation=20)

ax = axes[3, 1]
ax.barh(org_top.index[::-1], org_top.values[::-1], color="#17becf")
ax.set_title(tr("Top 15 organizacoes (mediana)", "Top 15 organizations (median)"))

ax = axes[3, 2]
rb15 = RANKING_BOOT.head(15).iloc[::-1]
y = np.arange(len(rb15))
ax.errorbar(rb15["rank_mediano_boot"], y,
            xerr=[rb15["rank_mediano_boot"] - rb15["rank_lo95"],
                  rb15["rank_hi95"] - rb15["rank_mediano_boot"]],
            fmt="o", color="#1f77b4", capsize=3)
ax.set_yticks(y)
ax.set_yticklabels([n[:25] for n in rb15["fullname"]], fontsize=8)
ax.set_title(tr("Top-15 - incerteza da posicao", "Top-15 - position uncertainty"))
ax.invert_xaxis()

fig2.suptitle(tr("LLMs sob o microscopio - painel consolidado",
                 "LLMs under the microscope - consolidated panel"),
              fontsize=22, y=0.995)
plt.tight_layout()
plt.savefig(FIG / "10_painel_consolidado.png", dpi=130, bbox_inches="tight")
plt.close()

print("OK")
