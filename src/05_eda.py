from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from i18n import tr, figures_dir, tables_dir

sns.set_theme(style="whitegrid", context="talk")

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
FIG = figures_dir(ROOT)
TAB = tables_dir(ROOT)

ldb = pd.read_parquet(IN / "leaderboard_valid.parquet")
ldb["organizacao"] = ldb["fullname"].astype(str).str.split("/").str[0]

bench = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]
label = {"ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5",
         "gpqa": "GPQA", "musr": "MuSR", "mmlu_pro": "MMLU-PRO"}

desc = ldb[bench + ["average"]].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(2)
desc.to_csv(TAB / "05_sumario_estatistico.csv")

fig, ax = plt.subplots(figsize=(13, 7))
data = [ldb[c].dropna().values for c in bench]
parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.85)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(sns.color_palette("Set2")[i])
    pc.set_alpha(0.75)
ax.set_xticks(range(1, len(bench) + 1))
ax.set_xticklabels([label[b] for b in bench], rotation=15)
ax.set_ylabel("Score (%)")
ax.set_title(tr(f"Distribuicao de scores - {len(ldb)} modelos",
                f"Score distribution - {len(ldb)} models"))
plt.tight_layout()
plt.savefig(FIG / "05_violino_benchmarks.png", dpi=140)
plt.close()

corr_s = ldb[bench].corr(method="spearman")
corr_p = ldb[bench].corr(method="pearson")
corr_s.to_csv(TAB / "05_correlacao_spearman.csv")
corr_p.to_csv(TAB / "05_correlacao_pearson.csv")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, c, t in zip(axes, [corr_s, corr_p], ["Spearman", "Pearson"]):
    sns.heatmap(c, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                vmin=-0.1, vmax=1, ax=ax, square=True,
                xticklabels=[label[b] for b in c.columns],
                yticklabels=[label[b] for b in c.index],
                cbar_kws={"shrink": 0.7})
    ax.set_title(t)
fig.suptitle(tr("Correlacao inter-benchmark", "Inter-benchmark correlation"), fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "05_correlacao_benchmarks.png", dpi=140)
plt.close()

sub = ldb[ldb["params_b"].notna() & (ldb["params_b"] > 0)]
fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(sub["params_b"], sub["average"], c=sub["average"],
                cmap="viridis", s=18, alpha=0.6, edgecolor="none")
ax.set_xscale("log")
ax.set_xlabel(tr("# parametros (B) - escala log", "# parameters (B) - log scale"))
ax.set_ylabel("Average (%)")
ax.set_title(tr("Average vs #parametros", "Average vs #parameters"))
fig.colorbar(sc, ax=ax, label="average")
rho, pv = stats.spearmanr(np.log(sub["params_b"]), sub["average"])
ax.text(0.02, 0.95, f"Spearman log(params)~average: rho={rho:.2f} (p={pv:.1e})",
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(facecolor="white", alpha=0.8))
plt.tight_layout()
plt.savefig(FIG / "05_average_vs_params.png", dpi=140)
plt.close()

org_grp = ldb.groupby("organizacao").agg(
    n=("fullname", "count"),
    avg_med=("average", "median"),
    avg_max=("average", "max"),
)
org_top = org_grp[org_grp["n"] >= 5].sort_values("avg_med", ascending=False).head(15)
org_top.to_csv(TAB / "05_top15_organizacoes.csv")

fig, ax = plt.subplots(figsize=(11, 8))
order = org_top.sort_values("avg_med").index
ax.barh(order, org_top.loc[order, "avg_med"], color="#1f77b4")
ax.scatter(org_top.loc[order, "avg_max"], range(len(order)), color="red", zorder=3,
           label=tr("maximo", "max"))
for i, o in enumerate(order):
    ax.text(org_top.loc[o, "avg_med"] + 0.3, i, f"n={int(org_top.loc[o, 'n'])}",
            va="center", fontsize=9)
ax.set_xlabel("Average (%)")
ax.set_title(tr("Top 15 organizacoes por mediana", "Top 15 organizations by median"))
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(FIG / "05_top15_organizacoes.png", dpi=140)
plt.close()

top30 = ldb.sort_values("average", ascending=False).head(30).set_index("fullname")
mat = top30[bench].copy()
mat.columns = [label[c] for c in mat.columns]
fig, ax = plt.subplots(figsize=(11, 14))
sns.heatmap(mat, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "score (%)"})
ax.set_title(tr("Top 30 modelos × benchmarks", "Top 30 models × benchmarks"))
plt.tight_layout()
plt.savefig(FIG / "05_heatmap_top30.png", dpi=140)
plt.close()

top200 = ldb.sort_values("average", ascending=False).head(200)[bench].copy()
top200.columns = [label[c] for c in bench]
g = sns.pairplot(top200, diag_kind="kde", plot_kws={"s": 12, "alpha": 0.5}, height=2.0)
g.fig.suptitle(tr("Relacoes pareadas - top 200", "Pairwise relationships - top 200"), y=1.02)
plt.savefig(FIG / "05_pairplot_top200.png", dpi=140, bbox_inches="tight")
plt.close()

order = ["<3B", "3-8B", "8-14B", "14-35B", "35-80B", ">=80B", "desconhecido"]
def bucket(p):
    if pd.isna(p): return "desconhecido"
    if p < 3:   return "<3B"
    if p < 8:   return "3-8B"
    if p < 14:  return "8-14B"
    if p < 35:  return "14-35B"
    if p < 80:  return "35-80B"
    return ">=80B"

ldb["tam_bucket"] = pd.Categorical(ldb["params_b"].apply(bucket), categories=order, ordered=True)
fig, ax = plt.subplots(figsize=(13, 7))
sns.boxplot(data=ldb, x="tam_bucket", y="average", order=order, palette="Set3", ax=ax)
ax.set_xlabel(tr("Bucket de #parametros", "#parameters bucket"))
ax.set_ylabel("Average (%)")
ax.set_title(tr("Average por faixa de tamanho", "Average by size bucket"))
plt.tight_layout()
plt.savefig(FIG / "05_average_por_tamanho.png", dpi=140)
plt.close()

print("OK")
