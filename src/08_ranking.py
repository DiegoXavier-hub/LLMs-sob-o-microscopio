from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from i18n import tr, figures_dir, tables_dir

sns.set_theme(style="whitegrid", context="talk")
rng = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
FIG = figures_dir(ROOT)
TAB = tables_dir(ROOT)

ldb = pd.read_parquet(IN / "leaderboard_valid.parquet")
tidy = pd.read_parquet(IN / "leaderboard_tidy.parquet")

bench = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]
label = {"ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5",
         "gpqa": "GPQA", "musr": "MuSR", "mmlu_pro": "MMLU-PRO"}
N_Q = {b: int(tidy[tidy["benchmark"] == b]["n"].iloc[0]) for b in bench}

top50 = ldb.sort_values("average", ascending=False).head(50).reset_index(drop=True)
M, B = len(top50), 2000

avg_samples = np.zeros((M, B))
for b in bench:
    n = N_Q[b]
    p = top50[b].values / 100
    avg_samples += rng.binomial(n, p[:, None], size=(M, B)) / n
avg_samples /= len(bench)
avg_ranks = M - np.argsort(np.argsort(avg_samples, axis=0), axis=0)

med = np.median(avg_ranks, axis=1)
lo = np.quantile(avg_ranks, 0.025, axis=1)
hi = np.quantile(avg_ranks, 0.975, axis=1)

df_rank = pd.DataFrame({
    "fullname": top50["fullname"],
    "average": top50["average"],
    "rank_observado": np.arange(1, M + 1),
    "rank_mediano_boot": med,
    "rank_lo95": lo,
    "rank_hi95": hi,
    "amplitude_95": hi - lo,
}).sort_values("rank_mediano_boot").reset_index(drop=True)
df_rank.to_csv(TAB / "08_ranking_bootstrap_top50.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 13))
plot_df = df_rank.head(30).iloc[::-1]
y = np.arange(len(plot_df))
ax.errorbar(plot_df["rank_mediano_boot"], y,
            xerr=[plot_df["rank_mediano_boot"] - plot_df["rank_lo95"],
                  plot_df["rank_hi95"] - plot_df["rank_mediano_boot"]],
            fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3)
ax.set_yticks(y)
ax.set_yticklabels([fn[:38] for fn in plot_df["fullname"]], fontsize=8)
ax.set_xlabel(tr("Posicao (bootstrap)", "Position (bootstrap)"))
ax.set_title(tr("Top-30 - posicao com IC 95% bootstrap",
                "Top-30 - position with 95% bootstrap CI"))
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(FIG / "08_ranking_bootstrap_top30.png", dpi=140)
plt.close()

df_sorted = df_rank.sort_values("rank_mediano_boot").reset_index(drop=True)
tiers = []
current = []
current_hi = None
for _, r in df_sorted.iterrows():
    if not current:
        current = [r["fullname"]]
        current_hi = r["rank_hi95"]
        continue
    if r["rank_lo95"] <= current_hi:
        current.append(r["fullname"])
        current_hi = max(current_hi, r["rank_hi95"])
    else:
        tiers.append(current)
        current = [r["fullname"]]
        current_hi = r["rank_hi95"]
if current:
    tiers.append(current)

tier_rows = [{"tier": ti, "fullname": m} for ti, members in enumerate(tiers, 1) for m in members]
tier_df = pd.DataFrame(tier_rows).merge(df_rank, on="fullname")
tier_df.to_csv(TAB / "08_tiers_estatisticos.csv", index=False)
tier_sizes = tier_df.groupby("tier").size()

fig, ax = plt.subplots(figsize=(13, 9))
palette = sns.color_palette("tab20", max(20, len(tiers)))
y = 0
for ti in sorted(tier_df["tier"].unique()):
    sub = tier_df[tier_df["tier"] == ti].sort_values("rank_mediano_boot")
    for _, r in sub.iterrows():
        ax.barh(y, r["rank_hi95"] - r["rank_lo95"] + 0.4, left=r["rank_lo95"] - 0.2,
                color=palette[(ti - 1) % len(palette)], alpha=0.7,
                edgecolor="black", linewidth=0.3)
        ax.text(r["rank_mediano_boot"], y, f" {r['fullname'][:32]}", va="center", fontsize=7)
        y += 1
    y += 0.5
ax.set_xlabel(tr("Posicao (1 = melhor)", "Position (1 = best)"))
ax.set_title(tr(f"Tiers estatisticos do top-50 - {len(tiers)} grupos",
                f"Statistical tiers of top-50 - {len(tiers)} groups"))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG / "08_tiers_swimlane.png", dpi=140)
plt.close()

top200 = ldb.sort_values("average", ascending=False).head(200)
ktau = pd.DataFrame(
    np.eye(len(bench)),
    index=[label[b] for b in bench],
    columns=[label[b] for b in bench],
)
for i, j in itertools.combinations(range(len(bench)), 2):
    a, b = bench[i], bench[j]
    sub = top200[[a, b]].dropna()
    tau, _ = stats.kendalltau(sub[a], sub[b])
    ktau.iloc[i, j] = tau
    ktau.iloc[j, i] = tau
ktau.to_csv(TAB / "08_kendall_tau_benchmarks.csv")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(ktau, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            vmin=0, vmax=1, ax=ax, square=True)
ax.set_title(tr("Kendall tau entre os 6 benchmarks (top-200)",
                "Kendall tau among the 6 benchmarks (top-200)"))
plt.tight_layout()
plt.savefig(FIG / "08_kendall_tau_benchmarks.png", dpi=140)
plt.close()

weights = {}
for b in bench:
    n = N_Q[b]
    p_med = top200[b].mean() / 100
    weights[b] = 1 / (p_med * (1 - p_med) / n)
w_total = sum(weights.values())
weights = {k: v / w_total for k, v in weights.items()}

ldb["score_composito"] = sum(ldb[b] * weights[b] for b in bench)
top_comp = ldb.sort_values("score_composito", ascending=False).head(30)[
    ["fullname", "average", "score_composito"] + bench
]
top_comp.to_csv(TAB / "08_top30_score_composito.csv", index=False)

top100_simple = ldb.sort_values("average", ascending=False).head(100)["fullname"].tolist()
top100_comp = ldb.sort_values("score_composito", ascending=False).head(100)["fullname"].tolist()
overlap = len(set(top100_simple) & set(top100_comp))

resumo = {
    "n_tiers": len(tiers),
    "maior_tier_size": int(tier_sizes.max()),
    "tier1_size": int(tier_sizes.iloc[0]),
    "tier1_membros": tiers[0][:5],
    "kendall_tau_mediano": float(
        ktau.where(np.triu(np.ones(ktau.shape, dtype=bool), k=1)).stack().median()
    ),
    "pesos_composito": {label[k]: round(v, 4) for k, v in weights.items()},
    "overlap_top100": overlap,
}
(TAB / "resumo_08_ranking.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(resumo)
