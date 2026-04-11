from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from i18n import tr, figures_dir, tables_dir

sns.set_theme(style="whitegrid", context="talk")
rng = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
FIG = figures_dir(ROOT)
TAB = tables_dir(ROOT)

tidy = pd.read_parquet(IN / "leaderboard_tidy.parquet")
ldb = pd.read_parquet(IN / "leaderboard_valid.parquet")

bench = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]
label = {"ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5",
         "gpqa": "GPQA", "musr": "MuSR", "mmlu_pro": "MMLU-PRO"}
N_Q = {b: int(tidy[tidy["benchmark"] == b]["n"].iloc[0]) for b in bench}


def cohens_h(p1, p2):
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


top200 = ldb.sort_values("average", ascending=False).head(200).copy()
top200["organizacao"] = top200["fullname"].astype(str).str.split("/").str[0]

ci_rows = []
for _, row in top200.iterrows():
    for b in bench:
        if pd.isna(row[b]): continue
        n = N_Q[b]
        p = row[b] / 100
        lo, hi = proportion_confint(int(round(p * n)), n, alpha=0.05, method="wilson")
        ci_rows.append({
            "fullname": row["fullname"], "organizacao": row["organizacao"],
            "benchmark": b, "n": n, "p_hat": p,
            "ci_lo": lo, "ci_hi": hi, "ci_width": hi - lo,
        })
pd.DataFrame(ci_rows).to_parquet(IN / "wilson_ci_top200.parquet", index=False)

B = 2000
top30 = ldb.sort_values("average", ascending=False).head(30).copy()
boot_rows = []
for _, row in top30.iterrows():
    for b in bench:
        if pd.isna(row[b]): continue
        n = N_Q[b]
        p = row[b] / 100
        k = int(round(p * n))
        x = np.concatenate([np.ones(k), np.zeros(n - k)])
        means = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        boot_rows.append({
            "fullname": row["fullname"], "benchmark": b, "n": n,
            "p_hat": p, "boot_lo": lo, "boot_hi": hi, "boot_se": means.std(),
        })
boot = pd.DataFrame(boot_rows)
boot.to_parquet(IN / "bootstrap_top30.parquet", index=False)


def forest(subset, color, title, fname):
    subset = subset.sort_values("p_hat").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 12))
    y = np.arange(len(subset))
    ax.errorbar(
        subset["p_hat"] * 100, y,
        xerr=[(subset["p_hat"] - subset["boot_lo"]) * 100,
              (subset["boot_hi"] - subset["p_hat"]) * 100],
        fmt="o", color=color, ecolor=color, alpha=0.7, capsize=3,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([fn[:40] for fn in subset["fullname"]], fontsize=8)
    ax.set_xlabel(tr("score (%) com IC 95% bootstrap", "score (%) with 95% bootstrap CI"))
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=140)
    plt.close()


forest(
    boot[boot["benchmark"] == "mmlu_pro"].merge(top30[["fullname"]], on="fullname"),
    "#1f77b4",
    tr("Forest plot - top 30 em MMLU-PRO", "Forest plot - top 30 in MMLU-PRO"),
    "06_forest_mmlu_pro.png",
)
forest(
    boot[boot["benchmark"] == "gpqa"].merge(top30[["fullname"]], on="fullname"),
    "#d62728",
    tr("Forest plot - top 30 em GPQA (n=448)", "Forest plot - top 30 in GPQA (n=448)"),
    "06_forest_gpqa.png",
)

top50 = ldb.sort_values("average", ascending=False).head(50).reset_index(drop=True)
pair_rows = []
for b in bench:
    n = N_Q[b]
    scores = top50[b].values / 100
    names = top50["fullname"].values
    for i, j in itertools.combinations(range(len(scores)), 2):
        p1, p2 = scores[i], scores[j]
        if np.isnan(p1) or np.isnan(p2): continue
        k1, k2 = int(round(p1 * n)), int(round(p2 * n))
        try:
            stat, pv = proportions_ztest([k1, k2], [n, n], alternative="two-sided")
        except Exception:
            stat, pv = np.nan, 1.0
        h = cohens_h(p1, p2)
        pair_rows.append({
            "benchmark": b, "model_a": names[i], "model_b": names[j],
            "p_a": p1, "p_b": p2, "delta_pp": (p1 - p2) * 100,
            "z": stat, "pvalue": pv, "cohens_h": h, "abs_h": abs(h),
        })

pairs = pd.DataFrame(pair_rows)
pairs["pvalue_fdr"] = np.nan
pairs["sig_fdr"] = False
for b, sub in pairs.groupby("benchmark"):
    rej, padj, _, _ = multipletests(sub["pvalue"].fillna(1).values, alpha=0.05, method="fdr_bh")
    pairs.loc[sub.index, "pvalue_fdr"] = padj
    pairs.loc[sub.index, "sig_fdr"] = rej
pairs.to_parquet(IN / "pares_top50_testes.parquet", index=False)

resumo_pares = pairs.groupby("benchmark").agg(
    n_pares=("pvalue", "size"),
    pct_sig_brut=("pvalue", lambda x: float((x < 0.05).mean() * 100)),
    pct_sig_fdr=("sig_fdr", lambda x: float(x.mean() * 100)),
    cohens_h_mediano=("abs_h", "median"),
).round(2)
resumo_pares.index = resumo_pares.index.map(label)
resumo_pares.to_csv(TAB / "06_pares_top50_resumo.csv")

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(resumo_pares))
w = 0.35
ax.bar(x - w/2, resumo_pares["pct_sig_brut"], w, label=tr("Brutos (alpha=0.05)", "Raw (alpha=0.05)"), color="#1f77b4")
ax.bar(x + w/2, resumo_pares["pct_sig_fdr"], w, label=tr("Apos FDR-BH", "After FDR-BH"), color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(resumo_pares.index, rotation=15)
ax.set_ylabel(tr("% de pares significativos", "% significant pairs"))
ax.set_title(tr("Top-50 - significancia antes e depois da correcao FDR",
                "Top-50 - significance before and after FDR correction"))
ax.legend()
for i, (a, c) in enumerate(zip(resumo_pares["pct_sig_brut"], resumo_pares["pct_sig_fdr"])):
    ax.text(i - w/2, a + 0.5, f"{a:.0f}%", ha="center", fontsize=10)
    ax.text(i + w/2, c + 0.5, f"{c:.0f}%", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "06_pct_significativos_fdr.png", dpi=140)
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, b in zip(axes.flat, bench):
    h = pairs.loc[pairs["benchmark"] == b, "abs_h"].values
    ax.hist(h, bins=50, color="#9467bd", alpha=0.85)
    ax.axvline(0.2, color="orange", ls="--", label=tr("h=0.20 (pequeno)", "h=0.20 (small)"))
    ax.axvline(0.5, color="red", ls="--", label=tr("h=0.50 (medio)", "h=0.50 (medium)"))
    ax.set_title(f"{label[b]}  |  med={np.median(h):.3f}")
    ax.set_xlabel("|Cohen's h|")
    ax.legend(fontsize=8)
fig.suptitle(tr("Tamanho de efeito - pares top-50", "Effect size - top-50 pairs"))
plt.tight_layout()
plt.savefig(FIG / "06_cohens_h.png", dpi=140)
plt.close()

top15_names = top50["fullname"].head(15).tolist()
sub_pv = pairs[(pairs["benchmark"] == "mmlu_pro")
               & pairs["model_a"].isin(top15_names)
               & pairs["model_b"].isin(top15_names)]
mat = pd.DataFrame(np.eye(15), index=top15_names, columns=top15_names)
for _, r in sub_pv.iterrows():
    mat.loc[r["model_a"], r["model_b"]] = r["pvalue_fdr"]
    mat.loc[r["model_b"], r["model_a"]] = r["pvalue_fdr"]
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(mat, annot=True, fmt=".2f", cmap="RdYlGn", center=0.05,
            vmin=0, vmax=0.5, ax=ax,
            xticklabels=[n[:25] for n in top15_names],
            yticklabels=[n[:25] for n in top15_names])
ax.set_title(tr("p-valores ajustados (FDR) - top 15 em MMLU-PRO",
                "Adjusted p-values (FDR) - top 15 in MMLU-PRO"))
plt.tight_layout()
plt.savefig(FIG / "06_heatmap_pvalores_top15.png", dpi=140)
plt.close()

power = NormalIndPower()
power_rows = []
for b in bench:
    s = top50[b].values / 100
    if np.isnan(s[0]) or np.isnan(s[9]): continue
    delta = abs(s[0] - s[9])
    if delta == 0: continue
    h = cohens_h(s[0], s[9])
    if h == 0: continue
    n_needed = power.solve_power(effect_size=abs(h), alpha=0.05, power=0.8, alternative="two-sided")
    power_rows.append({
        "benchmark": label[b],
        "p_top1": s[0], "p_top10": s[9],
        "delta_pp": delta * 100,
        "cohens_h": h,
        "n_atual": N_Q[b],
        "n_necessario": int(np.ceil(n_needed)) if not np.isnan(n_needed) else None,
    })
power_df = pd.DataFrame(power_rows)
power_df.to_csv(TAB / "06_analise_poder.csv", index=False)

B2 = 1000
king_rows = []
for b in bench:
    n = N_Q[b]
    sub = top50[["fullname", b]].dropna()
    names = sub["fullname"].values
    p = sub[b].values / 100
    samples = rng.binomial(n, p[:, None], size=(len(p), B2)) / n
    winner_idx = np.argmax(samples, axis=0)
    counts = pd.Series(winner_idx).value_counts().head(10)
    for idx, cnt in counts.items():
        king_rows.append({
            "benchmark": label[b],
            "fullname": names[idx],
            "p_hat": p[idx],
            "vezes_top1": int(cnt),
            "prob_top1_pct": round(100 * cnt / B2, 1),
        })
king = pd.DataFrame(king_rows)
king.to_csv(TAB / "06_rei_do_hill.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, b in zip(axes.flat, bench):
    sub = king[king["benchmark"] == label[b]].head(7)
    ax.barh(range(len(sub)), sub["prob_top1_pct"], color=sns.color_palette("Set2")[0])
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([n[:30] for n in sub["fullname"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("P(top-1) %")
    ax.set_title(label[b])
fig.suptitle(tr("Rei do hill - P(modelo = #1) sob reamostragem",
                "King of the hill - P(model = #1) under resampling"))
plt.tight_layout()
plt.savefig(FIG / "06_rei_do_hill.png", dpi=140)
plt.close()

resumo = {
    "n_pares_por_bench": int(len(top50) * (len(top50) - 1) / 2),
    "pct_sig_brut_medio": float(resumo_pares["pct_sig_brut"].mean()),
    "pct_sig_fdr_medio": float(resumo_pares["pct_sig_fdr"].mean()),
    "n_necessario_top1_vs_top10": power_df.set_index("benchmark")["n_necessario"].to_dict(),
}
(TAB / "resumo_06_inference.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(resumo)
