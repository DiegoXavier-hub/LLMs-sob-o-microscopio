from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from i18n import tr, figures_dir, tables_dir

sns.set_theme(style="whitegrid", context="talk")

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
FIG = figures_dir(ROOT)
TAB = tables_dir(ROOT)
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

tidy = pd.read_parquet(IN / "leaderboard_tidy.parquet")
ldb = pd.read_parquet(IN / "leaderboard_valid.parquet")

bench = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]
label = {"ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5",
         "gpqa": "GPQA", "musr": "MuSR", "mmlu_pro": "MMLU-PRO"}

z_a = stats.norm.ppf(0.975)
z_b = stats.norm.ppf(0.80)

rows = []
for b in bench:
    n = int(tidy[tidy["benchmark"] == b]["n"].iloc[0])
    lo50, hi50 = proportion_confint(int(0.5 * n), n, alpha=0.05, method="wilson")
    lo40, hi40 = proportion_confint(int(0.4 * n), n, alpha=0.05, method="wilson")
    moe50 = (hi50 - lo50) / 2 * 100
    moe40 = (hi40 - lo40) / 2 * 100
    dmd50 = (z_a + z_b) * np.sqrt(2 * 0.5 * 0.5 / n) * 100
    dmd40 = (z_a + z_b) * np.sqrt(2 * 0.4 * 0.6 / n) * 100
    rows.append({
        "benchmark": label[b], "n_questoes": n,
        "MOE_p50": round(moe50, 2), "MOE_p40": round(moe40, 2),
        "DMD_p50": round(dmd50, 2), "DMD_p40": round(dmd40, 2),
    })

moe_df = pd.DataFrame(rows).sort_values("n_questoes")
moe_df.to_csv(TAB / "04_margem_erro_teorica.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(moe_df))
w = 0.38
ax.bar(x - w/2, moe_df["MOE_p50"], w, label="MOE 95% (p=0.5)", color="#1f77b4")
ax.bar(x + w/2, moe_df["DMD_p50"], w, label=tr("Dif. min. detectavel", "Min. detectable diff."), color="#d62728")
ax.set_xticks(x)
ax.set_xticklabels(moe_df["benchmark"], rotation=20)
ax.set_ylabel(tr("pontos percentuais", "percentage points"))
ax.set_title(tr("Margem de erro teorica por benchmark", "Theoretical margin of error by benchmark"))
for i, (m, d, n) in enumerate(zip(moe_df["MOE_p50"], moe_df["DMD_p50"], moe_df["n_questoes"])):
    ax.text(i - w/2, m + 0.05, f"{m:.1f}", ha="center", fontsize=10)
    ax.text(i + w/2, d + 0.05, f"{d:.1f}", ha="center", fontsize=10)
    ax.text(i, -0.6, f"n={n}", ha="center", fontsize=9, color="gray")
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "04_moe_teorica.png", dpi=140)
plt.close()

top100 = ldb.sort_values("average", ascending=False).head(100)
pud_rows = []
for b in bench:
    n = int(tidy[tidy["benchmark"] == b]["n"].iloc[0])
    scores = top100[b].dropna().values / 100
    total = under = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            p_avg = (scores[i] + scores[j]) / 2
            dmd = (z_a + z_b) * np.sqrt(2 * p_avg * (1 - p_avg) / n)
            if abs(scores[i] - scores[j]) < dmd:
                under += 1
            total += 1
    pud_rows.append({
        "benchmark": label[b],
        "pares_total": total,
        "pares_indistinguiveis": under,
        "pct": round(100 * under / total, 1),
    })

pud = pd.DataFrame(pud_rows)
pud.to_csv(TAB / "04_pares_indistinguiveis_top100.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(pud["benchmark"], pud["pct"], color="#9467bd")
for i, v in enumerate(pud["pct"]):
    ax.text(v + 0.5, i, f"{v}%", va="center")
ax.set_xlabel(tr("% de pares do top-100 estatisticamente indistinguiveis",
                 "% of top-100 pairs statistically indistinguishable"))
ax.set_title(tr("Quanto do ranking e ruido?", "How much of the ranking is noise?"))
ax.set_xlim(0, max(pud["pct"]) * 1.15 + 5)
plt.tight_layout()
plt.savefig(FIG / "04_pct_pares_indistinguiveis.png", dpi=140)
plt.close()

cov = tidy.groupby("benchmark").size().to_frame("n_modelos").reset_index()
cov["benchmark"] = cov["benchmark"].map(label)
cov.to_csv(TAB / "04_cobertura_por_benchmark.csv", index=False)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, b in zip(axes.flat, bench):
    s = tidy[tidy["benchmark"] == b]["score_pct"]
    ax.hist(s, bins=50, color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.axvline(s.median(), color="red", linestyle="--", label=f"{tr('mediana','median')}={s.median():.1f}")
    ax.axvline(s.mean(), color="black", linestyle=":", label=f"{tr('media','mean')}={s.mean():.1f}")
    ax.set_title(label[b])
    ax.set_xlabel("score (%)")
    ax.legend(fontsize=9)
fig.suptitle(tr("Distribuicao de scores por benchmark", "Score distribution by benchmark"), fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "04_distribuicoes_scores.png", dpi=140)
plt.close()

norm_rows = []
for b in bench:
    s = tidy[tidy["benchmark"] == b]["score_pct"].dropna().values
    stat, p = stats.normaltest(s)
    norm_rows.append({
        "benchmark": label[b], "n": len(s),
        "skew": round(stats.skew(s), 3),
        "kurtosis": round(stats.kurtosis(s), 3),
        "dagostino_stat": round(stat, 1),
        "p_value": f"{p:.2e}",
        "rejeita_normalidade": p < 0.05,
    })
pd.DataFrame(norm_rows).to_csv(TAB / "04_testes_normalidade.csv", index=False)

resumo = {
    "moe_max_pp": float(moe_df["MOE_p50"].max()),
    "moe_min_pp": float(moe_df["MOE_p50"].min()),
    "pct_pares_indistinguiveis_medio": float(pud["pct"].mean()),
}
(TAB / "resumo_04_audit.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(resumo)
