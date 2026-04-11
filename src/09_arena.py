from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
from i18n import tr, figures_dir, tables_dir

sns.set_theme(style="whitegrid", context="talk")
rng = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed" / "arena"
LDB = ROOT / "data" / "processed" / "leaderboard_valid.parquet"
FIG = figures_dir(ROOT)
TAB = tables_dir(ROOT)

battles = pd.read_parquet(IN / "arena_battles.parquet")
decisive = battles[battles["winner"].isin(["model_a", "model_b"])].copy()

MODELS = sorted(pd.concat([battles["model_a"], battles["model_b"]]).unique())
idx = {m: i for i, m in enumerate(MODELS)}
K = len(MODELS)

W = np.zeros((K, K))
for _, r in decisive.iterrows():
    a, b = idx[r["model_a"]], idx[r["model_b"]]
    if r["winner"] == "model_a":
        W[a, b] += 1
    else:
        W[b, a] += 1

N = W + W.T
winrate = np.where(N > 0, W / np.where(N == 0, 1, N), np.nan)
wr_df = pd.DataFrame(winrate, index=MODELS, columns=MODELS)
wr_df.to_csv(TAB / "09_matriz_winrate.csv")

global_wins = np.nansum(W, axis=1)
global_n = np.nansum(N, axis=1)
global_wr = global_wins / np.where(global_n == 0, 1, global_n)
wilson = [proportion_confint(int(w), int(n), alpha=0.05, method="wilson") if n > 0 else (np.nan, np.nan)
          for w, n in zip(global_wins, global_n)]
global_df = pd.DataFrame({
    "model": MODELS,
    "win_rate": global_wr,
    "n_battles": global_n.astype(int),
    "wilson_lo": [x[0] for x in wilson],
    "wilson_hi": [x[1] for x in wilson],
}).sort_values("win_rate", ascending=False).reset_index(drop=True)
global_df.to_csv(TAB / "09_winrate_global.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 9))
y = np.arange(len(global_df))
ax.errorbar(global_df["win_rate"] * 100, y,
            xerr=[(global_df["win_rate"] - global_df["wilson_lo"]) * 100,
                  (global_df["wilson_hi"] - global_df["win_rate"]) * 100],
            fmt="o", color="#1f77b4", ecolor="#1f77b4", capsize=3)
ax.set_yticks(y)
ax.set_yticklabels(global_df["model"])
ax.set_xlabel(tr("Taxa de vitoria global (%) - IC Wilson 95%",
                 "Global win rate (%) - 95% Wilson CI"))
ax.axvline(50, ls="--", color="red", alpha=0.6, label="50%")
ax.set_title(tr(f"Arena - {len(decisive):,} batalhas decisivas, {K} modelos",
                f"Arena - {len(decisive):,} decisive battles, {K} models"))
ax.invert_yaxis()
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "09_winrate_global.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(wr_df.round(2), annot=True, fmt=".2f", cmap="RdYlGn", center=0.5,
            vmin=0, vmax=1, ax=ax, square=True,
            cbar_kws={"label": tr("P(linha vence coluna)", "P(row beats column)")})
ax.set_title(tr("Matriz de winrate pareada", "Pairwise win-rate matrix"))
plt.tight_layout()
plt.savefig(FIG / "09_matriz_winrate.png", dpi=140)
plt.close()


def bt_fit(W_mat, max_iter=500, tol=1e-8):
    K = W_mat.shape[0]
    p = np.ones(K)
    W_row = W_mat.sum(axis=1)
    for _ in range(max_iter):
        p_new = np.zeros(K)
        for i in range(K):
            denom = 0
            for j in range(K):
                if i == j:
                    continue
                n_ij = W_mat[i, j] + W_mat[j, i]
                if n_ij == 0:
                    continue
                denom += n_ij / (p[i] + p[j])
            if denom > 0:
                p_new[i] = W_row[i] / denom
        p_new = p_new / p_new.sum() * K
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            break
        p = p_new
    return p


p_hat = bt_fit(W)
bt_rating = np.log(p_hat) * (400 / np.log(10)) + 1000
bt_df = pd.DataFrame({
    "model": MODELS,
    "bt_strength": p_hat,
    "bt_rating_elo": bt_rating,
}).sort_values("bt_rating_elo", ascending=False).reset_index(drop=True)

B = 500
decisive_arr = decisive[["model_a", "model_b", "winner"]].values
boot_ratings = np.zeros((B, K))
n_dec = len(decisive_arr)
for b in range(B):
    samp = decisive_arr[rng.integers(0, n_dec, size=n_dec)]
    W_b = np.zeros((K, K))
    for row in samp:
        ai, bi = idx[row[0]], idx[row[1]]
        if row[2] == "model_a":
            W_b[ai, bi] += 1
        else:
            W_b[bi, ai] += 1
    try:
        p_b = bt_fit(W_b, max_iter=200, tol=1e-6)
        boot_ratings[b] = np.log(p_b) * (400 / np.log(10)) + 1000
    except Exception:
        boot_ratings[b] = np.nan

lo = np.nanquantile(boot_ratings, 0.025, axis=0)
hi = np.nanquantile(boot_ratings, 0.975, axis=0)
bt_df["bt_lo95"] = [lo[idx[m]] for m in bt_df["model"]]
bt_df["bt_hi95"] = [hi[idx[m]] for m in bt_df["model"]]
bt_df.to_csv(TAB / "09_bradley_terry.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 9))
y = np.arange(len(bt_df))
ax.errorbar(bt_df["bt_rating_elo"], y,
            xerr=[bt_df["bt_rating_elo"] - bt_df["bt_lo95"],
                  bt_df["bt_hi95"] - bt_df["bt_rating_elo"]],
            fmt="o", color="#d62728", ecolor="#d62728", capsize=3)
ax.set_yticks(y)
ax.set_yticklabels(bt_df["model"])
ax.set_xlabel(tr("Rating (escala Elo, IC 95% bootstrap)",
                 "Rating (Elo scale, 95% bootstrap CI)"))
ax.set_title(tr("Bradley-Terry dos 20 modelos da Arena", "Bradley-Terry for 20 Arena models"))
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG / "09_bradley_terry.png", dpi=140)
plt.close()


def elo_run(arr, K_factor=16, init=1000):
    r = {m: init for m in MODELS}
    for row in arr:
        ma, mb, w = row[0], row[1], row[2]
        ea = 1 / (1 + 10 ** ((r[mb] - r[ma]) / 400))
        sa = 1 if w == "model_a" else 0
        r[ma] += K_factor * (sa - ea)
        r[mb] += K_factor * ((1 - sa) - (1 - ea))
    return r


elos = [elo_run(decisive_arr[rng.permutation(len(decisive_arr))]) for _ in range(10)]
elo_df = pd.DataFrame(elos)
elo_summary = pd.DataFrame({
    "model": elo_df.columns,
    "elo_mean": elo_df.mean(),
    "elo_std": elo_df.std(),
    "elo_lo95": elo_df.quantile(0.025),
    "elo_hi95": elo_df.quantile(0.975),
}).sort_values("elo_mean", ascending=False).reset_index(drop=True)
elo_summary.to_csv(TAB / "09_elo_classico.csv", index=False)

pair_rows = []
for i, j in itertools.combinations(range(K), 2):
    n = int(N[i, j])
    if n < 20:
        continue
    wi, wj = int(W[i, j]), int(W[j, i])
    res = stats.binomtest(wi, n, p=0.5, alternative="two-sided")
    pair_rows.append({
        "model_a": MODELS[i], "model_b": MODELS[j],
        "n": n, "wins_a": wi, "wins_b": wj,
        "winrate_a": wi / n,
        "pvalue": res.pvalue,
    })
pair_df = pd.DataFrame(pair_rows)
rej, padj, _, _ = multipletests(pair_df["pvalue"].values, alpha=0.05, method="fdr_bh")
pair_df["pvalue_fdr"] = padj
pair_df["sig_fdr"] = rej
pair_df.to_csv(TAB / "09_pares_arena_fdr.csv", index=False)

pmat = pd.DataFrame(np.eye(K), index=MODELS, columns=MODELS)
for _, r in pair_df.iterrows():
    pmat.loc[r["model_a"], r["model_b"]] = r["pvalue_fdr"]
    pmat.loc[r["model_b"], r["model_a"]] = r["pvalue_fdr"]
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(pmat, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=0.2,
            ax=ax, square=True, cbar_kws={"label": tr("p-valor FDR", "FDR p-value")})
ax.set_title(tr("p-valores FDR - H0: A = B", "FDR p-values - H0: A = B"))
plt.tight_layout()
plt.savefig(FIG / "09_heatmap_pvalores.png", dpi=140)
plt.close()


def rank_por_sub(sub):
    W_s = np.zeros((K, K))
    for _, r in sub.iterrows():
        a, b = idx[r["model_a"]], idx[r["model_b"]]
        if r["winner"] == "model_a":
            W_s[a, b] += 1
        elif r["winner"] == "model_b":
            W_s[b, a] += 1
    tot = W_s + W_s.T
    wr = np.nansum(W_s, axis=1) / np.where(np.nansum(tot, axis=1) == 0, 1, np.nansum(tot, axis=1))
    return pd.Series(wr, index=MODELS)


langs = ["English", "Portuguese", "German", "Spanish"]
rank_per_lang = {}
for lang in langs:
    sub = decisive[decisive["language"] == lang]
    if len(sub) > 100:
        rank_per_lang[lang] = rank_por_sub(sub).rank(ascending=False)
rank_per_lang["Global"] = rank_por_sub(decisive).rank(ascending=False)
rank_lang_df = pd.DataFrame(rank_per_lang)
rank_lang_df.to_csv(TAB / "09_rank_por_lingua.csv")

tau_rows = []
for a, b in itertools.combinations(rank_lang_df.columns, 2):
    sub = rank_lang_df[[a, b]].dropna()
    if len(sub) > 3:
        tau, pv = stats.kendalltau(sub[a], sub[b])
        tau_rows.append({"lang_a": a, "lang_b": b, "kendall_tau": tau, "p": pv, "n": len(sub)})
tau_df = pd.DataFrame(tau_rows)
tau_df.to_csv(TAB / "09_kendall_tau_linguas.csv", index=False)

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(rank_lang_df.sort_values("Global"), annot=True, fmt=".0f",
            cmap="YlOrRd_r", ax=ax, cbar_kws={"label": tr("posicao", "rank")})
ax.set_title(tr("Ranking por lingua", "Ranking by language"))
plt.tight_layout()
plt.savefig(FIG / "09_rank_por_lingua.png", dpi=140)
plt.close()

top_judges = battles["judge"].value_counts().head(10).index.tolist()
rank_per_judge = {}
for j in top_judges:
    sub = decisive[decisive["judge"] == j]
    if len(sub) >= 30:
        rank_per_judge[j] = rank_por_sub(sub).rank(ascending=False)
rank_j_df = pd.DataFrame(rank_per_judge)
rank_j_df.to_csv(TAB / "09_rank_por_juiz.csv")

tau_judge = []
for a, b in itertools.combinations(rank_j_df.columns, 2):
    s = rank_j_df[[a, b]].dropna()
    if len(s) > 3:
        tau, _ = stats.kendalltau(s[a], s[b])
        tau_judge.append(tau)

ldb = pd.read_parquet(LDB)
overlap_map = {
    "vicuna-13b": "lmsys/vicuna-13b-v1.3",
    "wizardlm-13b": "WizardLMTeam/WizardLM-13B-V1.0",
    "dolly-v2-12b": "databricks/dolly-v2-12b",
}
cross_rows = []
for am, lm in overlap_map.items():
    ldb_row = ldb[ldb["fullname"] == lm]
    arena_row = global_df[global_df["model"] == am]
    if len(ldb_row) == 0 or len(arena_row) == 0:
        continue
    cross_rows.append({
        "arena_model": am,
        "ldb_model": lm,
        "arena_wr": float(arena_row["win_rate"].iloc[0]),
        "arena_n": int(arena_row["n_battles"].iloc[0]),
        "ldb_average": float(ldb_row["average"].iloc[0]),
    })
cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(TAB / "09_cross_validation.csv", index=False)

sig_brut = int((pair_df["pvalue"] < 0.05).sum())
sig_fdr = int(pair_df["sig_fdr"].sum())
total = len(pair_df)

resumo = {
    "n_batalhas": int(len(battles)),
    "n_decisivas": int(len(decisive)),
    "n_modelos": K,
    "top3_bt": bt_df.head(3)[["model", "bt_rating_elo"]].to_dict("records"),
    "pares_n_ge_20": total,
    "pct_sig_brut": round(100 * sig_brut / total, 1),
    "pct_sig_fdr": round(100 * sig_fdr / total, 1),
    "kendall_tau_medio_juizes": float(np.nanmean(tau_judge)) if tau_judge else None,
    "cross_validation_overlap": int(len(cross_df)),
}
(TAB / "resumo_09_arena.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
)
print(resumo)
