from pathlib import Path
from math import pi
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import xgboost as xgb
from i18n import tr, figures_dir, tables_dir

sns.set_theme(style="whitegrid", context="talk")
rng = np.random.default_rng(42)

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
FIG = figures_dir(ROOT)
TAB = tables_dir(ROOT)

ldb = pd.read_parquet(IN / "leaderboard_valid.parquet")
ldb["organizacao"] = ldb["fullname"].astype(str).str.split("/").str[0]
ldb = ldb[ldb["params_b"].notna() & (ldb["params_b"] > 0)].copy()
ldb["log_params"] = np.log10(ldb["params_b"])

bench = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]
label = {"ifeval": "IFEval", "bbh": "BBH", "math": "MATH Lvl 5",
         "gpqa": "GPQA", "musr": "MuSR", "mmlu_pro": "MMLU-PRO"}

top_orgs = ldb["organizacao"].value_counts().head(30).index
ldb["org_top"] = ldb["organizacao"].where(ldb["organizacao"].isin(top_orgs), other="OUTRA")

num_features = ["log_params"]
cat_features = ["type", "architecture", "precision", "org_top"]
X = ldb[num_features + cat_features].fillna("desconhecido")
y = ldb["average"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_features),
])

modelos = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=14, n_jobs=-1, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=42),
    "XGBoost": xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, verbosity=0,
    ),
}

resultados = []
pipe_xgb = None
for nome, mdl in modelos.items():
    pipe = Pipeline([("pre", pre), ("mdl", mdl)])
    pipe.fit(X_tr, y_tr)
    pred_te = pipe.predict(X_te)
    pred_tr = pipe.predict(X_tr)
    resultados.append({
        "modelo": nome,
        "rmse_test": float(np.sqrt(mean_squared_error(y_te, pred_te))),
        "mae_test": float(mean_absolute_error(y_te, pred_te)),
        "r2_test": float(r2_score(y_te, pred_te)),
        "r2_train": float(r2_score(y_tr, pred_tr)),
    })
    if nome == "XGBoost":
        pipe_xgb = pipe
        pred_xgb = pred_te

res_df = pd.DataFrame(resultados).sort_values("r2_test", ascending=False)
res_df.to_csv(TAB / "07_regressao_metricas.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(res_df["modelo"], res_df["r2_test"], color=sns.color_palette("Set2"))
for i, v in enumerate(res_df["r2_test"]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
ax.set_ylabel(tr("R2 (teste)", "R2 (test)"))
ax.set_title(tr("Quanto do average e explicado so por metadados?",
                "How much of average is explained by metadata only?"))
ax.set_ylim(0, max(res_df["r2_test"]) * 1.15)
plt.tight_layout()
plt.savefig(FIG / "07_r2_por_modelo.png", dpi=140)
plt.close()

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(y_te, pred_xgb, alpha=0.4, s=20, color="#1f77b4")
mn, mx = min(y_te.min(), pred_xgb.min()), max(y_te.max(), pred_xgb.max())
ax.plot([mn, mx], [mn, mx], "k--", label="ideal")
ax.set_xlabel("Average real (%)")
ax.set_ylabel(tr("Average predito (%)", "Predicted average (%)"))
ax.set_title(tr(f"XGBoost - predito vs real (R2={r2_score(y_te, pred_xgb):.3f})",
                f"XGBoost - predicted vs actual (R2={r2_score(y_te, pred_xgb):.3f})"))
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "07_pred_vs_real_xgb.png", dpi=140)
plt.close()

xgb_mdl = pipe_xgb.named_steps["mdl"]
ohe = pipe_xgb.named_steps["pre"].named_transformers_["cat"]
feat_names = num_features + list(ohe.get_feature_names_out(cat_features))
imps = pd.Series(xgb_mdl.feature_importances_, index=feat_names).sort_values(ascending=False).head(20)
imps.to_csv(TAB / "07_xgb_importancias.csv")

fig, ax = plt.subplots(figsize=(11, 9))
imps.iloc[::-1].plot.barh(ax=ax, color="#2ca02c")
ax.set_xlabel(tr("Ganho XGBoost", "XGBoost gain"))
ax.set_title(tr("Top 20 features", "Top 20 features"))
plt.tight_layout()
plt.savefig(FIG / "07_xgb_importancias.png", dpi=140)
plt.close()

top300 = ldb.sort_values("average", ascending=False).head(300).copy()
M = top300[bench].values
M_z = (M - M.mean(0)) / (M.std(0) + 1e-9)

sil_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    lab = km.fit_predict(M_z)
    sil_rows.append({"k": k, "silhouette": silhouette_score(M_z, lab)})
sil_df = pd.DataFrame(sil_rows)
sil_df.to_csv(TAB / "07_silhouette.csv", index=False)
k_opt = int(sil_df.sort_values("silhouette", ascending=False).iloc[0]["k"])

km = KMeans(n_clusters=k_opt, n_init=30, random_state=42)
top300["cluster"] = km.fit_predict(M_z)

pca = PCA(n_components=2)
pcs = pca.fit_transform(M_z)
top300["pc1"], top300["pc2"] = pcs[:, 0], pcs[:, 1]

palette = sns.color_palette("tab10", k_opt)
fig, ax = plt.subplots(figsize=(11, 8))
for c in sorted(top300["cluster"].unique()):
    sub = top300[top300["cluster"] == c]
    ax.scatter(sub["pc1"], sub["pc2"], color=palette[c],
               label=f"cluster {c} (n={len(sub)})", s=40, alpha=0.75)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
ax.set_title(f"Clusters (top-300), k={k_opt}, silhouette={sil_df['silhouette'].max():.3f}")
ax.legend(loc="best", fontsize=10)
plt.tight_layout()
plt.savefig(FIG / "07_clusters_pca.png", dpi=140)
plt.close()

profile = top300.groupby("cluster")[bench].mean()
profile.columns = [label[c] for c in bench]
profile.to_csv(TAB / "07_perfil_clusters.csv")

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(profile, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={"label": "score (%)"})
ax.set_title(tr("Perfil medio dos clusters (top-300)", "Average cluster profile (top-300)"))
plt.tight_layout()
plt.savefig(FIG / "07_perfil_clusters.png", dpi=140)
plt.close()

def radar(ax, values, labels, color, title):
    N = len(labels)
    angles = [n / N * 2 * pi for n in range(N)] + [0]
    values = list(values) + [values[0]]
    ax.plot(angles, values, color=color, linewidth=2)
    ax.fill(angles, values, color=color, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title(title, fontsize=11)

fig, axes = plt.subplots(1, k_opt, figsize=(5 * k_opt, 5), subplot_kw=dict(polar=True))
if k_opt == 1: axes = [axes]
maxv = profile.values.max()
for ax, c in zip(axes, profile.index):
    radar(ax, (profile.loc[c].values / maxv * 100).round(1),
          profile.columns.tolist(), palette[c], f"Cluster {c}")
fig.suptitle(tr("Personalidades de LLMs", "LLM personalities"), fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "07_radar_clusters.png", dpi=140)
plt.close()

exemplos = (
    top300.groupby("cluster")
    .apply(lambda g: g.nlargest(5, "average")[["fullname", "average"] + bench], include_groups=False)
    .reset_index()
)
exemplos.to_csv(TAB / "07_exemplos_por_cluster.csv", index=False)

feats = ldb[bench + ["log_params"]].values
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42)
ldb["anomaly_score"] = -iso.fit(feats).score_samples(feats)
ldb["is_outlier"] = iso.predict(feats) == -1

outliers = ldb.sort_values("anomaly_score", ascending=False).head(30)[
    ["fullname", "organizacao", "params_b", "average"] + bench + ["anomaly_score"]
]
outliers.to_csv(TAB / "07_top30_outliers.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(ldb.loc[~ldb["is_outlier"], "params_b"], ldb.loc[~ldb["is_outlier"], "average"],
           s=12, alpha=0.4, color="#1f77b4", label="normal")
ax.scatter(ldb.loc[ldb["is_outlier"], "params_b"], ldb.loc[ldb["is_outlier"], "average"],
           s=40, alpha=0.9, color="#d62728", label="outlier")
ax.set_xscale("log")
ax.set_xlabel(tr("# parametros (B)", "# parameters (B)"))
ax.set_ylabel("Average (%)")
ax.set_title(tr("Outliers - IsolationForest (contaminacao=2%)",
                "Outliers - IsolationForest (contamination=2%)"))
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "07_outliers.png", dpi=140)
plt.close()

resumo = {
    "n_modelos_ml": int(len(ldb)),
    "regressao_top": res_df.iloc[0].to_dict(),
    "k_clusters": k_opt,
    "silhouette_max": float(sil_df["silhouette"].max()),
    "n_outliers": int(ldb["is_outlier"].sum()),
    "top5_features": imps.head(5).to_dict(),
}
(TAB / "resumo_07_ml.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(resumo)
