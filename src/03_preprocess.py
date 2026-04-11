from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed"

ldb = pd.read_parquet(IN / "leaderboard_valid.parquet")
n_q = pd.read_csv(IN / "n_questoes_por_benchmark.csv").set_index("benchmark")["n_questoes"].to_dict()

bench = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]

ldb["organizacao"] = ldb["fullname"].astype(str).str.split("/").str[0]

def bucket(p):
    if pd.isna(p): return "desconhecido"
    if p < 3:   return "<3B"
    if p < 8:   return "3-8B"
    if p < 14:  return "8-14B"
    if p < 35:  return "14-35B"
    if p < 80:  return "35-80B"
    return ">=80B"

ldb["tam_bucket"] = ldb["params_b"].apply(bucket)

records = []
for _, row in ldb.iterrows():
    for b in bench:
        s = row[b]
        if pd.isna(s):
            continue
        p = float(s) / 100
        if not 0 <= p <= 1:
            continue
        n = n_q[b]
        records.append({
            "fullname": row["fullname"],
            "organizacao": row["organizacao"],
            "tam_bucket": row["tam_bucket"],
            "params_b": row["params_b"],
            "type": row.get("type"),
            "benchmark": b,
            "score_pct": float(s),
            "p": p,
            "n": int(n),
            "sucessos": int(round(p * n)),
        })

tidy = pd.DataFrame.from_records(records)
tidy.to_parquet(OUT / "leaderboard_tidy.parquet", index=False)

top100 = ldb.sort_values("average", ascending=False).head(100)
top30 = ldb.sort_values("average", ascending=False).head(30)
top100.to_parquet(OUT / "leaderboard_top100.parquet", index=False)
top30.to_parquet(OUT / "leaderboard_top30.parquet", index=False)
top100[["fullname", "organizacao", "tam_bucket", "params_b", "average"] + bench].to_csv(
    OUT / "leaderboard_top100.csv", index=False
)

org_stats = (
    ldb.groupby("organizacao")
    .agg(n_modelos=("fullname", "count"), avg_med=("average", "median"), avg_max=("average", "max"))
    .sort_values("n_modelos", ascending=False)
)
org_stats.to_csv(OUT / "estatisticas_por_organizacao.csv")

resumo = {
    "linhas_tidy": int(len(tidy)),
    "modelos_unicos": int(tidy["fullname"].nunique()),
    "benchmarks": bench,
    "top100_min_avg": float(top100["average"].min()),
    "top100_max_avg": float(top100["average"].max()),
    "n_organizacoes": int(ldb["organizacao"].nunique()),
}
(OUT / "resumo_03_preprocess.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(resumo)
