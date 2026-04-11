from pathlib import Path
import json
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"
OUT  = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ── auto-download ────────────────────────────────────────────────────────────
_LDB_PATH = RAW / "open_llm_leaderboard" / "train.parquet"
_LDB_REPO  = "open-llm-leaderboard/contents"
_LDB_FILE  = "data/train-00000-of-00001.parquet"
_LDB_REV   = "9c09a7cae43334062a82cb164f2ef255013dafa2"

def _ensure_leaderboard() -> Path:
    if _LDB_PATH.exists():
        return _LDB_PATH
    print(f"[01] Baixando leaderboard de {_LDB_REPO} (revision {_LDB_REV[:8]})…")
    from huggingface_hub import hf_hub_download
    _LDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=_LDB_REPO,
        filename=_LDB_FILE,
        repo_type="dataset",
        revision=_LDB_REV,
    )
    shutil.copy(cached, _LDB_PATH)
    print(f"[01] Salvo em {_LDB_PATH}")
    return _LDB_PATH

# ── mapeamento de colunas ────────────────────────────────────────────────────
rename = {
    "Average ⬆️": "average",
    "IFEval Raw": "ifeval_raw", "IFEval": "ifeval",
    "BBH Raw": "bbh_raw",      "BBH": "bbh",
    "MATH Lvl 5 Raw": "math_raw", "MATH Lvl 5": "math",
    "GPQA Raw": "gpqa_raw",    "GPQA": "gpqa",
    "MUSR Raw": "musr_raw",    "MUSR": "musr",
    "MMLU-PRO Raw": "mmlu_pro_raw", "MMLU-PRO": "mmlu_pro",
    "#Params (B)": "params_b",
    "Hub ❤️": "hub_likes",
    "CO₂ cost (kg)": "co2_kg",
    "Model": "model",
    "Type": "type",
    "Architecture": "architecture",
    "Precision": "precision",
    "Hub License": "license",
    "Flagged": "flagged",
    "MoE": "moe",
    "Merged": "merged",
    "Official Providers": "official_providers",
    "Base Model": "base_model",
    "Submission Date": "submission_date",
    "Upload To Hub Date": "upload_date",
}

bench = ["ifeval", "bbh", "math", "gpqa", "musr", "mmlu_pro"]

# ── ingestão ─────────────────────────────────────────────────────────────────
ldb = pd.read_parquet(_ensure_leaderboard()).rename(columns=rename)

keep = (
    ["fullname", "model", "type", "architecture", "precision",
     "params_b", "license", "flagged", "moe", "merged", "official_providers",
     "submission_date", "average"]
    + bench
    + [c + "_raw" for c in bench]
)
ldb = ldb[[c for c in keep if c in ldb.columns]].copy()

for c in bench + ["average"]:
    ldb[c] = pd.to_numeric(ldb[c], errors="coerce")
ldb["params_b"] = pd.to_numeric(ldb["params_b"], errors="coerce")
ldb["flagged"]  = ldb["flagged"].astype(bool, errors="ignore")

valid = ldb[(~ldb["flagged"].fillna(False)) & ldb["average"].notna()].drop_duplicates("fullname")

ldb.to_parquet(OUT / "leaderboard_full.parquet",  index=False)
valid.to_parquet(OUT / "leaderboard_valid.parquet", index=False)
valid.to_csv(OUT / "leaderboard_valid.csv", index=False, encoding="utf-8")

n_questoes = {
    "ifeval": 541, "bbh": 6511, "math": 1324,
    "gpqa": 448, "musr": 756, "mmlu_pro": 12032,
}
pd.DataFrame(
    [{"benchmark": k, "n_questoes": v} for k, v in n_questoes.items()]
).to_csv(OUT / "n_questoes_por_benchmark.csv", index=False)

resumo = {
    "linhas_brutas": int(len(ldb)),
    "linhas_validas": int(len(valid)),
    "benchmarks": bench,
    "n_questoes": n_questoes,
}
(OUT / "resumo_01_leaderboard.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(resumo)
