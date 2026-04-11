from pathlib import Path
import json
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw" / "arena"
OUT  = ROOT / "data" / "processed" / "arena"
OUT.mkdir(parents=True, exist_ok=True)

# ── auto-download ────────────────────────────────────────────────────────────
_ARENA_REPO = "lmsys/chatbot_arena_conversations"
_ARENA_FILE = "data/train-00000-of-00001-cced8514c7ed782a.parquet"

def _ensure_arena() -> Path:
    matches = list(RAW.glob("train-*.parquet"))
    if matches:
        return matches[0]
    RAW.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        print(f"[02] Baixando arena de {_ARENA_REPO}…")
        cached = hf_hub_download(
            repo_id=_ARENA_REPO,
            filename=_ARENA_FILE,
            repo_type="dataset",
        )
        dst = RAW / Path(_ARENA_FILE).name
        shutil.copy(cached, dst)
        print(f"[02] Salvo em {dst}")
        return dst
    except Exception as e:
        raise SystemExit(
            "\n[02] Dataset lmsys/chatbot_arena_conversations é restrito (gated).\n"
            "Para baixar:\n"
            f"  1. Aceite os termos em https://huggingface.co/datasets/{_ARENA_REPO}\n"
            "  2. Execute: huggingface-cli login\n"
            "  3. Rode este script novamente.\n"
            f"Erro: {e}"
        )

# ── ingestão ─────────────────────────────────────────────────────────────────
df = pd.read_parquet(_ensure_arena())

cols = ["question_id", "model_a", "model_b", "winner", "judge", "turn",
        "anony", "language", "tstamp"]
light = df[cols].copy()
light["tstamp"] = pd.to_datetime(light["tstamp"], unit="s", errors="coerce")
light["y"] = light["winner"].map(
    {"model_a": 1, "model_b": 0, "tie": 0.5, "tie (bothbad)": 0.5}
)
light["resultado"] = light["winner"].map(
    {"model_a": "A", "model_b": "B", "tie": "TIE", "tie (bothbad)": "TIE_BAD"}
)
light.to_parquet(OUT / "arena_battles.parquet", index=False)

models = (
    pd.concat([light["model_a"], light["model_b"]])
    .value_counts()
    .rename_axis("model")
    .reset_index(name="n_battles")
)
models.to_csv(OUT / "arena_modelos.csv", index=False)

decisive = light[light["winner"].isin(["model_a", "model_b"])].copy()

wins = pd.pivot_table(
    decisive.assign(won_by_a=(decisive["winner"] == "model_a").astype(int)),
    index="model_a", columns="model_b", values="won_by_a", aggfunc="sum", fill_value=0,
)
counts = pd.pivot_table(
    decisive.assign(c=1),
    index="model_a", columns="model_b", values="c", aggfunc="sum", fill_value=0,
)
wins.to_parquet(OUT / "matriz_vitorias.parquet")
counts.to_parquet(OUT / "matriz_contagens.parquet")

resumo = {
    "total_batalhas": int(len(df)),
    "modelos_unicos":  int(len(models)),
    "decisivas":       int(len(decisive)),
    "empates":         int(len(df) - len(decisive)),
    "linguas_top3":    df["language"].value_counts().head(3).to_dict(),
    "modelo_mais_batalhas": models.iloc[0]["model"],
    "n_juizes":        int(df["judge"].nunique()),
}
(OUT / "resumo_arena.json").write_text(
    json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(resumo)
