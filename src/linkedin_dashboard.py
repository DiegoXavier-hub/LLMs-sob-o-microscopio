"""
Dashboard visual para LinkedIn — LLMs sob o Microscópio
Gera: figures/linkedin_dashboard.png  (1080×1350 px, 150 dpi)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.image import imread
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG  = ROOT / "figures"

# ── Paleta ───────────────────────────────────────────────────────────────────
BG       = "#0B1929"   # azul-marinho fundo
SURFACE  = "#132338"   # painéis ligeiramente mais claros
ACCENT   = "#4FC3F7"   # azul-claro destaque
ACCENT2  = "#F06292"   # rosa para contraste
GOLD     = "#FFD54F"   # amarelo estatístico
TEXT_PRI = "#F0F4F8"   # branco frio principal
TEXT_SEC = "#90A4AE"   # cinza médio secundário
GREEN    = "#81C784"

# ── Canvas ───────────────────────────────────────────────────────────────────
W, H = 10.8, 13.5          # polegadas → 1080×1350 @100 dpi (print em 150 = 1620×2025)
fig  = plt.figure(figsize=(W, H), facecolor=BG)

gs = GridSpec(
    6, 2,
    figure=fig,
    top=0.97, bottom=0.03,
    left=0.04, right=0.96,
    hspace=0.45, wspace=0.08,
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def embed(ax, fname, title="", letter=""):
    img_path = FIG / fname
    if not img_path.exists():
        ax.set_facecolor(SURFACE)
        ax.text(0.5, 0.5, f"[{fname}]", transform=ax.transAxes,
                ha="center", va="center", color=TEXT_SEC, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        return
    img = imread(str(img_path))
    ax.imshow(img, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor(SURFACE)
    if title:
        ax.set_title(title, color=TEXT_PRI, fontsize=8.5, fontweight="bold",
                     pad=4, loc="left")
    if letter:
        ax.text(0.02, 0.97, letter, transform=ax.transAxes,
                color=ACCENT, fontsize=7, fontweight="bold",
                va="top", ha="left",
                bbox=dict(facecolor=BG, edgecolor="none", pad=1.5, alpha=0.7))


def stat_card(ax, number, label, color=ACCENT, unit="", sub=""):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.4)
    ax.text(0.5, 0.65, f"{number}{unit}", transform=ax.transAxes,
            ha="center", va="center", color=color,
            fontsize=22, fontweight="bold")
    ax.text(0.5, 0.28, label, transform=ax.transAxes,
            ha="center", va="center", color=TEXT_PRI,
            fontsize=7.5, wrap=True)
    if sub:
        ax.text(0.5, 0.10, sub, transform=ax.transAxes,
                ha="center", va="center", color=TEXT_SEC, fontsize=6.5)


# ── Linha 0 — Cabeçalho ──────────────────────────────────────────────────────
ax_hdr = fig.add_subplot(gs[0, :])
ax_hdr.set_facecolor(BG)
ax_hdr.set_xticks([]); ax_hdr.set_yticks([])
for sp in ax_hdr.spines.values(): sp.set_visible(False)

# linha decorativa
ax_hdr.axhline(0.12, color=ACCENT, linewidth=1.8, xmin=0.0, xmax=1.0, alpha=0.6)

ax_hdr.text(0.0, 0.90, "LLMs sob o Microscópio",
            transform=ax_hdr.transAxes,
            color=TEXT_PRI, fontsize=18, fontweight="bold", va="top")
ax_hdr.text(0.0, 0.52,
            "Incerteza Estatística, Poder e a Ilusão do Ranking Exato em Benchmarks de LLMs",
            transform=ax_hdr.transAxes,
            color=ACCENT, fontsize=9, va="top")
ax_hdr.text(0.0, 0.28,
            "Análise de 4.496 modelos · Open LLM Leaderboard v2 · LMSYS Chatbot Arena",
            transform=ax_hdr.transAxes,
            color=TEXT_SEC, fontsize=7.5, va="top")

# ── Linha 1 — Cartões de estatísticas ────────────────────────────────────────
cards = [
    ("4.496",  "modelos\nanalisados",     ACCENT,  "",   "Leaderboard v2"),
    ("33 k",   "batalhas\nArena",         GREEN,   "",   "LMSYS / 20 modelos"),
    ("±4,6",   "pp margem de erro\nGPQA", ACCENT2, "pp", "n=448 questões"),
    ("R²≈0,71","explicado por\nmetadados",GOLD,    "",   "XGBoost · tamanho+org"),
]
for col, (num, lbl, cor, unit, sub) in enumerate(cards):
    ax_c = fig.add_subplot(gs[1, col // 2] if len(cards) <= 2 else
                           fig.add_axes([0.04 + col * 0.235, 0.665, 0.215, 0.085]))
    stat_card(ax_c, num, lbl, color=cor, sub=sub)

# ── Linha 2 — Margem de erro + Pares indistinguíveis ─────────────────────────
ax_moe = fig.add_subplot(gs[2, 0])
embed(ax_moe, "04_moe_teorica.png",
      "① Margem de erro por benchmark (Wilson 95%)", "①")

ax_pai = fig.add_subplot(gs[2, 1])
embed(ax_pai, "04_pct_pares_indistinguiveis.png",
      "② % pares estatisticamente indistinguíveis · top-100", "②")

# ── Linha 3 — FDR + Bootstrap ranking ────────────────────────────────────────
ax_fdr = fig.add_subplot(gs[3, 0])
embed(ax_fdr, "06_pct_significativos_fdr.png",
      "③ Pares significativos: bruto vs. Benjamini-Hochberg", "③")

ax_boot = fig.add_subplot(gs[3, 1])
embed(ax_boot, "08_ranking_bootstrap_top30.png",
      "④ Ranking bootstrap top-30 · IC 95%", "④")

# ── Linha 4 — XGBoost pred vs real + Bradley-Terry Arena ─────────────────────
ax_xgb = fig.add_subplot(gs[4, 0])
embed(ax_xgb, "07_pred_vs_real_xgb.png",
      "⑤ XGBoost: predito vs. real (só metadados)", "⑤")

ax_bt = fig.add_subplot(gs[4, 1])
embed(ax_bt, "09_bradley_terry.png",
      "⑥ Arena · Bradley-Terry escala Elo · IC 95%", "⑥")

# ── Linha 5 — Achados-chave + Stack + Link ────────────────────────────────────
ax_ftr = fig.add_subplot(gs[5, :])
ax_ftr.set_facecolor(SURFACE)
ax_ftr.set_xticks([]); ax_ftr.set_yticks([])
for sp in ax_ftr.spines.values():
    sp.set_edgecolor(ACCENT)
    sp.set_linewidth(0.8)

insights = [
    ("GPQA precisaria de 11× mais questões", ACCENT2),
    ("Top-50 colapsa em apenas 2 tiers estatísticos", GOLD),
    ("Arena e Leaderboard concordam nos 3 modelos sobrepostos", GREEN),
]

ax_ftr.text(0.01, 0.92, "Achados-chave", transform=ax_ftr.transAxes,
            color=TEXT_PRI, fontsize=8, fontweight="bold", va="top")

for i, (txt, cor) in enumerate(insights):
    ax_ftr.text(0.01, 0.70 - i * 0.23, f"▸  {txt}",
                transform=ax_ftr.transAxes,
                color=cor, fontsize=7.2, va="top")

# stack
stack_txt = "Stack:  Python · pandas · scipy · statsmodels · scikit-learn · XGBoost · Bradley-Terry · Plotly · LaTeX"
ax_ftr.text(0.01, 0.14, stack_txt, transform=ax_ftr.transAxes,
            color=TEXT_SEC, fontsize=6.2, va="bottom")

# link
ax_ftr.text(0.99, 0.14, "⭐ github.com/DiegoXavier-hub/LLMs-sob-o-microscopio",
            transform=ax_ftr.transAxes,
            color=ACCENT, fontsize=6.8, va="bottom", ha="right",
            fontweight="bold")

# stat cards (abs positioned)
card_data = [
    (0.04,  "4.496",  "modelos\nanalisados",  ACCENT),
    (0.28,  "33 k",   "batalhas\nArena",       GREEN),
    (0.52,  "±4,6pp", "margem erro\nGPQA",     ACCENT2),
    (0.76,  "R²≈0,71","só\nmetadados",         GOLD),
]
for xpos, num, lbl, cor in card_data:
    ax_card = fig.add_axes([xpos, 0.645, 0.215, 0.095], facecolor=SURFACE)
    stat_card(ax_card, num, lbl, color=cor)

# ── Salvar ────────────────────────────────────────────────────────────────────
out = FIG / "linkedin_dashboard.png"
fig.savefig(out, dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close(fig)
print(f"Salvo: {out}  ({out.stat().st_size // 1024} KB)")
