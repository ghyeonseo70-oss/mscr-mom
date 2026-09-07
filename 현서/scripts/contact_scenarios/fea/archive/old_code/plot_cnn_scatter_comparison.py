"""3가지 접촉추정 CNN 버전(direct CNN-노이즈없음 / 2단계 이상적상한선 / 2단계 엔드투엔드)의
예측값 vs 실제값 산점도를 나란히 비교. 타겟(s, F_mag, Fx, Fy)별로 행을 나눔.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

TARGETS = ["s", "F_mag", "Fx", "Fy"]
UNITS = {"s": "mm", "F_mag": "N", "Fx": "N", "Fy": "N"}

# ── 1) direct CNN(노이즈 없음) - 저장된 모델 가중치로 검증셋 추론 재현 ──────────────
class ContactEstimator(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_out)
        )

    def forward(self, x):
        return self.regressor(self.features(x))


bfield = np.load(os.path.join(FEA_DATA_DIR, "fea_contact_bfield_dataset_nonoise.npz"))
B_free, B_load, y_all = bfield["B_free"], bfield["B_load"], bfield["y"]
delta = B_load - B_free
Xc = delta.reshape(-1, 5, 5, 3).transpose(0, 3, 1, 2)
Xc_norm = (Xc - Xc.mean()) / Xc.std()
yc_mean, yc_std = y_all.mean(axis=0), y_all.std(axis=0)

rng3 = np.random.default_rng(0)
idx = rng3.permutation(len(Xc_norm))
split = int(0.8 * len(idx))
val_idx = idx[split:]

model_direct = ContactEstimator(len(TARGETS))
model_direct.load_state_dict(torch.load(os.path.join(MODELS_DIR, "fea_contact_estimator_nonoise.pth")))
model_direct.eval()
with torch.no_grad():
    pred_norm = model_direct(torch.tensor(Xc_norm[val_idx]).float()).numpy()
pred_direct = pred_norm * yc_std + yc_mean
true_direct = y_all[val_idx]

# ── 2) 2단계 NN(이상적 상한선 / 엔드투엔드) - 저장된 학습이력에서 로드 ──────────────
hist = np.load(os.path.join(FEA_DATA_DIR, "fea_two_stage_train_history.npz"))
pred_ideal, pred_e2e, true_contact = hist["pred_ideal"], hist["pred_e2e"], hist["true_contact"]

# ── 3) 2단계 NN 엔드투엔드(수정판: NN2를 NN1의 실제 예측값으로 재학습) ──────────────
hist_fix = np.load(os.path.join(FEA_DATA_DIR, "fea_nn2_on_nn1_preds_history.npz"))
pred_fix, true_fix = hist_fix["pred"], hist_fix["true"]

YELLOW = "#eda100"
COLUMNS = [
    ("direct CNN\n(노이즈 없음)", true_direct, pred_direct, BLUE),
    ("2단계 NN\n(이상적 상한선)", true_contact, pred_ideal, ORANGE),
    ("2단계 NN\n(엔드투엔드, exposure bias)", true_contact, pred_e2e, AQUA),
    ("2단계 NN\n(엔드투엔드, 수정판)", true_fix, pred_fix, YELLOW),
]

fig, axes = plt.subplots(len(TARGETS), len(COLUMNS), figsize=(17, 15.5))

for row, tname in enumerate(TARGETS):
    for c, (cname, true_arr, pred_arr, color) in enumerate(COLUMNS):
        ax = axes[row, c]
        t = true_arr[:, row]
        p = pred_arr[:, row]
        r2 = r2_score(t, p)
        ax.scatter(t, p, s=6, color=color, alpha=0.25, edgecolors="none", zorder=3)
        lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "--", color=INK, linewidth=1.3, alpha=0.6, zorder=4)
        ax.set_title(f"R²={r2:.3f}", fontsize=10.5, fontweight="bold", color=color)
        if row == 0:
            ax.text(0.5, 1.28, cname, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=INK)
        if c == 0:
            ax.set_ylabel(f"{tname} 예측값 ({UNITS[tname]})", fontsize=9.5)
        ax.set_xlabel(f"{tname} 실제값 ({UNITS[tname]})", fontsize=9)
        ax.grid(True, linestyle=":", color=GRID, alpha=0.7)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_aspect("auto")

fig.suptitle("접촉추정 CNN 4가지 버전 — 예측값 vs 실제값 산점도 비교",
             fontweight="bold", fontsize=16, y=0.995)
fig.text(0.5, 0.975, "점선 = 완벽한 예측(y=x) / 점이 대각선에 가깝게 몰릴수록 정확",
         ha="center", fontsize=10, color=MUTED)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out_path = os.path.join(FEA_DATA_DIR, "cnn_scatter_comparison.png")
fig.savefig(out_path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
print(f"저장: {out_path}")
