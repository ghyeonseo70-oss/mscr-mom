"""
2단계 NN(위치추정+L_M -> s_f,F, 15000개, 이상적 상한선) 결과 시각화.
(1) 무작위 검증 사례 3개 - 실제 형상/접촉 vs 예측 접촉
(2) 예측 vs 실제 산점도 (전체 타깃)
주의: 위치추정(x,y,theta)에 노이즈가 없다고 가정한 "이상적 상한선" 결과 - 실전 배치용
CNN 직접추정(R2=0.633)과는 별개로, "이 방향이 잘 되면 어디까지 갈 수 있는지" 보여주는 목적.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "force_model"))
import force_model as fm

DATA_DIR = os.path.join(HERE, "..", "..", "data", "contact_scenarios")

feat = np.load(os.path.join(DATA_DIR, "shape_to_contact_features_15k.npz"))
X_raw, y_raw, targets = feat["X"], feat["y"], list(feat["targets"])
n = len(X_raw)

hist = np.load(os.path.join(DATA_DIR, "shape_to_contact_train_history_15k.npz"))
val_pred, val_true = hist["val_pred"], hist["val_true"]

# test_shape_to_contact_nn.py와 동일한 분할 로직으로 val_idx 복원
rng_split = np.random.default_rng(0)
idx = rng_split.permutation(n)
split = int(0.8 * n)
val_idx = idx[split:]
tcol = {c: i for i, c in enumerate(targets)}

R2 = {}
for i, t in enumerate(targets):
    R2[t] = 1 - np.sum((val_pred[:, i] - val_true[:, i]) ** 2) / np.sum((val_true[:, i] - val_true[:, i].mean()) ** 2)
AVG = np.mean(list(R2.values()))

# ══════════════════════════════════════════════════════════════
# (1) 무작위 검증 사례 3개
# ══════════════════════════════════════════════════════════════
rng = np.random.default_rng(999)
pick = rng.choice(len(val_idx), 3, replace=False)
PHI_EXAMPLE = 60.0

fig1, axes = plt.subplots(1, 3, figsize=(17, 6.3))
for panel, (ax, k) in enumerate(zip(axes, pick)):
    orig_i = val_idx[k]
    L_M = X_raw[orig_i, 0]  # X 첫 컬럼 = L_M

    s_true, Fx_true, Fy_true = val_true[k, tcol['s']], val_true[k, tcol['Fx']], val_true[k, tcol['Fy']]
    Fmag_true = val_true[k, tcol['F_mag']]
    s_pred, Fmag_pred = val_pred[k, tcol['s']], val_pred[k, tcol['F_mag']]

    r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI_EXAMPLE, loads=[], return_curve=True)
    r_load = fm.solve_shape_robust(L_M=L_M, phi_deg=PHI_EXAMPLE,
                                    loads=[{"type": "point", "s": s_true, "Fx": Fx_true, "Fy": Fy_true}],
                                    theta_L_hint_deg=r_free["theta_L_deg"], return_curve=True)

    ax.plot(r_free["curve_x_mm"], r_free["curve_y_mm"], "--", color="#bbbbbb", linewidth=1.5,
             label="무접촉 기준형상", zorder=2)
    # 실제 카테터처럼 두꺼운 튜브 모양으로 렌더링 (얇은 수학적 선이 아니라 실물 느낌)
    ax.plot(r_load["curve_x_mm"], r_load["curve_y_mm"], "-", color="#aacbe8", linewidth=11,
            solid_capstyle="round", zorder=2.5)
    ax.plot(r_load["curve_x_mm"], r_load["curve_y_mm"], "-", color="#2a78d6", linewidth=2.2,
            solid_capstyle="round", zorder=3, label="접촉시 카테터 형상")
    ax.plot(0, 0, "ks", markersize=10, zorder=5, label="베이스(고정단)")

    cs, cx, cy, cth = r_load["curve_s_mm"], r_load["curve_x_mm"], r_load["curve_y_mm"], r_load["curve_theta_deg"]
    order = np.argsort(cs)
    xt = np.interp(s_true, cs[order], cx[order])
    yt = np.interp(s_true, cs[order], cy[order])

    # 실제 접촉지점에 힘 방향 화살표 (실제 Fx,Fy 방향 그대로 - 밖에서 튜브를 미는 모습)
    f_vec = np.array([Fx_true, Fy_true])
    f_dir = f_vec / (np.linalg.norm(f_vec) + 1e-12)
    arrow_len = 16.0
    ax.annotate("", xy=(xt, yt), xytext=(xt - f_dir[0] * arrow_len, yt - f_dir[1] * arrow_len),
                arrowprops=dict(arrowstyle="-|>", color="#B23A32", linewidth=2.8, mutation_scale=22), zorder=8)
    ax.plot(xt, yt, "o", color="#e34948", markersize=12, zorder=7,
            markeredgecolor="white", markeredgewidth=1.5, label=f"실제 접촉 (s={s_true:.0f}mm)")

    s_pred_clip = np.clip(s_pred, cs.min(), cs.max())
    xp = np.interp(s_pred_clip, cs[order], cx[order])
    yp = np.interp(s_pred_clip, cs[order], cy[order])
    ax.plot(xp, yp, "x", color="#1E7B34", markersize=15, mew=3, zorder=7,
            label=f"예측 접촉 (s={s_pred:.0f}mm)")
    ax.plot([xt, xp], [yt, yp], ":", color="#444444", linewidth=1.3, zorder=4)

    err_s = abs(s_pred - s_true)
    err_f = abs(Fmag_pred - Fmag_true) * 1000
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.set_title(f"사례 {panel+1}: L_M={L_M:.0f}mm\n"
                 f"실제 F={Fmag_true*1000:.1f}mN vs 예측 F={Fmag_pred*1000:.1f}mN\n"
                 f"(s 오차 {err_s:.1f}mm, F 오차 {err_f:.1f}mN)",
                 fontweight="bold", fontsize=11)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7.8, loc="best")

fig1.suptitle("2단계 NN(위치추정+L_M) 무작위 검증 사례 3개 - 이상적 상한선(노이즈 없음 가정)",
              fontweight="bold", fontsize=14, y=1.04)
out1 = os.path.join(DATA_DIR, "nn2_random_cases.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"저장: {out1}")

# ══════════════════════════════════════════════════════════════
# (2) 예측 vs 실제 산점도 + 헤로 배너
# ══════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 6.6))
gs = fig2.add_gridspec(2, 4, height_ratios=[0.3, 1], hspace=0.2, wspace=0.35)

hero = fig2.add_subplot(gs[0, :])
hero.axis("off")
hero.add_patch(plt.Rectangle((0, 0), 1, 1, transform=hero.transAxes, facecolor="#e8eefa",
                              edgecolor="#2451A3", linewidth=1.5))
hero.text(0.02, 0.5, f"{AVG:.3f}", fontsize=32, fontweight="bold", color="#2451A3",
          transform=hero.transAxes, va="center", ha="left")
hero.text(0.15, 0.5, "2단계 NN 평균 R² (이상적 상한선, n=15,000)\n"
                     "위치추정 노이즈 0 가정 - 실전 검증은 별도 진행 예정",
          fontsize=12.5, fontweight="bold", color="#193A7D", transform=hero.transAxes,
          va="center", ha="left", linespacing=1.4)

colors = {'s': '#8e44ad', 'F_mag': '#2e7d32', 'Fx': '#2a78d6', 'Fy': '#e34948'}
units = {'s': 'mm', 'F_mag': 'N', 'Fx': 'N', 'Fy': 'N'}
for i, t in enumerate(targets):
    ax = fig2.add_subplot(gs[1, i])
    ax.scatter(val_true[:, i], val_pred[:, i], s=6, alpha=0.25, color=colors[t])
    lo, hi = val_true[:, i].min(), val_true[:, i].max()
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
    ax.set_xlabel(f"실제 {t} ({units[t]})")
    ax.set_ylabel(f"예측 {t} ({units[t]})")
    ax.set_title(f"{t}: R²={R2[t]:.3f}", fontweight="bold", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.5)

fig2.suptitle("2단계 NN(위치추정+L_M -> 접촉) 예측 정확도", fontweight="bold", fontsize=16, y=1.01)
out2 = os.path.join(DATA_DIR, "nn2_accuracy_scatter.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"저장: {out2}")
