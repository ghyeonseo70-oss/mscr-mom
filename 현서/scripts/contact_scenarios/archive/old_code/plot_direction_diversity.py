"""
2D 평면 안에서 힘 방향이 실제로 다양하게(0~360도 전체) 학습됐다는 걸 보여주는 시각화.
(A) 학습데이터의 힘 방향 분포 (극좌표 히스토그램) - 전 방향 커버 확인
(B) 서로 다른 방향에서 민 사례 6개 - 무접촉 기준형상 위에 실제/예측 접촉위치를 함께 표시
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
tcol = {c: i for i, c in enumerate(targets)}

hist = np.load(os.path.join(DATA_DIR, "shape_to_contact_train_history_15k.npz"))
val_pred, val_true = hist["val_pred"], hist["val_true"]

rng_split = np.random.default_rng(0)
idx = rng_split.permutation(n)
split = int(0.8 * n)
val_idx = idx[split:]

F_ang_all = np.degrees(np.arctan2(val_true[:, tcol['Fy']], val_true[:, tcol['Fx']]))

# ══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, height_ratios=[0.65, 1], hspace=0.45, wspace=0.3)

# ── (A) 힘 방향 분포 (극좌표) ──────────────────────────────
ax_polar = fig.add_subplot(gs[0, 0], projection="polar")
bins = np.linspace(-180, 180, 25)
counts, edges = np.histogram(F_ang_all, bins=bins)
theta = np.radians((edges[:-1] + edges[1:]) / 2)
ax_polar.bar(theta, counts, width=np.radians(15), color="#2a78d6", alpha=0.75, edgecolor="white")
ax_polar.set_theta_zero_location("E")
ax_polar.set_theta_direction(1)
ax_polar.set_title("(A) 학습데이터 힘 방향(F_ang) 분포\n- 0~360도 전 방향 고르게 포함됨",
                    fontweight="bold", fontsize=12, pad=20)

# ── 요약 텍스트 ──────────────────────────────
ax_txt = fig.add_subplot(gs[0, 1:])
ax_txt.axis("off")
ax_txt.text(0.0, 0.85, "2D 평면 내 방향 다양성 확인", fontsize=16, fontweight="bold", transform=ax_txt.transAxes)
ax_txt.text(0.0, 0.55,
            "학습 시 접촉힘 방향(F_ang)을 0~360도 전체에서 무작위로 샘플링 -> 같은 위치라도\n"
            "위/아래/옆 등 어느 쪽에서 밀렸는지가 다양하게 섞여 학습됨.\n\n"
            "오른쪽 6개 사례는 서로 다른 방향(약 60도 간격)에서 민 경우를 골라, 무접촉 기준형상\n"
            "위에 실제 접촉위치(빨강)와 예측 접촉위치(초록)를 함께 표시함.\n\n"
            "단, 이는 모두 2D 평면(보드 수평면) '안'에서의 방향이고, 평면을 뚫고 수직으로\n"
            "미는 힘(3D)은 이 모델의 범위 밖입니다.",
            fontsize=12.5, transform=ax_txt.transAxes, va="top", linespacing=1.9)

# ── (B) 방향별 사례 6개 (극좌표 60도 간격으로 가장 가까운 검증샘플 선택) ──────────────
target_angles = np.arange(-150, 181, 60)  # -150,-90,-30,30,90,150
chosen = []
for ta in target_angles:
    diffs = np.abs(((F_ang_all - ta + 180) % 360) - 180)
    chosen.append(np.argmin(diffs))

fig2, axes2 = plt.subplots(2, 3, figsize=(17, 10.5))
PHI_EXAMPLE = 60.0
for panel, (ax, k) in enumerate(zip(axes2.flat, chosen)):
    orig_i = val_idx[k]
    L_M = X_raw[orig_i, 0]
    s_true, Fx_true, Fy_true = val_true[k, tcol['s']], val_true[k, tcol['Fx']], val_true[k, tcol['Fy']]
    Fmag_true = val_true[k, tcol['F_mag']]
    s_pred = val_pred[k, tcol['s']]
    f_ang = np.degrees(np.arctan2(Fy_true, Fx_true))

    r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI_EXAMPLE, loads=[], return_curve=True)
    r_load = fm.solve_shape_robust(L_M=L_M, phi_deg=PHI_EXAMPLE,
                                    loads=[{"type": "point", "s": s_true, "Fx": Fx_true, "Fy": Fy_true}],
                                    theta_L_hint_deg=r_free["theta_L_deg"], return_curve=True)

    # 무접촉 기준형상 (얇은 튜브) - 여기 위에 실제/예측 접촉위치를 표시
    fcs, fcx, fcy = r_free["curve_s_mm"], r_free["curve_x_mm"], r_free["curve_y_mm"]
    forder = np.argsort(fcs)
    ax.plot(fcx, fcy, "-", color="#cfd8e3", linewidth=9, solid_capstyle="round", zorder=2)
    ax.plot(fcx, fcy, "--", color="#8a93a6", linewidth=1.6, zorder=2.5, label="무접촉 기준형상")
    ax.plot(r_load["curve_x_mm"], r_load["curve_y_mm"], "-", color="#2a78d6", linewidth=1.6,
            alpha=0.6, zorder=2.2, label="접촉시 실제형상(참고)")
    ax.plot(0, 0, "ks", markersize=9, zorder=5)

    xt = np.interp(s_true, fcs[forder], fcx[forder])
    yt = np.interp(s_true, fcs[forder], fcy[forder])
    f_dir = np.array([Fx_true, Fy_true]) / (np.hypot(Fx_true, Fy_true) + 1e-12)
    arrow_len = 16.0
    ax.annotate("", xy=(xt, yt), xytext=(xt - f_dir[0] * arrow_len, yt - f_dir[1] * arrow_len),
                arrowprops=dict(arrowstyle="-|>", color="#B23A32", linewidth=2.6, mutation_scale=20), zorder=8)
    ax.plot(xt, yt, "o", color="#e34948", markersize=12, zorder=7, markeredgecolor="white",
            markeredgewidth=1.5, label=f"실제 접촉 (s={s_true:.0f}mm)")

    s_pred_clip = np.clip(s_pred, fcs.min(), fcs.max())
    xp = np.interp(s_pred_clip, fcs[forder], fcx[forder])
    yp = np.interp(s_pred_clip, fcs[forder], fcy[forder])
    ax.plot(xp, yp, "x", color="#1E7B34", markersize=14, mew=3, zorder=7,
            label=f"예측 접촉 (s={s_pred:.0f}mm)")

    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.set_title(f"방향 {f_ang:.0f}도 | L_M={L_M:.0f}mm | F={Fmag_true*1000:.1f}mN\n"
                 f"s 오차 {abs(s_pred-s_true):.1f}mm", fontweight="bold", fontsize=10.5)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, loc="best")

fig2.suptitle("서로 다른 방향(약 60도 간격)에서 민 사례 6개 - 무접촉 기준형상 위 실제 vs 예측 접촉위치",
              fontweight="bold", fontsize=14.5, y=1.01)
fig2.tight_layout(rect=[0, 0, 1, 0.98])

out1 = os.path.join(DATA_DIR, "direction_coverage_summary.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
print(f"저장: {out1}")

out2 = os.path.join(DATA_DIR, "direction_diversity_cases.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"저장: {out2}")
