"""
검증셋에서 무작위로 3개 사례를 뽑아, 실제 카테터 형상(접촉위치·힘)과 모델이 예측한 값을
나란히 보여준다 - "정확히 어떤 상황을 감지할 수 있는지"를 숫자(R²)가 아니라 그림으로 체감.
현재 실전 배치 가능한 최선 모델(능동탐색 5프로브, 편향제거 15000개, 평균 R2=0.633) 기준.
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

# ── train_multiprobe_model.py와 동일한 분할 로직으로 val_idx 복원 ──────────────
d_full = np.load(os.path.join(DATA_DIR, "contact_multiprobe_5probe_15k_unbiased.npz"))
y_all, y_cols = d_full["y"], list(d_full["y_columns"])
ycol = {c: i for i, c in enumerate(y_cols)}
n_total = len(y_all)

rng_split = np.random.default_rng(0)
idx = rng_split.permutation(n_total)
split = int(0.8 * n_total)
val_idx = idx[split:]

hist = np.load(os.path.join(DATA_DIR, "multiprobe_train_history_15k_unbiased.npz"))
val_pred, val_true, targets = hist["val_pred"], hist["val_true"], list(hist["targets"])
tcol = {c: i for i, c in enumerate(targets)}

# ── 무작위 3개 선택 ──────────────────────────
rng = np.random.default_rng(20260728)
pick = rng.choice(len(val_idx), 3, replace=False)

fig, axes = plt.subplots(1, 3, figsize=(17, 6.3))

for panel, (ax, k) in enumerate(zip(axes, pick)):
    orig_i = val_idx[k]
    L_M = y_all[orig_i, ycol["L_M"]]

    s_true = val_true[k, tcol["s"]]
    Fx_true = val_true[k, tcol["Fx"]]
    Fy_true = val_true[k, tcol["Fy"]]
    Fmag_true = val_true[k, tcol["F_mag"]]

    s_pred = val_pred[k, tcol["s"]]
    Fx_pred = val_pred[k, tcol["Fx"]]
    Fy_pred = val_pred[k, tcol["Fy"]]
    Fmag_pred = val_pred[k, tcol["F_mag"]]

    # 대표 phi=60도에서 실제 형상 재구성 (phi=0은 무외력 상태가 우연히 일직선이 되는 특수케이스라
    # 오해 소지가 있어서, 무외력일 때도 자연스럽게 휘어있는 phi=60을 대표값으로 사용)
    PHI_EXAMPLE = 60.0
    r_free = fm.solve_shape(L_M=L_M, phi_deg=PHI_EXAMPLE, loads=[], return_curve=True)
    r_load = fm.solve_shape_robust(L_M=L_M, phi_deg=PHI_EXAMPLE,
                                    loads=[{"type": "point", "s": s_true, "Fx": Fx_true, "Fy": Fy_true}],
                                    theta_L_hint_deg=r_free["theta_L_deg"], return_curve=True)

    ax.plot(r_free["curve_x_mm"], r_free["curve_y_mm"], "--", color="#999999", linewidth=1.8,
             label="무접촉 기준형상", zorder=2)
    ax.plot(r_load["curve_x_mm"], r_load["curve_y_mm"], "-", color="#5b9bd5", linewidth=2.8,
             label="접촉시 실제형상", zorder=3)
    ax.plot(0, 0, "ks", markersize=9, zorder=5)

    # 실제 접촉점 표시 (s_true 위치를 곡선에서 보간)
    cs, cx, cy = r_load["curve_s_mm"], r_load["curve_x_mm"], r_load["curve_y_mm"]
    order = np.argsort(cs)
    xt = np.interp(s_true, cs[order], cx[order])
    yt = np.interp(s_true, cs[order], cy[order])
    ax.plot(xt, yt, "o", color="#e34948", markersize=13, zorder=6, label=f"실제 접촉 (s={s_true:.0f}mm)")

    # 모델이 예측한 접촉위치 표시 (같은 곡선 위, s_pred 지점 - 근사)
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

fig.suptitle("무작위 검증 사례 3개 - 실제 접촉 vs 모델 예측 (능동탐색 5프로브 모델, 평균 R²=0.633)",
             fontweight="bold", fontsize=14.5, y=1.04)

out_path = os.path.join(DATA_DIR, "random_detection_cases.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
for panel, k in enumerate(pick):
    print(f"사례{panel+1}: s실제={val_true[k,tcol['s']]:.1f} s예측={val_pred[k,tcol['s']]:.1f} | "
          f"F실제={val_true[k,tcol['F_mag']]*1000:.2f}mN F예측={val_pred[k,tcol['F_mag']]*1000:.2f}mN")
