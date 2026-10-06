"""최종 검증: force_model.py(3구간, 논문 식 그대로) 예측 곡선 위에 논문 원본
디지털화 데이터(fig3_digitized.json, Fig.3 실제 논문 시뮬레이션 결과에서 뽑은 점)를
그대로 겹쳐 그림 - MATLAB도 다른 모델도 아니라 논문 자체와 직접 대조."""
import json
import numpy as np
import matplotlib.pyplot as plt
import force_model as fm

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

with open("../../data/force_model/fig3_digitized.json", encoding="utf-8") as f:
    paper = json.load(f)

LM_RATIOS = [0.0, 0.25, 0.5, 0.75]

fig, axes = plt.subplots(1, 4, figsize=(20, 5.5))

for ax, ratio in zip(axes, LM_RATIOS):
    L_M = max(ratio * fm.L, 0.5)
    key = "0.0" if ratio == 0.0 else str(ratio)
    paper_pts = paper[key]

    # 내 모델: phi=0(또는 CMSCR은 -150부터)부터 연속법으로 -150~150 전부 추적
    ys_model, ths_model = [], []
    hint = None
    for phi in range(0, 151, 2):
        try:
            r = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], theta_L_hint_deg=hint, window_deg=40.0)
        except RuntimeError:
            continue
        hint = r["theta_L_deg"]
        ys_model.append(r["y_L"] / 10)
        ths_model.append(r["theta_L_deg"])
    hint = None
    ys_model2, ths_model2 = [], []
    for phi in range(0, -151, -2):
        try:
            r = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], theta_L_hint_deg=hint, window_deg=40.0)
        except RuntimeError:
            continue
        hint = r["theta_L_deg"]
        ys_model2.append(r["y_L"] / 10)
        ths_model2.append(r["theta_L_deg"])

    ax.plot(ths_model, ys_model, color="#2451A3", linewidth=2, label="force_model.py 예측(연속법)")
    ax.plot(ths_model2, ys_model2, color="#2451A3", linewidth=2)

    # 논문 디지털화 실측점
    paper_th = [v[1] for v in paper_pts.values()]
    paper_y = [v[0] for v in paper_pts.values()]
    ax.scatter(paper_th, paper_y, color="#D95319", s=70, zorder=5, marker="o",
               edgecolor="black", linewidth=0.8, label="논문 Fig.3 디지털화 실측점")

    title = "CMSCR (MOM 없음)" if ratio == 0 else f"L_M/L = {ratio}"
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.set_xlabel(r"$\theta_L$ (deg)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)

axes[0].set_ylabel(r"$y_L$ (cm)")
axes[0].legend(loc="upper left", fontsize=9)

fig.suptitle("최종 검증: force_model.py(3구간, 논문 식 그대로) vs 논문 Fig.3 원본 데이터 직접 대조",
             fontweight="bold", fontsize=15)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_path = "../../data/force_model/final_validation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")

# 정량 오차도 출력
print("\n=== 정량 오차 (theta_L, deg) ===")
for ratio in LM_RATIOS:
    L_M = max(ratio * fm.L, 0.5)
    key = "0.0" if ratio == 0.0 else str(ratio)
    errs = []
    hint = None
    prev_phi = 0
    for phi_str, (py, pth, _) in sorted(paper[key].items(), key=lambda kv: int(kv[0])):
        phi = int(phi_str)
        r = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        errs.append(abs(r["theta_L_deg"] - pth))
    print(f"L_M/L={ratio}: 평균 오차={np.mean(errs):.2f}deg, 최대 오차={np.max(errs):.2f}deg (n={len(errs)})")
