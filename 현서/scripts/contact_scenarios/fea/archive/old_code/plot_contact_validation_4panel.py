"""접촉(외력) 시나리오에서 "실제 s" vs "예측 s"를 비교하는 4-panel 그림(참고 이미지 스타일 재현).

이 컴퓨터에는 실측 파이프라인(실제 학습된 모델의 예측값)이 없어서, 오차값은 참고
이미지와 비슷한 수준(1~9mm)으로 예시용으로 임의 지정함 - 실제 검증은 다른 컴퓨터에서
이미 진행됨. 목적은 스타일(무접촉 기준형상 점선 + 접촉시 실제형상 15배 확대 + 실제/예측
마커 + 변위방향 화살표) 재현.

각 케이스: force_model.solve_shape로 무접촉(free) 곡선과, 실제 접촉점(s_actual)에 점하중을
가한 접촉(loaded) 곡선을 구하고, free 대비 displacement를 15배 확대해서 그린다.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "force_model"))
import force_model as fm

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

MAGNIFY = 15.0
ARROW_LEN_MM = 12.0

# (L_M, phi_deg, s_actual_mm, s_pred_mm, Fx_N, Fy_N) - Fx,Fy는 접촉힘(로컬좌표, 튜브를
# 누르는 방향으로 임의 지정). s_pred/오차는 예시용 임의값(다른 컴퓨터의 실측 결과 아님).
CASES = [
    (25.0, 90.0, 10.0, 4.6, 0.00006, -0.00002),
    (75.0, -90.0, 20.0, 28.3, -0.00004, 0.00005),
    (75.0, 120.0, 70.0, 71.1, 0.000006, 0.00001),
    (0.0, -90.0, 10.0, 13.3, 0.000002, 0.0000012),
]


def plot_case(ax, L_M, phi_deg, s_actual, s_pred, Fx, Fy):
    free = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, loads=[], return_curve=True)
    loaded = fm.solve_shape_ramped(L_M=L_M, phi_deg=phi_deg,
                                    loads=[{"type": "point", "s": s_actual, "Fx": Fx, "Fy": Fy}],
                                    theta_L_hint_deg=free["theta_L_deg"], n_steps=4, return_curve=True)

    fs, fx, fy = free["curve_s_mm"], free["curve_x_mm"], free["curve_y_mm"]
    ls, lx, ly = loaded["curve_s_mm"], loaded["curve_x_mm"], loaded["curve_y_mm"]
    # curve_s는 팁->베이스 내림차순 -> np.interp용으로 오름차순 정렬
    order_f, order_l = np.argsort(fs), np.argsort(ls)
    fs, fx, fy = fs[order_f], fx[order_f], fy[order_f]
    ls, lx, ly = ls[order_l], lx[order_l], ly[order_l]

    fx_at_l = np.interp(ls, fs, fx)
    fy_at_l = np.interp(ls, fs, fy)
    ex = fx_at_l + MAGNIFY * (lx - fx_at_l)
    ey = fy_at_l + MAGNIFY * (ly - fy_at_l)

    ax.plot(fx, fy, color="gray", linestyle="--", linewidth=1.5, label="무접촉 기준형상")
    ax.plot(ex, ey, color="#2451A3", linewidth=2.5, label=f"접촉 시 실제형상({MAGNIFY:.0f}배 확대)")
    ax.plot(0, 0, marker="s", color="black", markersize=9, zorder=6, label="베이스(고정단)")

    ax_actual, ay_actual = np.interp(s_actual, ls, ex), np.interp(s_actual, ls, ey)
    ax_pred, ay_pred = np.interp(s_pred, ls, ex), np.interp(s_pred, ls, ey)
    ax.plot(ax_actual, ay_actual, marker="o", color="#c0392b", markersize=10, zorder=7, label=f"실제 s={s_actual:.0f}mm")
    ax.plot(ax_pred, ay_pred, marker="x", color="#1b7a3d", markersize=11, markeredgewidth=3, zorder=7, label=f"예측 s={s_pred:.1f}mm")

    fmag = np.hypot(Fx, Fy)
    ang = np.degrees(np.arctan2(Fy, Fx))
    dx, dy = (Fx / fmag) * ARROW_LEN_MM, (Fy / fmag) * ARROW_LEN_MM
    ax.annotate("", xy=(ax_actual + dx, ay_actual + dy), xytext=(ax_actual, ay_actual),
                arrowprops=dict(arrowstyle="-|>", color="#c0392b", linewidth=2.2, mutation_scale=14), zorder=8)

    err = abs(s_actual - s_pred)
    ax.set_title(f"L_M={L_M:.0f}mm, phi={phi_deg:.0f}도 | 실제 s={s_actual:.0f}mm vs 예측 s={s_pred:.1f}mm\n"
                 f"오차 {err:.1f}mm (변위 방향 {ang:.0f}도)", fontsize=10)
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.set_xlim(-15, 100)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)


# 충돌(접촉) 없는 케이스 - 외력 자체가 없어서 접촉 시 형상 = 무접촉 기준형상과 완전히 동일.
NO_CONTACT_CASES = [(50.0, 90.0), (25.0, -60.0)]


def plot_no_contact_case(ax, L_M, phi_deg, color):
    free = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, loads=[], return_curve=True)
    fx, fy = free["curve_x_mm"], free["curve_y_mm"]

    ax.plot(fx, fy, color="gray", linestyle="--", linewidth=2.2, label="무접촉 기준형상")
    ax.plot(fx, fy, color=color, linewidth=2.2, alpha=0.75, label=f"접촉 시 실제형상({MAGNIFY:.0f}배 확대)")
    ax.plot(0, 0, marker="s", color="black", markersize=9, zorder=6, label="베이스(고정단)")

    ax.set_title(f"L_M={L_M:.0f}mm, phi={phi_deg:.0f}도", fontsize=10)
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.set_xlim(-15, 100)
    ax.set_ylim(-45, 45)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, case in zip(axes.flat, CASES):
        plot_case(ax, *case)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("접촉 시나리오 - 실제 s vs 예측 s (예시, 오차값은 임의 지정)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out_path = os.path.join(OUT_DIR, "contact_validation_4panel.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")

    NO_CONTACT_COLORS = ["#f2c744", "#e85d9c"]  # 노랑, 핑크 - 회색 점선이 비쳐 보이도록
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (L_M, phi_deg), color in zip(axes2, NO_CONTACT_CASES, NO_CONTACT_COLORS):
        plot_no_contact_case(ax, L_M, phi_deg, color)
    handles2, labels2 = axes2[0].get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig2.suptitle("무접촉(충돌 없음) 케이스 예시", fontsize=14, fontweight="bold")
    fig2.tight_layout(rect=[0, 0.08, 1, 0.94])

    out_path2 = os.path.join(OUT_DIR, "no_contact_2panel.png")
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path2}")
