"""force_model.solve_shape(loads=[], return_curve=True)로 얻은 외력 없는(free-state)
카테터 형상을, 튜브(두께 있는 리본)로 그리고 내부 자석 2개(main, MOM)를 실제 배치대로
표시한다.

자석 표현 2가지를 분리해서 그림:
- 몸체(회색 막대): 실제 하드웨어처럼 튜브 안에 축방향으로 박혀있다고 가정 -> 튜브 접선
  방향(theta_L_deg/theta_LM_deg)과 나란히 배치.
- N->S 극 방향(빨간 화살표): worker()의 magpylib 설정(main: polarization=(0,+Br,0),
  mom: polarization=(0,-Br,0), dimension의 height축=자석의 로컬 z, 즉 원통 "옆면"이
  착자된 diametric 자석)과 동일하게, 로컬 (0,+-1,0) 벡터를 Rotation.from_euler('z',-theta)로
  회전시켜 계산 - 몸체 축(접선 방향)과는 별개의 방향이 나온다(항상 수직은 아님, 실제 회전식
  그대로 계산). MOM은 극성이 반대(-)라 main과 화살표가 반대로 나온다 - "Opposite Magnet"인
  이유.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "force_model"))
import force_model as fm

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

TUBE_DIAM_MM = 4.0  # 실제 튜브 외경은 코드에 명시 안 됨 - 자석(최대 지름2mm)이 들어갈 정도로 가정

# 시각화 전용 크기(실제 치수 아님, 비율은 실제 지름 반영) - main(지름2mm=길이2mm, 원래도
# 짧고 굵은 자석)은 짧고 굵게, MOM(지름1mm x 길이8mm, 원래 길고 가는 자석)은 길고 가늘게.
MAIN_VISUAL_LEN_MM, MAIN_VISUAL_WIDTH_MM = 4.0, 4.0
MOM_VISUAL_LEN_MM, MOM_VISUAL_WIDTH_MM = fm.H_M, 2.2

CASES = [
    (25.0, 90.0), (75.0, -90.0), (50.0, 90.0), (75.0, 120.0),
    (50.0, 0.0),  # phi=0 -> 외력도 없고 자기토크에 의한 굽힘도 없어 완전 직선(theta=0) -
                  # 실선(와이어)과 점선(나머지 내강)이 한 직선 위에 겹치는 기준 케이스
]


def draw_tube(ax, cx, cy, theta_deg_arr):
    theta = np.radians(theta_deg_arr)
    nx, ny = -np.sin(theta), np.cos(theta)
    half = TUBE_DIAM_MM / 2
    ux, uy = cx + nx * half, cy + ny * half
    lx, ly = cx - nx * half, cy - ny * half
    poly_x = np.concatenate([ux, lx[::-1]])
    poly_y = np.concatenate([uy, ly[::-1]])
    ax.fill(poly_x, poly_y, facecolor="#cfe0f3", edgecolor="#2451A3", linewidth=1.5, zorder=1)


def draw_wire(ax, curve_s, curve_x, curve_y, L_M):
    """MOM을 s=L_M까지 밀고 당겨서 위치를 조절하는 제어 와이어 - 베이스(s=0)부터
    MOM(s=L_M)까지는 실선(실제 와이어), MOM 지나서 팁까지는 점선(와이어가 안 닿는
    나머지 내강(lumen) - 참고용 가이드선)으로 구분해서 그림."""
    mask_wire = curve_s <= L_M
    ax.plot(curve_x[mask_wire], curve_y[mask_wire], color="#333333", linewidth=1.3, zorder=3)
    mask_rest = curve_s >= L_M
    ax.plot(curve_x[mask_rest], curve_y[mask_rest], color="#333333", linewidth=1.0,
            linestyle=(0, (2, 2)), zorder=3)


N_COLOR, S_COLOR = "#d0231f", "#1f5fb0"  # 논문 컨셉도 스타일: 빨강=N, 파랑=S


def draw_magnet(ax, x, y, body_theta_deg, sign, visual_len_mm, visual_width_mm):
    """참고 이미지(논문 컨셉도) 스타일 - 원통 자석을 길이방향으로 반씩 빨강(N)/파랑(S)으로
    칠하고 양끝에 타원 캡을 붙여 입체감을 준 캡슐 아이콘. body_theta_deg(튜브 접선) 방향으로
    회전시켜 튜브 축에 나란히 삽입된 것처럼 배치. sign: +1(main)/-1(MOM) - 극성이 반대라
    두 자석의 빨강/파랑 순서도 반대(Opposite Magnet)."""
    theta = np.radians(body_theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    hl, hw = visual_len_mm / 2, visual_width_mm / 2

    near_color, far_color = (S_COLOR, N_COLOR) if sign > 0 else (N_COLOR, S_COLOR)
    for half_dir, color in [(-1, near_color), (+1, far_color)]:
        local = np.array([(0, -hw), (half_dir * hl, -hw), (half_dir * hl, hw), (0, hw)])
        pts = local @ rot.T + np.array([x, y])
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=color, edgecolor="black",
                                  linewidth=0.6, zorder=5))

    for local_cx, color in [(-hl, near_color), (hl, far_color)]:
        cx, cy = np.array([local_cx, 0]) @ rot.T + np.array([x, y])
        cap = Ellipse((cx, cy), width=hw * 0.9, height=hw * 2.0, angle=body_theta_deg,
                       facecolor=color, edgecolor="black", linewidth=0.6, zorder=6)
        ax.add_patch(cap)


def case_limits(L_M, phi_deg, margin=15):
    r = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, loads=[], return_curve=True)
    cx, cy = r["curve_x_mm"], r["curve_y_mm"]
    xlim = (min(cx.min(), -margin) - margin, cx.max() + margin)
    ylim = (min(cy.min(), -margin) - margin, cy.max() + margin)
    return xlim, ylim


def plot_case(ax, L_M, phi_deg, xlim, ylim):
    r = fm.solve_shape(L_M=L_M, phi_deg=phi_deg, loads=[], return_curve=True)
    cx, cy = r["curve_x_mm"], r["curve_y_mm"]

    draw_wire(ax, r["curve_s_mm"], cx, cy, L_M)
    ax.plot(0, 0, marker="s", color="black", markersize=9, zorder=6)

    draw_magnet(ax, r["x_LM"], r["y_LM"], r["theta_LM_deg"], -1.0, MOM_VISUAL_LEN_MM, MOM_VISUAL_WIDTH_MM)
    draw_magnet(ax, r["x_L"], r["y_L"], r["theta_L_deg"], +1.0, MAIN_VISUAL_LEN_MM, MAIN_VISUAL_WIDTH_MM)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm, 로컬)")
    ax.set_ylabel("y (mm, 로컬)")
    ax.set_title(f"L_M={L_M:.0f}mm, phi={phi_deg:.0f}도", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)


if __name__ == "__main__":
    # 모든 패널의 좌표축을 2번째 케이스(CASES[1]) 기준으로 통일. 베이스(고정단)는 항상
    # 로컬(0,0)이라 왼쪽 여백을 없애고 x=0에 딱 붙게 시작.
    _, shared_ylim = case_limits(*CASES[1])
    shared_xlim = (-3, case_limits(*CASES[1])[0][1])

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (L_M, phi_deg) in zip(axes.flat, CASES):
        plot_case(ax, L_M, phi_deg, shared_xlim, shared_ylim)
    for ax in axes.flat[len(CASES):]:
        ax.axis("off")

    legend_elems = [
        plt.Line2D([0], [0], marker="s", color="black", linestyle="", markersize=9, label="베이스(고정단)"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.3, label="제어 와이어(MOM 구동)"),
        plt.Line2D([0], [0], color="#333333", linewidth=1.0, linestyle=(0, (2, 2)), label="나머지 내강(가이드)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=N_COLOR, edgecolor="black", label="N극"),
        plt.Rectangle((0, 0), 1, 1, facecolor=S_COLOR, edgecolor="black", label="S극"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("MSCR-MOM 카테터 형상 + 내부 자석 배치(N/S 방향 포함)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    out_path = os.path.join(OUT_DIR, "free_shape_with_magnets_6panel.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"저장: {out_path}")
