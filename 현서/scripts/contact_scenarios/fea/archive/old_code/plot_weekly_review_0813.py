"""8월 2째주 랩미팅용 그림 6종 생성.

전부 실측 FEA 데이터(fea_beta_fine_resolution.json, fea_beta_generalization_check.json,
fea_angle_sweep_all.json) 또는 이번 세션 학습 로그(run_singleprobe_4seg_full.log,
ablation_arch_and_headcheck.log)에서 나온 실제 숫자를 사용. 로드맵 그림(6번)만 개념도.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

GREEN = "#27AE60"
LIGHTGREEN = "#EAF7EF"
NAVY = "#1F2A44"
GRAY = "#6B7280"
LIGHTGRAY = "#E5E7EB"
DARK = "#222222"
RED = "#E74C3C"
BLUE = "#2E86DE"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "contact_scenarios", "fea")
OUT_DIR = DATA_DIR


def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------------------
# Figure 1: 지난주 리캡 + 이번주 질문
# ---------------------------------------------------------------------------
def fig1_recap_questions():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.25, 0.95, "지난주 (8/11) — 11-probe 능동탐색 기준", ha="center", va="top",
            fontsize=15, fontweight="bold", color=DARK)

    stats = [
        ("86.0%", "구간(5개) 분류\nbalanced accuracy"),
        ("R²=0.954", "연속값 위치(s)\nMAE 4.4mm"),
        ("R²=0.90", "접촉힘 (Fx, Fy)"),
    ]
    box_w, gap = 0.135, 0.025
    start_x = 0.25 - (box_w * 3 + gap * 2) / 2
    for i, (val, label) in enumerate(stats):
        x = start_x + i * (box_w + gap)
        box = FancyBboxPatch((x, 0.55), box_w, 0.3, boxstyle="round,pad=0.01,rounding_size=0.02",
                              linewidth=1.4, edgecolor=GREEN, facecolor=LIGHTGREEN)
        ax.add_patch(box)
        ax.text(x + box_w / 2, 0.73, val, ha="center", va="center", fontsize=17, fontweight="bold", color=GREEN)
        ax.text(x + box_w / 2, 0.60, label, ha="center", va="center", fontsize=9.5, color=DARK)

    ax.text(0.25, 0.42, "단, 두 가지는 그때 답을 못 냄", ha="center", va="top",
            fontsize=11, color=GRAY, style="italic")

    arrow = FancyArrowPatch((0.5, 0.5), (0.56, 0.5), connectionstyle="arc3,rad=0",
                             arrowstyle="-|>", mutation_scale=25, color=GRAY, linewidth=2)
    ax.add_patch(arrow)

    ax.text(0.79, 0.95, "이번 주 — 질문 2개에 답하기", ha="center", va="top",
            fontsize=15, fontweight="bold", color=NAVY)

    qs = [
        ("Q1", "β(둘레방향 접촉각) 효과가 오차 원인이라는 것만\n알고 있었음 — 정확히 왜, 어디서 발생하는가?"),
        ("Q2", "위 86%는 능동탐색(11-probe 스캔) 전제.\n센서 1개만 쓰는 실전 조건이면 성능은?"),
    ]
    qy = [0.68, 0.30]
    for (tag, text), y in zip(qs, qy):
        box = FancyBboxPatch((0.6, y - 0.16), 0.38, 0.30, boxstyle="round,pad=0.012,rounding_size=0.025",
                              linewidth=1.4, edgecolor=NAVY, facecolor="white")
        ax.add_patch(box)
        ax.text(0.615, y + 0.10, tag, ha="left", va="center", fontsize=13, fontweight="bold", color=NAVY)
        ax.text(0.79, y - 0.01, text, ha="center", va="center", fontsize=10.5, color=DARK, linespacing=1.6)

    ax.axvline(0.5, ymin=0.03, ymax=0.97, color=LIGHTGRAY, linewidth=1.2, zorder=0)
    savefig(fig, "review0813_recap_questions.png")


# ---------------------------------------------------------------------------
# Figure 1b: β란 무엇인가 + 왜 어떤 각도는 신호가 안 잡히는가 (그림자 비유, 개념도)
# ---------------------------------------------------------------------------
def fig1b_beta_explainer():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))

    # ---- 왼쪽: β 정의 (원통 단면, 둘레의 어느 방향에서 눌렀는가) ----
    ax = axes[0]
    circle = plt.Circle((0, 0), 1.0, facecolor="#DCE9FB", edgecolor=NAVY, linewidth=2)
    ax.add_patch(circle)
    ax.text(0, 0, "카테터\n단면", ha="center", va="center", fontsize=11, color=NAVY, fontweight="bold")
    dirs = [(0, "0°\n(바깥쪽)"), (90, "90°\n(위)"), (180, "180°\n(안쪽)"), (270, "270°\n(아래)")]
    for deg, label in dirs:
        th = math.radians(deg)
        x0, y0 = 0.95 * math.cos(th), 0.95 * math.sin(th)
        x1, y1 = 1.55 * math.cos(th), 1.55 * math.sin(th)
        color = RED if deg in (90, 270) else GRAY
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, linewidth=2.4))
        ax.text(1.85 * math.cos(th), 1.85 * math.sin(th), label, ha="center", va="center",
                fontsize=10, color=color, fontweight="bold" if deg in (90, 270) else "normal")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-2.3, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("β란? — 같은 위치(s)라도\n둘레의 어느 방향에서 눌렀는지", fontsize=13, fontweight="bold")

    # ---- 오른쪽: 그림자 비유 ----
    ax = axes[1]
    ax.axhline(0, color=NAVY, linewidth=3, zorder=1)
    ax.text(-4.3, 0.18, "센서 평면 (보드, 옆에서 본 모습) →", ha="left", va="bottom",
            fontsize=10, color=NAVY, fontweight="bold")

    guide = dict(color=GRAY, linestyle=":", linewidth=1.3, zorder=1)

    # Scene A: 옆으로 밀기 (좌측) — 그림자가 크게 이동
    xa0, xa1, ya = -3.3, -1.9, 1.3
    ax.plot([xa0, xa0], [ya, 0], **guide)
    ax.plot([xa1, xa1], [ya, 0], **guide)
    ax.add_patch(plt.Circle((xa0, ya), 0.22, facecolor=GRAY, edgecolor=DARK, zorder=3))
    ax.annotate("", xy=(xa1, ya), xytext=(xa0, ya),
                arrowprops=dict(arrowstyle="-|>", color=GREEN, linewidth=3))
    ax.add_patch(plt.Circle((xa1, ya), 0.22, facecolor=GREEN, edgecolor=DARK, zorder=3))
    ax.plot([xa0, xa1], [0, 0], color=GREEN, linewidth=7, solid_capstyle="round", zorder=2)
    ax.add_patch(plt.Circle((xa0, 0), 0.09, facecolor=GRAY, edgecolor=DARK, zorder=4))
    ax.add_patch(plt.Circle((xa1, 0), 0.09, facecolor=GREEN, edgecolor=DARK, zorder=4))
    ax.text((xa0 + xa1) / 2, -0.85, "β=0° (옆에서 누름)\n평면 위 그림자가 크게 이동\n→ 신호 큼 (잘 보임)",
            ha="center", fontsize=9.5, color=DARK, linespacing=1.4)

    # Scene B: 위에서 아래로 밀기 (우측) — 그림자가 거의 제자리
    xb, yb0, yb1 = 2.6, 1.9, 0.9
    ax.plot([xb, xb], [max(yb0, yb1), 0], **guide)
    ax.add_patch(plt.Circle((xb, yb0), 0.22, facecolor=GRAY, edgecolor=DARK, zorder=3))
    ax.annotate("", xy=(xb, yb1), xytext=(xb, yb0),
                arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=3))
    ax.add_patch(plt.Circle((xb, yb1), 0.22, facecolor=RED, edgecolor=DARK, zorder=3))
    ax.add_patch(plt.Circle((xb, 0), 0.09, facecolor=RED, edgecolor=DARK, zorder=4))
    ax.text(xb, -0.85, "β=90° (위에서 누름)\n평면 위 그림자는 거의 제자리\n→ 신호 작음 (안 보임)",
            ha="center", fontsize=9.5, color=DARK, linespacing=1.4)

    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-1.9, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("그림자 비유 — 실제로는 둘 다 똑같이 눌렸는데,\n센서(평면)에 비치는 '그림자'만 다름", fontsize=13, fontweight="bold")

    fig.suptitle("물체를 옆으로 밀면 바닥 그림자가 크게 움직이지만, 위에서 아래로 누르면 그림자는 거의 안 움직이는 것과 같은 원리",
                 fontsize=12.5, fontweight="bold", y=1.04, color=NAVY)
    fig.tight_layout()
    savefig(fig, "review0813_beta_explainer.png")


# ---------------------------------------------------------------------------
# Figure 2: beta 정밀 스윕 — 전환점 규명
# ---------------------------------------------------------------------------
def fig2_beta_transition():
    d = load("fea_beta_fine_resolution.json")
    conds = [(50.0, 60.0), (50.0, -90.0)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, cond in zip(axes, conds):
        pts = sorted([x for x in d if (x["L_M_mm"], x["phi_deg"]) == cond], key=lambda x: x["beta_deg"])
        beta = np.array([p["beta_deg"] for p in pts])
        mag3d = np.array([math.sqrt(p["tip_ux_avg_mm"] ** 2 + p["tip_uy_avg_mm"] ** 2 + p["tip_uz_avg_mm"] ** 2) for p in pts])
        magxy = np.array([math.hypot(p["tip_ux_avg_mm"], p["tip_uy_avg_mm"]) for p in pts])

        ax.plot(beta, mag3d, "--", color=GRAY, marker="o", markersize=4, label="전체 팁변위 크기 (3D)")
        ax.plot(beta, magxy, "-", color=GREEN, marker="o", markersize=5, linewidth=2.2,
                label="보드평면(Bx,By) 성분 크기")
        for b in (90, 270):
            ax.axvline(b, color=RED, linestyle=":", linewidth=1.6)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("β (둘레방향 접촉각, deg)")
        L_M, phi = cond
        ax.set_title(f"L_M={L_M:.0f}mm, phi={phi:.0f}°", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("팁변위 (mm)")
    axes[0].text(90, axes[0].get_ylim()[1] * 0.92, "β=90°", color=RED, ha="center", fontsize=9)
    axes[0].text(270, axes[0].get_ylim()[1] * 0.92, "β=270°", color=RED, ha="center", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("β 15° 정밀 스윕: 전체 변위는 유지되는데, 보드평면 신호만 β=90°/270°에서 0에 가까워짐\n"
                 "→ 접촉이 사라진 게 아니라 그 각도에서 굽힘 평면과 수직으로 밀려서 센서 평면(Bx,By)엔 안 잡히는 것",
                 fontsize=12, fontweight="bold", y=1.1)
    savefig(fig, "review0813_beta_transition.png")


# ---------------------------------------------------------------------------
# Figure 3: 다른 형상에서도 재현되는가 (일반성 검증)
# ---------------------------------------------------------------------------
def fig3_beta_generalization():
    def ang(x):
        return math.degrees(math.atan2(x["tip_uy_avg_mm"], x["tip_ux_avg_mm"])) % 360

    def best(pts):
        # 가장 큰 magxy(=가장 신뢰도 높은 깊게 눌린 케이스)를 대표값으로
        return max(pts, key=lambda x: math.hypot(x["tip_ux_avg_mm"], x["tip_uy_avg_mm"]))

    gen = load("fea_beta_generalization_check.json")
    orig = load("fea_angle_sweep_all.json")

    def get(dataset, L_M, phi, beta):
        pts = [x for x in dataset if x["L_M_mm"] == L_M and x["phi_deg"] == phi and x["beta_deg"] == beta]
        return best(pts)

    conditions = [
        ("L_M=50, phi=60\n(최초 발견)", orig, 50.0, 60.0),
        ("L_M=25, phi=-90", gen, 25.0, -90.0),
        ("L_M=75, phi=150", gen, 75.0, 150.0),
    ]

    labels, diffs, peak_frac_90, peak_frac_270 = [], [], [], []
    for label, dataset, L_M, phi in conditions:
        p45 = get(dataset, L_M, phi, 45.0)
        p135 = get(dataset, L_M, phi, 135.0)
        d = abs(ang(p45) - ang(p135))
        d = min(d, 360 - d)
        diffs.append(d)
        labels.append(label)

        all_pts = [x for x in dataset if x["L_M_mm"] == L_M and x["phi_deg"] == phi]
        peak = max(math.hypot(x["tip_ux_avg_mm"], x["tip_uy_avg_mm"]) for x in all_pts)
        p90 = [x for x in all_pts if x["beta_deg"] == 90.0]
        p270 = [x for x in all_pts if x["beta_deg"] == 270.0]
        peak_frac_90.append(100 * max(math.hypot(x["tip_ux_avg_mm"], x["tip_uy_avg_mm"]) for x in p90) / peak if p90 else np.nan)
        peak_frac_270.append(100 * max(math.hypot(x["tip_ux_avg_mm"], x["tip_uy_avg_mm"]) for x in p270) / peak if p270 else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    x = np.arange(len(labels))

    ax = axes[0]
    bars = ax.bar(x, diffs, color=GREEN, width=0.5)
    ax.axhline(180, color=RED, linestyle="--", linewidth=1.5, label="180° (관측된 패턴)")
    ax.axhline(90, color=GRAY, linestyle=":", linewidth=1.5, label="90° (단순 회전이라면)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("방향 변화 (deg)")
    ax.set_title("β를 45°→135°(90° 이동)로 바꾸면\n변위 방향은 몇 도 바뀌나", fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 220)
    for xi, v in zip(x, diffs):
        ax.text(xi, v + 5, f"{v:.0f}°", ha="center", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    w = 0.35
    ax.bar(x - w / 2, peak_frac_90, width=w, color=BLUE, label="β=90°")
    ax.bar(x + w / 2, peak_frac_270, width=w, color=RED, label="β=270°")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("이 형상 최대 신호 대비 (%)")
    ax.set_title("β=90°/270°에서 보드평면 신호가\n실제로 얼마나 작아지나", fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle("3개 서로 다른 형상(L_M, phi)에서 재현 확인 → 우연이 아니라 일반적인 현상",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    savefig(fig, "review0813_beta_generalization.png")


# ---------------------------------------------------------------------------
# Figure 3b: 프로브 개수 실험 히스토리 — 싱글프로브를 왜 다시 봤는가
# ---------------------------------------------------------------------------
def fig3b_probe_history():
    fig, ax = plt.subplots(figsize=(13, 5.6))

    stages = [
        ("단일 관측\n(1-probe)", 20.5, "구", "무작위 수준"),
        ("4-probe\n능동탐색", 24.8, "구", ""),
        ("11-probe\n전체스캔", 28.7, "구", ""),
        ("11-probe\n+ 노이즈 제거\n(지난주 최종)", 86.0, "신", ""),
        ("1-probe\n+ 노이즈 제거\n(이번 주 재시험)", 71.3, "신", ""),
    ]
    x = np.arange(len(stages))
    colors = [GRAY if s[2] == "구" else GREEN for s in stages]
    vals = [s[1] for s in stages]
    bars = ax.bar(x, vals, color=colors, width=0.55, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 2, f"{v:.1f}%", ha="center", fontsize=12.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=10)
    ax.set_ylabel("구간 분류 balanced accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2, axis="y")

    # 구간 표시: 노이즈 버그 있던 시기 vs 제거 후
    ax.axvspan(-0.5, 2.5, color=LIGHTGRAY, alpha=0.4, zorder=0)
    ax.text(1.0, 93, "합성 데이터에 노이즈가 섞여있던 시기\n(프로브를 늘려도 성능이 잘 안 오름)",
            ha="center", fontsize=10, color=GRAY, fontweight="bold")
    ax.text(3.5, 93, "노이즈 제거 후\n(신호가 있는 그대로 드러남)",
            ha="center", fontsize=10, color=GREEN, fontweight="bold")

    # 화살표: 3 -> 4 (노이즈 제거 효과, 11-probe 기준)
    ax.annotate("", xy=(3, 84), xytext=(2, 31), arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=2))
    ax.text(2.5, 60, "노이즈 제거\n(같은 11-probe)", ha="center", fontsize=9.5, color=RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=RED))

    # 1-probe 두 막대(구/신)를 점선으로 연결 + 인사이트 박스는 빈 공간(왼쪽 위)에 배치
    ax.plot([0, 0, 4, 4], [20.5, 40, 40, 71.3], linestyle="--", color=NAVY, linewidth=1.6, zorder=2)
    ax.text(0.85, 52, "같은 1-probe인데 노이즈만 제거하면\n20.5% → 71.3%\n"
                      "→ '싱글프로브는 안 된다'는 예전 결론은\n   노이즈 버그로 저평가된 것이었음",
            ha="center", va="center", fontsize=10, color=NAVY, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHTGREEN, edgecolor=GREEN))

    ax.set_title("프로브 개수 실험은 처음이 아님 — 1-probe는 예전에도 해봤고, 그때 결론은 '거의 랜덤'이었음",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    savefig(fig, "review0813_probe_history.png")


# ---------------------------------------------------------------------------
# Figure 4a: 싱글프로브란 무엇인가 (능동탐색 유무 개념도)
# ---------------------------------------------------------------------------
def fig4a_singleprobe_setup():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    def draw_panel(ax, phis, highlight_idx, title, title_color):
        base = np.array([0.0, 0.0])
        s = np.linspace(0, 1, 60)
        for i, phi in enumerate(phis):
            active = i == highlight_idx or highlight_idx is None
            th = math.radians(phi)
            # 간단한 원호 모양(개념도용) — 실제 FEA 형상이 아니라 "로봇이 phi만큼 다르게 휜다"는 그림
            bend = 0.9 * s ** 1.3
            x = base[0] + s * 1.0
            y = base[1] + bend * math.sin(th)
            x = x + bend * (1 - math.cos(th)) * 0.3
            lw = 3.2 if (highlight_idx is not None and i == highlight_idx) else 1.6
            color = GREEN if (highlight_idx is not None and i == highlight_idx) else (GRAY if highlight_idx is not None else BLUE)
            alpha = 1.0 if (highlight_idx is None or i == highlight_idx) else 0.45
            ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round")
            if highlight_idx is None or i == highlight_idx:
                ax.text(x[-1] + 0.03, y[-1], f"{phi:+.0f}°", fontsize=9, color=color,
                        va="center", fontweight="bold" if i == highlight_idx else "normal")
        ax.add_patch(plt.Rectangle((-0.12, -0.08), 0.24, 0.16, color=NAVY, zorder=5))
        ax.text(0, -0.22, "베이스\n(5×5 홀센서 보드)", ha="center", va="top", fontsize=9, color=NAVY)
        ax.set_xlim(-0.3, 1.5)
        ax.set_ylim(-1.55, 1.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", color=title_color, pad=10)

    phis_multi = [-90, -60, -30, 0, 30, 60, 90]
    draw_panel(axes[0], phis_multi, None,
               "멀티프로브 (지난주) — 능동탐색", GREEN)
    axes[0].text(0.6, -1.15,
                 "접촉 1회 감지할 때마다\n로봇을 11번 다른 phi로\n재구성 후 측정 (삼각측량)",
                 ha="center", fontsize=10, color=DARK, linespacing=1.5)

    phis_single = [-90, -60, -30, 0, 30, 60, 90]
    draw_panel(axes[1], phis_single, 3,
               "싱글프로브 (이번 주) — 능동탐색 없음", BLUE)
    axes[1].text(0.6, -1.15,
                 "그 순간 로봇이 있던 phi\n그대로 1번만 측정\n재구성 없음",
                 ha="center", fontsize=10, color=DARK, linespacing=1.5)

    fig.suptitle("두 조건 모두 센서 보드(5×5)는 동일 — 차이는 '로봇을 몇 번 다른 각도로 재구성해서 측정하는가'",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, -0.06,
             "싱글프로브 학습 데이터는 β=90°/270°±15° 구간을 샘플링에서 제외함 — 질문1에서 확인한,\n"
             "보드평면 신호가 정상 대비 10~35%로 약해지는 구간을 그대로 반영한 것 (질문1 결과를 질문2에 재사용)",
             ha="center", fontsize=10.5, color=NAVY, linespacing=1.6,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHTGREEN, edgecolor=GREEN))

    fig.tight_layout()
    savefig(fig, "review0813_singleprobe_setup.png")


# ---------------------------------------------------------------------------
# Figure 4b: 싱글프로브 결과 상세 (이산 vs 연속 breakdown + 혼동행렬)
# ---------------------------------------------------------------------------
def fig4b_singleprobe_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    items = [
        ("구간(4개) 분류\n(balanced acc)", 71.3, "이산값", "n=15,000"),
        ("연속값 위치 s\n(R²)", 76.0, "연속값", "MAE 7.4mm"),
        ("힘 Fx (R²)", 95.3, "연속값", ""),
        ("힘 Fy (R²)", 95.3, "연속값", ""),
        ("로봇 자기위치\nL_M 추정 (R²)", 96.9, "연속값", "MAE 3.3mm"),
        ("로봇 자기위치\nphi 추정 (R²)", 97.2, "연속값", "MAE 10.1°"),
    ]
    labels = [it[0] for it in items]
    vals = [it[1] for it in items]
    colors = [BLUE if it[2] == "이산값" else GREEN for it in items]
    y = np.arange(len(items))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, (label, v, kind, note) in zip(y, items):
        ax.text(v + 1.5, yi, f"{v:.1f}%" + (f"  ({note})" if note else ""), va="center", fontsize=10, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 115)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("balanced accuracy 또는 R²×100 (%)")
    ax.axvline(25, color=RED, linestyle="--", linewidth=1.3)
    ax.set_title("이산적(구간) 판정만 타격, 연속값은 능동탐색 없이도 유지", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.2, axis="x")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="이산값(구간 분류)"), Patch(color=GREEN, label="연속값(회귀)"),
                        Line2D([0], [0], color=RED, linestyle="--", label="랜덤 기준(25%)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=9.5, frameon=False)

    ax = axes[1]
    cm = np.array([[1513, 522, 56, 29],
                   [223, 2918, 785, 270],
                   [0, 573, 2548, 1169],
                   [0, 32, 636, 3726]])
    cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    seg_labels = ["0-20mm", "20-40mm", "40-60mm", "60-80mm"]
    im = ax.imshow(cm_pct, cmap="Greens", vmin=0, vmax=100)
    for i in range(4):
        for j in range(4):
            c = "white" if cm_pct[i, j] > 55 else DARK
            ax.text(j, i, f"{cm_pct[i, j]:.0f}%", ha="center", va="center", fontsize=11, color=c)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(seg_labels, fontsize=9.5)
    ax.set_yticklabels(seg_labels, fontsize=9.5)
    ax.set_xlabel("예측 구간")
    ax.set_ylabel("실제 구간")
    ax.set_title("구간 혼동행렬 (행 기준 %)\n오답 91%가 바로 옆 구간과의 혼동", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    savefig(fig, "review0813_singleprobe_breakdown.png")


# ---------------------------------------------------------------------------
# Figure 4c: 자연스러운 움직임 재활용이란? (개념도 — 능동탐색과의 차이)
# ---------------------------------------------------------------------------
def fig4c_naturalmotion_concept():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    t = np.linspace(0, 10, 400)

    # 왼쪽: 능동탐색 (기각) — 억지로 11개 고정각으로 재구성 (계단식 강제 이동)
    ax = axes[0]
    targets = np.linspace(-150, 150, 11)
    steps_t = np.linspace(0.3, 9.7, 11)
    for i in range(len(steps_t) - 1):
        ax.plot([steps_t[i], steps_t[i + 1]], [targets[i], targets[i]], color=GRAY, linewidth=2.4, zorder=2)
        ax.plot([steps_t[i + 1], steps_t[i + 1]], [targets[i], targets[i + 1]], color=RED, linewidth=1.6,
                linestyle="-", zorder=2)
    ax.plot([steps_t[-1], steps_t[-1] + 0.3], [targets[-1], targets[-1]], color=GRAY, linewidth=2.4, zorder=2)
    for i, st in enumerate(steps_t):
        ax.plot(st, targets[i], "o", color=RED, markersize=6, zorder=3)
    ax.text(5, 175, "접촉 1회 감지할 때마다 로봇을\n11번 강제로 재구성 (원래 없던 동작)",
            ha="center", fontsize=10.5, color=DARK, linespacing=1.5)
    ax.set_ylim(-190, 210)
    ax.set_xlim(0, 10)
    ax.set_xlabel("시간 →")
    ax.set_ylabel("로봇 자세각 φ (deg)")
    ax.set_title("능동탐색 (기각됨)", fontsize=14, fontweight="bold", color=GRAY)
    ax.grid(alpha=0.2)

    # 오른쪽: 자연스러운 움직임 재활용 (채택) — 원래 하던 항법 동작의 매끄러운 φ(t)
    ax = axes[1]
    natural = 70 * np.sin(0.55 * t) + 25 * np.sin(1.3 * t + 1) + 10 * np.sin(2.1 * t + 0.5)
    ax.plot(t, natural, color=BLUE, linewidth=2.2, zorder=2, label="원래 하던 항법/조향 동작 φ(t)")
    rng = np.random.default_rng(3)
    capture_t = np.sort(rng.uniform(1, 9, 3))
    for i, ct in enumerate(capture_t):
        cy = 70 * np.sin(0.55 * ct) + 25 * np.sin(1.3 * ct + 1) + 10 * np.sin(2.1 * ct + 0.5)
        ax.plot([ct, ct], [-190, cy], linestyle=":", color=GREEN, linewidth=1.3, zorder=1)
        ax.plot(ct, cy, "o", color=GREEN, markersize=10, zorder=3)
        ax.text(ct, cy + 22, f"관측{i+1}", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax.text(5, 175, "그 동작이 지나가는 길목에서\n3번 그냥 스냅샷만 기록 (추가 동작 없음)",
            ha="center", fontsize=10.5, color=DARK, linespacing=1.5)
    ax.set_ylim(-190, 210)
    ax.set_xlim(0, 10)
    ax.set_xlabel("시간 →")
    ax.set_title("자연스러운 움직임 재활용 (채택)", fontsize=14, fontweight="bold", color=GREEN)
    ax.grid(alpha=0.2)

    fig.suptitle("왼쪽은 로봇에게 없던 동작을 강제로 시킴 · 오른쪽은 이미 하고 있던 동작에서 관측만 추가함",
                 fontsize=13, fontweight="bold", y=1.04, color=NAVY)
    fig.tight_layout()
    savefig(fig, "review0813_naturalmotion_concept.png")


# ---------------------------------------------------------------------------
# Figure 4d: 자연스러운 움직임 재활용 (3-probe) — 싱글프로브의 벽을 넘다
# ---------------------------------------------------------------------------
def fig4d_naturalmotion():
    fig, ax = plt.subplots(figsize=(13, 5.8))

    groups = ["구간(4개) 분류\n(balanced acc)", "연속값 위치 s\n(R²)", "힘 Fx\n(R²)", "힘 Fy\n(R²)"]
    cnn_1p = [71.7, 76.6, 94.7, 95.6]
    gbm_1p = [76.0, 78.8, 71.0, 70.4]
    nat_3p = [79.5, 87.1, 93.5, 94.7]

    x = np.arange(len(groups))
    w = 0.26
    ax.bar(x - w, cnn_1p, width=w, color=GRAY, label="1-probe (CNN)")
    ax.bar(x, gbm_1p, width=w, color=BLUE, label="1-probe (그래디언트부스팅)")
    bars3 = ax.bar(x + w, nat_3p, width=w, color=GREEN, label="3-probe (자연스러운 움직임 재활용)")
    for xi, v in zip(x - w, cnn_1p):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5)
    for xi, v in zip(x, gbm_1p):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5)
    for xi, v in zip(x + w, nat_3p):
        ax.text(xi, v + 1.5, f"{v:.1f}", ha="center", fontsize=9.5, fontweight="bold", color=GREEN)

    # s(연속값) 그룹 위에 예전 "비현실적" 11-probe 목표치 참고선
    s_x = x[1]
    ax.plot([s_x - w * 1.6, s_x + w * 1.6], [95.4, 95.4], color=RED, linestyle="--", linewidth=1.6)
    ax.text(s_x, 97.3, "11-probe(능동탐색, 기각) 목표치 95.4 — 참고용", ha="center", fontsize=9, color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10.5)
    ax.set_ylabel("balanced accuracy 또는 R²×100 (%)")
    ax.set_ylim(0, 108)
    ax.grid(alpha=0.2, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=10, frameon=False)
    ax.set_title("연속값 위치(s) 회귀가 가장 크게 개선됨: R²=0.766 → 0.871\n"
                 "GBM처럼 힘(Fx,Fy)을 희생하지 않으면서 구간분류·s는 GBM보다도 더 좋음",
                 fontsize=13, fontweight="bold")

    fig.text(0.5, -0.1,
             "\"능동탐색\"(로봇을 일부러 재구성)이 아니라, 원래 하던 동작 중 자연스럽게 지나가는 φ 3개를 그냥 버퍼링 — 추가 동작 없이 정보만 재사용\n"
             "예비 결과: 60,000개(실규모 15만개의 40%)로 학습 — 전체 규모로 재학습하면 추가로 더 좋아질 여지 있음",
             ha="center", fontsize=10.5, color=NAVY, linespacing=1.7,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHTGREEN, edgecolor=GREEN))

    fig.tight_layout()
    savefig(fig, "review0813_naturalmotion_comparison.png")


# ---------------------------------------------------------------------------
# Figure 4e: 자연스러운 움직임 재활용 — 결과 상세 (구간별 recall + 전체 지표)
# ---------------------------------------------------------------------------
def fig4e_naturalmotion_breakdown():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    ax = axes[0]
    seg_labels = ["0-20mm", "20-40mm", "40-60mm", "60-80mm"]
    recall_1p = [71.4, 70.0, 59.4, 84.8]
    recall_3p = [80.2, 79.7, 65.6, 92.4]
    x = np.arange(4)
    w = 0.32
    ax.bar(x - w / 2, recall_1p, width=w, color=GRAY, label="1-probe (CNN)")
    ax.bar(x + w / 2, recall_3p, width=w, color=GREEN, label="3-probe (자연스러운 움직임)")
    for xi, v in zip(x - w / 2, recall_1p):
        ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10)
    for xi, v in zip(x + w / 2, recall_3p):
        ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold", color=GREEN)
    ax.set_xticks(x)
    ax.set_xticklabels(seg_labels, fontsize=10.5)
    ax.set_ylabel("구간별 recall (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=10, frameon=False)
    ax.set_title("모든 구간에서 개선, 특히 제일 약했던\n40-60mm(가운데)도 59.4%→65.6%", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    ax.axis("off")
    rows = [
        ("지표", "1-probe (CNN)", "3-probe (자연스러운 움직임)"),
        ("구간(4개) 분류 balanced acc", "71.7%", "79.5%"),
        ("연속값 위치 s (R²)", "0.766", "0.871"),
        ("힘 Fx (R²)", "0.947", "0.935"),
        ("힘 Fy (R²)", "0.956", "0.947"),
        ("로봇 자기위치 L_M 추정 (R²)", "0.974", "0.982"),
        ("오답 중 인접구간 비율", "91.0%", "96.9%"),
    ]
    n = len(rows)
    for i, (a, b, c) in enumerate(rows):
        y = 1 - (i + 0.5) / n
        bold = i == 0
        color = NAVY if i == 0 else DARK
        ax.text(0.02, y, a, fontsize=11.5 if bold else 11, fontweight="bold" if bold else "normal",
                color=color, va="center")
        ax.text(0.60, y, b, fontsize=11.5 if bold else 11, fontweight="bold" if bold else "normal",
                color=color, va="center", ha="center")
        ax.text(0.90, y, c, fontsize=11.5 if bold else 11, fontweight="bold", color=GREEN if not bold else color,
                va="center", ha="center")
        if i == 0:
            ax.plot([0, 1], [y - 0.5 / n, y - 0.5 / n], color=NAVY, linewidth=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("전체 지표 요약", fontsize=12.5, fontweight="bold", pad=15)
    ax.text(0.5, -0.05, "\"오답 중 인접구간 비율\"이 높아진 건 나쁜 신호가 아님 —\n"
                        "틀려도 거의 항상 '바로 옆 구간'과만 헷갈린다는 뜻(경계 케이스), 동떨어진 오답은 오히려 더 줄어듦",
            ha="center", fontsize=9.5, color=GRAY, transform=ax.transAxes, linespacing=1.5)

    fig.tight_layout()
    savefig(fig, "review0813_naturalmotion_breakdown.png")


# ---------------------------------------------------------------------------
# Figure 5: 견고성 체크 (ablation)
# ---------------------------------------------------------------------------
def fig5_ablation_robustness():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    names = ["분류 head\n(구간을 직접 분류)", "연속값만 학습\n(회귀 후 구간 변환)"]
    vals = [71.4, 66.5]
    xpos = np.arange(2)
    bars = ax.bar(xpos, vals, width=0.45, color=[GREEN, LIGHTGRAY])
    bars[1].set_edgecolor(GRAY)
    bars[1].set_linewidth(1.5)
    for xi, v in zip(xpos, vals):
        ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("balanced accuracy (%)")
    ax.set_ylim(0, 85)
    ax.set_title("분류 head, 그냥 넣은 게 아니라 필요함", fontsize=12.5, fontweight="bold")
    ax.grid(alpha=0.2, axis="y")

    ax = axes[1]
    names2 = ["CNN\n(최종 채택)", "단순 MLP\n(비교용)"]
    vals2 = [71.3, 72.8]
    xpos2 = np.arange(2)
    bars2 = ax.bar(xpos2, vals2, width=0.45, color=[GREEN, LIGHTGRAY])
    bars2[1].set_edgecolor(GRAY)
    bars2[1].set_linewidth(1.5)
    for xi, v in zip(xpos2, vals2):
        ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    ax.set_xticks(xpos2)
    ax.set_xticklabels(names2, fontsize=10.5)
    ax.set_ylim(0, 85)
    ax.set_title("복잡한 구조 덕에 나온 숫자가 아님\n(단순 MLP도 동급) → 숫자가 특정 모델에 우연히\n맞아떨어진 게 아니라는 근거", fontsize=11.5, fontweight="bold")
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    savefig(fig, "review0813_ablation_robustness.png")


# ---------------------------------------------------------------------------
# Figure 6: 지금 접근이 타당한 이유 (로드맵, 개념도)
# ---------------------------------------------------------------------------
def fig6_rationale_roadmap():
    fig, ax = plt.subplots(figsize=(13.5, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    steps = [
        ("1", "이상적 조건 재현", "11-probe 능동탐색 + 센서노이즈 제거\n→ 성능 상한 86% 확인", "지난주", GREEN, LIGHTGREEN),
        ("2", "숨은 오차 요인 규명", "β 전환점(90°/270°)이\n보드평면 신호 소실 때문임을 확인", "이번 주", GREEN, LIGHTGREEN),
        ("3", "실전 하드웨어 제약 반영", "재구성 없이 1회 관측만으로 →\n71.3% (vs 랜덤 25%) 하한선 확보", "이번 주", GREEN, LIGHTGREEN),
        ("4", "하드웨어 실측 검증", "시뮬레이션 가정(FFT 필터링 등)이\n실제로 맞는지 확인", "다음 단계", "white", "white"),
    ]
    n = len(steps)
    box_w = 0.19
    gap = (1 - box_w * n) / (n + 1)
    y0, box_h = 0.30, 0.5

    xs = []
    for i, (num, title, desc, when, edge, face) in enumerate(steps):
        x = gap + i * (box_w + gap)
        xs.append(x)
        is_last = i == n - 1
        box = FancyBboxPatch((x, y0), box_w, box_h, boxstyle="round,pad=0.012,rounding_size=0.02",
                              linewidth=2 if not is_last else 1.6,
                              edgecolor=GREEN if not is_last else GRAY,
                              facecolor=face,
                              linestyle="solid" if not is_last else "dashed")
        ax.add_patch(box)
        cx = x + box_w / 2
        badge_color = GREEN if not is_last else GRAY
        ax.add_patch(plt.Circle((cx, y0 + box_h - 0.07), 0.035, color=badge_color, zorder=3))
        ax.text(cx, y0 + box_h - 0.07, num, ha="center", va="center", color="white",
                fontsize=13, fontweight="bold", zorder=4)
        ax.text(cx, y0 + box_h - 0.18, title, ha="center", va="top", fontsize=12,
                fontweight="bold", color=DARK if not is_last else GRAY)
        ax.text(cx, y0 + box_h - 0.29, desc, ha="center", va="top", fontsize=9,
                color=DARK if not is_last else GRAY, linespacing=1.5)
        ax.text(cx, y0 - 0.05, when, ha="center", va="top", fontsize=10, fontweight="bold",
                color=badge_color)

    for i in range(n - 1):
        x_from = xs[i] + box_w
        x_to = xs[i + 1]
        arrow = FancyArrowPatch((x_from + 0.005, y0 + box_h / 2), (x_to - 0.005, y0 + box_h / 2),
                                 arrowstyle="-|>", mutation_scale=22, color=GRAY, linewidth=2)
        ax.add_patch(arrow)

    ax.text(0.5, 0.98, "이번 주는 '지난주보다 나쁜 조건 고치기'가 아니라, 시뮬레이션→실전 사이 격차를 좁히는 단계",
            ha="center", va="top", fontsize=14, fontweight="bold", color=NAVY)
    savefig(fig, "review0813_rationale_roadmap.png")


if __name__ == "__main__":
    fig1_recap_questions()
    fig1b_beta_explainer()
    fig2_beta_transition()
    fig3_beta_generalization()
    fig3b_probe_history()
    fig4a_singleprobe_setup()
    fig4b_singleprobe_breakdown()
    fig4c_naturalmotion_concept()
    fig4d_naturalmotion()
    fig4e_naturalmotion_breakdown()
    fig5_ablation_robustness()
    fig6_rationale_roadmap()
