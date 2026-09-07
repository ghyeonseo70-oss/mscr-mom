"""find_safe_force_range.py가 뽑은 것과 완전히 같은 조합(같은 L_M, phi, s, 힘방향, 힘크기)을
그대로 FEA(점하중, run_point_load.py)로 재현해서, 해석모델의 판정과 FEA의 실제 결과를
케이스별로 나란히 비교하는 "스코어카드". 교수님께 논리 흐름을 한눈에 설명하기 위한 그림:
"같은 조건 → 해석모델은 반전이라는데 → FEA로 직접 풀어보니 실제로는 안 뒤집힘"을 케이스마다
반복해서 보여줌.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
GOOD = "#0ca30c"      # 반전 없음(정상)
CRITICAL = "#d03b3b"  # 반전(불안정)
GRAY = "#898781"      # 판정 불가(미수렴/타임아웃)

# batch_run_point_load.py의 CASES와 동일 (여기서는 케이스 메타데이터만 필요)
CASES_META = {
    2: (35.70, -120.0, 78.28, 0.0), 201: (79.44, -120.0, 48.95, 0.0),
    8: (39.62, -60.0, 33.68, 0.0), 194: (71.88, -60.0, 28.66, 0.0),
    0: (58.22, 0.0, 8.69, 0.0), 225: (56.15, 0.0, 94.87, 0.0),
    7: (57.51, 60.0, 74.81, 0.0), 173: (24.10, 60.0, 30.21, 0.0),
    9: (72.21, 120.0, 59.28, 0.0), 197: (76.07, 120.0, 76.96, 0.0),
    1: (50.71, 60.0, 17.97, 20.0), 93: (79.13, 60.0, 46.34, 20.0),
    121: (57.97, -60.0, 79.18, 20.0),
}

results = {}
agg_path = os.path.join(DATA_DIR, "point_load_validation_results.json")
if os.path.exists(agg_path):
    for r in json.load(open(agg_path)):
        results[r["seed"]] = r
for path in glob.glob(os.path.join(HERE, "pl_*_result.json")):
    r = json.load(open(path))
    results.setdefault(r["seed"], r)

rows = []
for seed, (L_M, phi, s, thr) in CASES_META.items():
    if seed in results:
        r = results[seed]
        status = r["fea_status"]
        fea_reversed = r.get("fea_reversed")
    else:
        status, fea_reversed = "running", None
    rows.append({"seed": seed, "L_M": L_M, "phi": phi, "s": s, "thr": thr,
                 "status": status, "fea_reversed": fea_reversed})

# 정렬: 해석모델 즉시반전 그룹 먼저(thr=0), 생존군 나중; 그 안에서는 seed순
rows.sort(key=lambda r: (r["thr"], r["seed"]))

n_pending = sum(1 for r in rows if r["status"] == "running")
n_resolved = len(rows) - n_pending

FIG_H = 0.62 * len(rows) + 3.6
fig, ax = plt.subplots(figsize=(11.5, FIG_H))

X_ANALYTIC, X_FEA = 0.28, 0.72
for i, r in enumerate(rows):
    y = len(rows) - i
    analytic_reversed = r["thr"] < 0.5  # 즉시반전 그룹은 항상 반전으로 판정됨
    a_color = CRITICAL if analytic_reversed else GOOD
    a_label = "반전(불안정)" if analytic_reversed else f"{r['thr']:.0f}mN까지 정상"

    if r["status"] == "ok":
        f_color = CRITICAL if r["fea_reversed"] else GOOD
        f_label = "반전" if r["fea_reversed"] else "반전 없음(정상)"
        line_color = "#c3c2b7"
    elif r["status"] == "running":
        f_color, f_label, line_color = "#c3c2b7", "진행 중…", "#e1e0d9"
    else:
        f_color = GRAY
        f_label = "타임아웃(판정불가)" if r["status"] in ("no_dat", "crashed") else "미수렴(판정불가)"
        line_color = "#e1e0d9"

    ax.plot([X_ANALYTIC, X_FEA], [y, y], "-", color=line_color, linewidth=1.5, zorder=1)
    ax.plot(X_ANALYTIC, y, "o", color=a_color, markersize=15, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.plot(X_FEA, y, "o", color=f_color, markersize=15, zorder=3,
            markeredgecolor="white", markeredgewidth=1.2)
    ax.text(X_ANALYTIC, y, "", ha="center", va="center")
    ax.text(X_ANALYTIC - 0.03, y, a_label, ha="right", va="center", fontsize=9.5, color=a_color)
    ax.text(X_FEA + 0.03, y, f_label, ha="left", va="center", fontsize=9.5,
            color=f_color if r["status"] != "running" else MUTED)

    case_str = f"L_M={r['L_M']:.0f}mm φ={r['phi']:.0f}° s={r['s']:.0f}mm"
    ax.text(0.01, y, case_str, ha="left", va="center", fontsize=9, color=MUTED,
            transform=ax.get_yaxis_transform())

    if r["status"] == "ok":
        match = "일치" if (analytic_reversed != r["fea_reversed"]) else "불일치"
        # analytic_reversed=True(반전예측) 이면서 fea_reversed=False(반전없음)이 "예상대로"인 케이스
        icon = "✓" if analytic_reversed and not r["fea_reversed"] else ("✓" if not analytic_reversed and not r["fea_reversed"] else "!")
        ax.text(0.97, y, icon, ha="center", va="center", fontsize=13, fontweight="bold",
                color=GOOD if icon == "✓" else CRITICAL, transform=ax.get_yaxis_transform())

ax.set_xlim(0, 1)
ax.set_ylim(0.3, len(rows) + 1.3)
ax.axis("off")

ax.text(X_ANALYTIC, len(rows) + 1.0, "해석적 모델\n(빔이론+슈팅법)", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=INK)
ax.text(X_FEA, len(rows) + 1.0, "FEA\n(같은 조건, 점하중 직접 재현)", ha="center", va="bottom",
        fontsize=11, fontweight="bold", color=INK)
ax.text(0.01, len(rows) + 1.0, "케이스(같은 L_M/φ/s/힘)", ha="left", va="bottom",
        fontsize=10, color=MUTED, transform=ax.get_yaxis_transform())

n_agree_good = sum(1 for r in rows if r["status"] == "ok" and r["thr"] < 0.5 and not r["fea_reversed"])
n_immediate_total = sum(1 for r in rows if r["thr"] < 0.5)
title = ("해석모델이 \"반전된다\"고 예측한 조건을 FEA로 그대로 재현하면, 실제로는 반전이 없다")
fig.suptitle(title, fontweight="bold", fontsize=14, y=0.995)
subtitle = (f"find_safe_force_range.py와 동일한 무작위 조합 13개(즉시반전 예측 10개 + 해석모델도 안전하다고 한 대조군 3개)를\n"
            f"FEA로 직접 재현. 지금까지 판정 완료된 즉시반전 케이스 중 {n_agree_good}/{sum(1 for r in rows if r['status']=='ok' and r['thr']<0.5)}건에서 FEA는 반전 없음.")
if n_pending:
    subtitle += f"  (아직 {n_pending}건 실행 중 — 완료되는 대로 갱신 예정)"
fig.text(0.5, 1 - 0.65 / FIG_H, subtitle, ha="center", fontsize=9.3, color=MUTED, linespacing=1.6)

fig.tight_layout(rect=[0, 0, 1, 1 - 1.35 / FIG_H])
out_path = os.path.join(DATA_DIR, "point_load_scorecard.png")
fig.savefig(out_path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
print(f"저장: {out_path}")
print(f"해결됨: {n_resolved}/{len(rows)}, 대기중: {n_pending}")
