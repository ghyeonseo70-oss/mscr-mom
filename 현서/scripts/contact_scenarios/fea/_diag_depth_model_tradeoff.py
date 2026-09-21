"""2026-09-21(36번): 깊이 랜덤화가 이득인지 손해인지 공정하게 판정.

문제: 깊이 랜덤 학습 후 동일 99행 기준 지표가 전부 하락함(bal acc 87.0->78.9% 등).
그런데 **그 99행은 전부 깊이 0.10mm**라, "0.10mm 전용 모델"과 "여러 깊이용 모델"을
0.10mm 시험지로만 채점한 셈 - 전자가 유리한 게 당연해서 이 비교만으론 판정 불가.

방법: 두 체크포인트(깊이 고정 학습본 / 깊이 랜덤 학습본)를 **두 시험지 모두**에 돌림.
  시험지 A: 깊이 0.10mm 행만 (기존 조건)
  시험지 B: 깊이 0.05/0.20mm 행만 (새 조건 - 실제 운용에서 충돌 강도가 변하는 상황)
깊이 랜덤이 A에서 조금 손해를 보더라도 B에서 크게 이득이면 전체적으로 이득.
(35번에서 서로게이트 단위로는 이미 확인됨: 깊이 축 없으면 R^2가 0.45~0.65로 붕괴.)
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "force_model"))
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_CLASSES, BIN_WIDTH_MM = 4, 20.0
CKPTS = [
    ("깊이 고정 학습본", "position_segment_classifier_singleprobe_beta0180_4seg_before_depth_0921.pth"),
    ("깊이 랜덤 학습본", "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
]


def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    return (int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000) < int(frac * 10000)


class Net(nn.Module):
    def __init__(self, n_shape=0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 + 2, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, N_CLASSES)
        self.force_head = nn.Linear(128, 2)
        self.s_head = nn.Linear(128, 1)
        if n_shape:
            self.shape_head = nn.Linear(128, n_shape)

    def forward(self, x, config):
        h = self.trunk(torch.cat([self.encoder(x[:, 0]), config], dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1)


sensors = magpy.Collection([magpy.Sensor(position=(x, y, 15))
                            for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)])
main_magnet = magpy.magnet.Cylinder(polarization=(0, 0.4, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -0.4, 0), dimension=(1, 8))
robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(robot, sensors) * 1e6


all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)
holdout = [r for r in all_rows if is_holdout_row(r)]

X, ys, fs, ss, cs, depths = [], [], [], [], [], []
for r in holdout:
    try:
        rf = fm.solve_shape(L_M=r["L_M_mm"], phi_deg=r["phi_deg"], loads=[])
    except Exception:
        continue
    d_xL, d_yL, d_thL = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"], -r["tip_theta_deg_board"]
    if "mom_ux_avg_mm" in r:
        d_xLM, d_yLM, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    else:
        fr = r["L_M_mm"] / 100.0
        d_xLM, d_yLM, d_thLM = d_xL * fr, d_yL * fr, d_thL * fr
    B0 = compute_B(rf["x_LM"], rf["y_LM"], rf["theta_LM_deg"], rf["x_L"], rf["y_L"], rf["theta_L_deg"])
    B1 = compute_B(rf["x_LM"] + d_xLM, rf["y_LM"] + d_yLM, rf["theta_LM_deg"] + d_thLM,
                    rf["x_L"] + d_xL, rf["y_L"] + d_yL, rf["theta_L_deg"] + d_thL)
    X.append((B1 - B0).reshape(5, 5, 3).transpose(2, 0, 1))
    ys.append(min(N_CLASSES - 1, int(r["contact_s_mm"] / BIN_WIDTH_MM)))
    fs.append([r["Fy_total_N"], r["Fx_total_N"]])
    ss.append(r["contact_s_mm"])
    cs.append([r["L_M_mm"], r["phi_deg"]])
    depths.append(round(r.get("push_depth_mm", 0.1), 3))

X = np.array(X, dtype=np.float32)
ys, ss = np.array(ys), np.array(ss, dtype=np.float32)
fs, cs = np.array(fs, dtype=np.float32), np.array(cs, dtype=np.float32)
depths = np.array(depths)

EXAMS = [("시험지A: 깊이 0.10mm만", depths == 0.10),
         ("시험지B: 깊이 0.05/0.20mm", depths != 0.10)]
print(f"홀드아웃 {len(ys)}행 = " + ", ".join(f"{n}({int(m.sum())}행)" for n, m in EXAMS) + "\n")

rows_out = []
for ck_label, ck_file in CKPTS:
    ck = torch.load(os.path.join(MODELS_DIR, ck_file), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    net = Net(n_shape=sd["shape_head.weight"].shape[0] if "shape_head.weight" in sd else 0)
    net.load_state_dict(sd)
    net.eval()
    with torch.no_grad():
        seg, force, s_pred = net(torch.tensor(((X - ck["X_mean"]) / ck["X_std"])[:, None]).float(),
                                  torch.tensor((cs - ck["c_mean"]) / ck["c_std"]).float())
    pred_cls = seg.argmax(dim=1).numpy()
    f_phys = force.numpy() * ck["f_std"] + ck["f_mean"]
    s_phys = s_pred.numpy() * ck["s_std"] + ck["s_mean"]
    for exam, mask in EXAMS:
        conf = np.zeros((N_CLASSES, N_CLASSES), int)
        for t, p in zip(ys[mask], pred_cls[mask]):
            conf[t, p] += 1
        bal = np.mean([conf[i, i] / max(1, conf[i].sum()) for i in range(N_CLASSES)])
        rows_out.append((ck_label, exam, int(mask.sum()), bal,
                         r2_score(ss[mask], s_phys[mask]), np.abs(s_phys[mask] - ss[mask]).mean(),
                         np.abs(f_phys[mask, 0] - fs[mask, 0]).mean() * 1000,
                         np.abs(f_phys[mask, 1] - fs[mask, 1]).mean() * 1000))

print(f"{'체크포인트':18s} {'시험지':24s} {'n':>4s} {'bal acc':>8s} {'s R²':>7s} {'s MAE':>9s} "
      f"{'Fx MAE':>10s} {'Fy MAE':>10s}")
for ck, exam, n, bal, r2, mae, fxmae, fymae in rows_out:
    print(f"{ck:18s} {exam:24s} {n:4d} {bal*100:7.1f}% {r2:7.3f} {mae:7.2f}mm "
          f"{fxmae:8.4f}mN {fymae:8.4f}mN")
print("\n※ 28번 교훈: 시험지가 다르면 정답값 분산도 달라 R²는 시험지 간 비교 불가. "
      "같은 시험지 안에서 두 체크포인트를 비교할 것(MAE는 단위가 같아 비교 가능).")
