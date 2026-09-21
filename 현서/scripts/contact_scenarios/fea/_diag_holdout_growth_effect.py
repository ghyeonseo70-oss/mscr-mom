"""2026-09-19: 데이터가 518->655행으로 늘면서 s 지표가 나빠져 보이는 게(0.910->0.844)
진짜 악화인지, 홀드아웃 구성이 바뀌어서 생긴 착시인지 확인.

배경: 23번(저-phi 팁조밀화) 스윕이 s=60-100mm 구간만 집중적으로 채웠기 때문에, 그 20%가
홀드아웃에 들어가면서 홀드아웃의 난이도 구성 자체가 바뀜(신규 26행 중 73%가 s>=60mm,
기존 99행은 34%). 모델이 그대로여도 평균이 내려갈 수 있는 구조 - PROJECT_STATUS.md 5번에서
똑같은 착시를 이미 겪었음.

방법: 현재 체크포인트로 (1) 전체 홀드아웃 (2) 기존 518개 시절에도 홀드아웃이던 행만
(3) 이번에 새로 홀드아웃에 들어온 행만 - 세 가지로 나눠서 같은 지표를 계산. (2)가 26번의
기준값(s R^2=0.910/MAE=5.77mm)과 비슷하면 착시, 확실히 낮으면 진짜 악화.
"""
import hashlib
import json
import os
import subprocess
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
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")
sys.path.insert(0, FORCE_MODEL_DIR)
import force_model as fm
import magpylib as magpy
from scipy.spatial.transform import Rotation

DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}
N_CLASSES = 4
BIN_WIDTH_MM = 20.0
# 518행 시절 커밋 - 이 시점 데이터에 있던 행이 "기존 행"
OLD_COMMIT = "0264d73"


def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    return (int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000) < int(frac * 10000)


def row_key(r):
    # 2026-09-21(36번) 수정: 깊이를 키에 포함. 안 그러면 같은 조합의 0.05/0.20mm 변형까지
    # "기존 행"으로 셈해져서(99개여야 할 게 129개로 나옴) 31번 기준값과 비교가 어긋남.
    return (r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"],
            round(r.get("push_depth_mm", 0.1), 3))


all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)

old_raw = subprocess.run(
    ["git", "show", f"{OLD_COMMIT}:현서/data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json"],
    capture_output=True, cwd=REPO_ROOT).stdout.decode("utf-8")
old_keys = {row_key(r) for r in json.loads(old_raw)}

holdout = [r for r in all_rows if is_holdout_row(r)]
print(f"전체 {len(all_rows)}행 중 홀드아웃 {len(holdout)}행 "
      f"(기존 {sum(1 for r in holdout if row_key(r) in old_keys)} / 신규 {sum(1 for r in holdout if row_key(r) not in old_keys)})")

ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                   map_location="cpu", weights_only=False)


class SingleProbeClassifier(nn.Module):
    def __init__(self, n_probes=1, n_classes=N_CLASSES, n_force=2, n_config_in=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * n_probes + n_config_in, 128),
                                    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        # 2026-09-21(36번): shape_head 추가 - 체크포인트 키를 맞추려면 여기도 있어야 함.
        self.shape_head = nn.Linear(128, 3)

    def forward(self, x, config):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds + [config], dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.shape_head(h)


# 2026-09-21(36번): shape_head 도입 과도기라 체크포인트에 shape_head가 있을 수도 없을 수도
# 있음(36번 이전 학습분은 없음). 키 존재 여부로 판단해서 양쪽 다 읽을 수 있게 함 -
# 이 진단은 shape를 안 쓰므로 없으면 랜덤 초기화된 채로 두고 무시하면 됨.
HAS_SHAPE_HEAD = "shape_head.weight" in ckpt["state_dict"]
cnn = SingleProbeClassifier()
cnn.load_state_dict(ckpt["state_dict"], strict=HAS_SHAPE_HEAD)
cnn.eval()
print(f"체크포인트 shape_head 포함 여부: {HAS_SHAPE_HEAD}")
X_mean2, X_std2 = ckpt["X_mean"], ckpt["X_std"]
f_mean, f_std = ckpt["f_mean"], ckpt["f_std"]
s_mean, s_std = ckpt["s_mean"], ckpt["s_std"]
c_mean, c_std = ckpt["c_mean"], ckpt["c_std"]
force_names = ckpt["force_names"]

SENSOR_HEIGHT_MM = 15
sensors = magpy.Collection([magpy.Sensor(position=(x, y, SENSOR_HEIGHT_MM))
                            for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)])
main_magnet = magpy.magnet.Cylinder(polarization=(0, 0.4, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -0.4, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


X, ys, fs, ss, cs, is_old = [], [], [], [], [], []
for r in holdout:
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    except Exception:
        continue
    d_xL, d_yL, d_thL = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"], -r["tip_theta_deg_board"]
    if "mom_ux_avg_mm" in r:
        d_xLM, d_yLM, d_thLM = r["mom_uy_avg_mm"], r["mom_ux_avg_mm"], -r["mom_theta_deg_board"]
    else:
        frac = L_M / 100.0
        d_xLM, d_yLM, d_thLM = d_xL * frac, d_yL * frac, d_thL * frac
    B_free = compute_B(r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"],
                        r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"])
    B_load = compute_B(r_free["x_LM"] + d_xLM, r_free["y_LM"] + d_yLM, r_free["theta_LM_deg"] + d_thLM,
                        r_free["x_L"] + d_xL, r_free["y_L"] + d_yL, r_free["theta_L_deg"] + d_thL)
    X.append((B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1))
    ys.append(min(N_CLASSES - 1, int(r["contact_s_mm"] / BIN_WIDTH_MM)))
    fs.append([r["Fy_total_N"], r["Fx_total_N"]])
    ss.append(r["contact_s_mm"])
    cs.append([L_M, phi])
    is_old.append(row_key(r) in old_keys)

X = np.array(X, dtype=np.float32)
ys, ss = np.array(ys), np.array(ss, dtype=np.float32)
fs, cs = np.array(fs, dtype=np.float32), np.array(cs, dtype=np.float32)
is_old = np.array(is_old)

with torch.no_grad():
    seg, force, s_pred, _ = cnn(torch.tensor(((X - X_mean2) / X_std2)[:, None]).float(),
                              torch.tensor((cs - c_mean) / c_std).float())
    pred_class = seg.argmax(dim=1).numpy()
    force_phys = force.numpy() * f_std + f_mean
    s_phys = s_pred.numpy() * s_std + s_mean

fx = force_phys[:, force_names.index("Fx_board_N")]
fy = force_phys[:, force_names.index("Fy_board_N")]

print(f"\n{'구분':28s} {'n':>4s} {'bal acc':>8s} {'s R²':>7s} {'s MAE':>8s} {'s>=60 MAE':>10s} "
      f"{'Fx R²':>7s} {'Fx MAE':>8s} {'Fy R²':>7s} {'Fy MAE':>8s}")
for label, mask in [("전체 홀드아웃", np.ones(len(ys), bool)),
                     ("기존 행만 (26번과 동일 비교)", is_old),
                     ("이번에 새로 들어온 행만", ~is_old)]:
    if mask.sum() < 3:
        continue
    conf = np.zeros((N_CLASSES, N_CLASSES), int)
    for t, p in zip(ys[mask], pred_class[mask]):
        conf[t, p] += 1
    bal = np.mean([conf[i, i] / max(1, conf[i].sum()) for i in range(N_CLASSES)])
    tip = mask & (ss >= 60)
    tip_mae = np.mean(np.abs(s_phys[tip] - ss[tip])) if tip.sum() >= 3 else float("nan")
    print(f"{label:28s} {int(mask.sum()):4d} {bal*100:7.1f}% {r2_score(ss[mask], s_phys[mask]):7.3f} "
          f"{np.mean(np.abs(s_phys[mask]-ss[mask])):7.2f}mm {tip_mae:9.2f}mm "
          f"{r2_score(fs[mask,0], fx[mask]):7.3f} {np.mean(np.abs(fx[mask]-fs[mask,0]))*1000:7.4f}mN "
          f"{r2_score(fs[mask,1], fy[mask]):7.3f} {np.mean(np.abs(fy[mask]-fs[mask,1]))*1000:7.4f}mN")

print("\n[26번 기준값(518개 학습, 기존 99행 홀드아웃)]: bal acc 83.0%, s R²=0.910/MAE=5.77mm, "
      "Fx_board R²=0.626, Fy_board R²=0.855")

# 28번(R^2 함정) 교훈: R^2가 이상하게 나쁠 땐 정답값 분산부터 볼 것 - 분산이 작으면
# 같은 절대오차도 R^2가 폭락함(신규 행의 Fy R^2=-92 같은 극단값이 여기서 설명됨).
print(f"\n[정답값 자체의 분산 - R^2 해석용]")
print(f"{'구분':24s} {'n':>4s} {'Fx정답 표준편차':>16s} {'Fy정답 표준편차':>16s} {'s정답 표준편차':>15s}")
for label, m in [("기존 행", is_old), ("신규 행", ~is_old)]:
    print(f"{label:24s} {int(m.sum()):4d} {fs[m,0].std()*1000:14.4f}mN {fs[m,1].std()*1000:14.4f}mN "
          f"{ss[m].std():13.2f}mm")
