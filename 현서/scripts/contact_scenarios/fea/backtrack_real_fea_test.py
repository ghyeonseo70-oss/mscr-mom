"""
학습된 구간분류기(position_segment_classifier_multiprobe_150k_11probe_nonoise.pth)를
합성데이터가 아니라 "진짜 FEA로 11개 phi 전부를 실제로 계산해둔" (L_M,s) 조합에 직접
넣어서 거꾸로 구간을 맞히는지 검증한다. 대체모델(surrogate) 근사가 전혀 안 들어간,
순수 실측(FEA) 기반 최종 확인.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
import force_model as fm
import magpylib as magpy

PHI_PROBES = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
BIN_WIDTH_MM = 20.0
N_CLASSES = 5


def s_to_bin(s_mm):
    return min(N_CLASSES - 1, int(s_mm / BIN_WIDTH_MM))


SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.36
main_magnet = magpy.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
mom = magpy.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
mscr_robot = magpy.Collection(main_magnet, mom)


def compute_B(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
    xLM_b, yLM_b = fm.to_board_frame(xLM_l, yLM_l)
    xL_b, yL_b = fm.to_board_frame(xL_l, yL_l)
    mom.position = (float(xLM_b), float(yLM_b), 0)
    mom.orientation = Rotation.from_euler("z", -thLM, degrees=True)
    main_magnet.position = (float(xL_b), float(yL_b), 0)
    main_magnet.orientation = Rotation.from_euler("z", -thL, degrees=True)
    return magpy.getB(mscr_robot, sensors) * 1e6


class MultiProbeClassifier(nn.Module):
    def __init__(self, n_probes=len(PHI_PROBES), n_classes=N_CLASSES):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        return self.classifier(torch.cat(embeds, dim=1))


ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_multiprobe_150k_11probe_nonoise.pth"),
                   map_location="cpu", weights_only=False)
model = MultiProbeClassifier()
model.load_state_dict(ckpt["state_dict"])
model.eval()
X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]

rows = json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_sweep_all.json")))
by_lm_s = {}
for r in rows:
    by_lm_s.setdefault((r["L_M_mm"], r["contact_s_mm"]), {})[r["phi_deg"]] = r

targets = [(lm, s) for (lm, s), v in by_lm_s.items() if all(p in v for p in PHI_PROBES)]
print(f"11개 phi 전부 실제 FEA로 존재하는 (L_M,s) 조합: {len(targets)}개 (대체모델 근사 없이 순수 실측)\n")

n_correct = 0
for L_M, s_true in sorted(targets):
    probes = np.zeros((len(PHI_PROBES), 3, 5, 5), dtype=np.float32)
    for pi, phi in enumerate(PHI_PROBES):
        r = by_lm_s[(L_M, s_true)][phi]
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        d_xL_local = r["tip_uy_avg_mm"]
        d_yL_local = r["tip_ux_avg_mm"]
        d_thL = -r["tip_theta_deg_board"]
        frac = L_M / 100.0
        d_xLM_local, d_yLM_local, d_thLM = d_xL_local * frac, d_yL_local * frac, d_thL * frac
        xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
        xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
        B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
        B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                            xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
        probes[pi] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)

    Xn = (probes[None] - X_mean) / X_std
    with torch.no_grad():
        logits = model(torch.tensor(Xn, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[0]
        pred_bin = int(probs.argmax())
    true_bin = s_to_bin(s_true)
    ok = pred_bin == true_bin
    n_correct += ok
    bin_labels = [f"{int(i*BIN_WIDTH_MM)}-{int((i+1)*BIN_WIDTH_MM)}mm" for i in range(N_CLASSES)]
    print(f"L_M={L_M:.0f}mm, 실제 s={s_true:.0f}mm (실제구간={bin_labels[true_bin]}) "
          f"-> 예측구간={bin_labels[pred_bin]} {'O' if ok else 'X'}  "
          f"(확률분포: {[f'{p*100:.0f}%' for p in probs.tolist()]})")

print(f"\n총 {len(targets)}개 중 {n_correct}개 정답 ({n_correct/len(targets)*100:.0f}%)")
