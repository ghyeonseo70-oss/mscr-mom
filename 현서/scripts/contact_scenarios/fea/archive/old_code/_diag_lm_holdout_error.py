"""가설 검증: L_M 실측 홀드아웃 R^2가 왜 낮은가(0.565, 합성-val 0.895와 큰 격차)?

가설: 실측 검증용 B-field를 만들 때 force_model.solve_shape(L_M,phi,loads=[])를 힌트 없이
호출하는데(get_bent_centerline.py, worker(), real_holdout 평가 블록 전부 이 방식), 이
함수가 특정 (L_M,phi) 조합에서 다중해(가지 점프) 문제로 "틀린 branch"의 형상을 계산할 수
있음(이미 62.5mm@150도, 87.5mm@120/150도에서 확인된 적 있음 - 그때는 해당 조합만 스윕에서
뺐지만, 그 6조합 이외에도 비슷한 문제가 있는 다른 (L_M,phi) 조합이 홀드아웃에 섞여있을 수
있음). 틀린 형상으로 B_free를 계산하면 그 지점의 B-field delta 자체가 오염되니, 모델이
아무리 좋아도 그 지점만은 L_M을 못 맞출 것.

방법: 연속법(신뢰 가능한 기준, phi=0부터 5도씩 추적)과 힌트 없는 solve_shape(실제 평가가
쓰는 방식)를 각 홀드아웃 행에서 비교해서 "가지 점프 의심" 행을 골라내고, 그 행들만 모델의
L_M 예측 오차가 유난히 큰지 확인."""
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

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


def is_holdout_row(r, frac=0.2):
    key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (h % 10000) < int(frac * 10000)


all_rows = []
for r in json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json"), encoding="utf-8")):
    row = dict(DEFAULTS)
    row.update(r)
    all_rows.append(row)

holdout_idx = [i for i, r in enumerate(all_rows) if is_holdout_row(r)]
real_holdout_rows = [all_rows[i] for i in holdout_idx]
print(f"실측 홀드아웃 재구성: n={len(real_holdout_rows)}")

# ---- 연속법(신뢰 기준) vs 힌트없음(실제 평가 방식) theta_L 비교 ----
continuation_cache = {}


def continuation_theta_L(L_M, phi_target):
    """phi=0에서 5도씩 target까지 연속법으로 추적."""
    key = (round(L_M, 3), 0 if phi_target >= 0 else 1)
    if key not in continuation_cache:
        step = 5 if phi_target >= 0 else -5
        hint = None
        phi = 0
        trace = {0: fm.solve_shape(L_M=L_M, phi_deg=0, loads=[])["theta_L_deg"]}
        while abs(phi) < 150:
            phi += step
            try:
                r = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[], theta_L_hint_deg=hint, window_deg=40.0)
            except RuntimeError:
                r = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
            hint = r["theta_L_deg"]
            trace[phi] = hint
        continuation_cache[key] = trace
    return continuation_cache[key]


BRANCH_JUMP_THRESHOLD = 15.0  # deg
meta = []
for r in real_holdout_rows:
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    theta_nohint = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])["theta_L_deg"]
    trace = continuation_theta_L(L_M, phi)
    phi_rounded = int(round(phi / 5.0) * 5)
    theta_cont = trace.get(phi_rounded)
    if theta_cont is None:
        closest = min(trace.keys(), key=lambda p: abs(p - phi))
        theta_cont = trace[closest]
    diff = abs(theta_nohint - theta_cont)
    meta.append({"L_M": L_M, "phi": phi, "s": r["contact_s_mm"], "diff": diff,
                 "suspect": diff > BRANCH_JUMP_THRESHOLD})

n_suspect = sum(1 for m in meta if m["suspect"])
print(f"가지 점프 의심(연속법과 {BRANCH_JUMP_THRESHOLD}도 이상 차이) 행: {n_suspect}/{len(meta)}")
for m in meta:
    if m["suspect"]:
        print(f"  L_M={m['L_M']}, phi={m['phi']}, s={m['s']}: diff={m['diff']:.1f}deg")

# ---- 모델 로드해서 각 행 L_M 예측 오차 계산 ----
ckpt = torch.load(os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"),
                   map_location="cpu", weights_only=False)


class SingleProbeClassifier(nn.Module):
    def __init__(self, n_probes=1, n_classes=N_CLASSES, n_force=2, n_config=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3))
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        self.config_head = nn.Linear(128, n_config)
        self.lm_zero_head = nn.Linear(128, 1)  # 2026-08-27 추가 - state_dict 키 맞추기용

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return (self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h),
                self.lm_zero_head(h).squeeze(-1))


cnn = SingleProbeClassifier()
cnn.load_state_dict(ckpt["state_dict"])
cnn.eval()
X_mean2, X_std2 = ckpt["X_mean"], ckpt["X_std"]
c_mean, c_std = ckpt["c_mean"], ckpt["c_std"]
config_names = ckpt["config_names"]
print("config_names:", config_names)

SENSOR_HEIGHT_MM = 15
sensor_positions = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
sensors = magpy.Collection([magpy.Sensor(position=pos) for pos in sensor_positions])
MAGNET_BR_TESLA = 0.4
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


real_X, real_c = [], []
kept_meta = []
for r, m in zip(real_holdout_rows, meta):
    L_M, phi = r["L_M_mm"], r["phi_deg"]
    try:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
    except Exception:
        continue
    d_xL_local, d_yL_local = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]
    d_thL = -r["tip_theta_deg_board"]
    frac = L_M / 100.0
    d_xLM_local, d_yLM_local, d_thLM = d_xL_local * frac, d_yL_local * frac, d_thL * frac
    xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
    xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
    B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
    B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                        xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
    real_X.append((B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1))
    real_c.append([L_M, phi])
    kept_meta.append(m)

real_X = np.array(real_X, dtype=np.float32)
real_c_arr = np.array(real_c, dtype=np.float32)
real_X_norm = (real_X - X_mean2) / X_std2

with torch.no_grad():
    rX = torch.tensor(real_X_norm[:, None]).float()
    _, _, _, r_config_pred, r_lm_zero_logit = cnn(rX)
    r_config_phys = r_config_pred.numpy() * c_std + c_mean
    r_lm_zero_pred = (r_lm_zero_logit.numpy() > 0)

lm_idx = config_names.index("L_M_mm")
lm_pred = r_config_phys[:, lm_idx]
lm_true = real_c_arr[:, lm_idx]
lm_abs_err = np.abs(lm_pred - lm_true)

suspect_mask = np.array([m["suspect"] for m in kept_meta])


def r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


print(f"\n전체(n={len(lm_true)}): L_M R^2={r2(lm_pred, lm_true):.3f}, MAE={lm_abs_err.mean():.2f}mm")
if suspect_mask.sum() >= 3:
    print(f"가지점프 의심 행(n={suspect_mask.sum()}): L_M R^2={r2(lm_pred[suspect_mask], lm_true[suspect_mask]):.3f}, "
          f"MAE={lm_abs_err[suspect_mask].mean():.2f}mm")
if (~suspect_mask).sum() >= 3:
    print(f"정상 행(n={(~suspect_mask).sum()}): L_M R^2={r2(lm_pred[~suspect_mask], lm_true[~suspect_mask]):.3f}, "
          f"MAE={lm_abs_err[~suspect_mask].mean():.2f}mm")

# phi 구간별로도 쪼개서 비교(가지점프 말고 다른 원인일 가능성 대비)
phi_arr = real_c_arr[:, config_names.index("phi_deg")]
high_phi = np.abs(phi_arr) >= 90
print(f"\n|phi|<90(n={(~high_phi).sum()}): L_M R^2={r2(lm_pred[~high_phi], lm_true[~high_phi]):.3f}, "
      f"MAE={lm_abs_err[~high_phi].mean():.2f}mm")
print(f"|phi|>=90(n={high_phi.sum()}): L_M R^2={r2(lm_pred[high_phi], lm_true[high_phi]):.3f}, "
      f"MAE={lm_abs_err[high_phi].mean():.2f}mm")

# L_M 값별 실제 분포(범위가 좁으면 R^2이 원래 불안정할 수 있음)
print(f"\n홀드아웃 L_M 실제값 분포: min={lm_true.min():.1f}, max={lm_true.max():.1f}, "
      f"mean={lm_true.mean():.1f}, std={lm_true.std():.1f}")

# 가장 오차 큰 10개 행 상세
order = np.argsort(-lm_abs_err)[:10]
print("\n오차 상위 10개:")
for i in order:
    print(f"  L_M_true={lm_true[i]:.1f} L_M_pred={lm_pred[i]:.1f} err={lm_abs_err[i]:.1f}  "
          f"phi={phi_arr[i]:.0f}  branch_suspect={kept_meta[i]['suspect']} (diff={kept_meta[i]['diff']:.1f}deg)")

# L_M=0(CMSCR, MOM 없음) 케이스만 따로 - 오차 상위가 전부 이거였음
lm0_mask = np.abs(lm_true) < 0.01
print(f"\n=== L_M=0(CMSCR) 케이스만: n={lm0_mask.sum()} ===")
print("예측값들:", np.round(lm_pred[lm0_mask], 1).tolist())
print(f"MAE={lm_abs_err[lm0_mask].mean():.2f}mm")
nz = ~lm0_mask
print(f"\n=== L_M!=0(MOM 있음)만: n={nz.sum()}, R^2={r2(lm_pred[nz], lm_true[nz]):.3f}, "
      f"MAE={lm_abs_err[nz].mean():.2f}mm ===")
