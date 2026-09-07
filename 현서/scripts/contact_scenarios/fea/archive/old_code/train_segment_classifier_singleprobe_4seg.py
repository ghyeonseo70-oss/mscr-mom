"""train_segment_classifier_multiprobe_auxreg_4seg.py의 단일 관측(능동탐색 없음) 버전.

11개 phi로 로봇을 재구성하며 스캔하는 "능동탐색"은 실제 하드웨어에서 접촉 감지 1회마다
로봇을 11번 재구성해야 해서 비현실적이라는 문제제기가 있었음. 원래 실험 기록(PROJECT_STATUS.md)의
"단일 관측 20.5%(무작위 수준)" 결과는 노이즈 제거 버그를 찾기 *이전*에 측정된 값이라 노이즈로
오염된 비교였을 가능성이 큼. 이 스크립트는 지금의 깨끗한(노이즈 없음+힘 멀티태스크+보조회귀)
설정을 그대로 두고 프로브 1개(그 순간 로봇이 실제로 있던 임의의 phi)만으로 재시험함.

원본 대비 변경점:
- N_PROBES: 11 -> 1. PHI_PROBES 고정 리스트 대신 케이스마다 phi를 무작위로 1개만 샘플링
  (FEA 스윕 학습범위인 -150~150도 내에서 균일분포 - 대체모델 외삽 방지).
- 11개 프로브 평균을 내던 힘(Fx,Fy) 계산도 단일 관측값 그대로 사용.
- Windows spawn 멀티프로세싱 버그 수정: worker를 모듈 최상위로 이동(기존 스크립트는
  `if __name__ == "__main__":` 블록 안에 정의되어 있어서 spawn 방식에서
  `AttributeError: module '__mp_main__' has no attribute 'worker'`로 실패했었음).
- force_model 모듈 경로 수정: 이 컴퓨터에서는 아직 저장소 루트 scripts/force_model/에 있고
  현서/scripts/force_model/로 옮겨지지 않은 상태라(git mv 누락, 알려진 이슈) 경로를 한 단계
  더 올려서 잡음.
- L_M, phi(로봇 자신의 구동 형상)도 예측 타깃에 추가: 텐던 액추에이터의 "명령한 값"과
  "실제 값"이 오차/슬랙으로 어긋날 수 있어서, 컨트롤러가 아는 값을 그대로 믿지 않고
  자기장으로부터 다시 확인(추정)하기 위함. 그림 속 5개 값(L_m, φ, s_f, Fx, Fy) 전부를
  단일 관측 하나로 동시 예측하도록 확장함.
- beta(원주각) 90도/270도 부근(±15도) 샘플링에서 제외: 이 두 각도는 미는 힘이 굽힘평면에
  완전히 수직(out-of-plane)이 되는 지점이라 실제로 튜브를 거의 안 굽힘(fea_beta_fine_resolution.json
  으로 확인 - 그 지점에서 tip 변위가 정상의 ~10%까지 떨어지고, ±15도 안쪽까지도 10~35%로 약함).
  대체모델도 이 부근을 계속 못 맞혀서(Fy_total_N R^2 계속 낮음, 데이터 추가해도 개선 안 됨)
  s=80~100mm 팁 구간과 같은 논리로 "실감지 어려운 구간"으로 보고 아예 샘플링에서 제외함.
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

BIN_WIDTH_MM = 20.0
N_CLASSES = 4  # 원본 5개(0-100mm)에서 팁쪽 80-100mm 제외 -> 4개(0-80mm)
PHI_RANGE = (-150.0, 150.0)  # FEA 스윕 학습범위와 동일 (외삽 방지)
N_PROBES = 1  # 능동탐색 없음 - 단일 관측
N_SAMPLES = int(os.environ.get("N_SAMPLES", 150_000))
N_WORKERS = int(os.environ.get("N_WORKERS", 32))
BETA_EXCLUDE_BANDS = [(75.0, 105.0), (255.0, 285.0)]  # 90/270 +-15도 - 굽힘에 거의 기여 안 함


def sample_beta(rng):
    while True:
        b = rng.uniform(0.0, 360.0)
        if not any(lo <= b <= hi for lo, hi in BETA_EXCLUDE_BANDS):
            return b


def s_to_bin(s_mm):
    return min(N_CLASSES - 1, int(s_mm / BIN_WIDTH_MM))


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


N_ENSEMBLE = 10


def worker(args):
    widx, n_chunk, surrogate_states, X_mean, X_std, y_mean, y_std = args
    sys.path.insert(0, FORCE_MODEL_DIR)
    import force_model as fm
    import magpylib as magpy
    from scipy.spatial.transform import Rotation

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

    surrogates = []
    for st in surrogate_states:
        m = SurrogateMLP(len(FEATURES), len(TARGETS))
        m.load_state_dict(st)
        m.eval()
        surrogates.append(m)

    def predict_surrogate(L_M, phi, beta, s, depth):
        x = np.array([[L_M, phi, beta, s, depth]])
        xn = (x - X_mean) / X_std
        xt = torch.tensor(xn, dtype=torch.float32)
        with torch.no_grad():
            pn = np.mean([m(xt).numpy()[0] for m in surrogates], axis=0)
        p = pn * y_std + y_mean
        return dict(zip(TARGETS, p))

    rng = np.random.default_rng(2000 + widx)
    L_M_range = (0.0, 100.0)
    s_range = (10.0, 80.0)  # 팁쪽 80-100mm 제외 (힘이 너무 약해 실제 감지 불가로 판단)
    FIXED_DEPTH = 0.10  # mm - 실제 FEA 스윕과 동일하게 고정 (힘/깊이는 추정 대상이 아님)

    free_cache = {}
    Xb = np.zeros((n_chunk, N_PROBES, 3, 5, 5), dtype=np.float32)
    yb = np.zeros(n_chunk, dtype=np.int64)
    fb = np.zeros((n_chunk, 3), dtype=np.float32)  # Fx_board, Fy_board, F_mag (단일 관측)
    sb = np.zeros(n_chunk, dtype=np.float32)  # 연속값 s(mm) - 보조회귀 타깃
    cb = np.zeros((n_chunk, 2), dtype=np.float32)  # L_M(mm), phi(deg) - 액추에이터 슬랙 보정용 타깃
    n_ok = 0
    while n_ok < n_chunk:
        L_M = rng.uniform(*L_M_range)
        s = rng.uniform(*s_range)
        beta = rng.uniform(0.0, 360.0)  # beta 제외 실험은 득이 없었음(구간분류 71.3%->69.5%) - 원상복구
        depth = FIXED_DEPTH
        phi = rng.uniform(*PHI_RANGE)  # 능동탐색 없이, 그 순간 로봇이 실제로 있던 임의의 phi 1개

        key = (round(L_M, 1), phi)
        if key in free_cache:
            r_free = free_cache[key]
        else:
            try:
                r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
            except Exception:
                continue
            free_cache[key] = r_free
            if len(free_cache) > 4000:
                free_cache.clear()

        pred = predict_surrogate(L_M, phi, beta, s, depth)
        d_xL_local = pred["tip_uy_avg_mm"]
        d_yL_local = pred["tip_ux_avg_mm"]
        d_thL = -pred["tip_theta_deg_board"]
        frac = L_M / 100.0
        d_xLM_local, d_yLM_local, d_thLM = d_xL_local * frac, d_yL_local * frac, d_thL * frac

        xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
        xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]

        B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
        B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                            xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
        Xb[n_ok, 0] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)

        yb[n_ok] = s_to_bin(s)
        fb[n_ok] = [pred["Fy_total_N"], pred["Fx_total_N"], pred["F_mag_N"]]  # 축교환: Fx_board=Fy_local, Fy_board=Fx_local
        sb[n_ok] = s
        cb[n_ok] = [L_M, phi]
        n_ok += 1
    return Xb, yb, fb, sb, cb


class SingleProbeClassifier(nn.Module):
    """구간분류(4-class) + 힘(Fx,Fy 보드좌표계) + 연속값 s(보조회귀) + 로봇 자신의 형상(L_M,phi,
    액추에이터 슬랙 보정용) 동시 예측 - 프로브 1개(능동탐색 없음)."""
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2, n_config=2):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 * n_probes, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        self.config_head = nn.Linear(128, n_config)  # [L_M, phi]

    def forward(self, x):  # x: (B, n_probes, 3, 5, 5)
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


if __name__ == "__main__":
    t_start = time.time()

    # ── 1) FEA 데이터 병합 + 대체모델(surrogate) 학습 ──
    # 원래 4개 소스로 되돌림(beta_generalization/lm10_gap/beta_fine_resolution 3개는 뺌) -
    # 이 3개를 추가하면 대체모델의 Fy_total_N R^2가 오히려 나빠지는 걸 이미 확인함
    # (compare_surrogate_before_after_gapfill.py 결과, 0.627->0.547). beta 제외 샘플링
    # (BETA_EXCLUDE_BANDS)의 순수 효과만 보기 위해 SOURCES는 원상복구.
    SOURCES = ["fea_lm_phi_pos_sweep_all.json", "fea_bent_contact_sweep.json",
               "fea_geom_sweep_all.json", "fea_angle_sweep_all.json"]
    all_rows = []
    for fname in SOURCES:
        path = os.path.join(FEA_DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        for r in json.load(open(path)):
            row = dict(DEFAULTS)
            row.update(r)
            all_rows.append(row)
    print(f"대체모델 학습 데이터: {len(all_rows)}개 ({', '.join(SOURCES)})")

    X = np.array([[r[f] for f in FEATURES] for r in all_rows])
    y = np.array([[r[t] for t in TARGETS] for r in all_rows])
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_std[X_std < 1e-9] = 1.0
    y_mean, y_std = y.mean(axis=0), y.std(axis=0)
    y_std[y_std < 1e-9] = 1.0
    Xn = (X - X_mean) / X_std
    yn = (y - y_mean) / y_std

    def train_mlp(X_tr, y_tr, seed, epochs=2000, lr=1e-3, weight_decay=1e-4):
        torch.manual_seed(seed)
        model = SurrogateMLP(X_tr.shape[1], y_tr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        Xt = torch.tensor(X_tr, dtype=torch.float32)
        yt = torch.tensor(y_tr, dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward()
            opt.step()
        return model

    def predict_ensemble(models, X_val):
        preds = []
        for m in models:
            m.eval()
            with torch.no_grad():
                preds.append(m(torch.tensor(X_val, dtype=torch.float32)).numpy())
        return np.mean(preds, axis=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds = np.zeros_like(yn)
    for tr_idx, val_idx in kf.split(Xn):
        fold_models = [train_mlp(Xn[tr_idx], yn[tr_idx], seed=i) for i in range(N_ENSEMBLE)]
        preds[val_idx] = predict_ensemble(fold_models, Xn[val_idx])
    print(f"=== 대체모델({N_ENSEMBLE}-앙상블) 5-fold R^2 (참고용, 타깃별) ===")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: R^2={r2_score(yn[:, i], preds[:, i]):.3f}")

    final_models = [train_mlp(Xn, yn, seed=i) for i in range(N_ENSEMBLE)]
    surrogate_states = [m.state_dict() for m in final_models]
    print(f"대체모델 학습 완료 ({time.time()-t_start:.0f}s)")

    # ── 2) 단일 관측(능동탐색 없음) 합성 시나리오 (멀티프로세싱) ────────────────────
    print(f"{N_SAMPLES}개 단일관측(능동탐색 없음) 합성 데이터 생성 시작 ({N_WORKERS}-way 병렬)...")
    t_gen = time.time()
    chunk = N_SAMPLES // N_WORKERS
    chunks = [chunk] * N_WORKERS
    chunks[-1] += N_SAMPLES - chunk * N_WORKERS
    tasks = [(i, chunks[i], surrogate_states, X_mean, X_std, y_mean, y_std) for i in range(N_WORKERS)]
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(worker, tasks)
    X_all = np.concatenate([r[0] for r in results], axis=0)
    y_all = np.concatenate([r[1] for r in results], axis=0)
    f_all = np.concatenate([r[2] for r in results], axis=0)
    s_all = np.concatenate([r[3] for r in results], axis=0)
    c_all = np.concatenate([r[4] for r in results], axis=0)  # [L_M, phi]
    print(f"합성 데이터 생성 완료: {len(y_all)}개 ({time.time()-t_gen:.0f}s)")
    print("구간별 샘플 수:", {c: int((y_all == c).sum()) for c in range(N_CLASSES)})

    np.savez(os.path.join(FEA_DATA_DIR, "segment_bfield_singleprobe_4seg.npz"),
             X=X_all, y=y_all, f=f_all, s=s_all, c=c_all)

    fxy_all = f_all[:, :2]
    f_mean, f_std = fxy_all.mean(axis=0), fxy_all.std(axis=0)
    f_std[f_std < 1e-12] = 1.0
    f_norm = (fxy_all - f_mean) / f_std

    s_mean, s_std = s_all.mean(), s_all.std()
    s_norm = (s_all - s_mean) / s_std

    c_mean, c_std = c_all.mean(axis=0), c_all.std(axis=0)  # L_M(mm), phi(deg) - 스케일이 달라 열별 정규화
    c_std[c_std < 1e-12] = 1.0
    c_norm = (c_all - c_mean) / c_std

    # ── 3) 최종 단일프로브 분류기 학습 ──────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"최종 분류기 학습 시작 (device={device})...")

    X_mean2, X_std2 = X_all.mean(), X_all.std()
    X_norm = (X_all - X_mean2) / X_std2

    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(X_norm))
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    N_EPOCHS = int(os.environ.get("N_EPOCHS", 60))

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_all[train_idx]).long(),
                      torch.tensor(f_norm[train_idx]).float(), torch.tensor(s_norm[train_idx]).float(),
                      torch.tensor(c_norm[train_idx]).float()),
        batch_size=256, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float().to(device)
    val_y = torch.tensor(y_all[val_idx]).long().to(device)
    val_f = torch.tensor(f_norm[val_idx]).float().to(device)
    val_f_phys = f_all[val_idx]
    val_s = torch.tensor(s_norm[val_idx]).float().to(device)
    val_s_phys = s_all[val_idx]
    val_c = torch.tensor(c_norm[val_idx]).float().to(device)
    val_c_phys = c_all[val_idx]

    model = SingleProbeClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    seg_criterion = nn.CrossEntropyLoss()
    force_criterion = nn.MSELoss()
    s_criterion = nn.MSELoss()
    config_criterion = nn.MSELoss()
    FORCE_LOSS_WEIGHT = 1.0
    S_LOSS_WEIGHT = 1.0
    CONFIG_LOSS_WEIGHT = 1.0

    t_train = time.time()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    for epoch in range(N_EPOCHS):
        model.train()
        for bx, by, bf, bs, bc in train_loader:
            bx, by, bf, bs, bc = bx.to(device), by.to(device), bf.to(device), bs.to(device), bc.to(device)
            optimizer.zero_grad()
            seg_logits, force_pred, s_pred, config_pred = model(bx)
            loss = (seg_criterion(seg_logits, by) + FORCE_LOSS_WEIGHT * force_criterion(force_pred, bf)
                    + S_LOSS_WEIGHT * s_criterion(s_pred, bs) + CONFIG_LOSS_WEIGHT * config_criterion(config_pred, bc))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_seg_logits, val_force_pred, val_s_pred, val_config_pred = model(val_X)
            val_seg_loss = seg_criterion(val_seg_logits, val_y).item()
            val_force_loss = force_criterion(val_force_pred, val_f).item()
            val_s_loss = s_criterion(val_s_pred, val_s).item()
            val_config_loss = config_criterion(val_config_pred, val_c).item()
            val_loss = (val_seg_loss + FORCE_LOSS_WEIGHT * val_force_loss + S_LOSS_WEIGHT * val_s_loss
                        + CONFIG_LOSS_WEIGHT * val_config_loss)
            val_acc = (val_seg_logits.argmax(dim=1) == val_y).float().mean().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:2d}/{N_EPOCHS}] ValAcc {val_acc*100:5.1f}%  SegLoss {val_seg_loss:.4f}  ForceLoss {val_force_loss:.4f}  SLoss {val_s_loss:.4f}  ConfigLoss {val_config_loss:.4f}")
    print(f"학습 완료 ({time.time()-t_train:.0f}s), 최적 epoch={best_epoch} (val loss 기준)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_seg_logits, val_force_pred, val_s_pred, val_config_pred = model(val_X)
        val_pred = val_seg_logits.argmax(dim=1)
        val_force_pred_phys = val_force_pred.cpu().numpy() * f_std + f_mean
        val_s_pred_phys = val_s_pred.cpu().numpy() * s_std + s_mean
        val_config_pred_phys = val_config_pred.cpu().numpy() * c_std + c_mean
    final_acc = (val_pred == val_y).float().mean().item()
    conf = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.int32)
    for t, p in zip(val_y.tolist(), val_pred.tolist()):
        conf[t, p] += 1
    bin_labels = [f"{int(i*BIN_WIDTH_MM)}-{int((i+1)*BIN_WIDTH_MM)}mm" for i in range(N_CLASSES)]
    per_class_recall = [conf[i, i].item() / max(1, conf[i].sum().item()) for i in range(N_CLASSES)]

    print(f"\n=== 최종 검증 정확도: {final_acc*100:.1f}% (n_val={len(val_idx)}), 무작위 기준선={100/N_CLASSES:.1f}% ===")
    print(f"balanced accuracy(구간별 recall 평균): {np.mean(per_class_recall)*100:.1f}%")
    print("구간별 recall:", {bin_labels[i]: f"{per_class_recall[i]*100:.1f}%" for i in range(N_CLASSES)})
    print(f"\n혼동행렬 (행=실제, 열=예측, {bin_labels}):")
    print(conf.numpy())
    adjacent = sum(conf[i, j].item() for i in range(N_CLASSES) for j in range(N_CLASSES) if abs(i - j) == 1)
    total_err = conf.sum().item() - torch.trace(conf).item()
    print(f"오답 중 인접구간 비율: {adjacent/max(1,total_err)*100:.1f}%")

    ss_res = np.sum((val_s_pred_phys - val_s_phys) ** 2)
    ss_tot = np.sum((val_s_phys - val_s_phys.mean()) ** 2)
    s_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    s_mae = np.mean(np.abs(val_s_pred_phys - val_s_phys))
    print(f"\n보조회귀 s(연속값): R^2={s_r2:.3f}, MAE={s_mae:.2f}mm")

    force_names = ["Fx_board_N", "Fy_board_N"]
    print(f"\n=== 힘(F) 회귀 성능 (보드좌표계, n_val={len(val_idx)}) ===")
    for i, name in enumerate(force_names):
        pred_i, true_i = val_force_pred_phys[:, i], val_f_phys[:, i]
        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(pred_i - true_i))
        print(f"  {name}: R^2={r2:.3f}, MAE={mae*1000:.4f}mN")

    pred_fmag = np.sqrt(val_force_pred_phys[:, 0] ** 2 + val_force_pred_phys[:, 1] ** 2)
    true_fmag_2d = np.sqrt(val_f_phys[:, 0] ** 2 + val_f_phys[:, 1] ** 2)
    ss_res = np.sum((pred_fmag - true_fmag_2d) ** 2)
    ss_tot = np.sum((true_fmag_2d - true_fmag_2d.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(pred_fmag - true_fmag_2d))
    print(f"  F_mag(유도): R^2={r2:.3f}, MAE={mae*1000:.4f}mN")

    config_names = ["L_M_mm", "phi_deg"]
    print(f"\n=== 로봇 형상(L_M, phi) 회귀 성능 (액추에이터 슬랙 보정용, n_val={len(val_idx)}) ===")
    for i, name in enumerate(config_names):
        pred_i, true_i = val_config_pred_phys[:, i], val_c_phys[:, i]
        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(pred_i - true_i))
        unit = "mm" if name == "L_M_mm" else "deg"
        print(f"  {name}: R^2={r2:.3f}, MAE={mae:.2f}{unit}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
                "f_mean": f_mean, "f_std": f_std, "s_mean": s_mean, "s_std": s_std,
                "c_mean": c_mean, "c_std": c_std,
                "bin_width_mm": BIN_WIDTH_MM, "n_classes": N_CLASSES, "phi_range": PHI_RANGE,
                "force_names": force_names, "config_names": config_names},
               os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_4seg.pth"))
    print(f"\n저장: {MODELS_DIR}/position_segment_classifier_singleprobe_4seg.pth")
    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분")
