"""train_segment_classifier_singleprobe_4seg.py의 "자연스러운 움직임 재활용" 버전.

"능동탐색"(11개 phi로 로봇을 일부러 재구성)은 비현실적이라 기각했었고, 순수 단일 관측(1개)은
정보 부족으로 구간분류 71%대, s R²=0.77대에서 벽에 부딪힘(CNN/MLP/그래디언트부스팅/beta제외/
장기학습 다 시도해도 안 넘어감 - 그래디언트부스팅만 76%/0.79로 소폭 개선).

이번 버전: 로봇을 "일부러" 재구성하지 않고, 원래 하던 동작(항법/조향) 중 짧은 시간 동안
자연스럽게 지나가는 phi 값 3개를 그냥 버퍼링해서 같이 쓴다는 가정. 물리적으로는 원본
멀티프로브(11개, PHI_PROBES 고정 그리드) 스크립트와 계산 방식이 같지만:
- N_PROBES: 11 -> 3 (일부러 재구성하는 게 아니라 "지나가다 보니 이미 3번 봤다"는 수준으로 최소화)
- PHI_PROBES 고정 그리드 대신, 케이스마다 3개를 무작위로 독립 샘플링(실제 동작 중 어떤 phi를
  지나갈지는 랜덤이므로).
- L_M, s, beta, depth는 3개 프로브 동안 동일(같은 접촉 이벤트, 로봇이 짧은 시간 동안 관측만
  이어감 - 접촉 자체나 L_M 배치가 그 사이 바뀐다고 가정 안 함).
- config 예측 타깃에서 phi는 제외(3개의 서로 다른 phi 중 "정답"이 하나로 정의되지 않으므로).
  L_M만 유지.
- 힘(Fx,Fy)은 3개 프로브의 대체모델 예측을 평균(원본 11프로브 스크립트와 동일한 방식 - phi에
  무관한 물리량이므로 평균내면 노이즈가 줄어듦).
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
N_CLASSES = 4
PHI_RANGE = (-150.0, 150.0)
N_PROBES = 3  # 자연스러운 움직임 중 지나간 phi 3개 버퍼링 (일부러 재구성 아님)
N_SAMPLES = int(os.environ.get("N_SAMPLES", 150_000))
N_WORKERS = int(os.environ.get("N_WORKERS", 32))


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
    s_range = (10.0, 80.0)
    FIXED_DEPTH = 0.10
    beta_range = (0.0, 360.0)

    free_cache = {}
    Xb = np.zeros((n_chunk, N_PROBES, 3, 5, 5), dtype=np.float32)
    yb = np.zeros(n_chunk, dtype=np.int64)
    fb = np.zeros((n_chunk, 3), dtype=np.float32)
    sb = np.zeros(n_chunk, dtype=np.float32)
    cb = np.zeros((n_chunk, 1), dtype=np.float32)  # L_M만 (phi는 프로브마다 달라서 단일 정답 없음)
    n_ok = 0
    while n_ok < n_chunk:
        L_M = rng.uniform(*L_M_range)
        s = rng.uniform(*s_range)
        beta = rng.uniform(*beta_range)
        depth = FIXED_DEPTH
        phi_list = rng.uniform(*PHI_RANGE, size=N_PROBES)  # 자연스러운 움직임 중 지나간 phi 3개

        ok = True
        probes = np.zeros((N_PROBES, 3, 5, 5), dtype=np.float32)
        fx_list, fy_list, fmag_list = [], [], []
        for pi, phi in enumerate(phi_list):
            key = (round(L_M, 1), round(float(phi), 1))
            if key in free_cache:
                r_free = free_cache[key]
            else:
                try:
                    r_free = fm.solve_shape(L_M=L_M, phi_deg=float(phi), loads=[])
                except Exception:
                    ok = False
                    break
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
            probes[pi] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)

            fx_list.append(pred["Fy_total_N"])  # 축교환: Fx_board=Fy_local
            fy_list.append(pred["Fx_total_N"])  # Fy_board=Fx_local
            fmag_list.append(pred["F_mag_N"])

        if not ok:
            continue
        Xb[n_ok] = probes
        yb[n_ok] = s_to_bin(s)
        fb[n_ok] = [np.mean(fx_list), np.mean(fy_list), np.mean(fmag_list)]
        sb[n_ok] = s
        cb[n_ok] = [L_M]
        n_ok += 1
    return Xb, yb, fb, sb, cb


class NaturalMotionClassifier(nn.Module):
    """구간분류(4-class) + 힘(Fx,Fy) + 연속값 s(보조회귀) + L_M(config, phi는 프로브마다
    달라 제외) 동시 예측 - 프로브 3개(자연스러운 움직임 중 버퍼링, 능동 재구성 아님)."""
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2, n_config=1):
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
        self.config_head = nn.Linear(128, n_config)

    def forward(self, x):  # x: (B, n_probes, 3, 5, 5)
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


if __name__ == "__main__":
    t_start = time.time()

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
    print(f"=== 대체모델({N_ENSEMBLE}-앙상블) 5-fold R^2 (참고용) ===")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: R^2={r2_score(yn[:, i], preds[:, i]):.3f}")

    final_models = [train_mlp(Xn, yn, seed=i) for i in range(N_ENSEMBLE)]
    surrogate_states = [m.state_dict() for m in final_models]
    print(f"대체모델 학습 완료 ({time.time()-t_start:.0f}s)")

    print(f"{N_SAMPLES}개 자연스러운움직임-{N_PROBES}프로브 합성 데이터 생성 시작 ({N_WORKERS}-way 병렬)...")
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
    c_all = np.concatenate([r[4] for r in results], axis=0)
    print(f"합성 데이터 생성 완료: {len(y_all)}개 ({time.time()-t_gen:.0f}s)")
    print("구간별 샘플 수:", {c: int((y_all == c).sum()) for c in range(N_CLASSES)})

    np.savez(os.path.join(FEA_DATA_DIR, "segment_bfield_naturalmotion3_4seg.npz"),
             X=X_all, y=y_all, f=f_all, s=s_all, c=c_all)

    fxy_all = f_all[:, :2]
    f_mean, f_std = fxy_all.mean(axis=0), fxy_all.std(axis=0)
    f_std[f_std < 1e-12] = 1.0
    f_norm = (fxy_all - f_mean) / f_std

    s_mean, s_std = s_all.mean(), s_all.std()
    s_norm = (s_all - s_mean) / s_std

    c_mean, c_std = c_all.mean(axis=0), c_all.std(axis=0)
    c_std[c_std < 1e-12] = 1.0
    c_norm = (c_all - c_mean) / c_std

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

    model = NaturalMotionClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    seg_criterion = nn.CrossEntropyLoss()
    force_criterion = nn.MSELoss()
    s_criterion = nn.MSELoss()
    config_criterion = nn.MSELoss()

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
            loss = (seg_criterion(seg_logits, by) + force_criterion(force_pred, bf)
                    + s_criterion(s_pred, bs) + config_criterion(config_pred, bc))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_seg_logits, val_force_pred, val_s_pred, val_config_pred = model(val_X)
            val_seg_loss = seg_criterion(val_seg_logits, val_y).item()
            val_force_loss = force_criterion(val_force_pred, val_f).item()
            val_s_loss = s_criterion(val_s_pred, val_s).item()
            val_config_loss = config_criterion(val_config_pred, val_c).item()
            val_loss = val_seg_loss + val_force_loss + val_s_loss + val_config_loss
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
    adjacent = sum(conf[i, j].item() for i in range(N_CLASSES) for j in range(N_CLASSES) if abs(i - j) == 1)
    total_err = conf.sum().item() - torch.trace(conf).item()
    print(f"오답 중 인접구간 비율: {adjacent/max(1,total_err)*100:.1f}%")

    ss_res = np.sum((val_s_pred_phys - val_s_phys) ** 2)
    ss_tot = np.sum((val_s_phys - val_s_phys.mean()) ** 2)
    s_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    s_mae = np.mean(np.abs(val_s_pred_phys - val_s_phys))
    print(f"\n보조회귀 s(연속값): R^2={s_r2:.3f}, MAE={s_mae:.2f}mm")

    force_names = ["Fx_board_N", "Fy_board_N"]
    print(f"\n=== 힘(F) 회귀 성능 (n_val={len(val_idx)}) ===")
    for i, name in enumerate(force_names):
        pred_i, true_i = val_force_pred_phys[:, i], val_f_phys[:, i]
        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(pred_i - true_i))
        print(f"  {name}: R^2={r2:.3f}, MAE={mae*1000:.4f}mN")

    print(f"\n=== L_M 회귀 성능 (n_val={len(val_idx)}) ===")
    pred_lm, true_lm = val_config_pred_phys[:, 0], val_c_phys[:, 0]
    ss_res = np.sum((pred_lm - true_lm) ** 2)
    ss_tot = np.sum((true_lm - true_lm.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = np.mean(np.abs(pred_lm - true_lm))
    print(f"  L_M_mm: R^2={r2:.3f}, MAE={mae:.2f}mm")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
                "f_mean": f_mean, "f_std": f_std, "s_mean": s_mean, "s_std": s_std,
                "c_mean": c_mean, "c_std": c_std,
                "bin_width_mm": BIN_WIDTH_MM, "n_classes": N_CLASSES, "phi_range": PHI_RANGE,
                "n_probes": N_PROBES, "force_names": force_names},
               os.path.join(MODELS_DIR, "position_segment_classifier_naturalmotion3_4seg.pth"))
    print(f"\n저장: {MODELS_DIR}/position_segment_classifier_naturalmotion3_4seg.pth")
    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분")
