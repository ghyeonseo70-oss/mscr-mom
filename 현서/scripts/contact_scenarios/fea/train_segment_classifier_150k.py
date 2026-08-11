"""
master_pipeline.py와 같은 구조(대체모델 -> 대량 합성데이터 -> 최종 CNN)를 재사용하되,
1) 목표를 회귀(정확한 s)에서 분류(20mm씩 5구간 중 어디)로 바꾸고
2) 클래스 불균형을 원천적으로 없애기 위해 s를 균등분포로 15만개 합성해서 학습
3) 합성 생성은 멀티프로세싱으로 병렬화(단일프로세스로 하면 5만개에 108분 걸렸던 전례가 있어
   15만개면 5시간+ 걸림 - 이 컴퓨터 64코어를 써서 수십 분으로 단축)

입력 FEA 데이터: fea_lm_phi_pos_sweep_all.json(195, LM 4 x phi 11 x s 10) +
기존 3개 스윕(fea_bent_contact_sweep/geom_sweep_all/angle_sweep_all, 총 413) = 608개.
"""
import glob
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

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

BIN_WIDTH_MM = 20.0
N_CLASSES = 5
N_SAMPLES = int(os.environ.get("N_SAMPLES", 150_000))
N_WORKERS = int(os.environ.get("N_WORKERS", 32))


def s_to_bin(s_mm):
    return min(N_CLASSES - 1, int(s_mm / BIN_WIDTH_MM))


class SurrogateMLP(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_out),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    t_start = time.time()

    # ── 1) FEA 데이터 병합 + 대체모델(surrogate) 학습 ───────────────────────
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

    def train_mlp(X_tr, y_tr, X_val, epochs=1000, lr=1e-3, weight_decay=1e-3):
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
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_val, dtype=torch.float32)).numpy()
        return pred, model

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    preds = np.zeros_like(yn)
    for tr_idx, val_idx in kf.split(Xn):
        pred, _ = train_mlp(Xn[tr_idx], yn[tr_idx], Xn[val_idx])
        preds[val_idx] = pred
    print("=== 대체모델 5-fold R^2 (참고용, 타깃별) ===")
    for i, t in enumerate(TARGETS):
        print(f"  {t}: R^2={r2_score(yn[:, i], preds[:, i]):.3f}")

    _, final_model = train_mlp(Xn, yn, Xn)
    surrogate_state = final_model.state_dict()
    print(f"대체모델 학습 완료 ({time.time()-t_start:.0f}s)")

    # ── 2) 15만개 합성 시나리오 (멀티프로세싱) ───────────────────────────
    def worker(args):
        widx, n_chunk, surrogate_state, X_mean, X_std, y_mean, y_std = args
        sys.path.insert(0, os.path.join(HERE, "..", "..", "force_model"))
        import force_model as fm
        import magpylib as magpy
        from scipy.spatial.transform import Rotation

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

        surrogate = SurrogateMLP(len(FEATURES), len(TARGETS))
        surrogate.load_state_dict(surrogate_state)
        surrogate.eval()

        def predict_surrogate(L_M, phi, beta, s, depth):
            x = np.array([[L_M, phi, beta, s, depth]])
            xn = (x - X_mean) / X_std
            with torch.no_grad():
                pn = surrogate(torch.tensor(xn, dtype=torch.float32)).numpy()[0]
            p = pn * y_std + y_mean
            return dict(zip(TARGETS, p))

        rng = np.random.default_rng(1000 + widx)
        L_M_range, phi_range, beta_range = (0.0, 100.0), (-150.0, 150.0), (0.0, 360.0)
        s_range, depth_range = (10.0, 100.0), (0.02, 0.20)

        free_cache = {}
        Xb = np.zeros((n_chunk, 3, 5, 5), dtype=np.float32)
        yb = np.zeros(n_chunk, dtype=np.int64)
        n_ok = 0
        while n_ok < n_chunk:
            L_M = rng.uniform(*L_M_range)
            phi = rng.uniform(*phi_range)
            beta = rng.uniform(*beta_range)
            s = rng.uniform(*s_range)
            depth = rng.uniform(*depth_range)

            key = (round(L_M, 1), round(phi, 1))
            if key in free_cache:
                r_free = free_cache[key]
            else:
                try:
                    r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
                except Exception:
                    continue
                free_cache[key] = r_free
                if len(free_cache) > 3000:
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
            noise_scale_free = np.max(np.abs(B_free)) * 0.05
            noise_scale_load = np.max(np.abs(B_load)) * 0.05
            B_free = B_free + rng.normal(0, noise_scale_free, B_free.shape)
            B_load = B_load + rng.normal(0, noise_scale_load, B_load.shape)

            Xb[n_ok] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)
            yb[n_ok] = s_to_bin(s)
            n_ok += 1
        return Xb, yb

    print(f"15만개 합성 데이터 생성 시작 ({N_WORKERS}-way 병렬)...")
    t_gen = time.time()
    chunk = N_SAMPLES // N_WORKERS
    chunks = [chunk] * N_WORKERS
    chunks[-1] += N_SAMPLES - chunk * N_WORKERS
    tasks = [(i, chunks[i], surrogate_state, X_mean, X_std, y_mean, y_std) for i in range(N_WORKERS)]
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(worker, tasks)
    X_all = np.concatenate([r[0] for r in results], axis=0)
    y_all = np.concatenate([r[1] for r in results], axis=0)
    print(f"합성 데이터 생성 완료: {len(y_all)}개 ({time.time()-t_gen:.0f}s)")
    print("구간별 샘플 수:", {c: int((y_all == c).sum()) for c in range(N_CLASSES)})

    np.savez(os.path.join(FEA_DATA_DIR, "segment_bfield_150k.npz"), X=X_all, y=y_all)

    # ── 3) 최종 CNN 분류기 학습 (GPU 사용) ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"최종 분류기 학습 시작 (device={device})...")

    X_mean2, X_std2 = X_all.mean(), X_all.std()
    X_norm = (X_all - X_mean2) / X_std2

    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(X_norm))
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_all[train_idx]).long()),
        batch_size=256, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float().to(device)
    val_y = torch.tensor(y_all[val_idx]).long().to(device)

    class SegmentClassifier(nn.Module):
        def __init__(self, n_classes=N_CLASSES):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Linear(32 * 5 * 5, 128), nn.BatchNorm1d(128), nn.ReLU(),
                nn.Dropout(0.3), nn.Linear(128, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = SegmentClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    t_train = time.time()
    for epoch in range(60):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:2d}/60] ValAcc {val_acc*100:5.1f}%")
    print(f"학습 완료 ({time.time()-t_train:.0f}s)")

    model.eval()
    with torch.no_grad():
        val_pred = model(val_X).argmax(dim=1)
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

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
                "bin_width_mm": BIN_WIDTH_MM, "n_classes": N_CLASSES},
               os.path.join(MODELS_DIR, "position_segment_classifier_150k.pth"))
    print(f"\n저장: {MODELS_DIR}/position_segment_classifier_150k.pth")
    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분")
