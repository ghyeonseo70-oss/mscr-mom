"""
면접촉(분포하중) 포함 확장판. 점접촉(width=0)과 면접촉(width>0)을 하나의 연속된 스펙트럼으로
다룬다 - test_distributed_load.py로 물리모델이 분포하중을 정확히 계산하는 것과, width->0일 때
점접촉과 정확히 일치하는 것 확인 완료.

지금까지 배운 것 전부 적용:
  - solve_shape_robust (5단계 캐스케이드, 실패율 ~0%)
  - L_M을 입력에 명시적으로 포함 (이미 아는 값이라 굳이 추측 안 시킴)
  - 2단계 NN 구조(위치추정 결과를 입력으로 받아 접촉정보 예측) - 직접추정보다 훨씬 정확했음
  - 데이터 15,000개 (3천개보다 확실히 좋았음)
  - 멀티프로세싱(8코어 병렬) - 이전 버전은 코어 1개만 써서 15000개에 4~5시간 걸렸는데,
    샘플끼리 서로 독립이라 병렬화하면 이론상 최대 8배 빨라짐

목표(타깃) 5개: s_center(접촉 중심위치), width(접촉 폭, 0=점접촉), F_mag, Fx, Fy
"""
import os
import sys
import time
import multiprocessing as mp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
FORCE_MODEL_DIR = os.path.join(HERE, "..", "force_model")

DATA_DIR = os.path.join(HERE, "..", "..", "data", "contact_scenarios")
MODELS_DIR = os.path.join(HERE, "..", "..", "models")

PHI_PROBES = [-120.0, -60.0, 0.0, 60.0, 120.0]
L_M_RANGE = (20.0, 80.0)
S_MARGIN = 5.0
F_MAX = 0.02
WIDTH_MAX = 30.0  # mm, 0(점접촉)~30mm(꽤 넓은 면접촉)
N_SAMPLES = 15000
SEED = 77


def generate_one(seed):
    """워커 프로세스에서 실행 - 샘플 하나를 무작위로 뽑아 5프로브 delta descriptor 계산.
    실패하면 None 반환."""
    import sys as _sys
    if FORCE_MODEL_DIR not in _sys.path:
        _sys.path.insert(0, FORCE_MODEL_DIR)
    import force_model as fm

    rng = np.random.default_rng(seed)
    L_M = rng.uniform(*L_M_RANGE)
    width = rng.uniform(0.0, WIDTH_MAX)
    lo = S_MARGIN + width / 2
    hi = fm.L - S_MARGIN - width / 2
    if lo >= hi:
        return None
    s_center = rng.uniform(lo, hi)
    F_mag = rng.uniform(0.0, F_MAX)
    F_ang = rng.uniform(0, 2 * np.pi)
    Fx, Fy = F_mag * np.cos(F_ang), F_mag * np.sin(F_ang)

    if width < 1e-6:
        load = {"type": "point", "s": s_center, "Fx": Fx, "Fy": Fy}
    else:
        load = {"type": "dist", "s_start": s_center - width / 2, "s_end": s_center + width / 2,
                "Fx": Fx, "Fy": Fy}

    feat = [L_M]
    for phi in PHI_PROBES:
        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        try:
            r_load = fm.solve_shape_robust(L_M=L_M, phi_deg=phi, loads=[load],
                                            theta_L_hint_deg=r_free["theta_L_deg"])
        except RuntimeError:
            return None
        feat.extend([
            r_load["x_L"] - r_free["x_L"], r_load["y_L"] - r_free["y_L"],
            r_load["theta_L_deg"] - r_free["theta_L_deg"],
            r_load["x_LM"] - r_free["x_LM"], r_load["y_LM"] - r_free["y_LM"],
            r_load["theta_LM_deg"] - r_free["theta_LM_deg"],
        ])
    target = [s_center, width, F_mag, Fx, Fy]
    return feat, target


def main():
    n_workers = mp.cpu_count()
    print(f"목표 {N_SAMPLES}개 생성 시작 (면접촉 폭 0~{WIDTH_MAX}mm 포함), 워커 {n_workers}개 병렬")
    t_start = time.time()

    X_list, y_list = [], []
    n_fail = 0
    seed_cursor = SEED * 1000

    with mp.Pool(processes=n_workers) as pool:
        while len(X_list) < N_SAMPLES:
            need = N_SAMPLES - len(X_list)
            # 실패율 감안해서 여유있게 더 많은 시드를 한 번에 투입(약 1.3배)
            batch = max(int(need * 1.3), n_workers * 4)
            seeds = range(seed_cursor, seed_cursor + batch)
            seed_cursor += batch
            for result in pool.imap_unordered(generate_one, seeds, chunksize=8):
                if result is None:
                    n_fail += 1
                    continue
                feat, target = result
                X_list.append(feat)
                y_list.append(target)
                if len(X_list) % 1000 == 0:
                    elapsed = time.time() - t_start
                    print(f"{len(X_list)}/{N_SAMPLES} (실패 {n_fail}건, 경과 {elapsed/60:.1f}분)", flush=True)
                if len(X_list) >= N_SAMPLES:
                    break

    X = np.array(X_list[:N_SAMPLES])
    y = np.array(y_list[:N_SAMPLES])
    targets = ["s_center", "width", "F_mag", "Fx", "Fy"]
    print(f"\n생성 완료: {len(X)}개 (실패 {n_fail}건), 총 {(time.time()-t_start)/60:.1f}분")

    np.savez(os.path.join(DATA_DIR, "surface_contact_features_15k.npz"),
             X=X, y=y, targets=np.array(targets))
    print("저장: surface_contact_features_15k.npz")

    # ── NN2 학습 ──────────────────────────────
    X_mean, X_std = X.mean(0), X.std(0)
    X_norm = (X - X_mean) / (X_std + 1e-9)
    y_mean, y_std = y.mean(0), y.std(0)
    y_norm = (y - y_mean) / y_std

    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(X_norm))
    split = int(0.8 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_norm[train_idx]).float()),
        batch_size=64, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float()
    val_y = torch.tensor(y_norm[val_idx]).float()

    class SurfaceContactNet(nn.Module):
        def __init__(self, n_in, n_out):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, n_out),
            )

        def forward(self, x):
            return self.net(x)

    model = SurfaceContactNet(X.shape[1], len(targets))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\n면접촉 NN2 학습 시작...")
    for epoch in range(200):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                vloss = criterion(model(val_X), val_y).item()
            print(f"Epoch [{epoch+1:3d}/200] Val Loss {vloss:.4f}")

    model.eval()
    with torch.no_grad():
        pred_norm = model(val_X).numpy()
    true_norm = val_y.numpy()
    pred = pred_norm * y_std + y_mean
    true = true_norm * y_std + y_mean

    print("\n[결과 - 면접촉 포함 2단계 NN, 이상적 상한선]")
    for i, t in enumerate(targets):
        mae = np.mean(np.abs(pred[:, i] - true[:, i]))
        r2 = 1 - np.sum((pred[:, i] - true[:, i]) ** 2) / np.sum((true[:, i] - true[:, i].mean()) ** 2)
        unit = 'mm' if t in ('s_center', 'width') else 'N'
        print(f"  {t}: R2={r2:.3f}  MAE={mae:.5f}{unit}")

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "surface_contact_nn2_15k.pth"))
    np.savez(os.path.join(DATA_DIR, "surface_contact_train_history_15k.npz"),
             val_pred=pred, val_true=true, targets=np.array(targets),
             X_mean=X_mean, X_std=X_std, y_mean=y_mean, y_std=y_std)
    print("\n저장 완료: models/surface_contact_nn2_15k.pth")


if __name__ == "__main__":
    main()
