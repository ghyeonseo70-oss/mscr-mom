"""단일 관측(single-probe) 캐시 데이터로 CNN 대신 그래디언트 부스팅 트리(HistGradientBoosting,
scikit-learn 내장 - LightGBM과 비슷한 방식, 추가 설치 불필요)를 시도.

입력이 5x5x3=75개 숫자짜리 정형 데이터에 가까워서, CNN의 위치-무관 가중치공유(convolution)
이점이 크게 없어 보인다는 게 이미 확인됨(CNN vs MLP 무승부). 트리 앙상블은 이런 소규모
정형 데이터에서 신경망보다 강할 때가 많아서 기준선으로 시도.

같은 캐시(segment_bfield_singleprobe_4seg.npz)와 같은 train/val 분할(seed=0, 90/10)을
써서 CNN 결과(구간분류 71.7%, s R²=0.766, Fx/Fy R²=0.947/0.956, L_M/phi R²=0.974/0.974)와
직접 비교 가능.
"""
import os
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

BIN_WIDTH_MM = 20.0
N_CLASSES = 4


def balanced_acc(pred, true, n_classes):
    recalls = []
    for c in range(n_classes):
        mask = true == c
        recalls.append((pred[mask] == c).mean() if mask.sum() > 0 else 0.0)
    return float(np.mean(recalls))


def r2(pred, true):
    ss_res = np.sum((pred - true) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


if __name__ == "__main__":
    data = np.load(os.path.join(FEA_DATA_DIR, "segment_bfield_singleprobe_4seg.npz"))
    X_all, y_all, f_all, s_all, c_all = data["X"], data["y"], data["f"], data["s"], data["c"]
    X_flat = X_all.reshape(len(X_all), -1)  # (N, 1,3,5,5) -> (N, 75)
    print(f"데이터: {len(y_all)}개, 입력 {X_flat.shape[1]}차원")

    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(X_flat))
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, X_val = X_flat[train_idx], X_flat[val_idx]

    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=300, random_state=0, early_stopping=True)
    clf.fit(X_tr, y_all[train_idx])
    pred_y = clf.predict(X_val)
    acc = (pred_y == y_all[val_idx]).mean()
    bacc = balanced_acc(pred_y, y_all[val_idx], N_CLASSES)
    print(f"[구간분류] acc={acc*100:.1f}% balanced={bacc*100:.1f}%  ({time.time()-t0:.0f}s)")

    def fit_reg(y_tr, y_val, name):
        t0 = time.time()
        reg = HistGradientBoostingRegressor(max_iter=300, random_state=0, early_stopping=True)
        reg.fit(X_tr, y_tr)
        pred = reg.predict(X_val)
        print(f"[{name}] R2={r2(pred, y_val):.3f} MAE={np.mean(np.abs(pred-y_val)):.4f}  ({time.time()-t0:.0f}s)")
        return pred

    fit_reg(s_all[train_idx], s_all[val_idx], "s(mm)")
    fit_reg(f_all[train_idx, 0], f_all[val_idx, 0], "Fx_board_N")
    fit_reg(f_all[train_idx, 1], f_all[val_idx, 1], "Fy_board_N")
    fit_reg(c_all[train_idx, 0], c_all[val_idx, 0], "L_M_mm")
    fit_reg(c_all[train_idx, 1], c_all[val_idx, 1], "phi_deg")

    print("\n[참고: CNN(150epoch) 결과] 구간분류 balanced=71.7%  s R2=0.766  Fx R2=0.947  Fy R2=0.956  L_M R2=0.974  phi R2=0.974")
