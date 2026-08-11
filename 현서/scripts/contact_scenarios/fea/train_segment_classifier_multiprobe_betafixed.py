"""
단일 자기장 스냅샷으로는 구간분류가 무작위 수준(21.9%, balanced 20.5%)이었음
(train_segment_classifier_150k.py 결과) -> 능동탐색(active sensing) 시도.

핵심 아이디어(test_active_sensing.py에서 이미 검증됨): 같은 물리적 접촉(로컬 좌표 s, 힘 고정)
이라도 로봇 구동상태(L_M,phi)에 따라 "구별 가능/불가능"이 달라짐 -> 접촉이 유지되는 동안
phi(외부자기장 방향, 전기적으로 빠르게 바꿀 수 있음)를 여러 각도로 스캔해서 자기장을 여러 번
관측하면 신호가 살아날 수 있음. compare_probe_combos.py가 예전(analytical force_model 기반)
찾아둔 "robust 4-probe 조합" phi=[-150,-90,-30,60]을 그대로 재사용.

케이스 하나 = (L_M, s, beta, depth) 고정(스캔 중 안 바뀜) + phi 4곳에서 각각 B_delta 계산
-> MultiProbeEstimator(compare_probe_combos.py 구조: probe별 공유 CNN 인코더 -> concat -> head)
   를 회귀 대신 5구간 분류로 학습.
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

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

BIN_WIDTH_MM = 20.0
N_CLASSES = 5
PHI_PROBES = [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0]  # 11개 전부(우리 FEA 그리드와 동일)
N_PROBES = len(PHI_PROBES)
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


N_ENSEMBLE = 10  # 대체모델 예측오차(RMSE)가 s 10mm당 실제 신호 변화량과 맞먹을 정도로 컸음
# (tip_ux RMSE=0.073mm vs s=20->30mm 실제변화=0.115mm) -> 앙상블 평균으로 노이즈를 줄임
# (실험 결과 큰 네트워크+5-앙상블로 RMSE 16~43% 감소 확인)


if __name__ == "__main__":
    t_start = time.time()

    # ── 1) FEA 데이터 병합 + 대체모델(surrogate) 학습 (기존과 동일) ──────────
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
    resid_phys = (preds * y_std + y_mean) - y
    print(f"  (참고 RMSE, mm/deg/N 단위): " +
          ", ".join(f"{t}={np.sqrt(np.mean(resid_phys[:,i]**2)):.4f}" for i, t in enumerate(TARGETS[:4])))

    final_models = [train_mlp(Xn, yn, seed=i) for i in range(N_ENSEMBLE)]
    surrogate_states = [m.state_dict() for m in final_models]
    print(f"대체모델 학습 완료 ({time.time()-t_start:.0f}s)")

    # ── 2) 15만개 멀티프로브 합성 시나리오 (멀티프로세싱) ────────────────────
    # 케이스 하나 = 물리적 접촉(L_M, s, beta, depth) 고정 + phi를 PHI_PROBES 4곳으로 스캔
    def worker(args):
        widx, n_chunk, surrogate_states, X_mean, X_std, y_mean, y_std = args
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
        s_range = (10.0, 100.0)
        FIXED_DEPTH = 0.10  # mm - 실제 FEA 스윕(fea_lm_phi_pos_sweep_all.json)과 동일하게 고정.
        # depth를 0.02~0.20mm로 무작위 섞었더니 s에 따른 신호(최대 266배 차이)보다 depth로 인한
        # 크기 변화가 더 커서(F_mag가 s=10~80mm 사이에서 0.0008~0.213mN까지 벌어지는데 depth
        # 범위만으로도 비슷한 배율이 나옴) 위치 신호를 덮어버리는 게 확인됨 -> 교수님 말대로
        # 힘/깊이는 추정 대상이 아니니 아예 고정해서 이 혼란 요인을 제거.
        FIXED_BETA = 0.0  # beta 확인용 실험: 무작위(0~360) 대신 고정해서 남은 교란요인인지 테스트

        free_cache = {}
        Xb = np.zeros((n_chunk, N_PROBES, 3, 5, 5), dtype=np.float32)
        yb = np.zeros(n_chunk, dtype=np.int64)
        # 힘 타깃(Fx,Fy,F_mag) - Fx/Fy는 프로브(phi)마다 국소좌표계가 달라서 그대로 평균내면
        # 안 되므로, force_model.to_board_frame과 같은 축 교환 규칙(Fx_board=Fy_local,
        # Fy_board=Fx_local - master_pipeline.py의 팁변위 변환과 동일한 규칙)으로 먼저 보드좌표계로
        # 통일한 뒤 11개 프로브 평균을 씀. F_mag는 방향 무관이라 그냥 평균.
        fb = np.zeros((n_chunk, 3), dtype=np.float32)  # Fx_board, Fy_board, F_mag 평균
        n_ok = 0
        while n_ok < n_chunk:
            # 스캔 중 고정되는 값들: 로봇이 그 순간 있던 L_M, 실제 접촉(s, beta, depth)
            L_M = rng.uniform(*L_M_range)
            s = rng.uniform(*s_range)
            beta = FIXED_BETA
            depth = FIXED_DEPTH

            ok = True
            probes = np.zeros((N_PROBES, 3, 5, 5), dtype=np.float32)
            fx_board_list, fy_board_list, fmag_list = [], [], []
            for pi, phi in enumerate(PHI_PROBES):
                key = (round(L_M, 1), phi)
                if key in free_cache:
                    r_free = free_cache[key]
                else:
                    try:
                        r_free = fm.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
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
                # 노이즈 미포함(사용자 지시) - 센서 잡음 없는 이상적인 상황에서의 신호만 확인
                probes[pi] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)

                fx_board_list.append(pred["Fy_total_N"])  # 축교환: Fx_board = Fy_local
                fy_board_list.append(pred["Fx_total_N"])  # Fy_board = Fx_local
                fmag_list.append(pred["F_mag_N"])

            if not ok:
                continue
            Xb[n_ok] = probes
            yb[n_ok] = s_to_bin(s)
            fb[n_ok] = [np.mean(fx_board_list), np.mean(fy_board_list), np.mean(fmag_list)]
            n_ok += 1
        return Xb, yb, fb

    print(f"15만개 멀티프로브({N_PROBES}개 phi={PHI_PROBES}) 합성 데이터 생성 시작 ({N_WORKERS}-way 병렬)...")
    t_gen = time.time()
    chunk = N_SAMPLES // N_WORKERS
    chunks = [chunk] * N_WORKERS
    chunks[-1] += N_SAMPLES - chunk * N_WORKERS
    tasks = [(i, chunks[i], surrogate_states, X_mean, X_std, y_mean, y_std) for i in range(N_WORKERS)]
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(worker, tasks)
    X_all = np.concatenate([r[0] for r in results], axis=0)
    y_all = np.concatenate([r[1] for r in results], axis=0)
    f_all = np.concatenate([r[2] for r in results], axis=0)  # Fx_board, Fy_board, F_mag (평균, 단위 N)
    print(f"합성 데이터 생성 완료: {len(y_all)}개 ({time.time()-t_gen:.0f}s)")
    print("구간별 샘플 수:", {c: int((y_all == c).sum()) for c in range(N_CLASSES)})

    np.savez(os.path.join(FEA_DATA_DIR, "segment_bfield_multiprobe_150k_11probe_betafixed.npz"),
             X=X_all, y=y_all, f=f_all)

    # F_mag(3번째 열)은 대체모델이 "항상 0 이상"이라는 물리적 제약 없이 예측한 값이라 22%가
    # 마이너스로 나오는 문제가 확인됨(특히 힘이 작은 케이스). Fx,Fy(부호 있는 값이라 이 문제
    # 없음)만 회귀 타깃으로 쓰고, F_mag은 예측된 Fx,Fy로부터 sqrt(Fx^2+Fy^2)로 유도.
    fxy_all = f_all[:, :2]
    f_mean, f_std = fxy_all.mean(axis=0), fxy_all.std(axis=0)
    f_std[f_std < 1e-12] = 1.0
    f_norm = (fxy_all - f_mean) / f_std

    # ── 3) 최종 멀티프로브 분류기 학습 (GPU) ──────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"최종 분류기 학습 시작 (device={device})...")

    X_mean2, X_std2 = X_all.mean(), X_all.std()
    X_norm = (X_all - X_mean2) / X_std2

    rng2 = np.random.default_rng(0)
    idx = rng2.permutation(len(X_norm))
    split = int(0.9 * len(idx))
    train_idx, val_idx = idx[:split], idx[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_all[train_idx]).long(),
                      torch.tensor(f_norm[train_idx]).float()),
        batch_size=256, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float().to(device)
    val_y = torch.tensor(y_all[val_idx]).long().to(device)
    val_f = torch.tensor(f_norm[val_idx]).float().to(device)
    val_f_phys = f_all[val_idx]  # 실제 단위(N) 참고용, 3열(Fx,Fy,F_mag) 그대로 - F_mag은 비교용

    class MultiProbeClassifier(nn.Module):
        """구간분류(5-class) + 힘(Fx,Fy 보드좌표계, 2개 연속값) 동시 예측 멀티태스크 모델.
        F_mag은 별도 예측하지 않고 예측된 Fx,Fy로부터 sqrt(Fx^2+Fy^2)로 유도(항상 양수 보장)."""
        def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2):
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

        def forward(self, x):  # x: (B, n_probes, 3, 5, 5)
            embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
            h = self.trunk(torch.cat(embeds, dim=1))
            return self.seg_head(h), self.force_head(h)

    model = MultiProbeClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    seg_criterion = nn.CrossEntropyLoss()
    force_criterion = nn.MSELoss()
    FORCE_LOSS_WEIGHT = 1.0  # 두 손실이 비슷한 스케일(정규화됨)이라 1:1로 시작

    t_train = time.time()
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    for epoch in range(60):
        model.train()
        for bx, by, bf in train_loader:
            bx, by, bf = bx.to(device), by.to(device), bf.to(device)
            optimizer.zero_grad()
            seg_logits, force_pred = model(bx)
            loss = seg_criterion(seg_logits, by) + FORCE_LOSS_WEIGHT * force_criterion(force_pred, bf)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_seg_logits, val_force_pred = model(val_X)
            val_seg_loss = seg_criterion(val_seg_logits, val_y).item()
            val_force_loss = force_criterion(val_force_pred, val_f).item()
            val_loss = val_seg_loss + FORCE_LOSS_WEIGHT * val_force_loss
            val_acc = (val_seg_logits.argmax(dim=1) == val_y).float().mean().item()
        if val_loss < best_val_loss:  # 에폭별로 크게 흔들릴 수 있어서(작은샘플 테스트에서 확인됨)
            best_val_loss = val_loss  # val loss 기준으로 제일 좋았던 시점의 모델을 따로 저장
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1:2d}/60] ValAcc {val_acc*100:5.1f}%  SegLoss {val_seg_loss:.4f}  ForceLoss {val_force_loss:.4f}")
    print(f"학습 완료 ({time.time()-t_train:.0f}s), 최적 epoch={best_epoch} (val loss 기준)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_seg_logits, val_force_pred = model(val_X)
        val_pred = val_seg_logits.argmax(dim=1)
        val_force_pred_phys = val_force_pred.cpu().numpy() * f_std + f_mean  # 정규화 해제 (N 단위)
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

    # 힘(Fx,Fy, 보드좌표계) 회귀 성능 - R^2와 MAE. F_mag은 예측된 Fx,Fy로부터 유도(항상 양수).
    force_names = ["Fx_board_N", "Fy_board_N"]
    print(f"\n=== 힘(F) 회귀 성능 (보드좌표계, n_val={len(val_idx)}) ===")
    for i, name in enumerate(force_names):
        pred_i, true_i = val_force_pred_phys[:, i], val_f_phys[:, i]
        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(pred_i - true_i))
        print(f"  {name}: R^2={r2:.3f}, MAE={mae*1000:.4f}mN")

    pred_fmag = np.sqrt(val_force_pred_phys[:, 0] ** 2 + val_force_pred_phys[:, 1] ** 2)  # 유도, 항상>=0
    true_fmag_2d = np.sqrt(val_f_phys[:, 0] ** 2 + val_f_phys[:, 1] ** 2)  # Fx,Fy만으로 계산한 실제크기(공정비교)
    true_fmag_3d = val_f_phys[:, 2]  # 대체모델이 Fz까지 포함해서 직접 예측했던 원래 F_mag(참고용)
    for name, true_i in [("F_mag(유도, Fx-Fy만 기준 비교)", true_fmag_2d),
                          ("F_mag(참고, Fz포함 원래값과 비교)", true_fmag_3d)]:
        ss_res = np.sum((pred_fmag - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(pred_fmag - true_i))
        print(f"  {name}: R^2={r2:.3f}, MAE={mae*1000:.4f}mN")
    print(f"  (유도된 F_mag 중 음수 비율: {(pred_fmag < 0).mean()*100:.1f}% - sqrt라 구조적으로 항상 0)")

    pred_ang = np.degrees(np.arctan2(val_force_pred_phys[:, 1], val_force_pred_phys[:, 0]))
    true_ang = np.degrees(np.arctan2(val_f_phys[:, 1], val_f_phys[:, 0]))
    ang_err = np.abs(((pred_ang - true_ang + 180) % 360) - 180)  # 각도차 -180~180 보정
    print(f"  힘 방향각(atan2(Fy,Fx), 유도값): 평균 오차 {ang_err.mean():.1f}도, 중앙값 {np.median(ang_err):.1f}도")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
                "f_mean": f_mean, "f_std": f_std,
                "bin_width_mm": BIN_WIDTH_MM, "n_classes": N_CLASSES, "phi_probes": PHI_PROBES,
                "force_names": force_names},
               os.path.join(MODELS_DIR, "position_segment_classifier_multiprobe_150k_11probe_betafixed.pth"))
    print(f"\n저장: {MODELS_DIR}/position_segment_classifier_multiprobe_150k_11probe_betafixed.pth")
    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분")
