"""train_segment_classifier_singleprobe_4seg.py 기반 + 교수님 피드백 반영판.

교수님 지시: 멀티프로브(3개/11개)는 그만두고 단일 관측 유지. beta(원주각)를 0도/180도
두 값으로만 제한(실험 조건 단순화). phi/L_M은 그대로 예측 대상 유지(빼지 말 것 - 처음에
phi를 0/180으로 제한하고 phi/L_M 예측도 빼는 걸로 잘못 이해했다가 정정함). 힘/위치도
그대로 유지. "기구학적으로 풀고 다이나믹한 건 하지 말라"는 지시는 이미 만족됨 -
force_model.py의 solve_shape()는 원래부터 정역학/기구학 모델.

원본(train_segment_classifier_singleprobe_4seg.py) 대비 유일한 차이:
- beta: rng.uniform(0,360) 균일분포 -> rng.choice([0,180]) 두 값 중 하나만.
  (beta 제외 실험(90/270 근처만 빼기)은 이미 별도로 해봤고 득이 없었음 - 이번은 그거랑
  다르게 아예 0/180 두 값으로 극단적으로 좁히는 실험)
"""
import hashlib
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
from sklearn.model_selection import GroupShuffleSplit, KFold
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")
MODELS_DIR = os.path.join(HERE, "..", "..", "..", "models")
HYUNSEO_DIR = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(HYUNSEO_DIR, ".."))
FORCE_MODEL_DIR = os.path.join(REPO_ROOT, "scripts", "force_model")

FEATURES = ["L_M_mm", "phi_deg", "beta_deg", "contact_s_mm", "push_depth_mm"]
TARGETS = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_uz_avg_mm", "tip_theta_deg_board",
           "Fx_total_N", "Fy_total_N", "Fz_total_N", "F_mag_N",
           "mom_ux_avg_mm", "mom_uy_avg_mm", "mom_uz_avg_mm", "mom_theta_deg_board"]
# 2026-08-26 추가: MOM(강체구간) 자체의 변위를 대체모델이 직접 예측하도록 타겟에 추가.
# 예전엔 "MOM은 팁 변위의 L_M/100만큼만 움직인다"는 frac 근사를 썼는데, L_M=0(MOM이 베이스
# 바로 옆)에서 frac~0이 되어 MOM 변위 신호가 거의 사라지는 바람에 L_M 실측 R^2가 0.565로
# 낮게 나온 원인이 됨(_diag_lm_holdout_error.py로 확인) - 실측 FEA(mom_*_avg_mm)가 있는
# 새 데이터부터는 이 값을 그대로 쓰고, 없는 옛 데이터는 하위호환을 위해 frac 근사로 대체.
MOM_TARGETS_MISSING_OLD_DATA = ["mom_ux_avg_mm", "mom_uy_avg_mm", "mom_uz_avg_mm", "mom_theta_deg_board"]
DEFAULTS = {"L_M_mm": 50.0, "phi_deg": 60.0, "beta_deg": 0.0}

BIN_WIDTH_MM = 20.0
N_CLASSES = 4
PHI_RANGE = (-150.0, 150.0)
BETA_VALUES = [0.0, 180.0]  # 교수님 지시: beta는 0도/180도만
N_PROBES = 1
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

    # 2026-08-19 비판적 리뷰 반영 #5: 대체모델(서로게이트) 10개 앙상블이 서로 크게 갈리는
    # 지점 = 그 (L_M,phi,s,beta) 근처에 학습 시 참고할 실측 FEA가 부족해서 외삽하고 있다는
    # 신호. 정규화된 공간에서 앙상블 표준편차를 같이 반환해서, 아래 worker 루프에서 이 불일치가
    # 큰 샘플은 버리고 다시 뽑도록(rejection sampling) 함 - "모르는 영역"이 15만개에 조용히
    # 섞여 들어가는 걸 막음.
    DISAGREEMENT_TARGETS = [TARGETS.index(t) for t in
                             ("tip_ux_avg_mm", "tip_uy_avg_mm", "tip_theta_deg_board",
                              "Fx_total_N", "Fy_total_N")]

    def predict_surrogate(L_M, phi, beta, s, depth):
        x = np.array([[L_M, phi, beta, s, depth]])
        xn = (x - X_mean) / X_std
        xt = torch.tensor(xn, dtype=torch.float32)
        with torch.no_grad():
            ens = np.array([m(xt).numpy()[0] for m in surrogates])
        pn = ens.mean(axis=0)
        disagreement = ens[:, DISAGREEMENT_TARGETS].std(axis=0).mean()  # 정규화 공간 기준
        p = pn * y_std + y_mean
        return dict(zip(TARGETS, p)), disagreement

    rng = np.random.default_rng(2000 + widx)
    L_M_range = (0.0, 100.0)
    s_range = (10.0, 80.0)  # 팁쪽 80-100mm 제외 (힘이 너무 약해 실제 감지 불가로 판단)
    FIXED_DEPTH = 0.10
    # 2026-08-19 비판적 리뷰 #5: 앙상블 불일치(정규화 표준편차) 임계값. 환경변수로 조절 가능
    # (기본 0.5 - 대략 상위 20~30% 정도 불일치가 큰 샘플을 거름, 데이터 분포에 따라 다름).
    DISAGREEMENT_THRESHOLD = float(os.environ.get("DISAGREEMENT_THRESHOLD", 0.5))
    n_rejected = 0

    free_cache = {}
    Xb = np.zeros((n_chunk, N_PROBES, 3, 5, 5), dtype=np.float32)
    yb = np.zeros(n_chunk, dtype=np.int64)
    fb = np.zeros((n_chunk, 3), dtype=np.float32)
    sb = np.zeros(n_chunk, dtype=np.float32)
    cb = np.zeros((n_chunk, 2), dtype=np.float32)  # L_M(mm), phi(deg)
    n_ok = 0
    while n_ok < n_chunk:
        L_M = rng.uniform(*L_M_range)
        s = rng.uniform(*s_range)
        beta = rng.choice(BETA_VALUES)  # 교수님 지시: 0도 또는 180도만
        depth = FIXED_DEPTH
        phi = rng.uniform(*PHI_RANGE)  # phi는 그대로 연속 샘플링(예측 대상 유지)

        pred, disagreement = predict_surrogate(L_M, phi, beta, s, depth)
        if disagreement > DISAGREEMENT_THRESHOLD:
            n_rejected += 1
            continue  # 서로게이트가 자신 없는(=실측 FEA가 부족한) 영역 - 다시 뽑음

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

        d_xL_local = pred["tip_uy_avg_mm"]
        d_yL_local = pred["tip_ux_avg_mm"]
        d_thL = -pred["tip_theta_deg_board"]
        # 2026-08-26: frac(=L_M/100) 근사 폐기 - 서로게이트가 이제 MOM 자체 변위도 직접
        # 예측하므로(TARGETS에 mom_* 추가) 그걸 그대로 씀(축교환 방식은 tip과 동일).
        d_xLM_local = pred["mom_uy_avg_mm"]
        d_yLM_local = pred["mom_ux_avg_mm"]
        d_thLM = -pred["mom_theta_deg_board"]

        xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
        xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]

        B_free = compute_B(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
        B_load = compute_B(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                            xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
        Xb[n_ok, 0] = (B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1)

        yb[n_ok] = s_to_bin(s)
        fb[n_ok] = [pred["Fy_total_N"], pred["Fx_total_N"], pred["F_mag_N"]]  # 축교환
        sb[n_ok] = s
        cb[n_ok] = [L_M, phi]
        n_ok += 1
    return Xb, yb, fb, sb, cb, n_rejected


class SingleProbeClassifier(nn.Module):
    """구간분류(4-class) + 힘(Fx,Fy 보드좌표계) + 연속값 s(보조회귀) + L_M,phi(액추에이터
    슬랙 보정용) 동시 예측 - 프로브 1개, beta는 0도/180도로 한정."""
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
        self.config_head = nn.Linear(128, n_config)

    def forward(self, x):
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds, dim=1))
        return self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1), self.config_head(h)


if __name__ == "__main__":
    t_start = time.time()

    # 2026-08-19: 옛 4개 파일(BR=0.36T, 논문피팅 K1/K2 기준, beta=0 고정)을 새 재료값
    # (BR=0.4T, MATLAB E*I 기준 K1/K2) 스윕 1개로 교체. 옛 파일들은 beta_deg가 전부 0.0으로
    # 고정돼있어 대체모델이 beta=180 입력을 한 번도 실제로 학습 못 하고 외삽만 했었는데(잠재
    # 버그), 이 새 파일은 beta=0(133개)/180(48개) 실측이 둘 다 있어서 그 문제도 같이 해결됨
    # (beta=180 대칭성은 check_beta180_symmetry.py로 검증됨, 별도 부호반전 증강 불필요).
    SOURCES = ["fea_lm_phi_pos_matv2_all.json"]
    all_rows = []
    for fname in SOURCES:
        path = os.path.join(FEA_DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        for r in json.load(open(path)):
            row = dict(DEFAULTS)
            row.update(r)
            if "mom_ux_avg_mm" not in row:
                # 2026-08-26 이전 FEA(MOM 절점 변위를 안 뽑던 시절)와의 하위호환용 - 그때 쓰던
                # frac 근사("MOM은 팁 변위의 L_M/100만큼만 움직인다")를 그대로 재현. 새 FEA는
                # 실측값을 그대로 쓰고, 이 근사는 옛 데이터에만 적용됨(row별로 다르게 처리하면
                # 코드가 사방에 흩어지니 로드 시점에 한 번만 통일).
                frac = row["L_M_mm"] / 100.0
                row["mom_ux_avg_mm"] = row["tip_ux_avg_mm"] * frac
                row["mom_uy_avg_mm"] = row["tip_uy_avg_mm"] * frac
                row["mom_uz_avg_mm"] = row.get("tip_uz_avg_mm", 0.0) * frac
                row["mom_theta_deg_board"] = row["tip_theta_deg_board"] * frac
            all_rows.append(row)

    # 2026-08-19 비판적 리뷰 반영 #2: 실측 FEA를 전부 대체모델 학습에 써버리면 최종 CNN을
    # "실제 물리"가 아니라 "대체모델의 자기 자신"으로만 검증하게 됨(순환검증). 20%를 대체모델
    # 학습에서 아예 빼고, 맨 끝에서 이 CNN을 대체모델 없이 순수 실측값으로만 평가하는 데 씀.
    #
    # 2026-08-25 수정: 예전엔 rng(42).permutation(len(all_rows))로 뽑았는데, 이러면 데이터가
    # 늘 때마다(예: 404->518개) 같은 시드를 써도 뽑히는 "행" 자체가 완전히 달라져서, 회차 간
    # 홀드아웃 R^2를 비교하는 게 원래 의미가 없었음(phi=90~150 s격자 조밀화 후 Fx_board R^2가
    # 0.729->0.462로 "악화"돼 보였던 게 실은 홀드아웃 구성이 바뀌어서였다는 게
    # _diag_holdout_phi_breakdown.py로 확인됨 - 저각도만 떼어보면 여전히 0.64로 정상).
    # 대신 각 행 고유키(L_M,phi,beta,s)를 해시해서 결정하면, 데이터가 늘어도 "기존 행"의
    # 홀드아웃 소속은 절대 안 바뀌고 새 행만 새로 배정됨 - 회차 간 비교가 처음으로 안정적이 됨.
    def is_holdout_row(r, frac=0.2):
        key = f"{r['L_M_mm']}_{r['phi_deg']}_{r['beta_deg']}_{r['contact_s_mm']}"
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return (h % 10000) < int(frac * 10000)

    holdout_idx = [i for i, r in enumerate(all_rows) if is_holdout_row(r)]
    fit_idx = [i for i, r in enumerate(all_rows) if not is_holdout_row(r)]
    real_holdout_rows = [all_rows[i] for i in holdout_idx]
    fit_rows = [all_rows[i] for i in fit_idx]
    print(f"대체모델 학습 데이터: {len(fit_rows)}개 (전체 {len(all_rows)}개 중 {len(real_holdout_rows)}개는 "
          f"순수 실측 검증용으로 분리, {', '.join(SOURCES)})")

    X = np.array([[r[f] for f in FEATURES] for r in fit_rows])
    y = np.array([[r[t] for t in TARGETS] for r in fit_rows])
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

    print(f"{N_SAMPLES}개 단일관측(beta=0/180도 한정) 합성 데이터 생성 시작 ({N_WORKERS}-way 병렬)...")
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
    n_rejected_total = sum(r[5] for r in results)
    print(f"합성 데이터 생성 완료: {len(y_all)}개 ({time.time()-t_gen:.0f}s), "
          f"서로게이트 불일치로 거른 샘플: {n_rejected_total}개")
    print("구간별 샘플 수:", {c: int((y_all == c).sum()) for c in range(N_CLASSES)})

    np.savez(os.path.join(FEA_DATA_DIR, "segment_bfield_singleprobe_beta0180_4seg.npz"),
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

    # 2026-08-25 실험: |phi|>=90 구간은 토크제로 특이점 근방이라 힘이 작고 불안정해서
    # 원래부터 예측이 어려움(실측 홀드아웃에서 이 구간만 R^2가 낮게 나옴, PROJECT_STATUS.md
    # 참고). 새 FEA 없이도 힘 손실 가중치만 이 구간에 더 줘서 모델이 더 집중하게 하면
    # 나아지는지 확인하는 실험(사용자 요청) - seg/s/config 손실은 그대로 둠(이미 잘 됨,
    # 굳이 건드려서 흔들 필요 없음).
    HIGH_PHI_WEIGHT = 3.0
    phi_weight_all = np.where(np.abs(c_all[:, 1]) >= 90, HIGH_PHI_WEIGHT, 1.0).astype(np.float32)
    print(f"|phi|>=90 힘 손실 가중치 {HIGH_PHI_WEIGHT}배 적용: "
          f"{int((phi_weight_all > 1).sum())}/{len(phi_weight_all)}개 샘플")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"최종 분류기 학습 시작 (device={device})...")

    X_mean2, X_std2 = X_all.mean(), X_all.std()
    X_norm = (X_all - X_mean2) / X_std2

    # 2026-08-19 비판적 리뷰 #3: 그냥 무작위 90/10을 하면 free_cache로 (L_M,phi) 형상이
    # 재사용되는 특성상 train/val에 "사실상 같은 형상, s/beta/depth만 다른" 샘플이 섞여
    # 들어가서 val 정확도가 "안 본 형상 일반화력"이 아니라 "본 형상의 보간력"을 재는 문제가
    # 있었음. (L_M,phi)를 10mm/10deg 격자로 묶어서 그룹 단위로 통째로 train 또는 val에
    # 배정 - 같은 형상이 양쪽에 걸치지 않게 함.
    L_M_bin = np.round(c_all[:, 0] / 10.0).astype(int)
    phi_bin = np.round(c_all[:, 1] / 10.0).astype(int)
    groups = L_M_bin * 1000 + phi_bin
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    train_idx, val_idx = next(gss.split(X_norm, groups=groups))
    print(f"train/val 분리: 형상(L_M,phi) 그룹 {len(np.unique(groups))}개 중 "
          f"train {len(np.unique(groups[train_idx]))}개 / val {len(np.unique(groups[val_idx]))}개 "
          f"(겹치는 그룹 {len(set(groups[train_idx]) & set(groups[val_idx]))}개)")

    N_EPOCHS = int(os.environ.get("N_EPOCHS", 60))

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_all[train_idx]).long(),
                      torch.tensor(f_norm[train_idx]).float(), torch.tensor(s_norm[train_idx]).float(),
                      torch.tensor(c_norm[train_idx]).float(),
                      torch.tensor(phi_weight_all[train_idx]).float()),
        batch_size=256, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float().to(device)
    val_y = torch.tensor(y_all[val_idx]).long().to(device)
    val_f = torch.tensor(f_norm[val_idx]).float().to(device)
    val_f_phys = f_all[val_idx]
    val_s = torch.tensor(s_norm[val_idx]).float().to(device)
    val_s_phys = s_all[val_idx]
    val_c = torch.tensor(c_norm[val_idx]).float().to(device)
    val_c_phys = c_all[val_idx]
    val_phi_weight = torch.tensor(phi_weight_all[val_idx]).float().to(device)

    # 2026-08-26 추가: 여기까지 최종 CNN(SingleProbeClassifier) 학습에는 시드 고정이 전혀
    # 없었음(대체모델 앙상블만 seed=i로 고정돼있었음) - 가중치 초기화, DataLoader shuffle이
    # 전부 매 실행마다 랜덤이라, 같은 코드/데이터로 재학습해도 holdout 지표가 그냥 시드
    # 차이만으로 크게 흔들릴 수 있었음(예: mom_* 타겟 추가 후 Fy_board R^2가 0.857->0.648로
    # "하락"한 게 실제 원인 때문인지 순수 시드 노이즈인지 구분이 안 됐던 문제 - 앞으로는
    # 이 시드를 고정해서 회차 간 비교가 "같은 시드, 다른 코드"가 되게 함).
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    model = SingleProbeClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    seg_criterion = nn.CrossEntropyLoss()
    s_criterion = nn.MSELoss()
    config_criterion = nn.MSELoss()
    # 2026-08-19 비판적 리뷰 #1: 대체모델 5-fold R^2가 Fy_total_N=-0.01(사실상 노이즈, 평균보다도
    # 못 맞춤)이라 이걸 "정답"으로 그대로 학습시키면 CNN이 서로게이트의 노이즈를 따라 배움.
    # 완전히 빼는 대신(F_mag=sqrt(Fx^2+Fy^2) 유도 로직이 Fx,Fy 둘 다 필요해서 구조를 안 바꿔도
    # 되게) 가중치를 1/10로 낮춰 신뢰 못 할 타겟이 학습을 왜곡하는 걸 줄임.
    # ⚠️ 열 순서 주의: fb/fxy_all은 [Fy_total_N(로컬), Fx_total_N(로컬)] 순서로 저장됨(위
    # worker()의 "축교환" 주석 - 보드좌표계 90도 회전 때문). 즉 0번 열이 R^2=-0.01인
    # Fy_total_N(로컬)이고, 이게 나중에 force_names=["Fx_board_N","Fy_board_N"]로 "표시"만
    # 될 뿐 실제 학습 순서는 그대로임 - 가중치를 반대로 넣으면 정작 나쁜 타겟이 그대로 살아있는
    # 채로 고친 척하게 됨.
    FORCE_LOSS_WEIGHTS = torch.tensor([0.1, 1.0], device=device)  # [Fy_total_N(로컬,나쁨), Fx_total_N(로컬,좋음)]

    def weighted_force_loss(pred, true, sample_weight=None):
        per_sample = (((pred - true) ** 2) * FORCE_LOSS_WEIGHTS).mean(dim=1)
        if sample_weight is not None:
            per_sample = per_sample * sample_weight
        return per_sample.mean()

    t_train = time.time()
    best_bal_acc = -1.0
    best_state = None
    best_epoch = -1
    for epoch in range(N_EPOCHS):
        model.train()
        for bx, by, bf, bs, bc, bw in train_loader:
            bx, by, bf, bs, bc, bw = (bx.to(device), by.to(device), bf.to(device), bs.to(device),
                                       bc.to(device), bw.to(device))
            optimizer.zero_grad()
            seg_logits, force_pred, s_pred, config_pred = model(bx)
            loss = (seg_criterion(seg_logits, by) + weighted_force_loss(force_pred, bf, bw)
                    + s_criterion(s_pred, bs) + config_criterion(config_pred, bc))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_seg_logits, val_force_pred, val_s_pred, val_config_pred = model(val_X)
            val_seg_loss = seg_criterion(val_seg_logits, val_y).item()
            val_force_loss = weighted_force_loss(val_force_pred, val_f, val_phi_weight).item()
            val_s_loss = s_criterion(val_s_pred, val_s).item()
            val_config_loss = config_criterion(val_config_pred, val_c).item()
            val_pred_epoch = val_seg_logits.argmax(dim=1)
            val_acc = (val_pred_epoch == val_y).float().mean().item()
            # 2026-08-19 비판적 리뷰 #4: 체크포인트를 "4개 손실 단순합" 대신 실제 목표인
            # 구간분류 balanced accuracy(클래스별 recall 평균) 기준으로 고름 - 손실 스케일이
            # 서로 다른 4개 태스크를 더한 값이 우연히 낮다고 해서 분류 성능이 최선이란 보장이
            # 없었음.
            conf_epoch = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.int32)
            for t, p in zip(val_y.tolist(), val_pred_epoch.tolist()):
                conf_epoch[t, p] += 1
            bal_acc = float(np.mean([conf_epoch[i, i].item() / max(1, conf_epoch[i].sum().item())
                                      for i in range(N_CLASSES)]))
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:2d}/{N_EPOCHS}] ValAcc {val_acc*100:5.1f}%  BalAcc {bal_acc*100:5.1f}%  SegLoss {val_seg_loss:.4f}  ForceLoss {val_force_loss:.4f}  SLoss {val_s_loss:.4f}  ConfigLoss {val_config_loss:.4f}")
    print(f"학습 완료 ({time.time()-t_train:.0f}s), 최적 epoch={best_epoch} (balanced accuracy 기준, {best_bal_acc*100:.1f}%)")

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
    print(f"\n=== 로봇 형상(L_M, phi) 회귀 성능 (n_val={len(val_idx)}) ===")
    for i, name in enumerate(config_names):
        pred_i, true_i = val_config_pred_phys[:, i], val_c_phys[:, i]
        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        mae = np.mean(np.abs(pred_i - true_i))
        unit = "mm" if name == "L_M_mm" else "deg"
        print(f"  {name}: R^2={r2:.3f}, MAE={mae:.2f}{unit}")

    # 2026-08-19 비판적 리뷰 반영 #2: 지금까지의 val_* 지표는 전부 대체모델이 만든 합성데이터
    # 안에서만 도는 순환검증(같은 서로게이트의 가정을 재확인하는 것)이라, 대체모델 학습에서
    # 아예 제외해둔 real_holdout_rows(실측 FEA, 대체모델이 한 번도 못 본 값)로 별도 평가함.
    # 이 블록은 worker()와 똑같은 로직(free-shape + 변위→B-field 차분)을 쓰되, 서로게이트
    # 예측 대신 실측 FEA 값을 그대로 씀 - "이 모델이 시뮬레이션 자기 자신이 아니라 진짜
    # FEA 물리에 맞는가"를 재는 유일한 지표.
    print(f"\n=== 순수 실측 FEA 검증 (대체모델 안 거침, n={len(real_holdout_rows)}) ===")
    sys.path.insert(0, FORCE_MODEL_DIR)
    import force_model as fm_eval
    import magpylib as magpy_eval
    from scipy.spatial.transform import Rotation as Rot_eval

    SENSOR_HEIGHT_MM = 15
    sensor_positions_eval = [(x, y, SENSOR_HEIGHT_MM) for y in np.linspace(180, 0, 5) for x in np.linspace(0, 180, 5)]
    sensors_eval = magpy_eval.Collection([magpy_eval.Sensor(position=pos) for pos in sensor_positions_eval])
    MAGNET_BR_TESLA = 0.4
    main_magnet_eval = magpy_eval.magnet.Cylinder(polarization=(0, MAGNET_BR_TESLA, 0), dimension=(2, 2))
    mom_eval = magpy_eval.magnet.Cylinder(polarization=(0, -MAGNET_BR_TESLA, 0), dimension=(1, 8))
    mscr_robot_eval = magpy_eval.Collection(main_magnet_eval, mom_eval)

    def compute_B_eval(xLM_l, yLM_l, thLM, xL_l, yL_l, thL):
        xLM_b, yLM_b = fm_eval.to_board_frame(xLM_l, yLM_l)
        xL_b, yL_b = fm_eval.to_board_frame(xL_l, yL_l)
        mom_eval.position = (float(xLM_b), float(yLM_b), 0)
        mom_eval.orientation = Rot_eval.from_euler("z", -thLM, degrees=True)
        main_magnet_eval.position = (float(xL_b), float(yL_b), 0)
        main_magnet_eval.orientation = Rot_eval.from_euler("z", -thL, degrees=True)
        return magpy_eval.getB(mscr_robot_eval, sensors_eval) * 1e6

    real_X, real_y, real_f, real_s, real_c = [], [], [], [], []
    for r in real_holdout_rows:
        L_M, phi = r["L_M_mm"], r["phi_deg"]
        s = r["contact_s_mm"]
        try:
            r_free = fm_eval.solve_shape(L_M=L_M, phi_deg=phi, loads=[])
        except Exception:
            continue
        d_xL_local, d_yL_local = r["tip_uy_avg_mm"], r["tip_ux_avg_mm"]  # 축교환(worker()와 동일)
        d_thL = -r["tip_theta_deg_board"]
        # 2026-08-26: frac 근사 폐기, all_rows 로딩 시점에 채워진 실측(또는 하위호환 근사)
        # mom_* 값을 그대로 씀(worker()와 동일한 축교환 방식).
        d_xLM_local = r["mom_uy_avg_mm"]
        d_yLM_local = r["mom_ux_avg_mm"]
        d_thLM = -r["mom_theta_deg_board"]
        xL_free, yL_free, thL_free = r_free["x_L"], r_free["y_L"], r_free["theta_L_deg"]
        xLM_free, yLM_free, thLM_free = r_free["x_LM"], r_free["y_LM"], r_free["theta_LM_deg"]
        B_free = compute_B_eval(xLM_free, yLM_free, thLM_free, xL_free, yL_free, thL_free)
        B_load = compute_B_eval(xLM_free + d_xLM_local, yLM_free + d_yLM_local, thLM_free + d_thLM,
                                 xL_free + d_xL_local, yL_free + d_yL_local, thL_free + d_thL)
        real_X.append((B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1))
        real_y.append(s_to_bin(s))
        real_f.append([r["Fy_total_N"], r["Fx_total_N"]])  # fb와 동일 순서(축교환)
        real_s.append(s)
        real_c.append([L_M, phi])  # config_names=["L_M_mm","phi_deg"]와 동일 순서

    if len(real_X) < 5:
        print(f"  free-shape 계산 성공 케이스가 {len(real_X)}개뿐이라 통계적으로 의미 있는 평가 불가")
    else:
        real_X = np.array(real_X, dtype=np.float32)
        real_X_norm = (real_X - X_mean2) / X_std2
        real_y_arr = np.array(real_y)
        real_f_arr = np.array(real_f, dtype=np.float32)
        real_s_arr = np.array(real_s, dtype=np.float32)
        real_c_arr = np.array(real_c, dtype=np.float32)

        model.eval()
        with torch.no_grad():
            rX = torch.tensor(real_X_norm[:, None]).float().to(device)  # (n,1,3,5,5) - probe 차원 추가
            r_seg_logits, r_force_pred, r_s_pred, r_config_pred = model(rX)
            r_pred_class = r_seg_logits.argmax(dim=1).cpu().numpy()
            r_force_phys = r_force_pred.cpu().numpy() * f_std + f_mean
            r_s_phys = r_s_pred.cpu().numpy() * s_std + s_mean
            r_config_phys = r_config_pred.cpu().numpy() * c_std + c_mean

        real_acc = float((r_pred_class == real_y_arr).mean())
        conf_r = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for t, p in zip(real_y_arr, r_pred_class):
            conf_r[t, p] += 1
        real_bal_acc = float(np.mean([conf_r[i, i] / max(1, conf_r[i].sum()) for i in range(N_CLASSES)]))
        print(f"  구간분류: acc={real_acc*100:.1f}%, balanced acc={real_bal_acc*100:.1f}% (n={len(real_y_arr)}, "
              f"합성-val 기준 balanced acc={best_bal_acc*100:.1f}%와 비교할 것)")

        ss_res = np.sum((r_s_phys - real_s_arr) ** 2)
        ss_tot = np.sum((real_s_arr - real_s_arr.mean()) ** 2)
        s_r2_real = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"  s(연속값): R^2={s_r2_real:.3f}, MAE={np.mean(np.abs(r_s_phys - real_s_arr)):.2f}mm")

        for i, name in enumerate(force_names):
            ss_res = np.sum((r_force_phys[:, i] - real_f_arr[:, i]) ** 2)
            ss_tot = np.sum((real_f_arr[:, i] - real_f_arr[:, i].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"  {name}: R^2={r2:.3f}, MAE={np.mean(np.abs(r_force_phys[:, i] - real_f_arr[:, i]))*1000:.4f}mN")

        # 2026-08-25 추가: config_head(L_M,phi)도 지금까지 seg/s/force와 달리 합성-val로만
        # 검증하고 실측 홀드아웃 검증이 빠져있었음 - 힘 추정 때 겪은 것과 같은 종류의 맹점이라
        # 똑같이 채움.
        for i, name in enumerate(config_names):
            pred_i, true_i = r_config_phys[:, i], real_c_arr[:, i]
            ss_res = np.sum((pred_i - true_i) ** 2)
            ss_tot = np.sum((true_i - true_i.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            unit = "mm" if name == "L_M_mm" else "deg"
            print(f"  {name}: R^2={r2:.3f}, MAE={np.mean(np.abs(pred_i - true_i)):.2f}{unit} "
                  f"(합성-val 기준 R^2={r2_score(val_c_phys[:, i], val_config_pred_phys[:, i]):.3f}와 비교할 것)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
                "f_mean": f_mean, "f_std": f_std, "s_mean": s_mean, "s_std": s_std,
                "c_mean": c_mean, "c_std": c_std,
                "bin_width_mm": BIN_WIDTH_MM, "n_classes": N_CLASSES, "phi_range": PHI_RANGE,
                "beta_values": BETA_VALUES, "force_names": force_names, "config_names": config_names},
               os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"))
    print(f"\n저장: {MODELS_DIR}/position_segment_classifier_singleprobe_beta0180_4seg.pth")
    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분")
