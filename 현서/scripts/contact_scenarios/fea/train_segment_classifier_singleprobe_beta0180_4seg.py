"""train_segment_classifier_singleprobe_4seg.py 기반 + 교수님 피드백 반영판.

beta(원주각)를 0도/180도
두 값으로만 제한(실험 조건 단순화). phi/L_M은 그대로 예측 대상 유지(빼지 말 것 - 처음에
phi를 0/180으로 제한하고 phi/L_M 예측도 빼는 걸로 잘못 이해했다가 정정함). 
힘/위치도 그대로 유지. "기구학적으로 풀고 다이나믹한 건 하지 말라"는 지시는 이미 만족됨 -
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

# 2026-09-21(36번) shape_head: "충돌 시 로봇이 어떤 형상이 되는가"를 CNN이 직접 예측.
# 목표를 힘 정량화에서 형상 추정으로 전환하면서 추가(PROJECT_STATUS.md 34~36번).
# 힘을 전혀 거치지 않는다는 게 핵심 - delta-B는 애초에 "자석이 얼마나 움직였는가"로부터
# 계산되는 신호라(worker()의 compute_B 참고), 변위를 되찾는 게 가장 직접적인 역문제임.
# 34번에서 확인: 변형의 "모양"은 (L_M,phi,s)가 100% 결정하고 깊이는 크기만 조절하므로
# (깊이 4배에도 팁변위/회전 비율 변동 0.011%), 모양 쪽은 known input으로 거의 결정됨.
# 서로게이트 난이도도 낮음(tip_theta R^2=0.97, tip_ux 0.88 vs 문제의 Fy_total_N 0.44).
SHAPE_NAMES = ["tip_ux_avg_mm", "tip_uy_avg_mm", "tip_theta_deg_board"]

# (예전엔 여기에 L_M=0 근처 연속회귀 실패 문제를 우회하는 lm_zero_head/LM_ZERO_THRESHOLD_MM가
# 있었음 - 2026-09-11에 L_M을 아예 예측 대상에서 빼고 known input으로 바꾸면서 그 우회
# 자체가 필요 없어져 제거함. 자세한 배경은 SingleProbeClassifier 클래스 docstring 참고.)
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


# 2026-09-07 추가: _diag_s_breakdown.py로 s=60-100mm(팁 근처) MAE=9.49mm(다른 구간
# 2.88~5.56mm)로 유독 나쁜 게 확인됨 - s>=40mm부터 60-100mm까지 부드럽게 커지는 연속
# 가중치를 s_head loss에 곱해서 그 구간 오차에 더 민감하게 반응하게 함. (HIGH_PHI_WEIGHT
# 실험에서 계단식 가중치가 정작 목표 구간을 못 고친 전례가 있어 이번엔 연속 램프로 설계함
# - 그래도 "된다"는 보장은 없음, 실측 홀드아웃 재확인 필수.)
S_WEIGHT_RAMP_START_MM = 40.0
S_WEIGHT_RAMP_END_MM = 100.0
S_WEIGHT_MAX = 3.0


def s_spatial_weight(s_mm):
    ramp = np.clip((s_mm - S_WEIGHT_RAMP_START_MM) / (S_WEIGHT_RAMP_END_MM - S_WEIGHT_RAMP_START_MM), 0.0, 1.0)
    return 1.0 + (S_WEIGHT_MAX - 1.0) * ramp


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
    # 2026-09-21(34번): 여태 FIXED_DEPTH=0.10으로 고정 샘플링해서, 합성 15만개가 전부
    # "같은 세기로 부딪힌" 데이터였음 - 충돌 강도를 구분할 근거가 학습에 아예 없었다는 뜻.
    # 33/34번에서 0.05/0.20mm FEA를 확보했으므로 이제 깊이도 랜덤 샘플링함.
    # 범위 근거(34번 분석):
    #  - 하한 0.08: 0.05mm는 힘의 67%가 0.001mN 미만(파이프라인 힘 MAE ~0.005mN보다 작음)이고
    #    MESH_SIZE_TUBE=0.3mm 대비 관입이 1/6이라 접촉 해석 자체가 부실 -> 그 구간은 피함.
    #  - 상한 0.20: 실측 FEA 앵커가 0.10(655행)과 0.20(74행)뿐이라 그 위는 순수 외삽.
    #    깊이-변형이 거의 선형(원점통과 R²=0.982)이라 0.10~0.20 사이 보간은 안전함.
    # 깊이는 CNN 입력이 아님(실제 운용에서 "얼마나 세게 부딪혔는지"는 모르는 값 = 감지 대상).
    # 서로게이트가 자신 없는 깊이 영역은 아래 DISAGREEMENT_THRESHOLD 리젝션이 알아서 걸러줌.
    DEPTH_RANGE = (float(os.environ.get("DEPTH_MIN", 0.08)),
                   float(os.environ.get("DEPTH_MAX", 0.20)))
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
    # 2026-09-21(36번): shape_head용 정답. 여태 팁 변위는 자기장을 만드는 데만 쓰고 버렸는데
    # (아래 d_xL_local 등), 이제 "충돌 시 로봇이 어떤 형상이 되는가"를 CNN이 직접 예측하도록
    # 정답으로도 저장함. 보드좌표계 원본값(축교환 전)을 그대로 씀 - 실측 FEA 행의
    # tip_ux_avg_mm/tip_uy_avg_mm/tip_theta_deg_board와 같은 정의여야 홀드아웃 평가가 맞음.
    shb = np.zeros((n_chunk, len(SHAPE_NAMES)), dtype=np.float32)
    n_ok = 0
    while n_ok < n_chunk:
        L_M = rng.uniform(*L_M_range)
        s = rng.uniform(*s_range)
        beta = rng.choice(BETA_VALUES)  # 교수님 지시: 0도 또는 180도만
        depth = rng.uniform(*DEPTH_RANGE)
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
        shb[n_ok] = [pred[t] for t in SHAPE_NAMES]
        n_ok += 1
    return Xb, yb, fb, sb, cb, shb, n_rejected


class SingleProbeClassifier(nn.Module):
    """구간분류(4-class) + 힘(Fx,Fy 보드좌표계) + 연속값 s(보조회귀) 예측 - 프로브 1개,
    beta는 0도/180도로 한정.

    2026-09-11 변경(교수님 피드백): L_M, phi는 더 이상 예측 대상이 아니라 네트워크
    입력으로 직접 받음. 둘 다 B-field에서 추정해야 할 미지수가 아니라 이미 아는 값 -
    phi는 조종자가 거는 외부자기장 방향(제어입력), L_M은 그 로봇의 고정된 MOM 위치
    스펙(하드웨어 상수). 이전엔 이 둘을 config_head로 "예측"하게 시켰는데, 그 예측오차가
    힘 성분 중 하나(Fx_board=F_mag*sinθ, θ는 L_M/phi의 비선형 함수)의 정확도를 깎아먹는
    원인으로 확인됨(_diag_force_geom_reconstruction.py: 예측 L_M/phi로 재구성 R^2=-0.35
    vs 진짜 L_M/phi로 재구성 R^2=0.75). L_M=0 근처 회귀가 불안정해서 넣었던 lm_zero_head도
    L_M을 더 이상 추정할 필요가 없어지면서 같이 제거됨(L_M=0 문제 자체가 사라짐).
    진짜 미지수(센서로 알아내야 하는 값)는 접촉위치(s)와 접촉힘(Fx,Fy)뿐."""
    def __init__(self, n_probes=N_PROBES, n_classes=N_CLASSES, n_force=2, n_config_in=2,
                 n_shape=len(SHAPE_NAMES)):
        super().__init__()
        self.n_probes = n_probes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * 5 * 5, 64), nn.ReLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 * n_probes + n_config_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )
        self.seg_head = nn.Linear(128, n_classes)
        self.force_head = nn.Linear(128, n_force)
        self.s_head = nn.Linear(128, 1)
        # 2026-09-21(36번) 신규: 충돌 시 팁 변위/회전(=형상) 직접 예측. SHAPE_NAMES 주석 참고.
        self.shape_head = nn.Linear(128, n_shape)

    def forward(self, x, config):
        """config: 정규화된 (L_M_mm, phi_deg) - 이미 아는 값, B-field와 함께 trunk에 들어감."""
        embeds = [self.encoder(x[:, p]) for p in range(self.n_probes)]
        h = self.trunk(torch.cat(embeds + [config], dim=1))
        return (self.seg_head(h), self.force_head(h), self.s_head(h).squeeze(-1),
                self.shape_head(h))


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
    sh_all = np.concatenate([r[5] for r in results], axis=0)  # 36번: shape_head 정답
    n_rejected_total = sum(r[6] for r in results)
    s_weight_all = s_spatial_weight(s_all).astype(np.float32)
    print(f"s 공간가중치(s={S_WEIGHT_RAMP_START_MM:.0f}mm부터 {S_WEIGHT_RAMP_END_MM:.0f}mm까지 "
          f"최대 {S_WEIGHT_MAX}배 램프): 평균={s_weight_all.mean():.2f}, "
          f"s>={S_WEIGHT_RAMP_END_MM:.0f}mm 샘플 수={int((s_all>=S_WEIGHT_RAMP_END_MM).sum())}개")
    print(f"합성 데이터 생성 완료: {len(y_all)}개 ({time.time()-t_gen:.0f}s), "
          f"서로게이트 불일치로 거른 샘플: {n_rejected_total}개")
    print("구간별 샘플 수:", {c: int((y_all == c).sum()) for c in range(N_CLASSES)})

    np.savez(os.path.join(FEA_DATA_DIR, "segment_bfield_singleprobe_beta0180_4seg.npz"),
             X=X_all, y=y_all, f=f_all, s=s_all, c=c_all, sh=sh_all)

    fxy_all = f_all[:, :2]
    f_mean, f_std = fxy_all.mean(axis=0), fxy_all.std(axis=0)
    f_std[f_std < 1e-12] = 1.0
    f_norm = (fxy_all - f_mean) / f_std

    # 36번: 타겟마다 스케일이 제각각이라(팁변위 ~0.2mm vs 힘 ~0.000005N) 정규화 없이 더하면
    # 숫자 큰 항만 학습됨 - 기존 f/s/c와 똑같이 평균0/표준편차1로 맞춤.
    sh_mean, sh_std = sh_all.mean(axis=0), sh_all.std(axis=0)
    sh_std[sh_std < 1e-12] = 1.0
    sh_norm = (sh_all - sh_mean) / sh_std
    print(f"shape_head 타겟 분포: " + ", ".join(
        f"{n}={m:+.4f}±{s:.4f}" for n, m, s in zip(SHAPE_NAMES, sh_mean, sh_std)))

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
    # 2026-08-27 재튜닝: 3.0으로는 balanced acc/Fy_board는 좋아졌지만 정작 목표였던
    # |phi|>=90 구간 Fx_board R^2=0.343으로 여전히 약했음(가중치가 너무 세서 그 구간
    # 노이즈에 과적합했을 가능성) - 1.5로 낮춰서 재시도. 이번에도 그 구간 Fx_board가
    # 안 좋아지면 이 레버 자체를 폐기하고 실측 FEA 추가 쪽으로 넘어갈 것(PROJECT_STATUS.md
    # 18번 참고).
    HIGH_PHI_WEIGHT = 1.5
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

    # c_norm(정규화된 L_M,phi)은 더 이상 학습 타겟이 아니라 model.forward()의 두 번째
    # 입력(known config)으로 씀 - DataLoader 텐서 순서/이름은 그대로 두되 역할만 바뀜.
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_norm[train_idx]).float(), torch.tensor(y_all[train_idx]).long(),
                      torch.tensor(f_norm[train_idx]).float(), torch.tensor(s_norm[train_idx]).float(),
                      torch.tensor(c_norm[train_idx]).float(),
                      torch.tensor(phi_weight_all[train_idx]).float(),
                      torch.tensor(s_weight_all[train_idx]).float(),
                      torch.tensor(sh_norm[train_idx]).float()),
        batch_size=256, shuffle=True)
    val_X = torch.tensor(X_norm[val_idx]).float().to(device)
    val_y = torch.tensor(y_all[val_idx]).long().to(device)
    val_f = torch.tensor(f_norm[val_idx]).float().to(device)
    val_f_phys = f_all[val_idx]
    val_s = torch.tensor(s_norm[val_idx]).float().to(device)
    val_s_phys = s_all[val_idx]
    val_c = torch.tensor(c_norm[val_idx]).float().to(device)  # known L_M,phi 입력(정규화)
    val_c_phys = c_all[val_idx]
    val_phi_weight = torch.tensor(phi_weight_all[val_idx]).float().to(device)
    val_s_weight = torch.tensor(s_weight_all[val_idx]).float().to(device)
    val_sh = torch.tensor(sh_norm[val_idx]).float().to(device)
    val_sh_phys = sh_all[val_idx]

    # 2026-08-26 추가: 여기까지 최종 CNN(SingleProbeClassifier) 학습에는 시드 고정이 전혀
    # 없었음(대체모델 앙상블만 seed=i로 고정돼있었음) - 가중치 초기화, DataLoader shuffle이
    # 전부 매 실행마다 랜덤이라, 같은 코드/데이터로 재학습해도 holdout 지표가 그냥 시드
    # 차이만으로 크게 흔들릴 수 있었음(예: mom_* 타겟 추가 후 Fy_board R^2가 0.857->0.648로
    # "하락"한 게 실제 원인 때문인지 순수 시드 노이즈인지 구분이 안 됐던 문제 - 앞으로는
    # 이 시드를 고정해서 회차 간 비교가 "같은 시드, 다른 코드"가 되게 함).
    FINAL_SEED = int(os.environ.get("FINAL_SEED", 42))
    torch.manual_seed(FINAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FINAL_SEED)
    model = SingleProbeClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    seg_criterion = nn.CrossEntropyLoss()
    shape_criterion = nn.MSELoss()  # 36번: shape_head. 정규화된 공간이라 가중치 1.0으로 시작.
    # 2026-08-19 비판적 리뷰 #1 (2026-09-14 재검토): 그 시점엔 대체모델 5-fold R^2가
    # Fy_total_N=-0.01(노이즈)이라 이 축 loss를 1/10로 깎았었음. 그 이후 데이터가 늘어서
    # (지금 518개) 같은 진단이 R^2=0.649로 나와 "노이즈라 못 배운다"는 전제가 깨졌길래
    # 0.1/0.5/1.0 세 값으로 재검증(시드43, 파인튜닝 없음, 0.5/1.0은 단일 실행):
    #   0.1: balanced acc 83.0%, |phi|>=90 Fx_board 0.440, Fy_board 0.855 (기존, 시드2개 교차검증됨)
    #   0.5: balanced acc 83.1%, |phi|>=90 Fx_board 0.477, Fy_board 0.673
    #   1.0: balanced acc 86.3%, |phi|>=90 Fx_board 0.382, Fy_board 0.704
    # 가중치를 올릴수록 목표(|phi|>=90 Fx_board)가 항상 좋아지는 것도 아니고(0.5>0.1>1.0,
    # 단조 아님), 대신 Fy_board(원래 제일 안정적이던 축)는 계속 나빠짐 - trunk 공유로 인한
    # 멀티태스크 트레이드오프. 0.5/1.0은 시드 1개뿐이라 이 파이프라인에서 이미 확인된 시드
    # 노이즈 폭(Fy_board가 같은 세팅에서도 0.608~0.855까지 흔들림) 안에 들어갈 수 있어 신뢰
    # 못 함. **결론: 가중치 튜닝은 여기서 폐기, 0.1(시드 2개로 검증된 유일한 설정)로 복귀.
    # |phi|>=90 문제는 loss weight가 아니라 그 구간 실측 FEA를 더 모으는 원래 처방으로.**
    # ⚠️ 열 순서 주의: fb/fxy_all은 [Fy_total_N(로컬), Fx_total_N(로컬)] 순서로 저장됨(위
    # worker()의 "축교환" 주석 - 보드좌표계 90도 회전 때문). 즉 0번 열이 Fy_total_N(로컬)이고,
    # 이게 나중에 force_names=["Fx_board_N","Fy_board_N"]로 "표시"만 될 뿐 실제 학습 순서는
    # 그대로임 - 가중치를 반대로 넣으면 정작 고치려는 축이 아닌 다른 축을 건드리게 됨.
    FORCE_LOSS_WEIGHTS = torch.tensor([0.1, 1.0], device=device)  # [Fy_total_N(로컬,=Fx_board), Fx_total_N(로컬,=Fy_board)]

    def weighted_force_loss(pred, true, sample_weight=None):
        per_sample = (((pred - true) ** 2) * FORCE_LOSS_WEIGHTS).mean(dim=1)
        if sample_weight is not None:
            per_sample = per_sample * sample_weight
        return per_sample.mean()

    def weighted_s_loss(pred, true, sample_weight):
        return (((pred - true) ** 2) * sample_weight).mean()

    t_train = time.time()
    best_bal_acc = -1.0
    best_state = None
    best_epoch = -1
    for epoch in range(N_EPOCHS):
        model.train()
        for bx, by, bf, bs, bc, bw, bsw, bsh in train_loader:
            bx, by, bf, bs, bc, bw, bsw, bsh = (bx.to(device), by.to(device), bf.to(device),
                                                 bs.to(device), bc.to(device), bw.to(device),
                                                 bsw.to(device), bsh.to(device))
            optimizer.zero_grad()
            seg_logits, force_pred, s_pred, shape_pred = model(bx, bc)
            loss = (seg_criterion(seg_logits, by) + weighted_force_loss(force_pred, bf, bw)
                    + weighted_s_loss(s_pred, bs, bsw) + shape_criterion(shape_pred, bsh))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_seg_logits, val_force_pred, val_s_pred, val_shape_pred = model(val_X, val_c)
            val_seg_loss = seg_criterion(val_seg_logits, val_y).item()
            val_force_loss = weighted_force_loss(val_force_pred, val_f, val_phi_weight).item()
            val_s_loss = weighted_s_loss(val_s_pred, val_s, val_s_weight).item()
            val_shape_loss = shape_criterion(val_shape_pred, val_sh).item()
            val_pred_epoch = val_seg_logits.argmax(dim=1)
            val_acc = (val_pred_epoch == val_y).float().mean().item()
            # 2026-08-19 비판적 리뷰 #4: 체크포인트를 "손실 단순합" 대신 실제 목표인
            # 구간분류 balanced accuracy(클래스별 recall 평균) 기준으로 고름 - 손실 스케일이
            # 서로 다른 태스크를 더한 값이 우연히 낮다고 해서 분류 성능이 최선이란 보장이
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
            print(f"Epoch [{epoch+1:2d}/{N_EPOCHS}] ValAcc {val_acc*100:5.1f}%  BalAcc {bal_acc*100:5.1f}%  SegLoss {val_seg_loss:.4f}  ForceLoss {val_force_loss:.4f}  SLoss {val_s_loss:.4f}  ShapeLoss {val_shape_loss:.4f}")
    print(f"학습 완료 ({time.time()-t_train:.0f}s), 최적 epoch={best_epoch} (balanced accuracy 기준, {best_bal_acc*100:.1f}%)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_seg_logits, val_force_pred, val_s_pred, val_shape_pred = model(val_X, val_c)
        val_pred = val_seg_logits.argmax(dim=1)
        val_force_pred_phys = val_force_pred.cpu().numpy() * f_std + f_mean
        val_s_pred_phys = val_s_pred.cpu().numpy() * s_std + s_mean
        val_shape_pred_phys = val_shape_pred.cpu().numpy() * sh_std + sh_mean
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

    # 2026-09-21(36번): shape_head - 충돌 시 형상(팁 변위/회전). 28번 교훈대로 R²와 MAE 병행.
    print(f"\n=== 형상(shape_head) 회귀 성능 (합성-val, n_val={len(val_idx)}) ===")
    for i, name in enumerate(SHAPE_NAMES):
        pred_i, true_i = val_shape_pred_phys[:, i], val_sh_phys[:, i]
        ss_res = np.sum((pred_i - true_i) ** 2)
        ss_tot = np.sum((true_i - true_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        unit = "deg" if "theta" in name else "mm"
        print(f"  {name}: R^2={r2:.3f}, MAE={np.mean(np.abs(pred_i - true_i)):.4f}{unit} "
              f"(정답값 표준편차={true_i.std():.4f}{unit})")

    # 2026-09-11: L_M,phi는 더 이상 모델이 예측하는 값이 아니라 known input(config_names
    # 순서로 val_c/val_c_phys에 그대로 들어있음)이라, 여기서 정확도를 "리포트"할 대상 자체가
    # 없음(입력=정답이라 R^2가 항상 1). L_M=0 근처 회귀 불안정 문제(이전 lm_zero_head로
    # 우회했던 것)도 L_M을 더 이상 추정하지 않으므로 자연히 사라짐.
    config_names = ["L_M_mm", "phi_deg"]  # known input 순서 (체크포인트에 기록용)

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

    # 2026-09-07 리팩터: 이 실측 행 -> B-field 배열 변환 로직을 함수로 뽑아서
    # real_holdout_rows뿐 아니라 fit_rows(파인튜닝용)에도 재사용.
    def rows_to_arrays(rows):
        Xs, ys, fs, ss, cs, shs = [], [], [], [], [], []
        for r in rows:
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
            Xs.append((B_load - B_free).reshape(5, 5, 3).transpose(2, 0, 1))
            ys.append(s_to_bin(s))
            fs.append([r["Fy_total_N"], r["Fx_total_N"]])  # fb와 동일 순서(축교환)
            ss.append(s)
            cs.append([L_M, phi])  # config_names=["L_M_mm","phi_deg"]와 동일 순서
            shs.append([r[t] for t in SHAPE_NAMES])  # 36번: 실측 FEA의 진짜 팁 변위/회전
        return Xs, ys, fs, ss, cs, shs

    real_X, real_y, real_f, real_s, real_c, real_sh = rows_to_arrays(real_holdout_rows)

    if len(real_X) < 5:
        print(f"  free-shape 계산 성공 케이스가 {len(real_X)}개뿐이라 통계적으로 의미 있는 평가 불가")
    else:
        real_X = np.array(real_X, dtype=np.float32)
        real_X_norm = (real_X - X_mean2) / X_std2
        real_y_arr = np.array(real_y)
        real_f_arr = np.array(real_f, dtype=np.float32)
        real_s_arr = np.array(real_s, dtype=np.float32)
        real_c_arr = np.array(real_c, dtype=np.float32)
        real_sh_arr = np.array(real_sh, dtype=np.float32)  # 36번: 형상 정답(실측 FEA)
        # L_M,phi는 실측 FEA 행에도 이미 정답으로 들어있는 known input이라(예측할 필요
        # 없음), 여기서 정규화해서 model.forward()의 config 입력으로 그대로 씀.
        real_c_norm = (real_c_arr - c_mean) / c_std

        # 2026-09-07 리팩터: 아래 평가 블록을 함수로 뽑음 - 실측 파인튜닝(다음 블록) 전/후
        # 성능을 같은 홀드아웃으로 두 번 비교해야 해서 재사용이 필요해짐.
        def evaluate_real(label):
            model.eval()
            with torch.no_grad():
                rX = torch.tensor(real_X_norm[:, None]).float().to(device)  # (n,1,3,5,5)
                rC = torch.tensor(real_c_norm).float().to(device)  # known L_M,phi 입력
                r_seg_logits, r_force_pred, r_s_pred, r_shape_pred = model(rX, rC)
                r_pred_class = r_seg_logits.argmax(dim=1).cpu().numpy()
                r_force_phys = r_force_pred.cpu().numpy() * f_std + f_mean
                r_s_phys = r_s_pred.cpu().numpy() * s_std + s_mean
                r_shape_phys = r_shape_pred.cpu().numpy() * sh_std + sh_mean

            print(f"\n--- [{label}] 순수 실측 FEA 검증 (n={len(real_y_arr)}) ---")
            real_acc = float((r_pred_class == real_y_arr).mean())
            conf_r = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
            for t, p in zip(real_y_arr, r_pred_class):
                conf_r[t, p] += 1
            real_bal_acc = float(np.mean([conf_r[i, i] / max(1, conf_r[i].sum()) for i in range(N_CLASSES)]))
            print(f"  구간분류: acc={real_acc*100:.1f}%, balanced acc={real_bal_acc*100:.1f}%")

            ss_res = np.sum((r_s_phys - real_s_arr) ** 2)
            ss_tot = np.sum((real_s_arr - real_s_arr.mean()) ** 2)
            s_r2_real = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            print(f"  s(연속값): R^2={s_r2_real:.3f}, MAE={np.mean(np.abs(r_s_phys - real_s_arr)):.2f}mm")
            # 2026-09-07 추가: s>=60mm(팁 근처, 23번 스윕 대상 구간) 따로 확인 - 전체
            # MAE만 보면 이 구간의 개선/악화가 다른 구간에 묻혀서 안 보일 수 있음.
            tip_mask = real_s_arr >= 60.0
            if tip_mask.sum() >= 3:
                tip_mae = np.mean(np.abs(r_s_phys[tip_mask] - real_s_arr[tip_mask]))
                print(f"    s>=60mm(팁 근처, n={int(tip_mask.sum())}) MAE={tip_mae:.2f}mm "
                      f"(재튜닝 전 기준값 9.49mm와 비교할 것)")

            for i, name in enumerate(force_names):
                ss_res = np.sum((r_force_phys[:, i] - real_f_arr[:, i]) ** 2)
                ss_tot = np.sum((real_f_arr[:, i] - real_f_arr[:, i].mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                print(f"  {name}: R^2={r2:.3f}, MAE={np.mean(np.abs(r_force_phys[:, i] - real_f_arr[:, i]))*1000:.4f}mN")
                # Fx_board(=force_names[0])는 |phi|>=90 vs |phi|<90 구간을 나눠서도 확인 -
                # 20/23번에서 계속 추적해온 구간(재튜닝 전 기준값 R^2=0.355).
                # 2026-09-14 발견(산점도 diff 확인 중): R^2만 보면 |phi|>=90이 훨씬 나빠
                # 보이는데, MAE(절대오차)로 보면 두 구간이 거의 동일함(예: 0.0047 vs
                # 0.0050mN) - |phi|>=90은 정답 값 자체의 분산이 작아서(실제값 표준편차
                # 0.014 vs 0.022mN) R^2 계산식(1-잔차/분산)이 구조적으로 불리하게 나오는
                # 통계적 함정일 가능성이 큼. **"고각도가 특별히 못 배운다"는 결론은 R^2만
                # 보고 오판한 것일 수 있어, 이후로는 R^2와 MAE를 항상 같이 보고 MAE로
                # 판단할 것.**
                if i == 0:
                    for label, mask in [("|phi|<90", np.abs(real_c_arr[:, 1]) < 90),
                                         ("|phi|>=90", np.abs(real_c_arr[:, 1]) >= 90)]:
                        if mask.sum() >= 3:
                            ss_res_m = np.sum((r_force_phys[mask, i] - real_f_arr[mask, i]) ** 2)
                            ss_tot_m = np.sum((real_f_arr[mask, i] - real_f_arr[mask, i].mean()) ** 2)
                            r2_m = 1 - ss_res_m / ss_tot_m if ss_tot_m > 0 else float("nan")
                            mae_m = np.mean(np.abs(r_force_phys[mask, i] - real_f_arr[mask, i])) * 1000
                            std_m = real_f_arr[mask, i].std() * 1000
                            print(f"    {label}(n={int(mask.sum())}) R^2={r2_m:.3f}, MAE={mae_m:.4f}mN "
                                  f"(실제값표준편차={std_m:.4f}mN, 기준값: R^2=0.355/0.720, MAE는 25/26번 로그 참고)")

            # 2026-09-21(36번): 형상(shape_head) - 실측 FEA 기준이 유일한 정직한 지표.
            # 합성-val은 "대체모델이 만든 데이터를 대체모델 기반 모델이 맞히는" 순환검증이라
            # 항상 좋게 나옴(8월에 크게 데인 부분). 28번 교훈대로 R^2/MAE/정답분산 병행.
            print(f"  --- 형상(shape_head) ---")
            for i, name in enumerate(SHAPE_NAMES):
                pred_i, true_i = r_shape_phys[:, i], real_sh_arr[:, i]
                ss_res_sh = np.sum((pred_i - true_i) ** 2)
                ss_tot_sh = np.sum((true_i - true_i.mean()) ** 2)
                r2_sh = 1 - ss_res_sh / ss_tot_sh if ss_tot_sh > 0 else float("nan")
                unit = "deg" if "theta" in name else "mm"
                print(f"    {name}: R^2={r2_sh:.3f}, MAE={np.mean(np.abs(pred_i - true_i)):.4f}{unit} "
                      f"(정답값표준편차={true_i.std():.4f}{unit})")

        evaluate_real("1단계(기존 파이프라인, 파인튜닝 전)")

        # ============================================================
        # 2026-09-07 신규 2단계: 실측 FEA(fit_rows, holdout 제외) 직접 파인튜닝.
        # 지금까지 최종 CNN은 150k 합성데이터로만 학습되고 실측은 검증에만 쓰였음 - 실측
        # 데이터가 gradient를 한 번도 직접 업데이트한 적이 없었다는 뜻. surrogate라는 병목을
        # 거치지 않고 실측을 직접 보여주는 "pretrain(합성) -> finetune(실측)" 2단계 시도.
        # |phi|>=90 실측 샘플은 오버샘플링해서 Fx_board 고각도 문제에 더 집중시킴.
        # (PROJECT_STATUS.md 20/23번 참고 - HIGH_PHI_WEIGHT 손실가중치 레버는 이미 실패했고,
        # 이건 배치 구성 자체를 바꾸는 다른 메커니즘이라 별도로 시도해볼 가치가 있음.)
        # ============================================================
        FINETUNE_ON_REAL = int(os.environ.get("FINETUNE_ON_REAL", 1))
        if FINETUNE_ON_REAL:
            ft_X, ft_y, ft_f, ft_s, ft_c, ft_sh = rows_to_arrays(fit_rows)
            if len(ft_X) < 20:
                print(f"\n2단계 파인튜닝 스킵: 실측 fit_rows 중 free-shape 계산 성공이 "
                      f"{len(ft_X)}개뿐(20개 미만)")
            else:
                ft_X = np.array(ft_X, dtype=np.float32)
                ft_X_norm = (ft_X - X_mean2) / X_std2
                ft_y_arr = np.array(ft_y)
                ft_f_arr = np.array(ft_f, dtype=np.float32)
                ft_f_norm = (ft_f_arr - f_mean) / f_std
                ft_s_arr = np.array(ft_s, dtype=np.float32)
                ft_s_norm = (ft_s_arr - s_mean) / s_std
                ft_c_arr = np.array(ft_c, dtype=np.float32)
                ft_c_norm = (ft_c_arr - c_mean) / c_std  # known L_M,phi 입력(정규화)
                ft_sh_norm = (np.array(ft_sh, dtype=np.float32) - sh_mean) / sh_std  # 36번
                ft_phi_weight = np.where(np.abs(ft_c_arr[:, 1]) >= 90, HIGH_PHI_WEIGHT, 1.0).astype(np.float32)
                ft_s_weight = s_spatial_weight(ft_s_arr).astype(np.float32)

                high_phi_mask = np.abs(ft_c_arr[:, 1]) >= 90
                FT_OVERSAMPLE = int(os.environ.get("FT_OVERSAMPLE", 4))
                base_idx = np.arange(len(ft_X))
                oversample_idx = np.concatenate(
                    [base_idx, np.repeat(base_idx[high_phi_mask], max(0, FT_OVERSAMPLE - 1))])
                print(f"\n2단계 파인튜닝 데이터: 실측 fit_rows {len(ft_X)}개 "
                      f"(|phi|>=90: {int(high_phi_mask.sum())}개, {FT_OVERSAMPLE}배 오버샘플링 후 "
                      f"배치풀 {len(oversample_idx)}개)")

                ft_tensors = [torch.tensor(ft_X_norm[:, None]).float(), torch.tensor(ft_y_arr).long(),
                              torch.tensor(ft_f_norm).float(), torch.tensor(ft_s_norm).float(),
                              torch.tensor(ft_c_norm).float(), torch.tensor(ft_phi_weight).float(),
                              torch.tensor(ft_s_weight).float(), torch.tensor(ft_sh_norm).float()]
                ft_dataset = TensorDataset(*[t[oversample_idx] for t in ft_tensors])
                FT_EPOCHS = int(os.environ.get("FT_EPOCHS", 15))
                FT_LR = float(os.environ.get("FT_LR", 5e-5))  # 기존 lr(1e-3)의 1/20 - 150k 합성
                # 데이터로 배운 일반 표현은 유지하고 미세보정만(오래 돌리면 fit_rows가
                # 400여개뿐이라 바로 과적합됨 - epoch도 짧게).
                ft_loader = DataLoader(ft_dataset, batch_size=16, shuffle=True)
                ft_optimizer = optim.Adam(model.parameters(), lr=FT_LR, weight_decay=1e-4)

                model.train()
                for ft_epoch in range(FT_EPOCHS):
                    for bx, by, bf, bs, bc, bw, bsw, bsh in ft_loader:
                        bx, by, bf, bs, bc, bw, bsw, bsh = (
                            bx.to(device), by.to(device), bf.to(device), bs.to(device),
                            bc.to(device), bw.to(device), bsw.to(device), bsh.to(device))
                        ft_optimizer.zero_grad()
                        seg_logits, force_pred, s_pred, shape_pred = model(bx, bc)
                        loss = (seg_criterion(seg_logits, by) + weighted_force_loss(force_pred, bf, bw)
                                + weighted_s_loss(s_pred, bs, bsw) + shape_criterion(shape_pred, bsh))
                        loss.backward()
                        ft_optimizer.step()
                print(f"2단계 파인튜닝 완료 ({FT_EPOCHS} epoch, lr={FT_LR})")
                evaluate_real("2단계(실측 파인튜닝 후, 신규)")
                print("\n※ 위 1단계 vs 2단계 수치를 직접 비교해서 파인튜닝이 실제로 도움이 "
                      "됐는지 판단할 것 - 특히 |phi|>=90 Fx_board R^2(기준값 0.355)와 "
                      "s>=60mm MAE(기준값 9.49mm). 도움 안 되면 FINETUNE_ON_REAL=0으로 꺼서 "
                      "1단계 체크포인트로 되돌릴 것.")
        else:
            print("\n2단계 파인튜닝 스킵됨 (FINETUNE_ON_REAL=0)")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "X_mean": X_mean2, "X_std": X_std2,
                "f_mean": f_mean, "f_std": f_std, "s_mean": s_mean, "s_std": s_std,
                # c_mean/c_std: 더 이상 예측 타겟 정규화용이 아니라 known input(L_M,phi)
                # 정규화용 - config_names 순서대로 (L_M_mm, phi_deg).
                "c_mean": c_mean, "c_std": c_std,
                # 36번: shape_head 출력을 물리 단위로 되돌리는 데 필요(진단 스크립트도 이걸 씀).
                "sh_mean": sh_mean, "sh_std": sh_std, "shape_names": SHAPE_NAMES,
                "bin_width_mm": BIN_WIDTH_MM, "n_classes": N_CLASSES, "phi_range": PHI_RANGE,
                "beta_values": BETA_VALUES, "force_names": force_names, "config_names": config_names},
               os.path.join(MODELS_DIR, "position_segment_classifier_singleprobe_beta0180_4seg.pth"))
    print(f"\n저장: {MODELS_DIR}/position_segment_classifier_singleprobe_beta0180_4seg.pth")
    print(f"\n총 소요시간: {(time.time()-t_start)/60:.1f}분")
