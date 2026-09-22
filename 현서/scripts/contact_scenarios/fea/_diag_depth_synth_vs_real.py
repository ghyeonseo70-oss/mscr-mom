"""2026-09-22(39번): depth_head가 깊이를 못 배운 원인이 "합성 데이터"인지 확인.

상황: 37번에서 **실측 FEA** delta-B로는 깊이를 70.0%로 맞힐 수 있음을 확인했는데(대조군
35.6%), 38번에서 depth_head를 붙여 학습시키니 실측 홀드아웃에서 44.9%에 그쳤고 예측값이
사실상 상수(0.12mm 근처)였음.

가설: CNN은 **합성** delta-B로 학습하는데, 합성은 대체모델이 예측한 변위로 자기장을
만들기 때문에, 대체모델이 깊이 의존성을 뭉개면 합성 데이터에는 깊이 정보가 실측보다
약하게 남음. 그러면 CNN이 배울 게 없는 게 당연함.

방법: 37번과 **똑같은 분류기(RandomForest)** 를 합성 데이터(학습 때 저장된 npz)에 적용.
- 합성에서도 잘 맞힌다 -> 데이터는 멀쩡, CNN 학습/구조 문제
- 합성에서 못 맞힌다   -> 합성 파이프라인이 깊이 정보를 잃고 있음(대체모델 문제)
"""
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                    "segment_bfield_singleprobe_beta0180_4seg.npz")

d = np.load(NPZ)
X, c, dp, s = d["X"], d["c"], d["dp"], d["s"]
print(f"합성 데이터 {len(dp)}개, 깊이 범위 {dp.min():.3f}~{dp.max():.3f}mm")

# 37번은 3클래스 분류였음. 합성은 연속값이라 같은 조건으로 맞추기 위해 3등분(동일 개수)으로
# 나눠 "얕음/중간/깊음"을 맞히게 함. 무작위 기준선은 똑같이 33.3%.
edges = np.quantile(dp, [1 / 3, 2 / 3])
y = np.digitize(dp, edges)
print(f"3등분 경계: {edges[0]:.3f}, {edges[1]:.3f}mm  (클래스별 개수 {np.bincount(y)})")

# 샘플 수가 많아 전부 쓰면 느림 - 무작위 부분표본으로 충분(경향만 보면 됨).
rng = np.random.default_rng(0)
sub = rng.choice(len(y), size=min(12000, len(y)), replace=False)
Xf = X[sub].reshape(len(sub), -1)
cf, yf, sf, dpf = c[sub], y[sub], s[sub], dp[sub]

# 같은 형상(L_M,phi)이 train/test에 걸치지 않게 그룹 분리(37번과 같은 취지).
groups = (np.round(cf[:, 0] / 10).astype(int) * 1000 + np.round(cf[:, 1] / 10).astype(int))

INPUTS = [
    ("(A) 전체 delta-B + known", np.hstack([Xf, cf])),
    ("(D) [대조군] known만", cf),
]
for label, Xin in INPUTS:
    preds = np.zeros_like(yf)
    for tr, te in GroupKFold(n_splits=4).split(Xin, yf, groups):
        clf = RandomForestClassifier(n_estimators=200, random_state=0,
                                      class_weight="balanced", n_jobs=-1)
        clf.fit(Xin[tr], yf[tr])
        preds[te] = clf.predict(Xin[te])
    print(f"{label:28s} balanced acc = {balanced_accuracy_score(yf, preds)*100:5.1f}%")

print("\n[비교] 37번 실측 FEA 기준: (A) 70.0%, 대조군 35.6%, 무작위 33.3%")
print("합성에서도 (A)가 높으면 데이터는 멀쩡 -> CNN 학습/구조 문제.")
print("합성에서 낮으면 대체모델이 깊이 의존성을 뭉개고 있다는 뜻.")

# 보조 확인: 합성 데이터에서 '자기장 크기'와 깊이의 관계가 실제로 남아 있는지
mag = np.linalg.norm(Xf, axis=1)
for lo, hi, lab in [(0, 1 / 3, "얕음"), (1 / 3, 2 / 3, "중간"), (2 / 3, 1, "깊음")]:
    m = (yf == {"얕음": 0, "중간": 1, "깊음": 2}[lab])
    print(f"  {lab}(깊이 {dpf[m].mean():.3f}mm): 자기장 크기 중앙값 {np.median(mag[m]):.4f}")
