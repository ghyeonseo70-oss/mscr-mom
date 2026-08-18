# 프로젝트 현황 (인수인계, 2026-08-18 최종 업데이트)

이 문서는 다른 컴퓨터/세션에서 새로 시작하는 Claude에게 상황을 알려주기 위한 것입니다.
**새 세션에서 클로드에게 할 말**: "PROJECT_STATUS.md 읽고 여기서부터 이어서 진행해줘"

## ⚠️ 2026-08-18 세션 업데이트 — 지금 당장 할 일 (이 컴퓨터, CalculiX 있음)

이전 세션(다른 컴퓨터)에서 교수님이 주신 MATLAB 코드(`현서/scripts/contact_scenarios/code (2).txt`)를
분석하다가 물리 상수를 바꿨고, 그 결과 **기존 FEA 664개가 새 상수와 안 맞게 됨**. 이걸 최소
규모로 바로잡는 게 이번 세션의 할 일.

### 1) 무엇이 바뀌었나
- `scripts/force_model/force_model.py`: `BR`(자석 잔류자속밀도) `0.36T → 0.4T`, K1/K2(굽힘강성)를
  논문 그림 디지타이징 피팅 대신 **교수님 MATLAB 코드 방식**(영률 E=850kPa 직접 대입,
  K2=E·I, K1=2·K2)으로 교체. 새 값: K1=1.2517e-06, K2=6.2586e-07 N·m² (옛 값 K1≈2.25~3.82e-6,
  K2≈8.53~9.89e-7 대비 더 유연함).
- `현서/scripts/contact_scenarios/fea/material_convert.py`: 이 파일이 FEA의 Neo-Hookean
  재료값(C10, D1)을 만드는 곳인데, **옛 K2(9.8872e-07)를 하드코딩**하고 있어서 force_model.py의
  새 K2와 어긋나 있었음(자유단 형상은 새 K2=더 유연, 접촉힘 계산은 옛 K2=더 뻣뻣한 재료 —
  물리적으로 모순). → **오늘 수정 완료**: 이제 force_model.py와 똑같은 `E=850kPa`를 직접 쓰도록
  고쳐서(K2를 거꾸로 역산하지 않음) 두 파일이 항상 같은 재료값을 공유하게 만듦. `run_contact.py`
  등이 `material_convert.py`를 매 실행시 새로 import하므로 **이 수정만으로 앞으로 돌리는 모든
  FEA는 자동으로 새 재료값을 씀** — 별도 조치 불필요.
- 문제: **기존 FEA 664개**(`fea_lm_phi_pos_sweep_all.json` 등 4개 파일, 대체모델/최종모델의
  학습 소스)는 전부 옛 재료값으로 이미 계산이 끝난 상태라 재사용 불가. 전체 664개를 다시 돌리는
  건 과함("600개는 에바" — 사용자 판단) — **지금 모델이 실제로 쓰는 건 beta=0°/180°뿐**이므로
  그것만 최소 규모로 다시 돈다.

### 2) 새로 만든 스윕 (beta=0/180 전용, 총 450케이스 — 옛 664개 대비 -32%)
`현서/scripts/contact_scenarios/fea/`에 3개 파일 추가:
- `sweep_lm_phi_position_matv2_worker.py` — 기존 `sweep_lm_phi_position_worker.py`와 동일한
  구조(조합 하나당 s 10곳), beta를 인자로 받도록만 바꿈. `material_convert.py`가 이미 새 E를
  쓰므로 이 워커는 그대로 실행하면 자동으로 새 재료값 사용.
- `run_lm_phi_position_matv2_sweep.sh` — 실행 스크립트. **beta=0**: L_M(4: 0,25,50,75mm) ×
  phi(9: 0,±30,±90,±120,±150°) × s(10) = 360케이스 (옛 스윕과 동일 격자밀도 그대로 — 이
  밀도가 91% 성능을 낸 격자라 더 줄이면 위험). **beta=180**: L_M(3: 0,50,75) × phi(3: 0,±90) ×
  s(10) = 90케이스만 — 전수 대신 "beta=0의 부호반전 = beta=180"이라는, 옛 재료값 데이터
  (`fea_beta_fine_resolution.json`)에서 이미 확인된 대칭성이 새 재료값에서도 유지되는지
  확인할 표본만.
  ```bash
  cd 현서/scripts/contact_scenarios/fea
  bash run_lm_phi_position_matv2_sweep.sh   # 결과: ../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json
  ```
- `check_beta180_symmetry.py` — 스윕 끝나면 실행. beta=180 90개 vs 대응 beta=0 케이스의
  부호반전 비교, F_mag 상대오차 중앙값 5% 이내면 "대칭성 유지 → beta=180 추가 스윕 불필요
  (beta=0 데이터를 부호반전해서 그대로 씀)", 아니면 "beta=180도 360개 전부 필요"라고 판정해줌
  (스크립트 안에 전체 스윕으로 확장하는 방법 설명 있음).
  ```bash
  python check_beta180_symmetry.py
  ```

### 3) 스윕 끝난 뒤 할 일
1. `check_beta180_symmetry.py` 판정 확인 (대칭성 유지되면 그대로, 아니면 beta=180 확장 필요).
2. `현서/scripts/contact_scenarios/fea/train_segment_classifier_singleprobe_beta0180_4seg.py`의
   `SOURCES` 리스트를 새 파일(`fea_lm_phi_pos_matv2_all.json`, 필요시 대칭성 보강분 포함)로
   바꾸고, 대체모델(surrogate) 재학습 → 15만개 합성데이터 재생성 → CNN 재학습(150k, GPU
   ~110분). 이전엔 새 K1/K2/Br 상수만 반영한 스모크테스트(500샘플/3에폭)까지만 하고, 이
   material_convert.py 불일치 문제 때문에 전체 재학습은 보류했었음 — 이제 이 스윕+수정이
   끝나면 바로 전체 재학습 진행하면 됨.
3. 재학습 후 성능이 기존(91.0% seg / R²=0.938 s, 옛 상수 기준)과 크게 다르면 — 상수가
   바뀌었으니 절대 수치는 달라질 수 있음, 큰 폭 저하가 아니면 정상.

## 프로젝트 개요

MSCR-MOM(자기 연성 카테터 로봇) 논문(Park et al. 2024) 재현 + 홀센서로 "충돌(접촉) 감지"
기능 확장 연구. 울산대 김현서 학생, 전기전자공학전공.

**⚠️ 파일 위치**: 예전 경로(`scripts/...`, `data/...`, `models/...`)는 저장소 루트 밑
**`현서/` 폴더 안**에 있습니다 (예: `현서/scripts/force_model/`). 이 이동이 `git mv`로
반영되지 않아서 `git status`에 대량의 "삭제됨/추적안함" 변경사항이 있는데, 원래 있던 파일
이동분은 그대로 두고(별도 정리 필요) 이번 세션에서 새로 만든 파일들만 골라서 커밋함.

- `현서/scripts/force_model/` — 논문 재현 물리모델 (K1,K2,M1 등)
- `현서/scripts/cnn_model/` — 홀센서→자석 위치추정 CNN (기존 성과, 잘 됨)
- `현서/scripts/contact_scenarios/` — 접촉(충돌) 감지 연구
- `현서/scripts/contact_scenarios/fea/` — CalculiX+Gmsh FEA 검증 (이번 세션의 주 작업 위치)

## 지금까지 확인된 것 (요약)

1. **이진 감지(접촉 유무)**: 93.8% 정확도로 성공 (`train_binary_detection.py`)
2. **위치·힘 정밀 추정(분석모델 기반, 예전 작업)**: 단일 관측으론 실패 → 능동탐색(phi 여러
   각도)으로 개선 → 2단계 NN 방식이 가장 좋음 (이상적 상한선 R²=0.87, **이번 세션에서 이 값도
   넘어섬**, 아래 참고)
3. **면접촉(폭) 추정**: 실패로 결론남 (자석 위치에 폭 정보가 거의 안 남음)
4. **`force_model.py`의 접촉힘 계산 버그**(phi≠0일 때 불안정) → FEA로 우회
5. **[신규] 구간(segment) 인지 + 정확한 위치(s) + 힘(Fx,Fy) 추정**: 이번 세션에서 전부
   실용적인 수준까지 끌어올림 — 아래 상세

## 이번 세션(2026-08-05~11)에서 한 일 — 최종 결과

### 핵심 파이프라인
1. `sweep_lm_phi_position_worker.py` — L_M(4)×phi(11: 0,±30,±60,±90,±120,±150°)×s(10구간)
   CalculiX 스윕. 케이스별 원본 산출물 자동삭제(디스크 안 불어남). 결과:
   `fea_lm_phi_pos_sweep_all.json`
2. `train_segment_classifier_multiprobe_auxreg.py` — **최종/최고 성능 스크립트**. FEA 608개
   → 대체모델(surrogate MLP, 10-앙상블) → 15만개 합성 B-field(magpylib, 11프로브 능동탐색)
   → CNN 멀티태스크(구간분류 + 연속값 s 회귀 + Fx,Fy 회귀) 학습.

### 최종 성능 (검증 15,000개 기준)
| 항목 | 결과 |
|---|---|
| 구간(5개, 20mm씩) 분류 balanced accuracy | **86.0%** (무작위 20%) |
| 연속값 s(정확한 위치, mm) 회귀 | **R²=0.954, MAE=4.37mm** |
| Fx (보드좌표) | R²=0.90 |
| Fy (보드좌표) | R²=0.88 |
| F_mag(힘 크기, Fx·Fy로부터 유도) | R²=0.92 |

**최종 모델**: `현서/models/position_segment_classifier_multiprobe_150k_11probe_auxreg.pth`
(구간 5-class + 연속값 s(1개) + Fx,Fy(2개) 동시 예측)

### 원인 규명 과정 (20.5% 무작위 → 86.0%)
| 단계 | balanced acc | 비고 |
|---|---|---|
| 단일 관측 | 20.5% | 무작위 수준 |
| 4프로브 능동탐색 | 24.8% | |
| +push_depth 고정 | 23.5% | 효과 없음(기각) |
| +대체모델 앙상블 | 26.3% | |
| 11프로브 전체스캔 | 28.7% | |
| **+센서 노이즈 제거** | **81.6%** | **결정적 원인** |
| +힘 멀티태스크 | 83.7~84.3% | |
| **+보조회귀(연속 s 동시학습)** | **86.0%** | 최종 |
| (참고) 더 큰 네트워크로만 재학습 | 83.1% | 오히려 악화, 기각 |
| (참고) beta 고정 | 84.1%(소표본) | 효과 미미, 기각 |

**결정적 원인**: 합성 데이터 생성 시 `magpy.getB()` 결과에 "센서 노이즈 시뮬레이션"으로
전체 자기장의 5% 무작위 노이즈를 넣고 있었는데, 이게 접촉으로 인한 실제 신호보다 훨씬 커서
신호를 거의 다 덮고 있었음. **실제 하드웨어는 FFT로 노이즈를 걸러내므로, 이런 합성 데이터는
노이즈 없이 만드는 게 오히려 더 정확한 시뮬레이션**(분류기가 받는 실제 입력과 유사) —
제거하자 28.7%→81.6%로 급상승. **→ 앞으로 이런 합성 B-field 데이터를 만들 때 인위적
노이즈를 넣지 말 것.**

**F_mag 버그 수정**: 대체모델이 F_mag을 직접 예측하면 "항상 0 이상"이라는 물리적 제약이 없어
22%가 마이너스로 나왔음 → Fx,Fy만 예측하고 F_mag=sqrt(Fx²+Fy²)로 유도하는 방식으로 해결
(R²0.54→0.92).

**보조회귀가 도움된 이유**: 오답의 98.4%(→최종모델은 99.9%)가 "바로 옆 구간"과의 혼동이라,
연속값 s도 같이 예측하게 하면 경계 근처 표현력이 보강됨 — 실제로 balanced accuracy 84.3%→
86.0%로 개선.

### 검증
- **실제 FEA(대체모델 근사 없는 순수 실측)로 검증**: `backtrack_real_fea_test.py` — 11개 phi
  전부 실제 계산된 (L_M,s) 조합 3개, 3/3 정답.
- 검증셋 5,000~15,000개 전수 산점도, 카테터 곡선 위 실제vs예측 시각화 다수 (아래 참고).

### 시각화 스크립트 (전부 `현서/scripts/contact_scenarios/fea/`, 결과는 `현서/data/contact_scenarios/fea/*.png`)
- `plot_concept_diagrams.py` → `concept_5segments.png`, `concept_beta.png`, `concept_force.png`
  (5구간/beta/힘 측정 개념도)
- `plot_validation_cases_and_scatter.py` → `validation_cases_3panel.png`,
  `scatter_force.png`, `scatter_segment.png` (실제vs예측 3사례 + 산점도)
- `plot_real_deformation_3panel.py` / `plot_real_deformation_with_prediction.py` →
  `real_deformation_3panel.png`, `real_deformation_with_prediction.png` (팁변위 기반 강체회전
  재구성으로 카테터 전체 변형형상 시각화, 실제FEA vs 모델예측 겹쳐그림)
- `plot_s_regression_6panel.py` → `s_regression_6panel.png` (연속값 s 예측, 6사례, phi=0 제외
  하고 다양한 L_M/phi로 선정. 화살표는 실측 팁변위(tip_ux,tip_uy) 방향과 반드시 일치하게 그릴 것
  — 처음에 "표면 법선" 같은 별개 값으로 그렸다가 화살표·변형모양이 안 맞아서 헤맸음)

### 안 써도 되는/실패한 시도 (참고용, 재시도 불필요)
- push_depth, beta를 고정해서 테스트 → 둘 다 효과 없었음(진짜 범인은 노이즈였음)
- FEA 재시도 시 INC(허용 증분수)를 늘려서 수렴 성공률 올리기 시도 → 실패시간만 늘어나고
  성공률 개선 안 됨(기각). s가 클수록(팁에 가까울수록) FEA 수렴 실패율이 원래 높음(91%→7%,
  지렛대 효과로 팁 근처는 물리적으로 신호가 원래 약함 - 버그 아님)
- CNN을 더 크게(32/64채널)+코사인LR스케줄로 재학습 → 83.1%로 오히려 악화(과적합 추정)

### 남은 일
- git 정리(파일 이동분) 안 됨 — 필요시 `git add`로 이동 반영
- 233GB 예전 CalculiX 원본산출물 여전히 디스크에 있음(재계산 가능, `.gitignore` 제외됨)
- **하드웨어 실측 검증이 다음 단계**: 지금까지는 전부 시뮬레이션(FEA+대체모델+magpylib) 기준.
  노이즈 제거 가정("FFT로 필터링할 것")도 실제로 확인 필요.
- 대용량 합성 데이터셋(`segment_bfield*.npz`, 케이스당 최대 476MB)은 git에 안 올림
  (`.gitignore` 추가함) — 필요시 학습 스크립트 재실행으로 재생성(15만개 기준 ~110분/GPU 학습
  포함).

## 세션 간 기억(memory) 시스템

`/home/user/.claude-hyunseo/projects/-home-user-mscr-mom/memory/`에 지속 메모리 있음. 특히:
- `feedback_no_sensor_noise.md` — 합성 B-field 데이터에 노이즈 넣지 말 것
- `project_segment_classification_investigation.md` — 구간분류 실험 전체 기록

## 환경 설정

```powershell
pip install -r requirements.txt
conda install -c conda-forge calculix
```
GPU(CUDA) 있으면 최종 CNN 학습이 몇 분 내로 끝남(이번 세션 컴퓨터: 64코어+CUDA, 15만개
합성+학습 총 ~110분/회).
