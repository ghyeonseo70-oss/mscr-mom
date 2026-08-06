# 프로젝트 현황 (인수인계, 2026-08-06 업데이트)

이 문서는 다른 컴퓨터/세션에서 새로 시작하는 Claude에게 상황을 알려주기 위한 것입니다.
**새 세션에서 클로드에게 할 말**: "PROJECT_STATUS.md 읽고 여기서부터 이어서 진행해줘"

## 프로젝트 개요

MSCR-MOM(자기 연성 카테터 로봇) 논문(Park et al. 2024) 재현 + 홀센서로 "충돌(접촉) 감지"
기능 확장 연구. 울산대 김현서 학생, 전기전자공학전공.

**⚠️ 중요: 파일 위치가 바뀌었습니다.** 예전 경로(`scripts/...`, `data/...`, `models/...`)는
전부 저장소 루트 밑 **`현서/` 폴더 안으로 옮겨졌습니다** (예: `현서/scripts/force_model/`).
근데 이 이동이 `git mv`로 반영되지 않아서 `git status`에 아직도 대량의 "삭제됨/추적안함"
변경사항이 잡혀 있습니다(약 500줄+) — **아직 해결 안 된 상태**로 남겨뒀습니다. 정리하려면
`git add`로 이동을 반영하거나 `.gitignore`를 점검한 뒤 커밋하면 됩니다.

- `현서/scripts/force_model/` — 논문 재현 물리모델 (K1,K2,M1 등)
- `현서/scripts/cnn_model/` — 홀센서→자석 위치추정 CNN (기존 성과, 잘 됨)
- `현서/scripts/contact_scenarios/` — 접촉(충돌) 감지 연구
- `현서/scripts/contact_scenarios/fea/` — CalculiX+Gmsh FEA 검증 (이번 세션의 주 작업 위치)

## 지금까지 확인된 것 (요약)

1. **이진 감지(접촉 유무)**: 93.8% 정확도로 성공 (`train_binary_detection.py`)
2. **위치·힘 정밀 추정(분석모델 기반, 예전 작업)**: 단일 관측으론 실패 → 능동탐색(phi 여러
   각도)으로 개선 → 2단계 NN 방식이 가장 좋음 (이상적 상한선 R²=0.87)
3. **면접촉(폭) 추정**: 실패로 결론남 (자석 위치에 폭 정보가 거의 안 남음)
4. **`force_model.py`의 접촉힘 계산 버그**(phi≠0일 때 불안정) → FEA로 우회하기로 결정,
   이번 세션에서 FEA 기반 파이프라인으로 상당히 진전시킴 (아래 참고)
5. **[신규] 구간(segment) 인지**: 교수님이 "정확한 위치 대신 대략 어느 구간에 접촉했는지만
   알아도 된다"고 하셔서 시도 → **최종 balanced accuracy 83.7%**까지 달성 (아래 상세)

## 이번 세션(2026-08-05~06)에서 한 일

### 1) 새 FEA 스윕: `현서/scripts/contact_scenarios/fea/sweep_lm_phi_position_worker.py`
L_M∈{0,25,50,75}mm × phi∈11개(0,±30,±60,±90,±120,±150°) × s∈{10~100mm, 10구간} 그리드로
CalculiX 접촉해석을 새로 돌림. beta(원주각)=0°, push_depth=0.10mm로 고정. 케이스 끝날 때마다
원본 CalculiX 산출물(.frd/.dat 등, 케이스당 ~700MB)을 자동삭제해서 디스크 안 불어나게 함.
결과: `현서/data/contact_scenarios/fea/fea_lm_phi_pos_sweep_all.json` (195/360 성공).

### 2) 구간분류(segment classification) 실험 시리즈
목표: 홀센서 5×5 자기장 변화(B_load-B_free)만 보고 카테터를 5구간(20mm씩)으로 나눴을 때
어디에 접촉했는지 분류. 대체모델(surrogate MLP, FEA 608개로 학습) → 15만개 합성 데이터
(magpylib) → CNN 분류기, 순서로 파이프라인 구성(`master_pipeline.py` 방식 재사용).

| 단계 | balanced accuracy | 비고 |
|---|---|---|
| 단일 관측 | 20.5% | 무작위 수준 |
| 4프로브 능동탐색 (phi 스캔) | 24.8% | `test_active_sensing.py` 아이디어 재사용 |
| + push_depth 고정 | 23.5% | 효과 없음(가설 기각) |
| + 대체모델 앙상블(정확도 개선) | 26.3% | |
| 11프로브(전체 phi) + 앙상블 | 28.7% | |
| **+ 센서 노이즈 제거** | **81.6%** | **핵심 원인이었음** (아래 참고) |
| + 힘(Fx,Fy) 동시 예측(멀티태스크) | 83.7% | 오히려 소폭 상승 |

**결정적 원인**: `magpy.getB()` 합성 데이터에 "센서 측정오차 시뮬레이션"으로 5%(전체 자기장
크기 기준) 무작위 노이즈를 넣고 있었는데, 이게 접촉으로 인한 실제 신호보다 훨씬 커서 신호를
거의 다 덮어버리고 있었음. 학생이 "어차피 실제 하드웨어에서 FFT로 노이즈 필터링할 거라 노이즈
없는 게 맞다"고 확인해줘서 제거 → 정확도 28.7%→81.6%로 급상승.
**→ 앞으로 이 종류의 합성 B-field 데이터를 만들 때 인위적 노이즈를 넣지 말 것.**

최종 스크립트: `현서/scripts/contact_scenarios/fea/train_segment_classifier_multiprobe.py`
최종 모델: `현서/models/position_segment_classifier_multiprobe_150k_11probe_force_v2.pth`
(멀티태스크: 구간분류 5-class + Fx,Fy 회귀 2개. F_mag은 별도 예측 안 하고 예측된 Fx,Fy로부터
sqrt(Fx²+Fy²)로 유도 — 대체모델이 F_mag을 직접 예측하면 "항상 0 이상"이라는 물리적 제약이
없어서 22%가 마이너스로 나오는 문제가 있었음, 유도 방식으로 해결)

**실제 FEA(대체모델 근사 없는 순수 실측)로도 검증함**: 11개 phi 전부 실제 계산된 (L_M,s)
조합 3개에 대해 3/3 정답 (`backtrack_real_fea_test.py`).

**한계**: 힘 크기(F_mag) 회귀는 상대적으로 약함, 방향각 오차도 평균 30도대. beta(원주방향
접촉각)가 무작위로 섞여있는데 이게 남은 주요 교란변수일 가능성이 있음(미검증).

### 3) 진행 중이던 마지막 실행
세션 종료 시점에 `train_segment_classifier_multiprobe.py`(힘 예측 v2, F_mag 유도 방식)가
백그라운드로 돌고 있었을 수 있음 — `현서/scripts/contact_scenarios/fea/train_segment_classifier_multiprobe_11probe_force_v2.log` 파일에서 완료 여부/결과 확인.

### 4) 미해결/남은 일
- **git 정리 안 됨** (위 참고, 500줄+ 변경사항 pending)
- **디스크**: `현서/scripts/contact_scenarios/fea/` 폴더에 예전 스윕들의 CalculiX 원본산출물이
  여전히 233GB 차지 중 (재계산 가능, `.gitignore`에도 제외돼 있음 — 필요시 삭제 가능)
- beta(원주각) confound 여부 미검증
- balanced accuracy 83.7%가 "실용적으로 충분한지"는 교수님 판단 필요

## 세션 간 기억(memory) 시스템

`/home/user/.claude-hyunseo/projects/-home-user-mscr-mom/memory/`에 이 프로젝트의 지속 메모리가
있음 (Claude Code의 memory 기능). 특히:
- `feedback_no_sensor_noise.md` — 합성 B-field 데이터에 노이즈 넣지 말 것 (위 참고)
- `project_segment_classification_investigation.md` — 구간분류 실험 전체 기록

## 환경 설정

```powershell
pip install -r requirements.txt
# Miniconda 설치 후:
conda install -c conda-forge calculix
```
CalculiX 실행은 `~/miniconda3` 경로를 가정. GPU(CUDA) 사용 가능하면 최종 CNN 학습이 몇 분 내로
끝남(이번 세션 컴퓨터는 64코어+CUDA 있었음, 15만개 합성+학습에 총 ~110분/회).
