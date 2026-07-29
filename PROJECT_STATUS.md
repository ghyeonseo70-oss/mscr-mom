# 프로젝트 현황 (컴퓨터 이사용 인수인계)

이 문서는 다른 컴�터에서 새로 시작하는 Claude에게 상황을 알려주기 위한 것입니다.
**새 컴퓨터에서 클로드에게 할 말**: "PROJECT_STATUS.md 읽고 여기서부터 이어서 진행해줘"

## 프로젝트 개요

MSCR-MOM(자기 연성 카테터 로봇) 논문(Park et al. 2024) 재현 + 홀센서로 "충돌(접촉) 감지"
기능 확장 연구. 울산대 김현서 학생, 전기전자공학전공.

- `scripts/force_model/` — 논문 재현 물리모델 (K1,K2,M1 등)
- `scripts/cnn_model/` — 홀센서→자석 위치추정 CNN (기존 성과, 잘 됨)
- `scripts/contact_scenarios/` — 접촉(충돌) 감지 연구 (지금 진행 중인 것)
- `scripts/contact_scenarios/fea/` — CalculiX+Gmsh FEA 검증

## 지금까지 확인된 것 (요약)

1. **이진 감지(접촉 유무)**: 93.8% 정확도로 성공 (`train_binary_detection.py`)
2. **위치·힘 정밀 추정**: 단일 관측으론 실패 → 능동탐색(phi 여러 각도)으로 개선
   → 2단계 NN(위치추정+L_M → 접촉정보) 방식이 가장 좋음 (이상적 상한선 R²=0.87)
3. **면접촉(폭) 추정**: 실패로 결론남 (자석 위치에 폭 정보가 거의 안 남음, 상관계수 0.05 미만)

## ⚠️ 현재 가장 중요한 미해결 문제

`force_model.py`의 **접촉힘(외력) 계산이 phi≠0(이미 자기장으로 휘어있는 상태)일 때 심각한
오류가 있음을 발견함**:
- 아주 작은 힘(마이크로뉴턴 단위)에서도 "미는 방향"과 "실제 변위 방향"이 반대로 나옴
- 단순 부호 버그는 아님(phi=0 일직선 기준에서는 정상 작동 확인함)
- `scripts/force_model/find_safe_force_range.py`로 확인한 결과, "안전한" 힘의 범위가
  극도로 좁음(대부분 조합에서 0.5mN도 이미 위험)
- 원인 후보: K1,K2가 너무 물러서 접촉 시 좌굴(buckling)에 가까운 불안정 영역에 있고,
  수치해법(슈팅법)이 불안정한 해로 수렴하는 것으로 추정 (미확정, 더 조사 필요)
- **그래서 지금까지 만든 "접촉힘 포함" 학습데이터(특히 Fx,Fy 방향)는 신뢰도가 낮음**
  (s(위치), F_mag(크기)는 상대적으로 덜 영향받았을 수 있음)

## 다음 할 일 (이 컴퓨터로 옮긴 이유)

빠른 물리모델 대신 **CalculiX FEA로 소규모라도 신뢰할 수 있는 접촉 데이터를 직접 생성**하기로
함 (FEA는 이 문제가 없음을 이미 검증함 - `plot_confusable_pair_fea.py` 결과 참고).

계획:
- L_M=50mm, phi=60도로 고정 (이미 메쉬 생성 코드 있음: `get_bent_centerline.py`,
  `make_bent_tube_mesh.py`, `make_bent_contact_scene.py`)
- 접촉위치(s) 9곳 × 깊이 10단계 = 90케이스 정도로 촘촘하게 스윕
- `sweep_bent_contact.py`를 확장해서 사용 (이미 `run_contact.py`에 `print_tip=True`로
  팁 변위까지 뽑는 기능 추가되어 있음)
- `run_contact.py`의 `run_ccx()`가 `OMP_NUM_THREADS=4`로 멀티스레드 쓰도록 이미 개선됨
  (더 좋은 컴퓨터면 이 값을 코어 수에 맞게 올리는 것도 검토)

## 환경 설정 (이 컴퓨터에서 처음 할 것)

```powershell
pip install -r requirements.txt
# Miniconda 설치 후:
conda install -c conda-forge calculix
# Gmsh는 pip로 이미 설치됨 (requirements.txt에 포함)
```

CalculiX 실행은 `~/miniconda3` 경로를 가정하므로, Miniconda를 기본 위치에 설치할 것.

## 참고: 데이터/모델 파일

`data/`, `models/` 폴더에 지금까지 만든 학습데이터·모델 다 들어있음 (git에 포함, `.frd` 등
CalculiX 원본 결과파일만 용량이 커서 제외했음 - 필요하면 다시 계산 가능).
