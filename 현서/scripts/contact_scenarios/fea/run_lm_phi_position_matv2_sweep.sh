#!/bin/bash
# 2026-08-18 신규: K1/K2/Br 상수 변경(E=850kPa 직접계산) + material_convert.py 수정 반영한
# 새 재료값으로, beta=0/180 두 경우에 대해서만 LM x phi x s 스윕을 다시 돈다(옛 664개 전체
# 재계산 대신, 지금 모델이 실제로 쓰는 beta=0/180만 최소 규모로).
#
# beta=0: LM(4: 0,25,50,75mm) x phi(9: 0,+-30,+-90,+-120,+-150deg) x s(10: 10~100mm)
#         = 360케이스 - 옛 스윕(run_lm_phi_position_sweep.sh)과 동일한 격자 밀도 그대로 유지
#         (이 밀도가 91% 성능을 낸 격자라 더 줄이면 보간 품질이 떨어질 위험).
# beta=180: LM(3: 0,50,75mm) x phi(3: -90,0,90deg) x s(10) = 90케이스 - 전수가 아니라
#         "beta=0 결과를 부호만 뒤집으면 beta=180과 같다"는 옛 재료값 기준 확인된 대칭성
#         (fea_beta_fine_resolution.json)이 새 재료값에서도 유지되는지 검증할 표본만.
#         이 90개가 beta=0의 대응 부분집합과 부호반전 일치하면 나머지 beta=180은 추가로
#         돌릴 필요 없음(대칭성으로 대체) - 안 맞으면 beta=180도 360개 전부 돌려야 함.
# 총 450케이스 (옛 664개 대비 -32%, 대칭성 확인되면 사실상 beta=0의 360개가 전부).
set -u
cd "$(dirname "$0")"
date +%s > lm_phi_pos_matv2_pipeline_start_time.txt

python3 - <<'PYEOF' > lm_phi_pos_matv2_combos.txt
LM_LIST_FULL = [0.0, 25.0, 50.0, 75.0]
PHI_LIST_FULL = [0.0, 30.0, -30.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0]
LM_LIST_SPOT = [0.0, 50.0, 75.0]
PHI_LIST_SPOT = [0.0, 90.0, -90.0]

def tag_of(lm, phi, beta):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"LM{int(lm)}_phi{sign}{int(abs(phi))}_b{int(beta)}"

for lm in LM_LIST_FULL:
    for phi in PHI_LIST_FULL:
        print(f"{lm} {phi} 0 {tag_of(lm, phi, 0)}")
for lm in LM_LIST_SPOT:
    for phi in PHI_LIST_SPOT:
        print(f"{lm} {phi} 180 {tag_of(lm, phi, 180)}")
PYEOF

N_COMBOS=$(wc -l < lm_phi_pos_matv2_combos.txt)
echo "=== LM x phi x beta x s 스윕 시작 (${N_COMBOS}조합, 동시 9개, 조합당 10케이스 = 총 $((N_COMBOS*10))케이스) ==="
cat lm_phi_pos_matv2_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$3" --threads 4 > "sweep_matv2_$3.log" 2>&1'

echo "=== 스윕 완료, 병합 ==="
python3 -c "
import json, glob
rows = []
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_*.json')):
    rows.extend(json.load(open(f)))
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'LM x phi x beta x s 스윕 병합 완료: {len(rows)}/$((N_COMBOS*10)) ->', out)
"
echo "=== DONE ==="
