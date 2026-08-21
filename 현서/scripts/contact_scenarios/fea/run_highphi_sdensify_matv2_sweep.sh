#!/bin/bash
# 2026-08-21 신규: phi=90~150 구간(±90/±120/±150) FEA 성공률이 낮은 채로 남아있어서
# (30~40%대, L_M 무관 - 물리적 토크제로 특이점 때문이지 메쉬 품질 문제가 아님, 메쉬 세밀화는
# 이미 실패 확인됨) 실패율 자체를 못 고치는 대신 시도 횟수를 늘려서 절대 성공 개수를 늘리는
# 전략. 기존 s 격자(10,20,...,100mm)는 이미 다 시도했으니, 그 사이 오프셋 격자
# (15,25,...,95mm, 9개)를 새로 시도 - 겹치지 않는 완전히 새로운 샘플이라 낭비 없음.
#
# 대상: phi in {90,-90,120,-120,150,-150} x L_M in {0,12.5,25,37.5,50,62.5,75,87.5}
# 단, 해석모델이 오답 branch를 계산하는 6조합(62.5mm의 ±150, 87.5mm의 ±120/±150)은 제외
# (run_lm_densify_matv2_sweep.sh와 동일한 이유 - PROJECT_STATUS.md 참고).
# = 8x6-6 = 42조합 x s(9) = 378케이스, beta=0만.
set -u
cd "$(dirname "$0")"
date +%s > highphi_sdensify_matv2_pipeline_start_time.txt

python3 - <<'PYEOF' > highphi_sdensify_matv2_combos.txt
LM_LIST = [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]
PHI_LIST = [90.0, -90.0, 120.0, -120.0, 150.0, -150.0]
BAD_COMBOS = {(62.5, 150.0), (62.5, -150.0), (87.5, 120.0), (87.5, -120.0), (87.5, 150.0), (87.5, -150.0)}

def tag_of(lm, phi):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"PHS{int(lm*10)}_phi{sign}{int(abs(phi))}_b0"

for lm in LM_LIST:
    for phi in PHI_LIST:
        if (lm, phi) in BAD_COMBOS:
            continue
        print(f"{lm} {phi} 0 {tag_of(lm, phi)}")
PYEOF

N_COMBOS=$(wc -l < highphi_sdensify_matv2_combos.txt)
echo "=== phi=90~150 s격자 조밀화 스윕 시작 (${N_COMBOS}조합, 동시 9개, 조합당 s9개 = 총 $((N_COMBOS*9))케이스, STABILIZE 기본 적용) ==="
cat highphi_sdensify_matv2_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$3" --threads 4 --s_list "15,25,35,45,55,65,75,85,95" > "sweep_highphisdensify_$3.log" 2>&1'

echo "=== 스윕 완료, 기존 fea_lm_phi_pos_matv2_all.json에 병합 ==="
python3 -c "
import json, glob
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8'))
before = len(rows)
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_PHS*.json')):
    rows.extend(json.load(open(f, encoding='utf-8')))
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'phi=90~150 s조밀화 병합 완료: {before}개 -> {len(rows)}개 (+{len(rows)-before}, 시도 $((N_COMBOS*9))개 중) ->', out)
"
echo "=== DONE ==="
