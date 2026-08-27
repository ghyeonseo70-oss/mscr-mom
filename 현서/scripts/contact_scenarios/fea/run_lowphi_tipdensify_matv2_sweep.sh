#!/bin/bash
# 2026-08-27 신규: _diag_s_breakdown.py로 확인한 "s=60-100mm(팁 근처) 오차가 유독 큼"
# 문제는 |phi|>=90 문제와 무관하고(저-phi에서도 MAE 거의 동일) phi 전 구간에 걸쳐 있음.
# 그런데 phi=90~150 구간은 이미 run_highphi_sdensify_matv2_sweep.sh로 s=15,25,...,95
# 오프셋 격자까지 조밀화했지만, 나머지 phi(0,+-30,+-60)는 원래 s=10,20,...,100(10mm 간격)
# 격자만 있어서 60-100mm 구간에 상대적으로 데이터가 성김 - 이 저-phi 구간의 s=60-100mm
# 근방을 오프셋 격자(65,75,85,95)로 조밀화해서 채우는 스윕.
#
# 대상: phi in {0,30,-30,60,-60} x L_M in {0,12.5,25,37.5,50,62.5,75,87.5} (bad-branch
# 조합 없음, run_lm_densify_matv2_sweep.sh의 BAD_COMBOS는 전부 phi=+-120/+-150이라 무관)
# = 5x8 = 40조합 x s(4, 60-100mm 구간만) = 160케이스, beta=0만.
set -u
cd "$(dirname "$0")"
date +%s > lowphi_tipdensify_matv2_pipeline_start_time.txt

python3 - <<'PYEOF' > lowphi_tipdensify_matv2_combos.txt
LM_LIST = [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]
PHI_LIST = [0.0, 30.0, -30.0, 60.0, -60.0]

def tag_of(lm, phi):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"LPT{int(lm*10)}_phi{sign}{int(abs(phi))}_b0"

for lm in LM_LIST:
    for phi in PHI_LIST:
        print(f"{lm} {phi} 0 {tag_of(lm, phi)}")
PYEOF

N_COMBOS=$(wc -l < lowphi_tipdensify_matv2_combos.txt)
echo "=== 저-phi s=60-100mm 팁근처 조밀화 스윕 시작 (${N_COMBOS}조합, 동시 9개, 조합당 s4개 = 총 $((N_COMBOS*4))케이스, STABILIZE 기본 적용) ==="
cat lowphi_tipdensify_matv2_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$3" --threads 4 --s_list "65,75,85,95" > "sweep_lowphitipdensify_$3.log" 2>&1'

echo "=== 스윕 완료, 기존 fea_lm_phi_pos_matv2_all.json에 병합 ==="
python3 -c "
import json, glob
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8'))
before = len(rows)
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_LPT*.json')):
    rows.extend(json.load(open(f, encoding='utf-8')))
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'저-phi 팁근처 조밀화 병합 완료: {before}개 -> {len(rows)}개 (+{len(rows)-before}, 시도 $((N_COMBOS*4))개 중) ->', out)
"
echo "=== DONE ==="
