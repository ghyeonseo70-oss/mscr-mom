#!/bin/bash
# 2026-09-07 신규 (PROJECT_STATUS.md "지금 할 일" 23번): 20번에서 HIGH_PHI_WEIGHT 레버(3.0->1.5)가
# |phi|>=90 Fx_board R²를 못 고친다고 확인돼서 폐기, "phi=90~150 실측 FEA를 더 늘리는
# 원래 검증된 방법"으로 복귀. 22번(s 오차 세분화)에서 s=60-100mm(팁 근처) 구간이
# MAE=9.49mm로 다른 구간(2.9~5.6mm) 대비 훨씬 나쁘다고 확인됨 - 두 문제를 한 번에
# 겨냥: phi=90~150 x s=60-100mm 조합만 집중 조밀화.
#
# 기존 s 격자는 이미 5mm 간격으로 10~100mm 전부 시도됨(원 격자 10,20,...,100 +
# run_highphi_sdensify_matv2_sweep.sh의 오프셋 격자 15,25,...,95 = 합쳐서 5mm 간격
# 전체). 이번엔 그 사이 2.5mm 오프셋 격자(62.5,67.5,...,97.5)로 s=60-100mm 구간만
# 겹치지 않게 새로 시도 - 전체 s범위(10-100)를 또 도는 건 낭비이므로 팁 근처만 집중.
#
# 대상: phi in {90,-90,120,-120,150,-150} x L_M in {0,12.5,25,37.5,50,62.5,75,87.5}
# 단, 해석모델이 오답 branch를 계산하는 6조합(62.5mm의 ±150, 87.5mm의 ±120/±150)은 제외
# (run_lm_densify_matv2_sweep.sh/run_highphi_sdensify_matv2_sweep.sh와 동일한 이유).
# = 8x6-6 = 42조합 x s(8, 60-100mm 2.5mm오프셋) = 336케이스, beta=0만.
set -u
cd "$(dirname "$0")"
date +%s > highphi_stipdensify_matv2_pipeline_start_time.txt

python3 - <<'PYEOF' > highphi_stipdensify_matv2_combos.txt
LM_LIST = [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]
PHI_LIST = [90.0, -90.0, 120.0, -120.0, 150.0, -150.0]
BAD_COMBOS = {(62.5, 150.0), (62.5, -150.0), (87.5, 120.0), (87.5, -120.0), (87.5, 150.0), (87.5, -150.0)}

def tag_of(lm, phi):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"PST{int(lm*10)}_phi{sign}{int(abs(phi))}_b0"

for lm in LM_LIST:
    for phi in PHI_LIST:
        if (lm, phi) in BAD_COMBOS:
            continue
        print(f"{lm} {phi} 0 {tag_of(lm, phi)}")
PYEOF

N_COMBOS=$(wc -l < highphi_stipdensify_matv2_combos.txt)
echo "=== phi=90~150 x s=60-100mm 팁조밀화 스윕 시작 (${N_COMBOS}조합, 동시 9개, 조합당 s8개 = 총 $((N_COMBOS*8))케이스, STABILIZE 기본 적용) ==="
cat highphi_stipdensify_matv2_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$3" --threads 4 --s_list "62.5,67.5,72.5,77.5,82.5,87.5,92.5,97.5" > "sweep_highphistipdensify_$3.log" 2>&1'

echo "=== 스윕 완료, 기존 fea_lm_phi_pos_matv2_all.json에 병합 ==="
python3 -c "
import json, glob
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8'))
before = len(rows)
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_PST*.json')):
    rows.extend(json.load(open(f, encoding='utf-8')))
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'phi=90~150 s팁조밀화 병합 완료: {before}개 -> {len(rows)}개 (+{len(rows)-before}, 시도 $((N_COMBOS*8))개 중) ->', out)
"
echo "=== DONE ==="
