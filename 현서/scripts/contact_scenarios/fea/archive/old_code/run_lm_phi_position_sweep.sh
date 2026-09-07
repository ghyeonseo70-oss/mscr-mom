#!/bin/bash
# LM(4: 0,25,50,75mm) x phi(9: 0,+-30,+-90,+-120,+-150deg) x s(10: 10~100mm) 스윕.
# push_depth=0.10mm, beta=0deg 고정(힘/각도는 중요하지 않다는 전제). 총 360케이스.
# 조합(36개)당 sweep_lm_phi_position_worker.py 한 프로세스, 동시 9개(각 4스레드=36코어 사용).
set -u
cd "$(dirname "$0")"
date +%s > lm_phi_pos_pipeline_start_time.txt

python3 - <<'PYEOF' > lm_phi_pos_combos.txt
LM_LIST = [0.0, 25.0, 50.0, 75.0]
PHI_LIST = [0.0, 30.0, -30.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0]
for lm in LM_LIST:
    for phi in PHI_LIST:
        sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
        tag = f"LM{int(lm)}_phi{sign}{int(abs(phi))}"
        print(f"{lm} {phi} {tag}")
PYEOF

echo "=== LM x phi x s 스윕 시작 (36조합, 동시 9개, 조합당 10케이스 = 총 360케이스) ==="
cat lm_phi_pos_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_worker.py --L_M "$0" --phi "$1" --tag "$2" --threads 4 > "sweep_lmphi_$2.log" 2>&1'

echo "=== 스윕 완료, 병합 ==="
python3 -c "
import json, glob
rows = []
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_sweep_*.json')):
    rows.extend(json.load(open(f)))
rows.sort(key=lambda r: (r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_sweep_all.json'
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print('LM x phi x s 스윕 병합 완료:', len(rows), '/360 ->', out)
"
echo "=== DONE ==="
