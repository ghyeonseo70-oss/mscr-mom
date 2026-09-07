#!/bin/bash
# 1) L_M=50/phi=60 고정, s(9)xdepth(10)=90케이스 스윕 (9-way 병렬)
# 2) L_M(5)xphi(5)=25조합 지오메트리 스윕, 조합당 s(3)xdepth(3)=9케이스 (동시 8개)
# 둘 다 print_tip=True로 팁 변위+회전각까지 저장.
set -u
cd "$(dirname "$0")"
date +%s > pipeline_start_time.txt

echo "=== [1단계] 90케이스(s x depth, L_M=50/phi=60) 스윕 시작 (9-way 병렬) ==="
rm -f fea_bent_contact_sweep_s*.log 2>/dev/null
rm -f ../../../data/contact_scenarios/fea/fea_bent_contact_sweep_s*.json 2>/dev/null
for s in 10 20 30 40 50 60 70 80 90; do
  python3 -u sweep_bent_contact_worker.py --s ${s}.0 --threads 6 --tag s${s} > sweep_s${s}.log 2>&1 &
done
wait
echo "=== [1단계] 90케이스 스윕 완료, 병합 ==="
python3 -c "
import json, glob
rows = []
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_bent_contact_sweep_s*.json')):
    rows.extend(json.load(open(f)))
rows.sort(key=lambda r: (r['contact_s_mm'], r['push_depth_mm']))
out = '../../../data/contact_scenarios/fea/fea_bent_contact_sweep.json'
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print('90케이스 병합 완료:', len(rows), '/90 ->', out)
"

echo "=== [2단계] L_M x phi 형상 스윕 시작 (25조합, 동시 8개) ==="
python3 - <<'PYEOF' > combos.txt
LM_LIST = [0.0, 25.0, 50.0, 75.0, 100.0]
PHI_LIST = [-120.0, -60.0, 0.0, 60.0, 120.0]
for lm in LM_LIST:
    for phi in PHI_LIST:
        sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
        tag = f"LM{int(lm)}_phi{sign}{int(abs(phi))}"
        print(f"{lm} {phi} {tag}")
PYEOF

cat combos.txt | xargs -P 8 -L 1 bash -c 'python3 -u geom_sweep_worker.py --L_M "$0" --phi "$1" --tag "$2" --threads 4 > "sweep_geom_$2.log" 2>&1'

echo "=== [2단계] 전체 형상 스윕 완료, 병합 ==="
python3 -c "
import json, glob
rows = []
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_geom_sweep_*.json')):
    rows.extend(json.load(open(f)))
rows.sort(key=lambda r: (r['L_M_mm'], r['phi_deg'], r['contact_s_mm'], r['push_depth_mm']))
out = '../../../data/contact_scenarios/fea/fea_geom_sweep_all.json'
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print('형상 스윕 병합 완료:', len(rows), '/225 ->', out)
"

echo "=== [3단계] 접촉각도(beta) 스윕 시작 (L_M=50/phi=60 고정, beta=90/180/270) ==="
for b in 90 180 270; do
  python3 -u angle_sweep_worker.py --beta ${b}.0 --threads 8 > sweep_angle_beta${b}.log 2>&1 &
done
wait

echo "=== [3단계] 각도 스윕 완료, 병합 ==="
python3 -c "
import json, glob
rows = []
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_angle_sweep_*.json')):
    rows.extend(json.load(open(f)))
rows.sort(key=lambda r: (r['beta_deg'], r['contact_s_mm'], r['push_depth_mm']))
out = '../../../data/contact_scenarios/fea/fea_angle_sweep_all.json'
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print('각도 스윕 병합 완료:', len(rows), '/27 ->', out)
"
echo "=== PIPELINE ALL DONE ==="
