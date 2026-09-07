#!/bin/bash
# 2026-08-19: PROJECT_STATUS.md "2번" 계획 - 메쉬 세밀화(1-B)가 0/12로 완전히 실패한 뒤
# 전환한 방향. phi 극단값(±90/±120/±150)은 근본적으로 CalculiX가 못 푸는 것으로 결론났으니
# 이번엔 "성공 케이스 수를 빠르게 늘리는 것"에 집중 - 기존 L_M 격자(0,25,50,75mm) 사이
# (12.5,37.5,62.5,87.5mm)를 잘 수렴하는 phi(0,±30,±60)로만 채운다.
# LM(4) x phi(5) x s(10) = 200케이스, beta=0만.
#
# sweep_lm_phi_position_matv2_worker.py를 그대로 재사용(output 파일명 패턴이 기존
# fea_lm_phi_pos_matv2_*.json과 동일해서 아래 병합 단계가 기존 all.json에 자동으로 합쳐짐 -
# 이번엔 별도 파일로 분리할 필요 없음, 목적 자체가 기존 데이터셋 보강이므로).
set -u
cd "$(dirname "$0")"
date +%s > lm_phi_pos_matv2_densify_start_time.txt

python3 - <<'PYEOF' > lm_phi_pos_matv2_densify_combos.txt
LM_LIST = [12.5, 37.5, 62.5, 87.5]
PHI_LIST = [0.0, 30.0, -30.0, 60.0, -60.0]

def tag_of(lm, phi, beta):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    lm_str = str(int(lm)) if lm == int(lm) else str(lm).replace(".", "p")
    return f"LM{lm_str}_phi{sign}{int(abs(phi))}_b{int(beta)}"

for lm in LM_LIST:
    for phi in PHI_LIST:
        print(f"{lm} {phi} 0 {tag_of(lm, phi, 0)}")
PYEOF

N_COMBOS=$(wc -l < lm_phi_pos_matv2_densify_combos.txt)
echo "=== L_M 조밀화 스윕 시작 (${N_COMBOS}조합, 동시 10개, 조합당 10케이스 = 총 $((N_COMBOS*10))케이스) ==="
cat lm_phi_pos_matv2_densify_combos.txt | xargs -P 10 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$3" --threads 4 > "sweep_matv2_densify_$3.log" 2>&1'

echo "=== 스윕 완료, 기존 all.json에 병합 ==="
# 주의(2026-08-19): all.json엔 이 스윕 밖에서 들어온 행도 있음(예: retry_failed_matv2_cases.py로
# 병합한 10개 - 개별 fea_lm_phi_pos_matv2_<tag>.json 파일로는 존재 안 하고 all.json에만 있음).
# 그래서 개별파일만으로 all.json을 다시 만들면 그 10개가 사라짐 - **기존 all.json을 베이스로
# 시작**하고, 개별파일(all.json 제외)에서 아직 없는 키만 추가하는 방식으로 안전하게 병합.
python3 -c "
import json, glob, os
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8')) if os.path.exists(out) else []
seen = {(r['L_M_mm'], r['phi_deg'], r['beta_deg'], r['contact_s_mm']) for r in rows}
added = 0
files = [f for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_*.json'))
         if not f.endswith('_all.json')]
for f in files:
    for r in json.load(open(f, encoding='utf-8')):
        key = (r['L_M_mm'], r['phi_deg'], r['beta_deg'], r['contact_s_mm'])
        if key not in seen:
            rows.append(r)
            seen.add(key)
            added += 1
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(rows, open(out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print(f'병합 완료: 기존 {len(rows) - added}개 + 신규 {added}개 = 총 {len(rows)}개 ->', out)
"
echo "=== DONE ==="
