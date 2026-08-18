#!/bin/bash
# 2026-08-18 신규: K1 구간(베이스~MOM)에 니티놀 와이어(실리콘 벽 두께 안쪽 임베드,
# include_wire=True)를 넣어서 run_lm_phi_position_matv2_sweep.sh(재료값-검증, 와이어 없음)와
# 똑같은 beta=0 그리드를 돈다: LM(4: 0,25,50,75mm, L_M=0은 K1 구간이 없어 와이어 없음과 동일한
# 결과가 나옴 - 사용자 요청으로 그리드 완전성을 위해 포함) x phi(9: 0,+-30,+-90,+-120,+-150deg)
# x s(10: 10~100mm) = 360케이스.
#
# 재료값-검증 스윕(run_lm_phi_position_matv2_sweep.sh)이 이미 동시에 돌고 있어서(9워커x4스레드
# =36스레드), 이 와이어 스윕은 동시성을 낮춰서(-P4, --threads4=16스레드) 자원을 나눠 씀
# (64코어 중 36+16=52, 여유 둠). 와이어 케이스는 훨씬 무거워서(케이스당 8~13분+, 무와이어
# 1~5분 대비) 이 정도 동시성으로도 전체 완료까지 상당히 오래 걸릴 것으로 예상됨.
#
# 결과 파일 접두사를 fea_matv2wire_*로 분리(재료값-검증 스윕의 fea_lm_phi_pos_matv2_* 와
# glob 패턴이 절대 겹치지 않게 - sweep_lm_phi_position_matv2_wire_worker.py 참고).
set -u
cd "$(dirname "$0")"
date +%s > lm_phi_pos_matv2_wire_pipeline_start_time.txt

python3 - <<'PYEOF' > lm_phi_pos_matv2_wire_combos.txt
LM_LIST_FULL = [0.0, 25.0, 50.0, 75.0]
PHI_LIST_FULL = [0.0, 30.0, -30.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0]

def tag_of(lm, phi, beta):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"LM{int(lm)}_phi{sign}{int(abs(phi))}_b{int(beta)}"

for lm in LM_LIST_FULL:
    for phi in PHI_LIST_FULL:
        print(f"{lm} {phi} 0 {tag_of(lm, phi, 0)}")
PYEOF

N_COMBOS=$(wc -l < lm_phi_pos_matv2_wire_combos.txt)
echo "=== 와이어 포함 LM x phi x s 스윕 시작 (${N_COMBOS}조합, 동시 4개, 조합당 10케이스 = 총 $((N_COMBOS*10))케이스) ==="
cat lm_phi_pos_matv2_wire_combos.txt | xargs -P 4 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_wire_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$3" --threads 4 > "sweep_matv2wire_$3.log" 2>&1'

echo "=== 스윕 완료, 병합 ==="
python3 -c "
import json, glob
rows = []
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_matv2wire_*.json')):
    rows.extend(json.load(open(f)))
rows.sort(key=lambda r: (r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
out = '../../../data/contact_scenarios/fea/fea_matv2wire_all.json'
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'와이어 스윕 병합 완료: {len(rows)}/$((N_COMBOS*10)) ->', out)
"
echo "=== DONE ==="
