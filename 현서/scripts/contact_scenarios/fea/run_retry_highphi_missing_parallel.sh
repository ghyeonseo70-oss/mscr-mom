#!/bin/bash
# 2026-09-14 신규: retry_highphi_missing_matv2.py(다른 컴퓨터 인수인계용, cae5212)는
# 조합을 순차(for loop, subprocess.run 하나씩)로 처리하도록 설계돼있어서 533개 재시도
# 대상을 케이스당 15~20분으로 계산하면 이 컴퓨터 혼자 다 돌리는 데 5~7일 걸림.
# 이 컴퓨터는 64코어라 기존 스윕들처럼 xargs 병렬로 돌릴 여유가 있어서, retry 스크립트의
# "목표 격자 대비 아직 없는 (L_M,phi,s) 계산" 로직만 그대로 재사용하고 실행은 조합 단위로
# 병렬화하는 래퍼. retry_highphi_missing_matv2.py 자체를 여러 개 동시 실행하면 안 됨
# (둘 다 같은 시작 상태에서 missing을 계산해서 같은 조합을 중복 시도 + 같은 출력파일에
# 동시 write하는 레이스 컨디션 생김) - 그래서 이 래퍼가 조합을 미리 나눠서 배정함.
#
# 다른 컴퓨터와의 인수인계 호환성 유지: 출력 파일명 태그(RETRY{lm*10}_phi{sign}{abs}_b0)와
# STABILIZE=0.01 기본값은 원본 스크립트와 동일하게 맞춤 - 이미 성공한 조합은 자동 제외되므로
# 이 컴퓨터가 병렬로 먼저 끝내도 다른 컴퓨터의 원본(순차) 스크립트가 재실행되면 자동으로
# 겹치지 않게 스킵됨.
set -u
cd "$(dirname "$0")"
PARALLEL="${1:-6}"  # 저-phi 스윕(36스레드)과 동시 실행 중이라 기본 6(x4스레드=24) - 64코어 중 여유분
date +%s > retry_highphi_missing_parallel_start_time.txt

python3 - <<'PYEOF' > retry_highphi_missing_parallel_combos.txt
import sys
sys.path.insert(0, ".")
from retry_highphi_missing_matv2 import compute_missing, tag_of

missing_by_combo, total_target, total_missing = compute_missing()
print(f"목표 {total_target}개, 재시도 대상 {total_missing}개, 조합 {len(missing_by_combo)}개", file=sys.stderr)
for (lm, phi), s_list in sorted(missing_by_combo.items()):
    s_csv = ",".join(str(s) for s in s_list)
    print(f"{lm} {phi} {s_csv} {tag_of(lm, phi)}")
PYEOF

N_COMBOS=$(wc -l < retry_highphi_missing_parallel_combos.txt)
N_CASES=$(python3 -c "
import sys
total=0
for line in open('retry_highphi_missing_parallel_combos.txt'):
    parts = line.split()
    total += len(parts[2].split(','))
print(total)
")
echo "=== |phi|>=90 실패조합 재시도 병렬 실행 (${N_COMBOS}조합, 동시 ${PARALLEL}개, 총 ${N_CASES}케이스, STABILIZE=0.01 명시) ==="
cat retry_highphi_missing_parallel_combos.txt | xargs -P "$PARALLEL" -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta 0 --tag "$3" --threads 4 --s_list "$2" --stabilize_value 0.01 > "retry_highphi_$3.log" 2>&1'

echo "=== 재시도 완료, 기존 fea_lm_phi_pos_matv2_all.json에 병합 ==="
python3 -c "
import json, glob
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8'))
before = len(rows)
existing = {(r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']) for r in rows}
added = 0
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_RETRY*.json')):
    for r in json.load(open(f, encoding='utf-8')):
        key = (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm'])
        if key in existing:
            continue
        rows.append(r)
        existing.add(key)
        added += 1
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'재시도 병합 완료: {before}개 -> {len(rows)}개 (+{added}, 시도 ${N_CASES}개 중) ->', out)
"
echo "=== DONE ==="
