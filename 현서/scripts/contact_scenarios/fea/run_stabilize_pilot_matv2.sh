#!/bin/bash
# 2026-08-21 신규: stabilize=True(CalculiX 자동감쇠)로도 여전히 실패하는 phi=90~150 케이스
# 20개(L_M=0/25/50/75 각 5개, 실측 데이터에 없는 = 이미 실패 확인된 (L_M,phi,s) 조합 중
# 무작위 추출)를 골라, 명시적 STABILIZE 감쇠계수 3가지(0.001, 0.005, 0.02)로 다시 시도해서
# 자동감쇠보다 나은 값이 있는지 확인하는 작은 파일럿. 성공하면 물리는 그대로(STABILIZE는
# 수치기법이지 모델을 바꾸는 게 아님)라 바로 본 데이터셋에 병합해도 됨.
# 20조합 x 3값 = 60케이스(각각 s는 1개 고정값만, --s_list로 지정).
set -u
cd "$(dirname "$0")"
date +%s > stabilize_pilot_matv2_start_time.txt

python3 - <<'PYEOF' > stabilize_pilot_matv2_combos.txt
# 아래 20개는 fea_lm_phi_pos_matv2_all.json에 없는(=stabilize=True로도 실패 확인된) 조합
# 중 L_M=0/25/50/75에서 5개씩 무작위 추출한 것 (2026-08-21 확인, 시드 고정 재현 가능)
COMBOS = [
    (0.0, -90.0, 40.0), (0.0, 90.0, 40.0), (0.0, 150.0, 80.0), (0.0, -120.0, 80.0), (0.0, -90.0, 100.0),
    (25.0, 150.0, 40.0), (25.0, 120.0, 80.0), (25.0, 90.0, 90.0), (25.0, 150.0, 80.0), (25.0, -90.0, 20.0),
    (50.0, -90.0, 60.0), (50.0, -120.0, 50.0), (50.0, -120.0, 100.0), (50.0, -120.0, 60.0), (50.0, 90.0, 90.0),
    (75.0, 120.0, 90.0), (75.0, -120.0, 40.0), (75.0, -150.0, 100.0), (75.0, 150.0, 80.0), (75.0, -120.0, 50.0),
]
STABILIZE_VALUES = [0.001, 0.005, 0.02]

def tag_of(lm, phi, s, stab):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    stab_tag = str(stab).replace(".", "")
    return f"STB{int(lm*10)}_phi{sign}{int(abs(phi))}_s{int(s)}_v{stab_tag}_b0"

for lm, phi, s in COMBOS:
    for stab in STABILIZE_VALUES:
        print(f"{lm} {phi} 0 {s} {stab} {tag_of(lm, phi, s, stab)}")
PYEOF

N_COMBOS=$(wc -l < stabilize_pilot_matv2_combos.txt)
echo "=== STABILIZE 감쇠계수 파일럿 시작 (${N_COMBOS}케이스, 동시 9개) ==="
cat stabilize_pilot_matv2_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$5" --threads 4 --s_list "$3" --stabilize_value "$4" > "sweep_stabpilot_$5.log" 2>&1'

echo "=== 파일럿 완료, 조합별로 어느 stabilize 값이 통했는지 집계 + 기존 데이터셋에 병합 ==="
python3 -c "
import json, glob, re
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8'))
before = len(rows)
have = set((r['L_M_mm'], r['phi_deg'], r['contact_s_mm']) for r in rows if r['beta_deg']==0.0)
added_by_combo = {}
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_STB*.json')):
    m = re.search(r'_v([0-9]+)_b0', f)
    stab_val = m.group(1) if m else '?'
    for r in json.load(open(f, encoding='utf-8')):
        key = (r['L_M_mm'], r['phi_deg'], r['contact_s_mm'])
        if key in have:
            continue  # 이미 있으면 중복 추가 안 함(여러 stabilize 값이 동시에 성공한 경우)
        have.add(key)
        rows.append(r)
        added_by_combo.setdefault(key, []).append(stab_val)
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'STABILIZE 파일럿 병합 완료: {before}개 -> {len(rows)}개 (+{len(rows)-before}, 시도 $((N_COMBOS))개 중)')
print('조합별 성공한 stabilize 값(첫 성공 기준):')
for k, v in sorted(added_by_combo.items()):
    print(' ', k, '->', v[0])
"
echo "=== DONE ==="
