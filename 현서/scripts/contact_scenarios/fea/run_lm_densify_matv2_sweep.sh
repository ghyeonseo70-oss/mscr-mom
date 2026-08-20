#!/bin/bash
# 2026-08-20 신규: L_M 격자 조밀화 스윕. 기존 L_M(0,25,50,75mm) 사이에 새 값
# (12.5,37.5,62.5,87.5mm)을 끼워넣어 대체모델이 L_M 축을 더 촘촘하게 배우게 함
# (목적: tip_ux/tip_uy 변위 예측 정확도 개선, 현재 R^2=0.555/0.877 - 0.8 이상이 목표).
#
# phi는 9개 전부(0,±30,±60,±90,±120,±150) 포함 - 처음엔 0,±30,±60만 하려 했다가,
# 실제 배포되는 15만개 합성데이터가 phi를 -150~150 전체에서 균일 샘플링한다는 걸 감안하면
# phi=90~150을 계속 안 뽑으면 그 구간에서 대체모델이 영원히 실측 없이 외삽만 하게 됨
# (오늘 하루 종일 고친 "서로게이트 환각 학습" 문제를 phi 축에서 재현하는 꼴) - 사용자 판단으로
# 성공률이 낮아도(phi=90~150은 30~40%대) 포함하기로 함.
#
# sweep_lm_phi_position_matv2_worker.py가 이제 stabilize=True를 기본으로 쓰도록 고쳐놔서
# (이전엔 나중에 별도 재시도 라운드로 STABILIZE를 붙였는데) 이번엔 처음부터 켜진 채로 돎 -
# 별도 재시도 라운드 불필요.
#
# LM(4) x phi(11, 0/+-30/+-60/+-90/+-120/+-150) x s(10) = 440케이스가 원래 최대치이나,
# 아래 BAD_COMBOS 6개(L_M=62.5mm의 +-150, L_M=87.5mm의 +-120/+-150 - 해석모델이 잘못된
# 해의 branch를 계산하는 격자점, 자세한 근거는 아래 파이썬 블록 주석 참고)를 제외해
# 실제로는 38조합 x s(10) = 380케이스. beta=0만(beta=180은 이미 대칭성 검증 완료, 재확인 불필요).
set -u
cd "$(dirname "$0")"
date +%s > lm_densify_matv2_pipeline_start_time.txt

python3 - <<'PYEOF' > lm_densify_matv2_combos.txt
LM_LIST = [12.5, 37.5, 62.5, 87.5]
PHI_LIST = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0]

# 2026-08-20 추가: force_model.py의 힌트 없는 solve_shape()(실제 이 워커가 쓰는 방식)로
# 격자점을 직접 확인한 결과, 아래 6개 조합은 해석모델이 잘못된 해(branch)를 계산함
# (theta_L이 phi 증가에도 갑자기 줄어드는 등 물리적으로 불가능한 역전 발생).
# FEA 메쉬는 만들어지고 성공할 수도 있지만, 애초에 틀린 형상 위에 만든 것이라 틀린 물리를
# 학습시키게 되므로 스윕에서 제외:
#   L_M=62.5mm: phi=+-150 (연속법 기준 130->135도 부근에서 이미 jump, 격자점 기준 150도가 오답)
#   L_M=87.5mm: phi=+-120, +-150 (연속법 기준 100->105도 부근에서 jump, 두 격자점 다 오답)
BAD_COMBOS = {(62.5, 150.0), (62.5, -150.0), (87.5, 120.0), (87.5, -120.0), (87.5, 150.0), (87.5, -150.0)}

def tag_of(lm, phi):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"LMD{int(lm*10)}_phi{sign}{int(abs(phi))}_b0"  # 12.5mm -> LMD125 (점 없이, 파일명 안전하게)

for lm in LM_LIST:
    for phi in PHI_LIST:
        if (lm, phi) in BAD_COMBOS:
            continue
        print(f"{lm} {phi} 0 {tag_of(lm, phi)}")
PYEOF

N_COMBOS=$(wc -l < lm_densify_matv2_combos.txt)
echo "=== L_M 조밀화 스윕 시작 (${N_COMBOS}조합, 동시 9개, 조합당 10케이스 = 총 $((N_COMBOS*10))케이스, STABILIZE 기본 적용) ==="
cat lm_densify_matv2_combos.txt | xargs -P 9 -L 1 bash -c \
  'python3 -u sweep_lm_phi_position_matv2_worker.py --L_M "$0" --phi "$1" --beta "$2" --tag "$3" --threads 4 > "sweep_lmdensify_$3.log" 2>&1'

echo "=== 스윕 완료, 기존 fea_lm_phi_pos_matv2_all.json에 병합 ==="
python3 -c "
import json, glob
out = '../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_all.json'
rows = json.load(open(out, encoding='utf-8'))
before = len(rows)
for f in sorted(glob.glob('../../../data/contact_scenarios/fea/fea_lm_phi_pos_matv2_LMD*.json')):
    rows.extend(json.load(open(f, encoding='utf-8')))
rows.sort(key=lambda r: (r['beta_deg'], r['L_M_mm'], r['phi_deg'], r['contact_s_mm']))
json.dump(rows, open(out, 'w'), indent=2, ensure_ascii=False)
print(f'L_M 조밀화 병합 완료: {before}개 -> {len(rows)}개 (+{len(rows)-before}, 시도 $((N_COMBOS*10))개 중) ->', out)
"
echo "=== DONE ==="
