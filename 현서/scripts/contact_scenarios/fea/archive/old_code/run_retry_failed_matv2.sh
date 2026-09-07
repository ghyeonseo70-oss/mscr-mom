#!/bin/bash
# 2026-08-19: fea_lm_phi_pos_matv2_all.json 스윕의 실패 269개(45개 조합)를, 더 촘촘한
# 하중 증분(inc=300, initial_inc=0.002 - 기존 100/0.01보다 훨씬 조심스럽게)으로 재시도.
# retry_failed_matv2_cases.py 참고(원인 분석 및 접근 근거 주석 있음).
set -u
cd "$(dirname "$0")"
date +%s > run_retry_failed_matv2_start_time.txt

N_COMBOS=45
echo "=== 실패 케이스 재시도 시작 (${N_COMBOS}개 조합, 동시 10개) ==="
seq 0 $((N_COMBOS - 1)) | xargs -P 10 -I{} bash -c \
  'python3 -u retry_failed_matv2_cases.py --combo-index {} > "retry_matv2_combo{}.log" 2>&1'

echo "=== 재시도 완료, 기존 all.json에 병합 ==="
python3 retry_failed_matv2_cases.py --combo-index 0 --merge
echo "=== DONE ==="
