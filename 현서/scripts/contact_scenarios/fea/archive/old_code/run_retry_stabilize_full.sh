#!/bin/bash
# 2026-08-19: STABILIZE 파일럿(5/12=42%, 셋 중 최선)을 phi 전체 실패케이스(218개)+phi=+-60
# 신규(80개) = 298개 전체에 적용. retry_stabilize_full.py 참고(파일럿 근거 주석 있음).
set -u
cd "$(dirname "$0")"
date +%s > run_retry_stabilize_full_start_time.txt

N_COMBOS=43
echo "=== STABILIZE 전체 재시도 시작 (${N_COMBOS}개 조합, 298케이스, 동시 14개) ==="
seq 0 $((N_COMBOS - 1)) | xargs -P 14 -I{} bash -c \
  'python3 -u retry_stabilize_full.py --combo-index {} > "retry_stab_combo{}.log" 2>&1'

echo "=== 재시도 완료, 기존 all.json에 병합 ==="
python3 retry_stabilize_full.py --merge
echo "=== DONE ==="
