#!/bin/bash
# FEA 스윕(1단계: L_M=50/phi=60 s×depth 90케이스, 2단계: L_M×phi 25조합 225케이스,
# 3단계: 접촉각도 beta=90/180/270 27케이스, 총 342케이스)
# 진행 상황 + 남은시간 예측을 실시간으로 보여줌. Ctrl+C로 종료.
cd "$(dirname "$0")"
DATA_DIR="../../../data/contact_scenarios/fea"

while true; do
  clear
  echo "===== FEA 스윕 진행 상황 ($(date +%H:%M:%S)) ====="
  echo ""
  echo "--- 파이프라인 로그 (마지막 5줄) ---"
  tail -5 run_full_pipeline.log 2>/dev/null
  echo ""

  s1=0
  if ls "$DATA_DIR"/fea_bent_contact_sweep_s*.json >/dev/null 2>&1; then
    echo "--- 1단계: s x depth (L_M=50, phi=60), 목표 90케이스 ---"
    s1=$(python3 -c "
import json, glob
total = 0
for f in sorted(glob.glob('$DATA_DIR/fea_bent_contact_sweep_s*.json')):
    d = json.load(open(f))
    tag = f.split('_s')[-1].replace('.json','')
    print(f'  s{tag}: {len(d)}/10')
    total += len(d)
print(f'  합계: {total}/90')
print(total)
" 2>/dev/null | tee /dev/stderr | tail -1)
    echo ""
  fi

  s2=0
  if ls "$DATA_DIR"/fea_geom_sweep_*.json >/dev/null 2>&1; then
    echo "--- 2단계: L_M x phi 형상, 목표 225케이스(25조합 x 9) ---"
    s2=$(python3 -c "
import json, glob
total = 0
for f in sorted(glob.glob('$DATA_DIR/fea_geom_sweep_*.json')):
    d = json.load(open(f))
    tag = f.split('fea_geom_sweep_')[-1].replace('.json','')
    print(f'  {tag}: {len(d)}/9')
    total += len(d)
print(f'  합계: {total}/225')
print(total)
" 2>/dev/null | tee /dev/stderr | tail -1)
    echo ""
  fi

  s3=0
  if ls "$DATA_DIR"/fea_angle_sweep_*.json >/dev/null 2>&1; then
    echo "--- 3단계: 접촉각도(beta), 목표 27케이스(beta 3개 x 9) ---"
    s3=$(python3 -c "
import json, glob
total = 0
for f in sorted(glob.glob('$DATA_DIR/fea_angle_sweep_*.json')):
    d = json.load(open(f))
    tag = f.split('fea_angle_sweep_')[-1].replace('.json','')
    print(f'  {tag}: {len(d)}/9')
    total += len(d)
print(f'  합계: {total}/27')
print(total)
" 2>/dev/null | tee /dev/stderr | tail -1)
    echo ""
  fi

  echo "--- 실행 중인 ccx 작업 ---"
  ps -eo pid,etime,pcpu,cmd | grep '^ *[0-9]* .*ccx ' | grep -v grep
  echo ""

  echo "--- 예상 남은 시간 ---"
  if [ -f pipeline_start_time.txt ]; then
    python3 -c "
import time
start = int(open('pipeline_start_time.txt').read().strip())
elapsed = time.time() - start
done = ${s1:-0} + ${s2:-0} + ${s3:-0}
total = 342
if done > 0:
    rate = elapsed / done
    remain = (total - done) * rate
    def fmt(sec):
        h, r = divmod(int(sec), 3600)
        m, s = divmod(r, 60)
        return f'{h}시간 {m}분'
    print(f'경과: {fmt(elapsed)} | 완료: {done}/{total} | 케이스당 평균 {rate:.0f}초 | 예상 남은시간: {fmt(remain)} (단계별 병렬 개수가 달라 참고용 수치임)')
else:
    print('아직 완료된 케이스가 없어 예측 불가')
"
  else
    echo "(pipeline_start_time.txt 없음 - 시간 예측 불가)"
  fi
  echo ""
  echo "(5초마다 갱신, 종료하려면 Ctrl+C)"
  sleep 5
done

