"""44개 (L_M,phi) 조합 각각에 대해, 성공률 낮았던 s값(60~100mm) 중 아직 없는 것만 재시도.
동시성 8(각 4스레드=32코어) - 이미 돌고 있는 다른 백그라운드 작업과 공존하도록 여유를 둠."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

LM_LIST = [0.0, 25.0, 50.0, 75.0]
PHI_LIST = [0.0, 30.0, -30.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 60.0, -60.0]
S_TARGET = [60.0, 70.0, 80.0, 90.0, 100.0]
CONCURRENCY = 8

rows = json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_sweep_all.json")))
have = set((r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]) for r in rows)

tasks = []
for lm in LM_LIST:
    for phi in PHI_LIST:
        missing = [s for s in S_TARGET if (lm, phi, s) not in have]
        if not missing:
            continue
        sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
        tag = f"LM{int(lm)}_phi{sign}{int(abs(phi))}"
        tasks.append((lm, phi, missing, tag))

print(f"재시도할 (L_M,phi) 조합: {len(tasks)}개, 총 케이스: {sum(len(t[2]) for t in tasks)}개", flush=True)

running = []
idx = 0
t_start = time.time()
while idx < len(tasks) or running:
    while len(running) < CONCURRENCY and idx < len(tasks):
        lm, phi, missing, tag = tasks[idx]
        s_str = ",".join(str(s) for s in missing)
        logf = open(os.path.join(HERE, f"retry_lowsucc_{tag}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-u", "retry_low_success_s_worker.py",
             "--L_M", str(lm), "--phi", str(phi), "--s_list", s_str, "--tag", tag, "--threads", "4"],
            cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
        running.append((p, logf, tag))
        print(f"[{time.strftime('%H:%M:%S')}] 시작: {tag} (s={missing})", flush=True)
        idx += 1
    time.sleep(10)
    still = []
    for p, logf, tag in running:
        if p.poll() is None:
            still.append((p, logf, tag))
        else:
            logf.close()
            print(f"[{time.strftime('%H:%M:%S')}] 완료: {tag}", flush=True)
    running = still

print(f"전체 완료 ({(time.time()-t_start)/60:.1f}분). 병합 시작...", flush=True)

import glob
merged = []
for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, "fea_retry_lowsucc_*.json"))):
    merged.extend(json.load(open(f)))
print(f"재시도로 새로 성공한 케이스: {len(merged)}개", flush=True)

# 기존 fea_lm_phi_pos_sweep_all.json에 합쳐서 갱신
all_rows = rows + merged
all_rows.sort(key=lambda r: (r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]))
out_path = os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_sweep_all.json")
json.dump(all_rows, open(out_path, "w"), indent=2, ensure_ascii=False)
print(f"병합 완료: 총 {len(all_rows)}개 -> {out_path}", flush=True)
