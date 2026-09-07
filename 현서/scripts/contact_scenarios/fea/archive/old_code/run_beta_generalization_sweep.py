"""A) beta 일반화 검증: (L_M,phi) = (25,-90),(75,150) x depth(0.05,0.10,0.15) = 6개 워커,
각 beta(7) x s(2) = 14케이스씩, 총 84케이스."""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

COMBOS = [(25.0, -90.0), (75.0, 150.0)]
DEPTHS = [0.05, 0.10, 0.15]
CONCURRENCY = 6

tasks = []
for lm, phi in COMBOS:
    for depth in DEPTHS:
        sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
        tag = f"LM{int(lm)}_phi{sign}{int(abs(phi))}_d{int(depth*100)}"
        tasks.append((lm, phi, depth, tag))

print(f"beta 일반화 스윕 시작: {len(tasks)}개 워커 (동시 {CONCURRENCY}), 워커당 14케이스 = 총 {len(tasks)*14}케이스", flush=True)

running = []
idx = 0
t_start = time.time()
while idx < len(tasks) or running:
    while len(running) < CONCURRENCY and idx < len(tasks):
        lm, phi, depth, tag = tasks[idx]
        logf = open(os.path.join(HERE, f"sweep_betagen_{tag}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-u", "sweep_beta_generalization_worker.py",
             "--L_M", str(lm), "--phi", str(phi), "--depth", str(depth), "--tag", tag, "--threads", "4"],
            cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
        running.append((p, logf, tag))
        print(f"[{time.strftime('%H:%M:%S')}] 시작: {tag}", flush=True)
        idx += 1
    time.sleep(15)
    still = []
    for p, logf, tag in running:
        if p.poll() is None:
            still.append((p, logf, tag))
        else:
            logf.close()
            print(f"[{time.strftime('%H:%M:%S')}] 완료: {tag}", flush=True)
    running = still

print(f"전체 완료 ({(time.time()-t_start)/60:.1f}분). 병합 시작...", flush=True)
merged = []
for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, "fea_beta_generalization_check_LM*.json"))):
    merged.extend(json.load(open(f)))
merged.sort(key=lambda r: (r["L_M_mm"], r["phi_deg"], r["beta_deg"], r["contact_s_mm"], r["push_depth_mm"]))
out = os.path.join(FEA_DATA_DIR, "fea_beta_generalization_check.json")
json.dump(merged, open(out, "w"), indent=2, ensure_ascii=False)
print(f"병합 완료: {len(merged)}개 -> {out}", flush=True)
print("DONE", flush=True)
