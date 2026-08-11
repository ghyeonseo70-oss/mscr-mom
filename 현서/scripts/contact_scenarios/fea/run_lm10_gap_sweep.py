"""B) L_M=10mm 그리드 공백 검증: phi 11개 x s 3개 = 33케이스."""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

PHI_LIST = [0.0, 30.0, -30.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 60.0, -60.0]
CONCURRENCY = 9


def tag_for(phi):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"LM10_phi{sign}{int(abs(phi))}"


tasks = [(phi, tag_for(phi)) for phi in PHI_LIST]
print(f"L_M=10 공백검증 스윕 시작: {len(tasks)}개 (동시 {CONCURRENCY}), 각 3케이스 = 총 {len(tasks)*3}케이스", flush=True)

running = []
idx = 0
t_start = time.time()
while idx < len(tasks) or running:
    while len(running) < CONCURRENCY and idx < len(tasks):
        phi, tag = tasks[idx]
        logf = open(os.path.join(HERE, f"sweep_{tag}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-u", "sweep_lm10_gap_worker.py",
             "--phi", str(phi), "--tag", tag, "--threads", "4"],
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
for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, "fea_lm10_gap_check_LM10_*.json"))):
    merged.extend(json.load(open(f)))
merged.sort(key=lambda r: (r["phi_deg"], r["contact_s_mm"]))
out = os.path.join(FEA_DATA_DIR, "fea_lm10_gap_check.json")
json.dump(merged, open(out, "w"), indent=2, ensure_ascii=False)
print(f"병합 완료: {len(merged)}개 -> {out}", flush=True)
print("DONE", flush=True)
