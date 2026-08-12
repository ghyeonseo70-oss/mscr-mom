"""beta 정밀분해능 스윕: (L_M=50,phi=60)과 (L_M=50,phi=-60) 두 조건에서 beta를 15도 간격
0~345도(24개)로 전부 스캔. s=30mm/depth=0.10mm 고정. 총 48케이스.
각 조건의 24개를 4묶음(6개씩)으로 쪼개서 병렬화 -> 총 8워커, 동시 8개."""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

CONDITIONS = [(50.0, 60.0, "base"), (50.0, -60.0, "shifted")]
BETA_ALL = [i * 15.0 for i in range(24)]  # 0,15,...,345
N_CHUNKS = 4
CONCURRENCY = 8

tasks = []
for lm, phi, label in CONDITIONS:
    chunk_size = len(BETA_ALL) // N_CHUNKS
    for c in range(N_CHUNKS):
        chunk = BETA_ALL[c * chunk_size:(c + 1) * chunk_size]
        tag = f"{label}_c{c}"
        tasks.append((lm, phi, chunk, tag))

print(f"beta 정밀분해능 스윕 시작: {len(tasks)}개 워커 (동시 {CONCURRENCY}), "
      f"워커당 {len(BETA_ALL)//N_CHUNKS}케이스 = 총 {len(BETA_ALL)*len(CONDITIONS)}케이스", flush=True)

running = []
idx = 0
t_start = time.time()
while idx < len(tasks) or running:
    while len(running) < CONCURRENCY and idx < len(tasks):
        lm, phi, chunk, tag = tasks[idx]
        beta_str = ",".join(str(b) for b in chunk)
        logf = open(os.path.join(HERE, f"sweep_betafine_{tag}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-u", "sweep_beta_fine_worker.py",
             "--L_M", str(lm), "--phi", str(phi), "--beta_list", beta_str, "--tag", tag, "--threads", "4"],
            cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
        running.append((p, logf, tag))
        print(f"[{time.strftime('%H:%M:%S')}] 시작: {tag} (beta={chunk})", flush=True)
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
for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, "fea_beta_fine_resolution_*.json"))):
    merged.extend(json.load(open(f)))
merged.sort(key=lambda r: (r["phi_deg"], r["beta_deg"]))
out = os.path.join(FEA_DATA_DIR, "fea_beta_fine_resolution.json")
json.dump(merged, open(out, "w"), indent=2, ensure_ascii=False)
print(f"병합 완료: {len(merged)}개 -> {out}", flush=True)
print("DONE", flush=True)
