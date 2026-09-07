"""L_M=100mm 전용 보강 스윕: phi 11개 x s 10구간 = 110케이스.
기존 데이터 분석 결과 L_M=100 행이 거의 비어있었던 것(11개 phi 중 7개가 0개)을 메우기 위함.
run_lm_phi_position_sweep.sh와 동일한 패턴(동시성 9)이지만 L_M=100 하나만 대상."""
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

PHI_LIST = [0.0, 30.0, -30.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 60.0, -60.0]
L_M = 100.0
CONCURRENCY = 9


def tag_for(phi):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"LM100_phi{sign}{int(abs(phi))}"


tasks = [(L_M, phi, tag_for(phi)) for phi in PHI_LIST]
print(f"L_M=100 스윕 시작: {len(tasks)}개 조합 (동시 {CONCURRENCY}개), 조합당 10케이스 = 총 {len(tasks)*10}케이스", flush=True)

with open(os.path.join(HERE, "lm100_sweep_start_time.txt"), "w") as f:
    f.write(str(time.time()))

running = []
idx = 0
t_start = time.time()
while idx < len(tasks) or running:
    while len(running) < CONCURRENCY and idx < len(tasks):
        lm, phi, tag = tasks[idx]
        logf = open(os.path.join(HERE, f"sweep_{tag}.log"), "w")
        p = subprocess.Popen(
            [sys.executable, "-u", "sweep_lm_phi_position_worker.py",
             "--L_M", str(lm), "--phi", str(phi), "--tag", tag, "--threads", "4"],
            cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
        running.append((p, logf, tag))
        print(f"[{time.strftime('%H:%M:%S')}] 시작: {tag} (phi={phi})", flush=True)
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

rows = json.load(open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_sweep_all.json")))
have = set((r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]) for r in rows)
new_rows = []
for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_sweep_LM100_*.json"))):
    new_rows.extend(json.load(open(f)))
added = [r for r in new_rows if (r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]) not in have]
rows.extend(added)
rows.sort(key=lambda r: (r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]))
json.dump(rows, open(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_sweep_all.json"), "w"),
          indent=2, ensure_ascii=False)
print(f"L_M=100 신규 성공: {len(new_rows)}개 (그중 신규 {len(added)}개), 병합 후 총 {len(rows)}개", flush=True)
print("DONE", flush=True)
