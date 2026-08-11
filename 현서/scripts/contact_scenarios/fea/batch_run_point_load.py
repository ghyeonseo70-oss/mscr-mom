"""find_safe_force_range.py와 같은 시드로 재현한 13개 케이스(즉시반전 10개+생존군 3개)를
run_point_load.run_case()로 하나씩(별도 프로세스로, 동시성 4) 돌려서 결과를 모은다.
각 케이스가 독자적인 tag(pl_{seed:03d})를 쓰므로 파일 충돌 없이 병렬 실행 가능.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

CASES = [
    {"seed": 2, "L_M": 35.70, "phi": -120.0, "s": 78.28, "F_ang_deg": 33.1, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 201, "L_M": 79.44, "phi": -120.0, "s": 48.95, "F_ang_deg": 213.1, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 8, "L_M": 39.62, "phi": -60.0, "s": 33.68, "F_ang_deg": 283.9, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 194, "L_M": 71.88, "phi": -60.0, "s": 28.66, "F_ang_deg": 173.4, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 0, "L_M": 58.22, "phi": 0.0, "s": 8.69, "F_ang_deg": 5.9, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 225, "L_M": 56.15, "phi": 0.0, "s": 94.87, "F_ang_deg": 82.8, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 7, "L_M": 57.51, "phi": 60.0, "s": 74.81, "F_ang_deg": 81.1, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 173, "L_M": 24.10, "phi": 60.0, "s": 30.21, "F_ang_deg": 177.0, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 9, "L_M": 72.21, "phi": 120.0, "s": 59.28, "F_ang_deg": 279.9, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 197, "L_M": 76.07, "phi": 120.0, "s": 76.96, "F_ang_deg": 205.6, "F_mag_mN": 0.5, "analytic_threshold_mN": 0.0},
    {"seed": 1, "L_M": 50.71, "phi": 60.0, "s": 17.97, "F_ang_deg": 341.5, "F_mag_mN": 20.0, "analytic_threshold_mN": 20.0},
    {"seed": 93, "L_M": 79.13, "phi": 60.0, "s": 46.34, "F_ang_deg": 156.5, "F_mag_mN": 20.0, "analytic_threshold_mN": 20.0},
    {"seed": 121, "L_M": 57.97, "phi": -60.0, "s": 79.18, "F_ang_deg": 209.1, "F_mag_mN": 20.0, "analytic_threshold_mN": 20.0},
]

CASES_PATH = os.path.join(HERE, "point_load_cases.json")
RESULTS_PATH = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                             "point_load_validation_results.json")
CONCURRENCY = 4


def worker_main(idx):
    from run_point_load import run_case
    case = CASES[idx]
    result = run_case(case)
    out_path = os.path.join(HERE, f"pl_{case['seed']:03d}_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(json.dumps(result))


def main():
    running = []
    idx = 0
    results = []
    logf_map = {}
    while idx < len(CASES) or running:
        while len(running) < CONCURRENCY and idx < len(CASES):
            case = CASES[idx]
            logf = open(os.path.join(HERE, f"pl_{case['seed']:03d}_batch.log"), "w")
            p = subprocess.Popen([sys.executable, __file__, "--worker", str(idx)],
                                  cwd=HERE, stdout=logf, stderr=subprocess.STDOUT)
            running.append((p, case, logf))
            LOG(f"시작: seed={case['seed']} L_M={case['L_M']:.1f} phi={case['phi']} s={case['s']:.1f} "
                f"F={case['F_mag_mN']}mN (동시실행 {len(running)}개)")
            idx += 1
        time.sleep(10)
        still = []
        for p, case, logf in running:
            if p.poll() is None:
                still.append((p, case, logf))
            else:
                logf.close()
                res_path = os.path.join(HERE, f"pl_{case['seed']:03d}_result.json")
                if os.path.exists(res_path):
                    r = json.load(open(res_path))
                    results.append(r)
                    LOG(f"완료: seed={case['seed']} status={r.get('fea_status')} "
                        f"fea_reversed={r.get('fea_reversed')} (해석모델 threshold={case['analytic_threshold_mN']}mN)")
                else:
                    LOG(f"완료(결과파일 없음, 실패 추정): seed={case['seed']}")
                    results.append({**case, "fea_status": "crashed"})
        running = still

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOG(f"전체 13개 완료, 저장: {RESULTS_PATH}")
    n_ok = sum(1 for r in results if r.get("fea_status") == "ok")
    LOG(f"성공 {n_ok}/13")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker_main(int(sys.argv[2]))
    else:
        main()
