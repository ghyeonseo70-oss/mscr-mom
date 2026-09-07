"""접촉위치 스윕(bent_sweep_s{tag}_05 = push_depth 0.10mm) 3개 케이스의 원본 .frd에서
실제 변형된 중심선을 뽑아냄. .frd는 케이스당 300MB+라 매번 전체를 읽지 않도록, 마지막
증분(수렴 완료 시점)의 DISP 블록 줄 범위만 grep으로 찾아 그 부분만 파싱한다.

방법: N_TUBE_OUTER(바깥 표면 절점, 원주 전체 고르게 분포)의 원좌표+변위를 무하중 기준
중심선(bent_centerline.json, 40개 점)에 최근접 매칭해서 버킷으로 묶고, 버킷별 평균을 내면
그 버킷의 "중심선 위치"에 근사(원주에 고르게 분포한 링의 평균 = 축 중심)한다는 원리.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_contact import _parse_nset, _parse_node_coords  # noqa: E402

DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea")

CASES = {"s10": 10.0, "s30": 30.0, "s80": 80.0}
# 참고: s40/s60의 현재 .frd/.sta는 이후 재시도(retry)로 덮어써져서 TOT TIME=1.0(완전수렴)에
# 도달하지 못한 상태(s40은 0.567, s60은 0.02에서 멈춤)로 남아있음 - 즉 json에 저장된 성공값과
# 지금 디스크의 .frd가 서로 다른 실행 결과라 여기서 다시 뽑으면 안 됨(실제로 겪음: s40의
# 팁 변위를 재구성해보니 수십mm로 터무니없이 커서 발견). .sta의 TOT TIME=1.0 확인된
# s10/s20/s30/s80만 사용.
JOB_INDEX = 5  # PUSH_DEPTH_LIST[4] = 0.10mm (sweep_bent_contact_worker.py)


def find_last_disp_range(frd_path):
    out = subprocess.run(["grep", "-n", "^ -4", frd_path], capture_output=True, text=True)
    lines = out.stdout.splitlines()
    disp_starts = [int(l.split(":")[0]) for l in lines if "DISP" in l]
    all_markers = sorted(int(l.split(":")[0]) for l in lines)
    start = disp_starts[-1]
    end = min(m for m in all_markers if m > start)  # 다음 -4 블록(STRESS/FORC 등) 시작 줄
    return start, end


def parse_disp_block(frd_path, start, end, wanted_ids):
    proc = subprocess.run(["sed", "-n", f"{start},{end}p", frd_path], capture_output=True, text=True)
    disp = {}
    for line in proc.stdout.splitlines():
        # 고정폭 포맷(FORTRAN E12.5) - 음수는 부호가 앞 필드에 바로 붙어 공백 없이 나올 수
        # 있어서 split()으로 나누면 값이 섞여버림(실제로 겪음: 33326개 중 132개만 파싱됨).
        if not line.startswith(" -1") or len(line) < 49:
            continue
        try:
            nid = int(line[3:13])
            ux = float(line[13:25])
            uy = float(line[25:37])
            uz = float(line[37:49])
        except ValueError:
            continue
        if nid in wanted_ids:
            disp[nid] = (ux, uy, uz)
    return disp


with open(os.path.join(HERE, "bent_centerline.json")) as f:
    cl = json.load(f)
ref_pts = np.array([[p["x"], p["y"], p["z"]] for p in cl["points"]])
ref_s = np.array([p["s"] for p in cl["points"]])

results = {}
for tag, s_val in CASES.items():
    print(f"=== {tag} (contact_s={s_val}mm) ===")
    frd_path = os.path.join(HERE, f"bent_sweep_{tag}_{JOB_INDEX:02d}.frd")
    sets_path = os.path.join(HERE, f"bent_contact_node_sets_{tag}.inp")
    inp_path = os.path.join(HERE, f"bent_contact_mesh_{tag}.inp")

    outer_ids = set(_parse_nset(sets_path, "N_TUBE_OUTER"))
    print(f"  N_TUBE_OUTER 절점 수: {len(outer_ids)}")

    start, end = find_last_disp_range(frd_path)
    print(f"  마지막 DISP 블록: 줄 {start}~{end}")
    disp = parse_disp_block(frd_path, start, end, outer_ids)
    print(f"  파싱된 변위 절점 수: {len(disp)}")

    ids = list(disp.keys())
    orig = _parse_node_coords(inp_path, ids)
    ids = [i for i in ids if i in orig]

    orig_xyz = np.array([orig[i] for i in ids])
    disp_xyz = np.array([disp[i] for i in ids])
    deformed_xyz = orig_xyz + disp_xyz

    # 각 절점을 무하중 기준 중심선(40점)에 최근접 매칭 -> 버킷
    d2 = ((orig_xyz[:, None, :] - ref_pts[None, :, :]) ** 2).sum(axis=2)
    bucket = d2.argmin(axis=1)

    bucket_s, bucket_orig, bucket_def, bucket_n = [], [], [], []
    for b in range(len(ref_pts)):
        mask = bucket == b
        n = mask.sum()
        if n == 0:
            continue
        bucket_s.append(float(ref_s[b]))
        bucket_orig.append(orig_xyz[mask].mean(axis=0).tolist())
        bucket_def.append(deformed_xyz[mask].mean(axis=0).tolist())
        bucket_n.append(int(n))

    order = np.argsort(bucket_s)
    results[tag] = {
        "contact_s_mm": s_val,
        "s": [bucket_s[i] for i in order],
        "orig_xyz": [bucket_orig[i] for i in order],
        "deformed_xyz": [bucket_def[i] for i in order],
        "n_nodes_per_bucket": [bucket_n[i] for i in order],
    }
    print(f"  버킷 {len(bucket_s)}/{len(ref_pts)}개 채워짐 (버킷당 평균 {np.mean(bucket_n):.0f}개 절점)")

out_path = os.path.join(DATA_DIR, "fea_deformed_centerline_examples.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"저장: {out_path}")
