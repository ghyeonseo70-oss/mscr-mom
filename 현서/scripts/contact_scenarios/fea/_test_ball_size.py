"""2026-10-06: 인덴터(볼) 크기를 키우면 같은 push_depth에서 수렴이 더 잘 되는지 + 힘/변위가
얼마나 커지는지 확인. 같은 조건(L_M=12.5, phi=-120, s=30, depth=0.20mm - 이전에 수렴 성공
확인된 케이스)에서 ball_r만 바꿔가며 비교.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

L_M, PHI, S, DEPTH = 12.5, -120.0, 30.0, 0.20
BALL_RADII = [0.4, 0.8]  # mm (기존 0.4 대조군 포함, 1.0은 일단 빼고 시간 절약)
MIN_INC = 1e-8
CONTACT_MESH_SIZE = 0.08
# 2026-10-06 버그 수정: contact_refine_radius 기본값(ball_r*2.5)으로 두면 ball_r=0.4에서도
# 1.0mm(검증 안 된 더 공격적인 값)가 돼서 메쉬가 20만개 요소로 폭증, 1시간 넘게 메싱만 함.
# 검증된 0.6mm(ball_r=0.4 기준)에 맞춰 ball_r+0.3mm로 고정 - 볼이 커져도 반경이 과하게 안 늘어남.

centerline_path = os.path.join(HERE, "matv2_centerline_BALLTEST.json")
subprocess.run(
    [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
     "--L_M", str(L_M), "--phi", str(PHI), "--out", centerline_path],
    check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
)

for ball_r in BALL_RADII:
    tag = f"BALLTEST_r{int(ball_r*100)}"
    inp_name = f"matv2_mesh_{tag}.inp"
    sets_name = f"matv2_node_sets_{tag}.inp"
    job_name = f"matv2test_{tag}"

    print(f"\n=== ball_r={ball_r}mm (지름 {ball_r*2}mm), depth={DEPTH}mm ===")
    t0 = time.time()
    try:
        scene_info = scene.build_mesh(
            contact_s=S, ball_r=ball_r, verbose=False, centerline_path=centerline_path,
            beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
            contact_mesh_size=CONTACT_MESH_SIZE, contact_refine_radius=ball_r + 0.3,
        )
        normal = scene_info["normal"]
        res = rc.run_case(
            DEPTH, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
            timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=4,
            print_tip=True, print_mom=True, stabilize=True, min_inc=MIN_INC,
        )
    except Exception as e:
        res = None
        print(f"  예외 발생: {e}")
    dt = time.time() - t0
    if res is None:
        print(f"  ball_r={ball_r}: 실패 ({dt:.1f}s)")
    else:
        tip_mag = (res["tip_ux_avg_mm"]**2 + res["tip_uy_avg_mm"]**2) ** 0.5
        print(f"  ball_r={ball_r}: 성공! F_mag={res['F_mag_N']*1000:.4f}mN, "
              f"팁변위={tip_mag:.4f}mm, 팁회전={res['tip_theta_deg']:.3f}deg ({dt:.1f}s)")

try:
    os.remove(centerline_path)
except OSError:
    pass

print("\n=== 테스트 완료 ===")
