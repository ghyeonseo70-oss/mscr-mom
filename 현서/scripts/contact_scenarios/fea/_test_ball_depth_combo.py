"""2026-10-06: 볼 크기 + push_depth 조합 테스트.
1) 같은 조건(L_M=12.5,phi=-120,s=30)에서 depth=0.30mm(한 번도 시도 안 한 깊이)를
   ball_r=0.4(기존) vs ball_r=0.8(큰 볼)로 비교 - 큰 볼이 더 깊은 depth에서도 도움되는지.
2) 아예 안 됐던 극단 케이스(L_M=0,phi=120,s=85, depth=0.20mm, 음수 자코비안까지 갔던 것)를
   ball_r=0.8로 재시도 - 큰 볼이 이 케이스를 살릴 수 있는지.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bent_contact_scene as scene
import run_contact as rc

MIN_INC = 1e-8
CONTACT_MESH_SIZE = 0.08

CASES = [
    dict(L_M=12.5, phi=-120.0, s=30.0, depth=0.30, ball_r=0.4, tag="COMBO1_r40_d30"),
    dict(L_M=12.5, phi=-120.0, s=30.0, depth=0.30, ball_r=0.8, tag="COMBO2_r80_d30"),
    dict(L_M=0.0, phi=120.0, s=85.0, depth=0.20, ball_r=0.8, tag="COMBO3_hardcase_r80"),
]

for case in CASES:
    tag = case["tag"]
    centerline_path = os.path.join(HERE, f"matv2_centerline_{tag}.json")
    subprocess.run(
        [sys.executable, os.path.join(HERE, "get_bent_centerline.py"),
         "--L_M", str(case["L_M"]), "--phi", str(case["phi"]), "--out", centerline_path],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    inp_name = f"matv2_mesh_{tag}.inp"
    sets_name = f"matv2_node_sets_{tag}.inp"
    job_name = f"matv2test_{tag}"

    print(f"\n=== {tag}: L_M={case['L_M']}, phi={case['phi']}, s={case['s']}, "
          f"depth={case['depth']}mm, ball_r={case['ball_r']}mm ===")
    t0 = time.time()
    try:
        scene_info = scene.build_mesh(
            contact_s=case["s"], ball_r=case["ball_r"], verbose=False,
            centerline_path=centerline_path, beta_deg=0.0, inp_name=inp_name, sets_name=sets_name,
            contact_mesh_size=CONTACT_MESH_SIZE, contact_refine_radius=case["ball_r"] + 0.3,
        )
        normal = scene_info["normal"]
        res = rc.run_case(
            case["depth"], inp_name=inp_name, sets_name=sets_name, job_name=job_name,
            timeout=1800, verbose=False, push_dir=tuple(normal), n_threads=4,
            print_tip=True, print_mom=True, stabilize=True, min_inc=MIN_INC,
        )
    except Exception as e:
        res = None
        print(f"  예외 발생: {e}")
    dt = time.time() - t0
    if res is None:
        print(f"  [{tag}] 실패 ({dt:.1f}s)")
    else:
        tip_mag = (res["tip_ux_avg_mm"]**2 + res["tip_uy_avg_mm"]**2) ** 0.5
        print(f"  [{tag}] 성공! F_mag={res['F_mag_N']*1000:.4f}mN, "
              f"팁변위={tip_mag:.4f}mm, 팁회전={res['tip_theta_deg']:.3f}deg ({dt:.1f}s)")
    try:
        os.remove(centerline_path)
    except OSError:
        pass

print("\n=== 테스트 완료 ===")
