"""
접촉위치(z), 누르는 깊이(push_depth), 구 크기(ball_r)를 바꿔가며 CalculiX 접촉해석을 반복 실행해서
'진짜 FEA 기반' 충돌 시나리오 데이터셋을 만든다.

케이스마다: 메쉬를 새로 생성(make_contact_scene.build_mesh) -> CalculiX 실행(run_contact.run_case)
-> 총 접촉력(Fx_total_N) 등을 기록. 메쉬/해석 둘 다 매번 새로 돌아야 해서 순수 물리모델
스윕(contact_force_sweep.py)보다 훨씬 느림 - 그래서 케이스 수를 작게 잡음.
"""
import json
import os
import time

import make_contact_scene as scene
import run_contact as rc

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 스윕 조합 ──────────────────────────────────────────────
# 실제 로봇 전체 길이(100mm) 기준 접촉위치: 고정단 가까이/중간/자유단 가까이 (이전 30mm 시험편과
# 같은 비율 25%/50%/75% 지점 유지)
CONTACT_Z_LIST = [25.0, 50.0, 75.0]    # mm
PUSH_DEPTH_LIST = [0.05, 0.10, 0.15]   # mm, 누르는 깊이(=대략적인 충돌 세기)
BALL_R = 0.4                            # mm, 구 크기는 일단 고정(필요하면 나중에 스윕 추가)

results = []
total = len(CONTACT_Z_LIST) * len(PUSH_DEPTH_LIST)
n = 0
t_start = time.time()

for contact_z in CONTACT_Z_LIST:
    # 접촉위치가 같으면 메쉬는 재사용 가능(깊이만 바뀌어도 메쉬 자체는 동일 - 초기간격은 고정)
    ball_center = scene.build_mesh(contact_z=contact_z, ball_r=BALL_R, verbose=False)
    for push_depth in PUSH_DEPTH_LIST:
        n += 1
        job_name = f"sweep_{n:03d}"
        t0 = time.time()
        print(f"[{n}/{total}] z={contact_z}mm, push_depth={push_depth}mm ...", flush=True)
        try:
            res = rc.run_case(push_depth, job_name=job_name, timeout=300, verbose=False)
        except Exception as e:
            print(f"  실패: {e}", flush=True)
            res = None
        dt = time.time() - t0
        if res is None:
            print(f"  [실패] ({dt:.1f}s)", flush=True)
            continue
        row = {
            "contact_z_mm": contact_z,
            "ball_r_mm": BALL_R,
            "push_depth_mm": push_depth,
            "ball_center": ball_center,
            "Fx_total_N": res["Fx_total_N"],
            "ux_avg_mm": res["ux_avg_mm"],
            "wall_time_s": dt,
        }
        results.append(row)
        print(f"  Fx={res['Fx_total_N']*1000:.4f}mN  ({dt:.1f}s)", flush=True)

t_total = time.time() - t_start
print(f"\n스윕 완료: {len(results)}/{total} 성공, 총 {t_total/60:.1f}분", flush=True)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea", "fea_contact_sweep.json")
out_path = os.path.abspath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"저장: {out_path}")
