"""
sweep_contact.py의 굽은 튜브 버전. 접촉위치(s, 중심선 호길이)와 누르는 깊이(push_depth)를
바꿔가며 CalculiX 접촉해석을 반복 실행해서 '진짜 FEA 기반' 굽은 튜브 충돌 데이터셋을 만든다.

직선 튜브 스윕과 같은 위치 비율(전체길이의 25%/50%/75%)·같은 깊이 후보를 써서 두 결과를
비교할 수 있게 함. s마다 접촉점의 법선(push_dir)이 달라서 make_bent_contact_scene.build_mesh()가
반환하는 normal을 그대로 run_contact.run_case에 넘겨야 함(방향 틀리면 힘이 0으로 나오는 버그
이미 두 번 겪음 - run_contact.py 주석 참고).
"""
import json
import os
import time

import make_bent_contact_scene as scene
import run_contact as rc

HERE = os.path.dirname(os.path.abspath(__file__))

# ── 스윕 조합 (PROJECT_STATUS.md 계획: s 9곳 x depth 10단계 = 90케이스) ──
# s: 0(베이스)/100(팁) 경계는 경계조건/자유단 특이점이라 제외하고 10% 간격으로 9곳
S_LIST = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]  # mm, 중심선 호길이
# 기존 3단계(0.05~0.15mm)와 같은 0.02mm 간격으로 10단계까지 확장
PUSH_DEPTH_LIST = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]  # mm
BALL_R = 0.4                            # mm

results = []
total = len(S_LIST) * len(PUSH_DEPTH_LIST)
n = 0
t_start = time.time()

for contact_s in S_LIST:
    # 접촉위치(s)가 같으면 메쉬(튜브+구) 재사용 가능(깊이만 바뀌어도 초기간격은 고정)
    scene_info = scene.build_mesh(contact_s=contact_s, ball_r=BALL_R, verbose=False)
    normal = scene_info["normal"]
    for push_depth in PUSH_DEPTH_LIST:
        n += 1
        job_name = f"bent_sweep_{n:03d}"
        t0 = time.time()
        print(f"[{n}/{total}] s={contact_s}mm, push_depth={push_depth}mm, normal={normal} ...", flush=True)
        try:
            res = rc.run_case(
                push_depth,
                inp_name="bent_contact_mesh.inp",
                sets_name="bent_contact_node_sets.inp",
                job_name=job_name,
                timeout=900,
                verbose=False,
                push_dir=tuple(normal),
            )
        except Exception as e:
            print(f"  실패: {e}", flush=True)
            res = None
        dt = time.time() - t0
        if res is None:
            print(f"  [실패] ({dt:.1f}s)", flush=True)
            continue
        row = {
            "contact_s_mm": contact_s,
            "ball_r_mm": BALL_R,
            "push_depth_mm": push_depth,
            "normal": normal,
            "ball_center": scene_info["ball_center"],
            "Fx_total_N": res["Fx_total_N"],
            "Fy_total_N": res["Fy_total_N"],
            "Fz_total_N": res["Fz_total_N"],
            "F_mag_N": res["F_mag_N"],
            "ux_avg_mm": res["ux_avg_mm"],
            "wall_time_s": dt,
        }
        results.append(row)
        print(f"  F_mag={res['F_mag_N']*1000:.4f}mN  ({dt:.1f}s)", flush=True)

        # 케이스마다 즉시 저장(장시간 스윕 중 중간에 끊겨도 그때까지 결과는 보존)
        out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                                 "fea_bent_contact_sweep.json")
        out_path = os.path.abspath(out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

t_total = time.time() - t_start
print(f"\n스윕 완료: {len(results)}/{total} 성공, 총 {t_total/60:.1f}분", flush=True)
print(f"저장: {out_path}")
