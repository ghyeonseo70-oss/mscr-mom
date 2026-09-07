"""
find_confusable_pair.py가 찾은 "빔이론 단순모델은 거의 똑같다고 보는" 두 접촉위치(s=10mm,
s=30mm)에서, 실제 FEA로 눌러봤을 때 팁(자유단) 변위가 힘 크기 대비 정말 비슷하게 나오는지
비교한다. 정확히 같은 힘(16mN vs 2mN)을 맞추기는 어려워서(변위제어라 힘은 결과값), 각 위치에서
여러 깊이로 눌러 (F_mag, 팁변위) 곡선을 얻고 같은 F_mag대에서 팁변위를 비교하는 방식을 씀.

결과 해석:
- 같은 F_mag에서 두 위치의 팁변위가 비슷 -> 단순모델 말이 맞음(물리적으로 진짜 구별 안 됨,
  FEA 데이터를 아무리 모아도 소용없음 - 센서/구조를 바꿔야 함)
- 뚜렷이 다름 -> 단순모델이 정보를 버린 것 -> FEA 기반 데이터가 실제로 도움이 됨 (오래 걸려도
  투자할 가치 있음)
"""
import json
import os
import time

import make_bent_contact_scene as scene
import run_contact as rc

HERE = os.path.dirname(os.path.abspath(__file__))

S_LIST = [10.0, 30.0]
DEPTH_LIST = [0.05, 0.10, 0.15]

results = []
t_start = time.time()

for s in S_LIST:
    inp_name = f"tip_test_s{int(s)}.inp"
    sets_name = f"tip_test_s{int(s)}_sets.inp"
    scene_info = scene.build_mesh(contact_s=s, ball_r=0.4, inp_name=inp_name, sets_name=sets_name, verbose=True)
    normal = scene_info["normal"]
    for depth in DEPTH_LIST:
        job_name = f"tip_test_s{int(s)}_d{str(depth).replace('.', '')}"
        t0 = time.time()
        print(f"[s={s}mm, depth={depth}mm] 실행 중...", flush=True)
        try:
            res = rc.run_case(depth, inp_name=inp_name, sets_name=sets_name, job_name=job_name,
                               timeout=900, verbose=False, push_dir=tuple(normal), print_tip=True)
        except Exception as e:
            print(f"  실패: {e}", flush=True)
            res = None
        dt = time.time() - t0
        if res is None:
            print(f"  [실패] ({dt:.1f}s)", flush=True)
            continue
        row = {"s": s, "depth": depth, "normal": normal,
               "F_mag_N": res["F_mag_N"], "Fx_total_N": res["Fx_total_N"], "Fy_total_N": res["Fy_total_N"],
               "tip_ux_mm": res["tip_ux_avg_mm"], "tip_uy_mm": res["tip_uy_avg_mm"],
               "tip_uz_mm": res["tip_uz_avg_mm"], "wall_time_s": dt}
        row["tip_shift_mag_mm"] = (row["tip_ux_mm"]**2 + row["tip_uy_mm"]**2 + row["tip_uz_mm"]**2) ** 0.5
        results.append(row)
        print(f"  F_mag={res['F_mag_N']*1000:.5f}mN, tip_shift={row['tip_shift_mag_mm']:.5f}mm ({dt:.1f}s)",
              flush=True)

        out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea",
                                 "tip_confusable_test.json")
        with open(os.path.abspath(out_path), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

t_total = time.time() - t_start
print(f"\n완료: {len(results)}/{len(S_LIST)*len(DEPTH_LIST)} 성공, 총 {t_total/60:.1f}분", flush=True)

print("\n=== 요약: F_mag 대비 팁변위 (s=10 vs s=30) ===")
for s in S_LIST:
    sub = [r for r in results if r["s"] == s]
    for r in sub:
        print(f"  s={s}mm: F_mag={r['F_mag_N']*1000:.5f}mN -> tip_shift={r['tip_shift_mag_mm']:.5f}mm "
              f"(변위/힘 비율={r['tip_shift_mag_mm']/(r['F_mag_N']*1000+1e-12):.4f}mm/mN)")
