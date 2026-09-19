"""2026-09-19 신규: push_depth(충돌 깊이) 축 데이터 확보 - 형상 추정 전환용.

배경: 기존 FEA 518개는 전부 push_depth=0.10mm 고정이었음(sweep_lm_phi_position_matv2_worker.py의
원래 주석: "힘/깊이는 덜 중요하다는 전제"). 목표가 "접촉 위치 감지"였을 땐 합리적인 선택이었지만,
"충돌 시 로봇이 어떤 형상이 되는가"로 방향을 틀면서 문제가 됨 - 지금 데이터로는 "어느 방향으로
휘는지"는 배울 수 있어도 "얼마나 세게 부딪혔을 때 얼마나 더 휘는지"는 깊이 값이 하나뿐이라
학습 자체가 불가능함(PROJECT_STATUS.md 30번).

전략: 새 격자를 처음부터 만드는 대신, **이미 0.10mm에서 수렴에 성공한 조합**만 골라서 다른
깊이로 재실행. 이러면 (a) 수렴 실패로 낭비되는 시간이 거의 없고, (b) 같은 (L_M,phi,s)에 대해
깊이만 다른 "짝지어진" 데이터가 생겨서 깊이의 효과를 깨끗하게 분리해 배울 수 있음.

선택은 L_M / |phi| 고저 / s 구간으로 층화추출(stratified)해서 공간을 골고루 덮음.

사용법:
    python sweep_push_depth_matv2.py --dry-run              # 대상/개수/예상시간만 확인
    python sweep_push_depth_matv2.py --max-attempts 60      # 실제 실행 (예산 제한)
    python sweep_push_depth_matv2.py --depths 0.05,0.2      # 깊이 목록 지정
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea"))
OUT_PATH = os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json")

BASE_DEPTH = 0.10          # 기존 518개가 전부 쓰던 깊이
DEFAULT_DEPTHS = [0.05, 0.20]  # 기존의 절반/두 배 - 로그 스케일로 고르게 벌림
# 층화추출 구간: (이름, s 하한, s 상한)
S_BINS = [("베이스쪽", 10.0, 40.0), ("중간", 40.0, 70.0), ("팁쪽", 70.0, 101.0)]


def load_rows():
    return json.load(open(OUT_PATH, encoding="utf-8"))


def existing_keys(rows):
    """이미 확보된 (L_M, phi, beta, s, depth) 조합."""
    return {(round(r["L_M_mm"], 1), round(r["phi_deg"], 1), round(r["beta_deg"], 1),
             round(r["contact_s_mm"], 1), round(r.get("push_depth_mm", BASE_DEPTH), 3))
            for r in rows}


def select_configs(rows, n_configs, seed=0):
    """0.10mm에서 성공한 케이스 중, L_M / |phi|고저 / s구간으로 층화추출."""
    base = [r for r in rows
            if round(r.get("push_depth_mm", BASE_DEPTH), 3) == BASE_DEPTH and r["beta_deg"] == 0.0]
    strata = {}
    for r in base:
        hi_phi = abs(r["phi_deg"]) >= 90
        s_bin = next((name for name, lo, hi in S_BINS if lo <= r["contact_s_mm"] < hi), None)
        if s_bin is None:
            continue
        strata.setdefault((r["L_M_mm"], hi_phi, s_bin), []).append(r)

    rng = np.random.default_rng(seed)
    keys = sorted(strata.keys(), key=lambda k: (k[0], k[1], k[2]))
    picked = []
    # 층을 라운드로빈으로 돌며 하나씩 뽑아 - 특정 층에 쏠리지 않게.
    round_idx = 0
    while len(picked) < n_configs:
        added_this_round = False
        for k in keys:
            if len(picked) >= n_configs:
                break
            bucket = strata[k]
            if round_idx < len(bucket):
                if round_idx == 0:
                    rng.shuffle(bucket)
                picked.append(bucket[round_idx])
                added_this_round = True
        if not added_this_round:
            break
        round_idx += 1
    return picked


def tag_of(lm, phi, depth):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"DEPTH{int(lm * 10)}_phi{sign}{int(abs(phi))}_d{int(depth * 1000)}_b0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true",
                         help="FEA는 안 돌리고 이미 있는 DEPTH*.json만 all.json에 병합 - "
                              "git 충돌 해결 후 양쪽 데이터를 합칠 때 사용(중복 제거됨).")
    parser.add_argument("--depths", type=str, default=None,
                         help=f"쉼표구분 깊이(mm) 목록 (기본 {DEFAULT_DEPTHS})")
    parser.add_argument("--n-configs", type=int, default=30,
                         help="깊이별로 재실행할 (L_M,phi,s) 조합 개수 (기본 30)")
    parser.add_argument("--config-start", type=int, default=0,
                         help="선택된 조합 목록에서 앞의 N개를 건너뜀 - 여러 컴퓨터가 "
                              "겹치지 않게 나눠 돌리기 위한 옵션. 층화추출은 시드 고정이라 "
                              "같은 목록이 결정적으로 재현되므로, A컴퓨터가 "
                              "--config-start 0 --n-configs 30, B컴퓨터가 "
                              "--config-start 30 --n-configs 120 식으로 나누면 중복 없음.")
    parser.add_argument("--max-attempts", type=int, default=None,
                         help="이번 실행에서 시도할 케이스 개수 상한 - 케이스당 ~15~20분")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minutes-per-case", type=float, default=18.0,
                         help="예상시간 계산용(이 컴퓨터 실측 ~17.7분)")
    parser.add_argument("--parallel", type=int, default=4,
                         help="예상시간 계산용 동시 실행 개수(실제 병렬 실행은 셸에서)")
    args = parser.parse_args()

    if args.merge_only:
        merge_into_all()
        return

    depths = [float(v) for v in args.depths.split(",")] if args.depths else list(DEFAULT_DEPTHS)
    rows = load_rows()
    have = existing_keys(rows)
    # 앞부분을 건너뛰려면 그만큼 더 뽑아야 함(층화추출 목록은 결정적이라 앞 N개는 항상 동일).
    configs = select_configs(rows, args.config_start + args.n_configs)[args.config_start:]

    print(f"기존 데이터 {len(rows)}행 (그중 depth={BASE_DEPTH}mm: "
          f"{sum(1 for r in rows if round(r.get('push_depth_mm', BASE_DEPTH), 3) == BASE_DEPTH)}행)")
    print(f"선택된 기준 조합: {len(configs)}개 "
          f"(전체 목록에서 {args.config_start}번째부터), 새로 돌릴 깊이: {depths}")

    # (L_M, phi, depth)로 묶어서 s를 한 번에 넘김 - centerline 재계산을 아껴줌.
    todo = {}
    for depth in depths:
        for r in configs:
            key = (r["L_M_mm"], r["phi_deg"], depth)
            s = r["contact_s_mm"]
            if (round(r["L_M_mm"], 1), round(r["phi_deg"], 1), 0.0, round(s, 1), round(depth, 3)) in have:
                continue  # 이미 있음(재실행 시 이어하기 가능)
            todo.setdefault(key, []).append(s)

    total = sum(len(v) for v in todo.values())
    if args.max_attempts is not None and args.max_attempts < total:
        capped, budget = {}, args.max_attempts
        for key in sorted(todo.keys()):
            if budget <= 0:
                break
            take = todo[key][:budget]
            capped[key] = take
            budget -= len(take)
        todo = capped
        total = sum(len(v) for v in todo.values())
        print(f"--max-attempts {args.max_attempts} 적용 -> 이번엔 {total}개만")

    est_h = total * args.minutes_per_case / max(1, args.parallel) / 60
    print(f"\n총 {total}케이스 ({len(todo)}회 워커 호출), "
          f"예상 {est_h:.1f}시간 (동시 {args.parallel}개, 케이스당 {args.minutes_per_case:.0f}분 가정)")
    print("\n[실행 목록]")
    for (lm, phi, depth), s_list in sorted(todo.items()):
        print(f"  L_M={lm:5.1f}, phi={phi:6.1f}, depth={depth:.3f}mm -> s={sorted(s_list)}")

    if args.dry_run or total == 0:
        return

    print(f"\n=== 깊이 스윕 시작 (동시 {args.parallel}개 x 스레드 {args.threads}개) ===")
    t0 = time.time()

    def run_one(item):
        (lm, phi, depth), s_list = item
        tag = tag_of(lm, phi, depth)
        s_arg = ",".join(str(s) for s in sorted(s_list))
        with open(os.path.join(HERE, f"sweep_depth_{tag}.log"), "w", encoding="utf-8") as logf:
            subprocess.run(
                [sys.executable, "-u", os.path.join(HERE, "sweep_lm_phi_position_matv2_worker.py"),
                 "--L_M", str(lm), "--phi", str(phi), "--beta", "0", "--tag", tag,
                 "--threads", str(args.threads), "--s_list", s_arg, "--push_depth", str(depth)],
                stdout=logf, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            )
        return tag, len(s_list)

    # 워커를 동시에 여러 개 띄움(각 워커는 ccx를 --threads개 스레드로 돌리므로
    # parallel x threads <= 논리코어 수로 맞출 것 - 이 컴퓨터는 16코어라 4x4가 기본).
    items = sorted(todo.items())
    done = 0
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, it): it for it in items}
        for fut in as_completed(futures):
            (lm, phi, depth), s_list = futures[fut]
            done += 1
            try:
                fut.result()
                status = "완료"
            except Exception as e:  # 워커 자체가 죽은 경우(수렴실패는 워커가 알아서 건너뜀)
                status = f"실패({e})"
            print(f"[{done}/{len(items)}] L_M={lm}, phi={phi}, depth={depth}mm "
                  f"(s {len(s_list)}개) {status}  경과 {(time.time()-t0)/60:.1f}분", flush=True)
    print(f"=== 완료 ({(time.time() - t0) / 60:.1f}분) ===")

    merge_into_all()


def merge_into_all():
    """DEPTH*.json들을 all.json에 병합. (L_M,phi,beta,s,depth) 키로 중복 제거하므로
    여러 번 실행해도 안전하고, 두 컴퓨터가 각자 병합한 뒤 git 충돌이 났을 때도
    아무 쪽이나 고른 다음 이 함수만 다시 돌리면 양쪽 데이터가 온전히 합쳐짐."""
    print("=== 기존 fea_lm_phi_pos_matv2_all.json에 병합 (중복 제거) ===")
    merged = load_rows()
    before = len(merged)
    seen = {(round(r["L_M_mm"], 1), round(r["phi_deg"], 1), round(r["beta_deg"], 1),
             round(r["contact_s_mm"], 1), round(r.get("push_depth_mm", BASE_DEPTH), 3)): r
            for r in merged}
    added = 0
    for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_DEPTH*.json"))):
        for r in json.load(open(f, encoding="utf-8")):
            key = (round(r["L_M_mm"], 1), round(r["phi_deg"], 1), round(r["beta_deg"], 1),
                   round(r["contact_s_mm"], 1), round(r.get("push_depth_mm", BASE_DEPTH), 3))
            if key not in seen:
                seen[key] = r
                added += 1
    merged = sorted(seen.values(), key=lambda r: (r["beta_deg"], r["L_M_mm"], r["phi_deg"],
                                                   r.get("push_depth_mm", BASE_DEPTH), r["contact_s_mm"]))
    json.dump(merged, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"병합 완료: {before}개 -> {len(merged)}개 (신규 {added}개)")


if __name__ == "__main__":
    main()
