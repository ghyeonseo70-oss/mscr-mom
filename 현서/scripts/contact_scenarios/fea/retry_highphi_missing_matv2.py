"""2026-09-14 신규: |phi|>=90 Fx_board 문제 전용 - "실패한 특정 조합만 재시도"
(PROJECT_STATUS.md 23번에서 "급하게 손대지 말 것"이라 미뤄뒀던 그 접근).

run_highphi_sdensify_matv2_sweep.sh(s=15~95 오프셋 격자 조밀화)까지 다 끝난 뒤에도
phi=90~150 구간은 FEA 접촉해석 수렴 성공률이 30~40%대로 낮게 남아있음(물리적
토크제로 특이점 때문, 메쉬 세밀화로는 해결 안 됨을 이미 확인함). 이 스크립트는
"목표 격자점 중 아직 실측 데이터가 없는 (L_M,phi,s) 조합"을 데이터 파일에서 직접
계산해서 뽑아낸 뒤, 기존 워커가 기본으로 쓰던 자동감쇠(stabilize=True) 대신
명시적 STABILIZE 값(--stabilize_value)으로 재시도한다. 자동감쇠가 이미 실패한
케이스라 같은 설정으로 또 돌려봐야 결과가 똑같을 뿐이라, 감쇠계수를 명시값으로
바꿔서 수렴 거동 자체를 바꿔보는 시도.

사용법:
    python retry_highphi_missing_matv2.py --dry-run   # 목표/기존/재시도 대상 개수만 확인
    python retry_highphi_missing_matv2.py             # 실제로 재시도 실행 (오래 걸림)
    python retry_highphi_missing_matv2.py --stabilize 0.01 --threads 4  # 감쇠값/스레드 지정
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FEA_DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea"))
OUT_PATH = os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_all.json")

# run_highphi_sdensify_matv2_sweep.sh / sweep_lm_phi_position_matv2_worker.py와
# 동일한 목표 격자(주석 참고) - 여기 숫자를 바꾸면 저기도 같이 맞춰야 함.
LM_LIST = [0.0, 12.5, 25.0, 37.5, 50.0, 62.5, 75.0, 87.5]
PHI_LIST = [90.0, -90.0, 120.0, -120.0, 150.0, -150.0]
# 해석모델이 잘못된 해(branch)를 계산하는 조합 - PROJECT_STATUS.md 물리 상수 현황 참고,
# 스윕 대상에서 항상 제외.
BAD_COMBOS = {(62.5, 150.0), (62.5, -150.0), (87.5, 120.0), (87.5, -120.0), (87.5, 150.0), (87.5, -150.0)}
# 기존 스윕들이 시도했던 s 격자 전부(기본 10mm 간격 + 오프셋 격자) 합집합.
S_LIST_FULL = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
               15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0]


def valid_combos():
    return [(lm, phi) for lm in LM_LIST for phi in PHI_LIST if (lm, phi) not in BAD_COMBOS]


def load_existing_keys():
    if not os.path.exists(OUT_PATH):
        return set()
    rows = json.load(open(OUT_PATH, encoding="utf-8"))
    have = set()
    for r in rows:
        if r.get("beta_deg", 0.0) == 0.0:
            have.add((round(r["L_M_mm"], 1), round(r["phi_deg"], 1), round(r["contact_s_mm"], 1)))
    return have


def compute_missing():
    have = load_existing_keys()
    combos = valid_combos()
    missing_by_combo = {}
    for lm, phi in combos:
        missing_s = [s for s in S_LIST_FULL if (round(lm, 1), round(phi, 1), round(s, 1)) not in have]
        if missing_s:
            missing_by_combo[(lm, phi)] = missing_s
    total_target = len(combos) * len(S_LIST_FULL)
    total_missing = sum(len(v) for v in missing_by_combo.values())
    return missing_by_combo, total_target, total_missing


def tag_of(lm, phi):
    sign = "N" if phi < 0 else ("0" if phi == 0 else "P")
    return f"RETRY{int(lm * 10)}_phi{sign}{int(abs(phi))}_b0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="재시도 없이 목표/기존/누락 개수만 출력")
    parser.add_argument("--stabilize", type=float, default=0.01,
                         help="*STATIC,STABILIZE에 쓸 명시적 감쇠계수 (기본 0.01) - "
                              "자동감쇠(stabilize=True)로 이미 실패한 조합이라 다른 값으로 재시도")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=None,
                         help="이번 실행에서 시도할 (L_M,phi,s) 케이스 개수 상한 - 케이스당 "
                              "~15~20분 걸리므로 전체(수백개)를 한 번에 돌리는 건 비현실적. "
                              "지정 안 하면 전부 시도(매우 오래 걸림, 권장 안 함).")
    args = parser.parse_args()

    missing_by_combo, total_target, total_missing = compute_missing()
    print(f"목표 격자: 유효조합 {len(valid_combos())}개 x s {len(S_LIST_FULL)}개 = {total_target}개")
    print(f"기존 성공: {total_target - total_missing}개, 재시도(=미확보) 대상: {total_missing}개")

    if args.max_attempts is not None and args.max_attempts < total_missing:
        # 앞에서부터(정렬된 조합 순서) max_attempts개만 채우고 나머지는 다음 실행으로 미룸 -
        # 여러 번 나눠 돌릴 때 --dry-run으로 매번 다시 계산하면 이미 성공한 건 자동으로
        # 빠지므로 이어서 돌리면 됨(체크포인트 별도 관리 불필요).
        capped, budget = {}, args.max_attempts
        for (lm, phi), s_list in sorted(missing_by_combo.items()):
            if budget <= 0:
                break
            take = s_list[:budget]
            capped[(lm, phi)] = take
            budget -= len(take)
        missing_by_combo = capped
        n_selected = sum(len(v) for v in missing_by_combo.values())
        print(f"--max-attempts {args.max_attempts} 적용: 이번엔 {n_selected}개만 시도 "
              f"(나머지 {total_missing - n_selected}개는 다음에 다시 실행)")

    print(f"재시도 대상 조합 수: {len(missing_by_combo)}개 (조합당 누락 s 목록):")
    for (lm, phi), s_list in sorted(missing_by_combo.items()):
        print(f"  L_M={lm}, phi={phi}: s={s_list}")

    if args.dry_run or total_missing == 0:
        return

    print(f"\n=== 재시도 시작 (STABILIZE={args.stabilize}, threads={args.threads}) ===")
    t0 = time.time()
    for i, ((lm, phi), s_list) in enumerate(sorted(missing_by_combo.items()), 1):
        tag = tag_of(lm, phi)
        s_arg = ",".join(str(s) for s in s_list)
        print(f"[{i}/{len(missing_by_combo)}] L_M={lm}, phi={phi}, s={s_arg} ...", flush=True)
        log_path = os.path.join(HERE, f"retry_highphi_{tag}.log")
        with open(log_path, "w", encoding="utf-8") as logf:
            subprocess.run(
                [sys.executable, "-u", os.path.join(HERE, "sweep_lm_phi_position_matv2_worker.py"),
                 "--L_M", str(lm), "--phi", str(phi), "--beta", "0", "--tag", tag,
                 "--threads", str(args.threads), "--s_list", s_arg,
                 "--stabilize_value", str(args.stabilize)],
                stdout=logf, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace",
            )
    print(f"=== 재시도 완료 ({(time.time() - t0) / 60:.1f}분) ===")

    print("=== 기존 fea_lm_phi_pos_matv2_all.json에 병합 ===")
    rows = json.load(open(OUT_PATH, encoding="utf-8")) if os.path.exists(OUT_PATH) else []
    before = len(rows)
    for f in sorted(glob.glob(os.path.join(FEA_DATA_DIR, "fea_lm_phi_pos_matv2_RETRY*.json"))):
        rows.extend(json.load(open(f, encoding="utf-8")))
    rows.sort(key=lambda r: (r["beta_deg"], r["L_M_mm"], r["phi_deg"], r["contact_s_mm"]))
    json.dump(rows, open(OUT_PATH, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"병합 완료: {before}개 -> {len(rows)}개 (+{len(rows) - before}, 시도 {total_missing}개 중) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
