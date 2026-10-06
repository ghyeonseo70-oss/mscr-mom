"""2026-10-06(43번 예정): 42번의 "다음 세션에서 할 일" 1번+2번을 합쳐서 실행하는 파일럿.
1. s 경계 찾기(1-a): L_M=12.5,phi=-120 고정, depth=0.30mm, s=40~90mm을 훑어서 s=30(성공)과
   s=80(4전4패) 사이 어디서부터 실패하는지 확인.
2. depth=0.40mm 전반적 성공률 파일럿(2번): 지금까지 검증이 거의 s=30 한 점 위주였으므로,
   다양한 L_M/phi 조합 x 안전한 s(<=50mm, 팁근처 제외)로 10케이스 실행해서 0.40mm가
   일반적으로도 쓸만한지 확인.
결과는 별도 all.json 병합 없이(실험용) 로그로만 남기고, 성공한 케이스는 CASES 리스트에
태그가 있으니 필요하면 나중에 sweep_lm_phi_position_matv2_worker.py로 재실행해 정식
데이터로 편입 가능.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# (1) s 경계 찾기: L_M=12.5, phi=-120, depth=0.30 고정
BOUNDARY_CASES = [
    dict(L_M=12.5, phi=-120.0, s=40.0, depth=0.30, tag="PILOT_BND_s40_d30"),
    dict(L_M=12.5, phi=-120.0, s=50.0, depth=0.30, tag="PILOT_BND_s50_d30"),
    dict(L_M=12.5, phi=-120.0, s=60.0, depth=0.30, tag="PILOT_BND_s60_d30"),
    dict(L_M=12.5, phi=-120.0, s=70.0, depth=0.30, tag="PILOT_BND_s70_d30"),
    dict(L_M=12.5, phi=-120.0, s=80.0, depth=0.30, tag="PILOT_BND_s80_d30"),
    dict(L_M=12.5, phi=-120.0, s=90.0, depth=0.30, tag="PILOT_BND_s90_d30"),
]

# (2) depth=0.40mm 일반 성공률 파일럿: 다양한 L_M/phi, s<=50(안전구간)
GENERAL_CASES = [
    dict(L_M=0.0, phi=0.0, s=20.0, depth=0.40, tag="PILOT_GEN_LM0_phi0_s20"),
    dict(L_M=0.0, phi=60.0, s=30.0, depth=0.40, tag="PILOT_GEN_LM0_phi60_s30"),
    dict(L_M=0.0, phi=-90.0, s=40.0, depth=0.40, tag="PILOT_GEN_LM0_phiN90_s40"),
    dict(L_M=25.0, phi=30.0, s=20.0, depth=0.40, tag="PILOT_GEN_LM25_phi30_s20"),
    dict(L_M=25.0, phi=-150.0, s=50.0, depth=0.40, tag="PILOT_GEN_LM25_phiN150_s50"),
    dict(L_M=50.0, phi=90.0, s=30.0, depth=0.40, tag="PILOT_GEN_LM50_phi90_s30"),
    dict(L_M=50.0, phi=-30.0, s=40.0, depth=0.40, tag="PILOT_GEN_LM50_phiN30_s40"),
    dict(L_M=75.0, phi=120.0, s=20.0, depth=0.40, tag="PILOT_GEN_LM75_phi120_s20"),
    dict(L_M=75.0, phi=-60.0, s=50.0, depth=0.40, tag="PILOT_GEN_LM75_phiN60_s50"),
    dict(L_M=87.5, phi=0.0, s=30.0, depth=0.40, tag="PILOT_GEN_LM87p5_phi0_s30"),
]

ALL_CASES = BOUNDARY_CASES + GENERAL_CASES

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["boundary", "general", "all"], default="all")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    cases = {"boundary": BOUNDARY_CASES, "general": GENERAL_CASES, "all": ALL_CASES}[args.only]
    print(f"=== 파일럿 {len(cases)}케이스 (threads={args.threads}) ===", flush=True)
    for c in cases:
        print(f"{c['L_M']} {c['phi']} {c['s']} {c['depth']} {c['tag']}")
