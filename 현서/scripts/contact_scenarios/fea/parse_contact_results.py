"""contact_test.dat에서 마지막 증분(전체 눌러넣은 상태)의 반력을 합산해서 총 접촉력을 구함."""
import re

with open("contact_test.dat", encoding="latin-1") as f:
    dat = f.read()

# 가장 마지막 "forces" 블록(전체 변위 다 적용된 시점, time=1.0)만 추출
blocks = dat.split("forces (fx,fy,fz) for set N_BALL_ALL and time")
last_block = blocks[-1]
lines = last_block.splitlines()[1:]  # 첫 줄은 "  0.1000000E+01" 시간 표시

fx_total = fy_total = fz_total = 0.0
n = 0
for line in lines:
    parts = line.split()
    if len(parts) == 4:
        try:
            node_id = int(parts[0])
            fx, fy, fz = float(parts[1]), float(parts[2]), float(parts[3])
            fx_total += fx
            fy_total += fy
            fz_total += fz
            n += 1
        except ValueError:
            break  # 숫자 아닌 줄 나오면 블록 끝

print(f"집계한 절점 수: {n}")
print(f"총 반력(구 전체 절점 합) Fx={fx_total:.6f} N, Fy={fy_total:.6f} N, Fz={fz_total:.6f} N")
print(f"-> 이게 실리콘관을 누르는 실제 접촉력(뉴턴 제3법칙): 약 {abs(fx_total)*1000:.3f} mN")

# 변위도 마지막 증분에서 확인 (프리스크라이브한 대로 -0.15mm 다 들어갔는지)
disp_blocks = dat.split("displacements (vx,vy,vz) for set N_BALL_ALL and time")
last_disp = disp_blocks[-1].splitlines()[1:]
ux_vals = []
for line in last_disp:
    parts = line.split()
    if len(parts) == 4:
        try:
            ux_vals.append(float(parts[1]))
        except ValueError:
            break
if ux_vals:
    print(f"\n구 절점들의 x변위 평균: {sum(ux_vals)/len(ux_vals):.5f} mm (목표: -0.15000 mm)")
