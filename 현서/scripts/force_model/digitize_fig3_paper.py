"""Fig.3(e)-(h) 전체 패널에서 축 라벨 텍스트 블롭으로 픽셀<->데이터 캘리브레이션을 구하고,
6가지 phi색상 마커의 중심좌표를 검출해 (LM/L, phi) -> (yL_cm, thetaL_deg) 테이블을 만든다."""
import json
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = Image.open("../../data/force_model/fig_embed_0.jpeg").convert("RGB")
W, H = SRC.size

COLORS = {
    0:   (0, 114, 189),
    30:  (217, 83, 25),
    60:  (237, 177, 32),
    90:  (126, 47, 142),
    120: (119, 172, 48),
    150: (77, 190, 238),
}

# 좌측 축 세로선(spine)의 절대 x픽셀 위치(연속run 검출로 실측, _find_spine.py 결과)
SPINES = {0.0: 182, 0.25: 1239, 0.5: 2293, 0.75: 3359}

# spine 기준으로 모든 패널에 동일한 상대 배치 적용: spine이 crop 내 rel=150에 오도록
PANEL_CROPS = {ratio: (sp - 150, 850, sp + 900, 1761) for ratio, sp in SPINES.items()}

Y_VALUES = [150, 100, 50, 0, -50, -100, -150]
X_VALUES = [-10, -5, 0, 5, 10]


def blobs(mask, min_size, dilate=8):
    d = ndimage.binary_dilation(mask, iterations=dilate)
    labeled, n = ndimage.label(d)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        if len(ys) < min_size:
            continue
        out.append((ys.mean(), xs.mean(), len(ys)))
    return out


def calibrate_axes(arr):
    """crop 배열(H,W,3) 안에서 y축(좌측 0~85열), x축(하단 750~800행) 라벨을 찾아 선형변환 계수 반환."""
    gray = arr.mean(axis=2)
    dark = gray < 140

    # y축: spine(rel=150) 기준 좌측(20~130) 스트립. title 텍스트 섞여 나오면 마지막 7개만 채택
    sub_y = dark[90:730, 20:130]
    cy = blobs(sub_y, min_size=15)
    cy.sort(key=lambda c: c[0])
    if len(cy) > len(Y_VALUES):
        cy = cy[-len(Y_VALUES):]
    assert len(cy) == len(Y_VALUES), f"y라벨 개수 불일치: {len(cy)}"
    rows = np.array([c[0] + 90 for c in cy])
    # 선형회귀: value = a*row + b
    a_y, b_y = np.polyfit(rows, Y_VALUES, 1)

    # x축: 하단 스트립(크롭 전체 폭). 큰 이상치(대시선과 겹친 '0') 제외
    sub_x = dark[750:800, :]
    cx = blobs(sub_x, min_size=15)
    cx = [c for c in cx if c[2] < 5000]
    cx.sort(key=lambda c: c[1])
    assert len(cx) == len(X_VALUES), f"x라벨 개수 불일치: {len(cx)} {cx}"
    cols = np.array([c[1] + 0 for c in cx])
    a_x, b_x = np.polyfit(cols, X_VALUES, 1)

    return (a_x, b_x), (a_y, b_y)


def extract_markers(arr, xcal, ycal, exclude_box=None):
    """exclude_box: (row0,row1,col0,col1) - 이 사각형 안의 픽셀은 범례 아이콘으로 간주하고 제외."""
    a_x, b_x = xcal
    a_y, b_y = ycal
    rgb = arr.astype(np.int16)
    found = {}
    for phi_mag, color in COLORS.items():
        dist = np.sqrt(((rgb - np.array(color)) ** 2).sum(axis=2))
        mask = dist < 45
        if exclude_box is not None:
            r0, r1, c0, c1 = exclude_box
            mask[r0:r1, c0:c1] = False
        comps = blobs(mask, min_size=60, dilate=6)
        # 같은 라벨(phi)에 여러 블롭이 잡히면(노이즈) 가장 큰 것만 채택
        candidates = {}
        for row, col, sz in comps:
            yL = a_x * col + b_x
            thL = a_y * row + b_y
            label = 0 if phi_mag == 0 else (phi_mag if thL >= 0 else -phi_mag)
            if label not in candidates or sz > candidates[label][2]:
                candidates[label] = (round(float(yL), 3), round(float(thL), 2), int(sz))
        found.update(candidates)
    return found


results = {}
for ratio, box in PANEL_CROPS.items():
    crop = SRC.crop(box)
    arr = np.array(crop)
    xcal, ycal = calibrate_axes(arr)
    # LM/L=0 패널에만 범례 상자(같은 6색 아이콘)가 있어 마커 검출과 혼동됨 -> 그 영역 제외
    exclude = (300, 760, 480, 1080) if ratio == 0.0 else None
    markers = extract_markers(arr, xcal, ycal, exclude_box=exclude)
    results[ratio] = markers
    print(f"\n=== LM/L={ratio} ===")
    for phi in sorted(markers, key=lambda k: (abs(k), k)):
        yL, thL, sz = markers[phi]
        print(f"  phi={phi:5d}  yL={yL:7.3f} cm  theta_L={thL:8.2f} deg  (blob px={sz})")

with open("../../data/force_model/fig3_digitized.json", "w", encoding="utf-8") as f:
    json.dump({str(k): {str(kk): vv for kk, vv in v.items()} for k, v in results.items()}, f,
               indent=2, ensure_ascii=False)
print("\n저장: ../../data/force_model/fig3_digitized.json")
