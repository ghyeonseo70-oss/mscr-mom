"""2026-09-21: 두 컴퓨터가 동시에 돌린 push_depth 스윕의 git 충돌 해결.

원인: `sweep_push_depth_matv2.py`의 태그가 (L_M, phi, depth)만 담고 **s를 안 담아서**,
조합 자체는 안 겹치는데(--config-start로 분담해서 s가 서로 다름) 파일 이름만 같아짐
-> git이 add/add 충돌로 봄. 각 파일의 양쪽 버전은 서로 다른 s 행을 담고 있으므로
**둘을 합치는 게 정답**(어느 한쪽을 버리면 그쪽 FEA 결과가 통째로 날아감).

이 스크립트는 충돌난 DEPTH*.json 각각에 대해 git의 양쪽 스테이지(:2=ours, :3=theirs)를
읽어 (L_M,phi,beta,s,depth) 키로 합집합을 만들어 쓴다. all.json은 여기서 건드리지 않고,
이후 `sweep_push_depth_matv2.py --merge-only`로 재구성하면 됨.
"""
import json
import os
import subprocess
import sys

# 한글 경로 때문에 cp949 기본 디코딩이 깨지므로 encoding 명시(ccx 때와 같은 문제).
REPO = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                       encoding="utf-8", errors="replace").stdout.strip()


def key_of(r):
    return (round(r["L_M_mm"], 1), round(r["phi_deg"], 1), round(r["beta_deg"], 1),
            round(r["contact_s_mm"], 1), round(r.get("push_depth_mm", 0.1), 3))


def read_stage(stage, path):
    """git 충돌 중인 파일의 특정 스테이지(2=ours, 3=theirs)를 읽음."""
    out = subprocess.run(["git", "show", f":{stage}:{path}"],
                          capture_output=True, cwd=REPO)
    if out.returncode != 0:
        return []
    return json.loads(out.stdout.decode("utf-8"))


conflicted = subprocess.run(["git", "-c", "core.quotepath=false", "diff", "--name-only",
                              "--diff-filter=U"], capture_output=True, cwd=REPO,
                             encoding="utf-8", errors="replace").stdout.split("\n")
targets = [p for p in conflicted if p.strip() and "_DEPTH" in p and p.endswith(".json")]

print(f"충돌난 DEPTH 파일 {len(targets)}개 합치는 중...\n")
total_ours = total_theirs = total_merged = 0
for path in targets:
    ours, theirs = read_stage(2, path), read_stage(3, path)
    merged = {}
    for r in ours + theirs:
        merged[key_of(r)] = r
    rows = sorted(merged.values(), key=lambda r: r["contact_s_mm"])
    full = os.path.join(REPO, path)
    with open(full, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    total_ours += len(ours)
    total_theirs += len(theirs)
    total_merged += len(rows)
    print(f"  {os.path.basename(path):52s} 내쪽 {len(ours)}행 + 저쪽 {len(theirs)}행 -> {len(rows)}행")
    subprocess.run(["git", "add", path], cwd=REPO)

print(f"\n합계: 내쪽 {total_ours}행 + 저쪽 {total_theirs}행 -> {total_merged}행 "
      f"(중복 {total_ours + total_theirs - total_merged}행 제거)")
print("\n다음: sweep_push_depth_matv2.py --merge-only 로 all.json 재구성")
