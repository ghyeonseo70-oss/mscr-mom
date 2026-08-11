"""굽은 튜브 메쉬가 force_model 예측 중심선과 잘 맞는지 시각적으로 확인."""
import json
import os
import gmsh
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "bent_centerline.json")) as f:
    cl = json.load(f)

gmsh.initialize()
gmsh.open(os.path.join(HERE, "bent_tube_mesh.msh"))
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
coords = np.array(node_coords).reshape(-1, 3)
gmsh.finalize()

fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.15, color="#5b9bd5", label="FEA 메쉬 절점(표면)")

cx = [p["x"] for p in cl["points"]]
cy = [p["y"] for p in cl["points"]]
ax.plot(cx, cy, "-", color="#e34948", linewidth=2.5, label="force_model 예측 중심선", zorder=5)
ax.plot(0, 0, "ks", markersize=10, label="베이스(고정단)", zorder=6)
ax.plot(cl["x_L"], cl["y_L"], "r*", markersize=18, label="예측 팁 위치", zorder=6)

ax.set_xlabel("x (mm)")
ax.set_ylabel("y (mm)")
ax.set_title(f"굽은 튜브 FEA 메쉬 vs 물리모델 예측 중심선\nL_M={cl['L_M']}mm, phi={cl['phi_deg']}deg",
             fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(loc="best", fontsize=9)

out_path = os.path.join(HERE, "..", "..", "..", "data", "contact_scenarios", "fea", "bent_mesh_check.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"저장: {out_path}")
