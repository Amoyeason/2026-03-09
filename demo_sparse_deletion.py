import numpy as np
import pyvista as pv
import os
from scipy.spatial import KDTree

# ================= 配置区域 =================
DATA_DIR = "E:\\ansys_data_final"
FILE_PATH = os.path.join(DATA_DIR, "digital_twin_data.npz")

STEP_SIZE = 0.25
MARGIN = 0.08
SAFETY_CHECK_RADIUS = 0.07

# 🟢 [核心] 双重筛选阈值
# 1. 角度阈值 (收紧到 150度)
MAX_ANGLE_GAP = 150
# 2. 重心偏移阈值 (超过 15mm 说明重心偏了 -> 边缘)
MAX_CENTROID_OFFSET = 0.015


def estimate_mesh_density(nodes):
    sample = nodes[np.random.choice(len(nodes), min(1000, len(nodes)), replace=False)]
    tree = KDTree(nodes)
    dists, _ = tree.query(sample, k=2)
    avg_spacing = np.mean(dists[:, 1])
    print(f"   📏 [环境检测] 当前网格平均间距: {avg_spacing * 1000:.1f} mm")
    return avg_spacing


def check_safety(center, tree_3d):
    """
    双重检测：角度包围 + 重心偏移
    """
    # 1. 获取邻居
    indices = tree_3d.query_ball_point(center, SAFETY_CHECK_RADIUS)

    # 点太少直接删
    if len(indices) < 4:
        return False, "Too few nodes"

    neighbors = tree_3d.data[indices]

    # ----------------------------------------
    # A. 重心偏移检测 (Centroid Check) - 对稀疏网格极强
    # ----------------------------------------
    centroid = np.mean(neighbors, axis=0)
    # 只看 XY 平面的偏移
    offset_dist = np.linalg.norm(centroid[:2] - center[:2])

    if offset_dist > MAX_CENTROID_OFFSET:
        return False, f"Centroid Shift {offset_dist * 1000:.1f}mm"

    # ----------------------------------------
    # B. 角度包围检测 (Topology Check)
    # ----------------------------------------
    dx = neighbors[:, 0] - center[0]
    dy = neighbors[:, 1] - center[1]
    angles = np.arctan2(dy, dx)
    angles.sort()

    diffs = np.diff(angles)
    last_gap = (angles[0] + 2 * np.pi) - angles[-1]
    max_gap_rad = max(np.max(diffs) if len(diffs) > 0 else 0, last_gap)
    max_gap_deg = np.rad2deg(max_gap_rad)

    if max_gap_deg > MAX_ANGLE_GAP:
        return False, f"Angle Gap {max_gap_deg:.0f}°"

    return True, "OK"


def generate_strict_layout(nodes):
    print(f"🔄 V12 双重筛选 (角度<{MAX_ANGLE_GAP}°, 重心<{MAX_CENTROID_OFFSET * 1000}mm)...")
    estimate_mesh_density(nodes)

    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()

    xs = np.arange(x_min + MARGIN, x_max - MARGIN, STEP_SIZE)
    ys = np.arange(y_min + MARGIN, y_max - MARGIN, STEP_SIZE)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    tree_2d = KDTree(nodes[:, :2])
    tree_3d = KDTree(nodes)

    valid_candidates = []
    rejected_candidates = []

    print(f"   🛡️ 开始筛选...")

    for gx, gy in flat_grid:
        dist, node_idx = tree_2d.query([gx, gy])
        if dist > 0.10: continue
        candidate_node = nodes[node_idx]

        # 核心判定
        safe, reason = check_safety(candidate_node, tree_3d)

        if safe:
            valid_candidates.append(candidate_node)
        else:
            # 记录下来画图
            rejected_candidates.append(candidate_node)

    valid_candidates = np.array(valid_candidates)
    if len(valid_candidates) > 0:
        valid_candidates = np.unique(valid_candidates, axis=0)

    print(f"   ✅ 最终保留: {len(valid_candidates)} 个")
    print(f"   🗑️ 严格剔除: {len(rejected_candidates)} 个")

    return valid_candidates, rejected_candidates


def main():
    if not os.path.exists(FILE_PATH):
        print("❌ 文件不存在！")
        return

    data = np.load(FILE_PATH)
    nodes = data['nodes']

    valid_pts, rejected_pts = generate_strict_layout(nodes)
    if len(valid_pts) == 0: return

    # ================= 可视化 =================
    print("\n🎨 启动 V12 视图...")
    pl = pv.Plotter()
    pl.set_background('white')

    pl.add_mesh(pv.PolyData(nodes), color="#333333", point_size=2,
                opacity=0.3, render_points_as_spheres=True, label="Mesh")

    # 绘制保留点
    spheres_ok = [pv.Sphere(radius=0.06, center=c) for c in valid_pts]
    if spheres_ok:
        pl.add_mesh(spheres_ok[0].merge(spheres_ok[1:]), color="green", opacity=0.6,
                    smooth_shading=True, label="Valid (Balanced)")

    # 绘制剔除点 (用红色显示)
    if len(rejected_pts) > 0:
        spheres_bad = [pv.Sphere(radius=0.06, center=c) for c in rejected_pts]
        if spheres_bad:
            pl.add_mesh(spheres_bad[0].merge(spheres_bad[1:]), color="red", opacity=0.2,
                        style='wireframe', label="Deleted (Unbalanced)")

    pl.add_legend(bcolor='white', size=(0.2, 0.3))
    pl.add_text("V12 Centroid + Angle Filter", color='black')

    pl.add_axes()
    pl.view_xy()
    print("👉 窗口已弹出。请检查红色点是否把所有边缘悬空都抓出来了。")
    pl.show()


if __name__ == "__main__":
    main()