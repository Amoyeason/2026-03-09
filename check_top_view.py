import numpy as np
import pyvista as pv
import os
from scipy.spatial import KDTree

# ================= 配置 =================
DATA_DIR = "E:\\ansys_data_final"
FILE_PATH = os.path.join(DATA_DIR, "digital_twin_data.npz")

# 参数 (保持一致)
STEP_SIZE = 0.2
MARGIN = 0.08
FIXTURE_RADIUS = 0.06


def generate_comparison(nodes):
    print(f"🔄 生成对比数据...")

    # 1. 生成理想的均匀网格 (Ideal Grid)
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()

    xs = np.arange(x_min + MARGIN, x_max - MARGIN, STEP_SIZE)
    ys = np.arange(y_min + MARGIN, y_max - MARGIN, STEP_SIZE)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # 理想网格点 (Z轴设为模型平均高度，方便显示)
    z_avg = np.mean(nodes[:, 2])
    ideal_points = np.column_stack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, z_avg)])

    # 2. 生成实际吸附点 (Snapped Candidates)
    tree_2d = KDTree(nodes[:, :2])
    actual_candidates = []
    valid_ideal_indices = []  # 记录哪些理想点是有效的（没掉进坑里）

    for i, (gx, gy, _) in enumerate(ideal_points):
        dist, node_idx = tree_2d.query([gx, gy])

        # 只有未落空的点才算数
        if dist <= 0.05:
            actual_candidates.append(nodes[node_idx])
            valid_ideal_indices.append(i)

    return ideal_points[valid_ideal_indices], np.array(actual_candidates)


def main():
    if not os.path.exists(FILE_PATH):
        print("❌ 文件不存在")
        return

    data = np.load(FILE_PATH)
    nodes = data['nodes']

    # 生成对比数据
    ideal_pts, actual_pts = generate_comparison(nodes)

    print("\n🎨 启动正交俯视图 (Orthographic Top-Down View)...")
    pl = pv.Plotter()
    pl.set_background('white')

    # 1. 绘制模型轮廓 (灰色背景)
    pl.add_mesh(pv.PolyData(nodes), color="green", point_size=2,
                render_points_as_spheres=True, opacity=0.5)

    # 2. 绘制 理想网格 (蓝色十字)
    # 这代表了我们原本想放的位置
    pl.add_mesh(pv.PolyData(ideal_pts), color="blue", point_size=12,
                render_points_as_spheres=False, style='points', label="Ideal Grid (200mm)")

    # 3. 绘制 实际夹具 (红色空心圈/球)
    # 这代表了吸附到节点后的位置
    pl.add_mesh(pv.PolyData(actual_pts), color="red", point_size=15,
                render_points_as_spheres=True, opacity=0.6, label="Snapped Candidate")

    # 4. 绘制连线 (显示偏移量)
    # 画一条线连接理想点和实际点，让你看清楚偏移了多少
    lines = []
    for p_ideal, p_actual in zip(ideal_pts, actual_pts):
        lines.append(pv.Line(p_ideal, p_actual))

    if lines:
        pl.add_mesh(lines[0].merge(lines[1:]), color="black", line_width=1)

    # 5. 设置正交视图 (关键!)
    pl.enable_parallel_projection()  # 开启平行投影，消除近大远小
    pl.view_xy()  # 强制俯视
    pl.add_legend(bcolor='white', size=(0.2, 0.2))
    pl.add_text("Top-Down View: Ideal Grid vs Actual Node Snap", color='black')

    print("👉 窗口已弹出。")
    print("   - 🔵 蓝点: 完美的 200mm 均匀网格")
    print("   - 🔴 红点: 实际吸附到的物理节点")
    print("   - ➖ 黑线: 吸附偏移路径")
    pl.show()


if __name__ == "__main__":
    main()