# import numpy as np
# import pyvista as pv
# import os
# from scipy.spatial import KDTree
#
# # ================= 配置区域 =================
# DATA_DIR = "E:\\ansys_data_final"
# FILE_PATH = os.path.join(DATA_DIR, "digital_twin_data.npz")
#
# # 🟢 关键参数
# STEP_SIZE = 0.25  # 候选点间距
# MARGIN = 0.08  # 边缘避让
# FIXTURE_RADIUS = 0.06  # 吸盘半径 60mm
#
# # 🟢 [升级] 悬空容差 (收紧到 10mm)
# MAX_EDGE_DIST = 0.01
#
#
# def generate_candidates(nodes):
#     print(f"🔄 计算候选点 (间距={STEP_SIZE}m, 半径={FIXTURE_RADIUS}m)...")
#
#     x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
#     y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
#
#     xs = np.arange(x_min + MARGIN, x_max - MARGIN, STEP_SIZE)
#     ys = np.arange(y_min + MARGIN, y_max - MARGIN, STEP_SIZE)
#     grid_x, grid_y = np.meshgrid(xs, ys)
#     flat_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
#
#     tree_2d = KDTree(nodes[:, :2])
#
#     valid_candidates = []
#     leaking_candidates = []
#
#     # 🟢 [升级] 12点雷达检测 (Clock-face Check)
#     # 生成 0 到 360 度，每隔 30 度一个检测点
#     angles = np.linspace(0, 2 * np.pi, 13)[:-1]  # 12个点
#     offsets = []
#     for ang in angles:
#         dx = FIXTURE_RADIUS * np.cos(ang)
#         dy = FIXTURE_RADIUS * np.sin(ang)
#         offsets.append((dx, dy))
#
#     print(f"   🛡️ 启用 12 点全周扫描模式...")
#
#     for gx, gy in flat_grid:
#         # A. 检查圆心
#         dist, node_idx = tree_2d.query([gx, gy])
#         if dist > 0.05: continue  # 落入空洞
#
#         center_node = nodes[node_idx]
#
#         # B. 全周扫描
#         is_leaking = False
#         for dx, dy in offsets:
#             check_x = gx + dx
#             check_y = gy + dy
#             # 查询圆周点距离最近节点的距离
#             dist_edge, _ = tree_2d.query([check_x, check_y])
#
#             # 只要有一个点悬空，整个夹具判定为漏气
#             if dist_edge > MAX_EDGE_DIST:
#                 is_leaking = True
#                 break
#
#         if is_leaking:
#             leaking_candidates.append(center_node)
#         else:
#             valid_candidates.append(center_node)
#
#     print(f"   ✅ 有效候选点: {len(valid_candidates)} 个")
#     print(f"   ⚠️ 漏气候选点: {len(leaking_candidates)} 个 (已严格剔除)")
#
#     return np.array(valid_candidates), np.array(leaking_candidates)
#
#
# def main():
#     print(f"📂 读取数据: {FILE_PATH} ...")
#     if not os.path.exists(FILE_PATH):
#         print("❌ 文件不存在！")
#         return
#
#     data = np.load(FILE_PATH)
#     nodes = data['nodes']
#
#     # 1. 生成候选点
#     valid_candidates, leaking_candidates = generate_candidates(nodes)
#     if len(valid_candidates) == 0:
#         print("❌ 没有生成有效候选点！")
#         return
#
#     # 2. 计算覆盖率
#     print("🔍 计算节点覆盖率...")
#     tree_3d = KDTree(nodes)
#     if len(valid_candidates) > 0:
#         indices_list = tree_3d.query_ball_point(valid_candidates, FIXTURE_RADIUS)
#         covered_mask = np.zeros(len(nodes), dtype=bool)
#         for idxs in indices_list:
#             covered_mask[idxs] = True
#     else:
#         covered_mask = np.zeros(len(nodes), dtype=bool)
#
#     covered_count = np.sum(covered_mask)
#     print(f"   📊 有效覆盖统计: {covered_count}/{len(nodes)} 节点 ({covered_count / len(nodes) * 100:.1f}%)")
#
#     # ================= 3. 可视化 =================
#     print("\n🎨 启动高严苛检测视图...")
#     pl = pv.Plotter()
#     pl.set_background('white')
#
#     # A. 背景节点
#     uncovered_nodes = nodes[~covered_mask]
#     if len(uncovered_nodes) > 0:
#         pl.add_mesh(pv.PolyData(uncovered_nodes), color="#333333", point_size=3,
#                     render_points_as_spheres=True, opacity=0.3)
#
#     # B. 有效区
#     covered_nodes = nodes[covered_mask]
#     if len(covered_nodes) > 0:
#         pl.add_mesh(pv.PolyData(covered_nodes), color="#00FF00", point_size=4,
#                     render_points_as_spheres=True, label="Supported Area")
#
#     # C. 漏气夹具 (红色)
#     if len(leaking_candidates) > 0:
#         spheres_leak = [pv.Sphere(radius=FIXTURE_RADIUS, center=c) for c in leaking_candidates]
#         if spheres_leak:
#             combined_leak = spheres_leak[0].merge(spheres_leak[1:])
#             pl.add_mesh(combined_leak, color="red", opacity=0.6,
#                         smooth_shading=True, label="Strictly Rejected")
#         pl.add_mesh(pv.PolyData(leaking_candidates), color="red", point_size=10,
#                     render_points_as_spheres=True)
#
#     # D. 有效夹具 (绿色)
#     if len(valid_candidates) > 0:
#         spheres_valid = [pv.Sphere(radius=FIXTURE_RADIUS, center=c) for c in valid_candidates]
#         if spheres_valid:
#             combined_valid = spheres_valid[0].merge(spheres_valid[1:])
#             pl.add_mesh(combined_valid, color="green", opacity=0.4,
#                         smooth_shading=True, label="Valid Fixtures")
#         pl.add_mesh(pv.PolyData(valid_candidates), color="black", point_size=8,
#                     render_points_as_spheres=True)
#
#     # 基准点 (保持不变)
#     x_min, y_min, z_avg = nodes[:, 0].min(), nodes[:, 1].min(), nodes[:, 2].mean()
#     pl.add_mesh(pv.Sphere(radius=0.05, center=[0, 0, 0]), color="black", label="Origin")
#     pl.add_mesh(pv.Sphere(radius=0.05, center=[x_min, y_min, z_avg]), color="orange", label="Corner")
#     pl.add_mesh(pv.Sphere(radius=0.06, center=[x_min + MARGIN, y_min + MARGIN, z_avg]), color="cyan", label="Start")
#
#     pl.add_legend(bcolor='white', size=(0.2, 0.3))
#     pl.add_text(f"Strict Check (12-Point Radar)\nTolerance: {MAX_EDGE_DIST * 1000}mm", color='black')
#
#     pl.add_axes()
#     pl.view_xy()
#     print("👉 窗口已弹出。这次应该非常干净了。")
#     pl.show()
#
#
# if __name__ == "__main__":
#     main()
import numpy as np
import pyvista as pv
import os
from scipy.spatial import KDTree

# ================= 配置区域 =================
DATA_DIR = "E:\\ansys_data_final"
FILE_PATH = os.path.join(DATA_DIR, "digital_twin_data.npz")

# 🟢 基础参数
STEP_SIZE = 0.25
MARGIN = 0.08
FIXTURE_RADIUS = 0.06  # 吸盘物理半径 (60mm)
MAX_EDGE_DIST = 0.01  # 悬空容差 (10mm)

# 🟢 [关键升级] 救援标准
# 只有当一个点周围 SAFETY_CHECK_RADIUS 范围内都有节点时，才算救援成功。
# 70mm = 60mm(吸盘) + 10mm(安全间隙)
SAFETY_CHECK_RADIUS = 0.07

# 搜索范围 (保持大范围搜索)
SEARCH_RADIUS = 0.15


def check_integrity(center, tree_2d, check_radius):
    """
    通用检测函数：判断以 center 为圆心，check_radius 为半径的圆是否悬空
    """
    # 1. 检查圆心
    dist, _ = tree_2d.query(center[:2])
    if dist > 0.05: return False

    # 2. 检查圆周 (12点雷达)
    angles = np.linspace(0, 2 * np.pi, 13)[:-1]
    for ang in angles:
        # 使用传入的 check_radius 进行检测
        check_x = center[0] + check_radius * np.cos(ang)
        check_y = center[1] + check_radius * np.sin(ang)
        dist_edge, _ = tree_2d.query([check_x, check_y])

        # 如果边缘点距离最近节点太远，说明这个半径的圆盖不住实体
        if dist_edge > MAX_EDGE_DIST:
            return False
    return True


def generate_and_fix(nodes):
    print(f"🔄 V9 带安全边距救援 (物理半径: 60mm, 安全验收半径: {SAFETY_CHECK_RADIUS * 1000}mm)...")

    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()

    xs = np.arange(x_min + MARGIN, x_max - MARGIN, STEP_SIZE)
    ys = np.arange(y_min + MARGIN, y_max - MARGIN, STEP_SIZE)
    grid_x, grid_y = np.meshgrid(xs, ys)
    flat_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    tree_2d = KDTree(nodes[:, :2])
    tree_3d = KDTree(nodes)

    final_candidates = []
    fixed_history = []
    failed_points = []

    print(f"   🛡️ 开始检测与修复...")

    for gx, gy in flat_grid:
        dist, node_idx = tree_2d.query([gx, gy])
        if dist > 0.05: continue
        current_node = nodes[node_idx]

        # 1. 自身体检
        # 初始点我们要求它满足"安全半径" (70mm)，而不仅仅是物理半径
        if check_integrity(current_node, tree_2d, SAFETY_CHECK_RADIUS):
            final_candidates.append(current_node)
            continue

        # 2. 广域救援 (目标：找一个满足 SAFETY_CHECK_RADIUS 的点)
        neighbor_indices = tree_3d.query_ball_point(current_node, SEARCH_RADIUS)

        if not neighbor_indices:
            failed_points.append(current_node)
            continue

        neighbors = nodes[neighbor_indices]
        dists = np.linalg.norm(neighbors - current_node, axis=1)
        sorted_indices = np.argsort(dists)

        found_fix = False
        for idx in sorted_indices:
            candidate = neighbors[idx]
            if dists[idx] < 1e-6: continue

            # 🟢 [核心修改] 救援条件升级
            # 我们不只是查 60mm，而是查 70mm。
            # 只有当候选点能通过 70mm 的测试时，才选用它。
            # 这样就保证了它离边缘至少有 10mm 的余量。
            if check_integrity(candidate, tree_2d, SAFETY_CHECK_RADIUS):
                final_candidates.append(candidate)
                fixed_history.append((current_node, candidate))
                found_fix = True
                break

        if not found_fix:
            # 如果实在找不到满足 70mm 的，可以考虑降级策略 (比如只满足60mm)，
            # 但为了严格执行您的要求，这里我们选择"宁缺毋滥"，直接剔除。
            failed_points.append(current_node)

    # 去重
    final_candidates = np.array(final_candidates)
    if len(final_candidates) > 0:
        final_candidates = np.unique(final_candidates, axis=0)

    print(f"   ✅ 最终有效动作空间: {len(final_candidates)} 个")
    print(f"   🔧 成功抢救: {len(fixed_history)} 个")
    print(f"   🗑️ 无法满足安全间隙: {len(failed_points)} 个")

    return final_candidates, fixed_history, failed_points


def main():
    if not os.path.exists(FILE_PATH):
        print("❌ 文件不存在！")
        return

    data = np.load(FILE_PATH)
    nodes = data['nodes']

    candidates, history, failed_pts = generate_and_fix(nodes)
    if len(candidates) == 0: return

    # ================= 可视化 =================
    print("\n🎨 启动 V9 安全边距视图...")
    pl = pv.Plotter()
    pl.set_background('white')

    pl.add_mesh(pv.PolyData(nodes), color="#F0F0F0", opacity=0.6, label="Part Surface")

    # 1. 绘制最终点
    spheres = [pv.Sphere(radius=FIXTURE_RADIUS, center=c) for c in candidates]
    if spheres:
        pl.add_mesh(spheres[0].merge(spheres[1:]), color="green", opacity=0.6,
                    smooth_shading=True, label="Safe Candidates (Margin OK)")
    pl.add_mesh(pv.PolyData(candidates), color="black", point_size=5)

    # 2. 绘制修复路径
    if len(history) > 0:
        arrows = []
        old_positions = []
        for old_p, new_p in history:
            old_positions.append(old_p)
            vec = new_p - old_p
            mag = np.linalg.norm(vec)
            if mag > 1e-5:
                arrows.append(pv.Arrow(start=old_p, direction=vec, scale=mag))

        if old_positions:
            spheres_bad = [pv.Sphere(radius=FIXTURE_RADIUS, center=c) for c in old_positions]
            pl.add_mesh(spheres_bad[0].merge(spheres_bad[1:]), color="red", opacity=0.2,
                        style='wireframe')
        if arrows:
            combined = arrows[0]
            for i in range(1, len(arrows)):
                combined = combined.merge(arrows[i])
            pl.add_mesh(combined, color="blue", label="Deep Rescue Vector")

    # 3. 失败点
    if len(failed_pts) > 0:
        pl.add_mesh(pv.PolyData(failed_pts), color="black", point_size=15,
                    render_points_as_spheres=True, label="Rejected (Too Narrow)")

    pl.add_legend(bcolor='white', size=(0.2, 0.3))
    pl.add_text(f"V9 Safety Rescue\nCheck Radius: {SAFETY_CHECK_RADIUS * 1000}mm (+10mm Gap)", color='black')

    pl.add_axes()
    pl.view_xy()
    print("👉 窗口已弹出。请观察蓝色箭头，这次它们应该把点拉得更靠里了。")
    pl.show()


if __name__ == "__main__":
    main()