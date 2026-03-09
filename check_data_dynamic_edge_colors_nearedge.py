import os
import math
import numpy as np
import pyvista as pv

from scipy.spatial import KDTree
from scipy import ndimage

# =========================
# 配置
# =========================
CAD_PATH = r"E:\ZJU\Learning baogao\112_1.igs"
CACHE_DIR = r"E:\ansys_data_offset_cache"
CACHE_NPZ = os.path.join(CACHE_DIR, "nodes_cache.npz")

EXPORT_DIR = os.path.join(CACHE_DIR, "exports")
EXPORT_CSV_ALL = os.path.join(EXPORT_DIR, "grid_all_2d.csv")
EXPORT_CSV_INSIDE = os.path.join(EXPORT_DIR, "grid_inside_2d.csv")
EXPORT_CSV_OK = os.path.join(EXPORT_DIR, "grid_ok_2d.csv")
EXPORT_CSV_OUTER = os.path.join(EXPORT_DIR, "outer_boundary_2d.csv")
EXPORT_CSV_OFFSET70 = os.path.join(EXPORT_DIR, "offset70mm_2d.csv")
EXPORT_CSV_SUPPORT_MOVED = os.path.join(EXPORT_DIR, "support_moved_all.csv")
EXPORT_CSV_RING_SUPPORTS = os.path.join(EXPORT_DIR, "ring_supports.csv")
EXPORT_CSV_OFFSET_SCAN = os.path.join(EXPORT_DIR, "offset_scan_summary.csv")
EXPORT_SCAN_VIEW_DIR = os.path.join(EXPORT_DIR, "scan_views_625")
EXPORT_BEST_VIEW = os.path.join(EXPORT_DIR, "best_layout_topview.png")

# ANSYS网格参数（仅首次提取点云）
THICKNESS = 0.005
MESH_SIZE = 0.01

# 网格参数
GRID_SPACING_M = 0.25  # 250mm
GRID_ROWS = 5
GRID_COLS = 24

# 夹具参数
FIXTURE_RADIUS_M = 0.0625
SAFETY_MARGIN_M = 0.0

# 动态边缘邻域：在当前孔阵偏移下，薄板边缘附近的 POGO（无论当前在板内还是板外）都允许 XY 移动；内部深处 POGO 只允许升降
EDGE_NEAR_BAND_M = 0.12

# 偏移线
OFFSET_M = 0.07

# 网格原点
GRID_ORIGIN_MODE = "aligned"
GRID_ORIGIN_CUSTOM = (0.0, 0.0)

# ============= 平台孔阵相对位置扫描（dx,dy in [0, spacing)） =============
OFFSET_SCAN_ENABLE = True
# 扫描步长（建议 0.01=10mm；更精细可用 0.005=5mm 或 0.002=2mm）
OFFSET_SCAN_STEP_M = 0.002
# 优化目标： "ok"=最大化原位可吸附孔位数(ok_flag)；"kept"=最大化最终有效支撑数(keep_flag)；"ring"=最大化最终一圈支撑数量
OFFSET_SCAN_OBJECTIVE = "kept"
# 目标相同的情况下的次级排序（从左到右）
OFFSET_SCAN_TIEBREAK = ("kept", "ring", "ok")

# ============= 支撑点离散移动参数 =============
MOVE_DISTANCES_M = [0.0, 0.03, 0.06, 0.09]  # 0/30/60/90mm
MOVE_DIRECTIONS = {
    "R":  (1.0, 0.0),
    "L":  (-1.0, 0.0),
    "U":  (0.0, 1.0),
    "D":  (0.0, -1.0),
    "UR": (1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
    "UL": (-1.0 / math.sqrt(2), 1.0 / math.sqrt(2)),
    "DR": (1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
    "DL": (-1.0 / math.sqrt(2), -1.0 / math.sqrt(2)),
}

# 支撑点可行性约束：支撑也要能放下 R=60mm
SUPPORT_REQUIRE_CLEARANCE = True

# 注：不再用“离 70mm 偏移线太远”作为 keep 的淘汰条件。
# 下面两个参数仅用于在多个可行移动候选里优先选择更靠近 offset70 的位置。
MAX_DIST_TO_OFFSET_M = 0.06  # 仅保留作兼容/记录，不参与 keep 判定
NO_MOVE_IF_WITHIN_M = 0.01  # 原位已贴近 offset70 时优先不动

# 最终“一圈支撑”沿绿线的间距（弧长分箱）
RING_SPACING_ALONG_OFFSET_M = 0.1  # 100mm

# 可视化：为了不太乱，只画“最终一圈支撑”的移动灰线
DRAW_MOVE_SEGMENTS_FOR = "all"  # "all" / "ring" - 显示所有有效点的移动

# 是否显示“每个 ring 点的 move 标签”
SHOW_MOVE_LABELS = True

# 扫描阶段是否保存625张俯视图
SAVE_SCAN_VIEWS = False
# 是否在文件名中写入指标
SCAN_VIEW_FILENAME_WITH_SCORE = True
# 俯视图分辨率
VIEW_IMAGE_SIZE = (1600, 1000)


# =========================
# 1) IGES -> nodes（带缓存）
# =========================
def extract_nodes_from_iges_mapdl(cad_path: str,
                                 work_dir: str,
                                 thickness: float,
                                 mesh_size: float,
                                 do_pca_align: bool = True) -> np.ndarray:
    from ansys.mapdl.core import launch_mapdl
    import numpy as _np

    mapdl = launch_mapdl(
        run_location=work_dir,
        nproc=1,
        additional_switches="-smp",
        override=True,
        cleanup_on_exit=True
    )
    mapdl.ignore_errors = False

    try:
        print("🏗️ MAPDL 导入 IGES ...")
        mapdl.clear()
        mapdl.aux15()
        mapdl.ioptn("GTOL", 0.05)
        mapdl.ioptn("IGES", "SMOOTH")
        mapdl.igesin(cad_path)
        mapdl.finish()
        mapdl.prep7()

        mapdl.allsel()
        xmin = mapdl.get_value("KP", 0, "MNLOC", "X")
        xmax = mapdl.get_value("KP", 0, "MXLOC", "X")
        raw_len = xmax - xmin
        print(f"📏 原始长度: {raw_len:.3f}")
        if raw_len > 10.0:
            print("⚠️ 疑似 mm，缩放到 m (x0.001)")
            mapdl.arscale("ALL", "", "", 0.001, 0.001, 0.001, "", "", 1)

        print("🕸️ 网格划分 ...")
        mapdl.et(1, "SHELL181")
        mapdl.sectype(1, "SHELL")
        mapdl.secdata(thickness)
        mapdl.esize(mesh_size)
        mapdl.amesh("ALL")

        nodes = _np.asarray(mapdl.mesh.nodes, dtype=float)
        print(f"✅ 节点数: {len(nodes)}")

    finally:
        mapdl.exit()

    # PCA对齐：你现在的“建系”就在这里（PCA坐标系 + 平移到最小为0）
    if do_pca_align:
        center = nodes.mean(axis=0)
        centered = nodes - center
        cov = np.cov(centered.T)
        _, eig_vecs = np.linalg.eigh(cov)
        new_z = eig_vecs[:, 0]
        new_x = eig_vecs[:, 2]
        new_y = np.cross(new_z, new_x)
        R = np.column_stack([new_x, new_y, new_z])
        nodes = centered @ R
        nodes = nodes - nodes.min(axis=0)

    return nodes


def extract_or_load_nodes(force_rebuild: bool = False, do_pca_align: bool = True) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)

    if (not force_rebuild) and os.path.exists(CACHE_NPZ):
        print(f"📦 从缓存读取: {CACHE_NPZ}")
        data = np.load(CACHE_NPZ)
        return data["nodes"]

    print("🚀 缓存不存在或强制重建，调用 MAPDL 提取节点 ...")
    nodes = extract_nodes_from_iges_mapdl(
        cad_path=CAD_PATH,
        work_dir=CACHE_DIR,
        thickness=THICKNESS,
        mesh_size=MESH_SIZE,
        do_pca_align=do_pca_align
    )
    np.savez_compressed(CACHE_NPZ, nodes=nodes)
    print(f"✅ 已写入缓存: {CACHE_NPZ}")
    return nodes


# =========================
# 2) mask + dist_in
# =========================
def build_part_mask_and_distance(nodes_xyz: np.ndarray,
                                 grid_step_m: float,
                                 inside_radius_m: float,
                                 close_px: int):
    pts2d = nodes_xyz[:, :2]
    tree2d = KDTree(pts2d)

    pad = 3 * grid_step_m
    min_xy = pts2d.min(axis=0) - pad
    max_xy = pts2d.max(axis=0) + pad

    xs = np.arange(min_xy[0], max_xy[0] + grid_step_m, grid_step_m)
    ys = np.arange(min_xy[1], max_xy[1] + grid_step_m, grid_step_m)
    X, Y = np.meshgrid(xs, ys)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    dist, _ = tree2d.query(grid_points, k=1)
    mask = (dist.reshape(X.shape) <= inside_radius_m)

    mask = ndimage.binary_fill_holes(mask)

    structure = np.ones((close_px, close_px))
    mask = ndimage.binary_closing(mask, structure=structure)

    dist_in = ndimage.distance_transform_edt(mask) * grid_step_m
    return mask, dist_in, xs, ys


# =========================
# 2.1) 外轮廓 + 偏移线（等值线）
# =========================
def _extract_longest_contour_from_image(img: np.ndarray, level: float, xs: np.ndarray, ys: np.ndarray):
    try:
        from skimage import measure
        contours = measure.find_contours(img, level=level)
        if not contours:
            return None
        c = max(contours, key=lambda a: a.shape[0])
        rows = c[:, 0]
        cols = c[:, 1]
        x = np.interp(cols, np.arange(len(xs)), xs)
        y = np.interp(rows, np.arange(len(ys)), ys)
        return np.column_stack([x, y])
    except Exception:
        pass

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure()
    cs = plt.contour(X, Y, img, levels=[level])
    best = None
    best_len = 0
    for col in cs.collections:
        for path in col.get_paths():
            v = path.vertices
            if v.shape[0] > best_len:
                best_len = v.shape[0]
                best = v
    plt.close(fig)
    return best.astype(float) if best is not None else None


def compute_outer_and_offset_curves(mask: np.ndarray, dist_in: np.ndarray, xs: np.ndarray, ys: np.ndarray, offset_m: float):
    outer2d = _extract_longest_contour_from_image(mask.astype(float), level=0.5, xs=xs, ys=ys)
    if outer2d is None or len(outer2d) < 30:
        raise ValueError("外轮廓提取失败：mask 断裂或参数不合适。")

    inner2d = _extract_longest_contour_from_image(dist_in, level=offset_m, xs=xs, ys=ys)
    if inner2d is None or len(inner2d) < 30:
        raise ValueError("偏移线提取失败：offset 太大或 mask 太窄/断裂。")

    return outer2d, inner2d


def _downsample_polyline(points2d: np.ndarray, max_points: int = 1500):
    if points2d is None or len(points2d) <= max_points:
        return points2d
    idx = np.linspace(0, len(points2d) - 1, max_points).astype(int)
    return points2d[idx]


# =========================
# 3) 网格点（All/Inside/OK）
# =========================
def generate_grid_points(min_xy, max_xy, spacing, origin_mode="aligned", origin_xy=None, origin_offset_xy=(0.0, 0.0),
                         rows: int = GRID_ROWS, cols: int = GRID_COLS):
    min_x, min_y = float(min_xy[0]), float(min_xy[1])

    if origin_mode == "bbox":
        x0, y0 = min_x, min_y
    elif origin_mode == "aligned":
        x0 = math.floor(min_x / spacing) * spacing
        y0 = math.floor(min_y / spacing) * spacing
    elif origin_mode == "custom":
        if origin_xy is None:
            raise ValueError("origin_mode='custom' 需要 origin_xy=(x0,y0)")
        x0, y0 = float(origin_xy[0]), float(origin_xy[1])
    else:
        raise ValueError("origin_mode 必须是 bbox/aligned/custom")

    dx, dy = float(origin_offset_xy[0]), float(origin_offset_xy[1])
    x0 += dx
    y0 += dy

    gx = x0 + np.arange(cols, dtype=float) * spacing
    gy = y0 + np.arange(rows, dtype=float) * spacing
    Xg, Yg = np.meshgrid(gx, gy)
    grid2d_all = np.column_stack([Xg.ravel(), Yg.ravel()])
    return grid2d_all, (x0, y0), gx, gy


def classify_grid_points(grid2d_all, mask, dist_in, xs, ys,
                         fixture_radius_m=FIXTURE_RADIUS_M,
                         safety_margin_m=SAFETY_MARGIN_M):
    xi = np.clip(np.searchsorted(xs, grid2d_all[:, 0]) - 1, 0, len(xs) - 1)
    yi = np.clip(np.searchsorted(ys, grid2d_all[:, 1]) - 1, 0, len(ys) - 1)

    inside_flag = mask[yi, xi]
    clearance = dist_in[yi, xi]
    need = fixture_radius_m + safety_margin_m
    ok_flag = inside_flag & (clearance >= need)
    return inside_flag, ok_flag, clearance, need



def compute_distance_to_outer(points2d: np.ndarray, outer2d: np.ndarray):
    outer_tree = KDTree(outer2d)
    dist_to_outer, _ = outer_tree.query(points2d, k=1)
    return dist_to_outer


def split_grid_states(inside_flag: np.ndarray,
                      dist_to_outer: np.ndarray,
                      edge_near_band_m: float = EDGE_NEAR_BAND_M):
    """
    状态定义（按用户最新工艺意图）：
      1) movable_candidate：只要靠近薄板外轮廓，不管当前在板内还是板外，都允许 XY 离散移动尝试进入可行位置
      2) inner_fixed：在薄板内部深处，只允许升降，不允许 XY 移动
      3) outside_far：在薄板外且离边界较远，不参与
    """
    near_edge_flag = dist_to_outer <= edge_near_band_m
    movable_candidate_flag = near_edge_flag
    inner_fixed_flag = inside_flag & (~near_edge_flag)
    outside_far_flag = (~inside_flag) & (~near_edge_flag)
    return outside_far_flag, movable_candidate_flag, inner_fixed_flag, near_edge_flag




def process_support_points_dynamic(grid2d_all: np.ndarray,
                                   inside_flag: np.ndarray,
                                   clearance: np.ndarray,
                                   need: float,
                                   offset2d: np.ndarray,
                                   outer2d: np.ndarray,
                                   mask, dist_in, xs, ys,
                                   edge_near_band_m: float = EDGE_NEAR_BAND_M):
    dist_to_outer = compute_distance_to_outer(grid2d_all, outer2d)
    outside_far_flag, movable_candidate_flag, inner_fixed_flag, near_edge_flag = split_grid_states(
        inside_flag=inside_flag,
        dist_to_outer=dist_to_outer,
        edge_near_band_m=edge_near_band_m
    )
    offset_tree = KDTree(offset2d)

    moved2d = np.array(grid2d_all, dtype=float, copy=True)
    keep_flag = np.zeros((len(grid2d_all),), dtype=bool)
    infos = [None] * len(grid2d_all)

    # 1) 边缘邻域：无论当前在板内还是板外，都允许 XY 离散移动
    movable_idx = np.where(movable_candidate_flag)[0]
    if len(movable_idx) > 0:
        moved_cand2d, keep_cand, infos_cand = move_points_towards_offset_with_priority(
            base_points2d=grid2d_all[movable_idx],
            offset_curve2d=offset2d,
            mask=mask, dist_in=dist_in, xs=xs, ys=ys,
            require_clearance=SUPPORT_REQUIRE_CLEARANCE,
            clearance_need=need,
            max_dist_to_offset_m=MAX_DIST_TO_OFFSET_M,
            no_move_if_within_m=NO_MOVE_IF_WITHIN_M,
            distances_m=MOVE_DISTANCES_M
        )
        moved2d[movable_idx] = moved_cand2d
        keep_flag[movable_idx] = keep_cand
        for gi, row in zip(movable_idx, infos_cand):
            row = dict(row)
            row["state"] = "movable_candidate"
            row["dist_to_outer_m"] = float(dist_to_outer[gi])
            row["origin_inside"] = int(bool(inside_flag[gi]))
            infos[gi] = row

    # 2) 内部深处：只允许升降，不允许 XY 移动；只要原位可吸附就保留
    inner_idx = np.where(inner_fixed_flag)[0]
    if len(inner_idx) > 0:
        d0, _ = offset_tree.query(grid2d_all[inner_idx], k=1)
        for local_i, gi in enumerate(inner_idx):
            p = grid2d_all[gi]
            clear0 = float(clearance[gi])
            dist0 = float(d0[local_i])
            feasible0 = bool(inside_flag[gi]) and (clear0 >= need)
            keep = feasible0
            keep_flag[gi] = keep
            infos[gi] = {
                "index": int(gi),
                "x0": float(p[0]), "y0": float(p[1]),
                "x1": float(p[0]), "y1": float(p[1]),
                "dir": "FIXED",
                "move_mm": 0.0,
                "clearance_m": clear0,
                "dist_to_offset_m": dist0,
                "dist_to_outer_m": float(dist_to_outer[gi]),
                "origin_inside": int(bool(inside_flag[gi])),
                "keep": int(keep),
                "reason": "inner_fixed_keep" if keep else "inner_fixed_no_clearance",
                "state": "inner_fixed"
            }

    # 3) 板外且远离边界：不参与
    outside_far_idx = np.where(outside_far_flag)[0]
    if len(outside_far_idx) > 0:
        d0, _ = offset_tree.query(grid2d_all[outside_far_idx], k=1)
        for local_i, gi in enumerate(outside_far_idx):
            p = grid2d_all[gi]
            infos[gi] = {
                "index": int(gi),
                "x0": float(p[0]), "y0": float(p[1]),
                "x1": float(p[0]), "y1": float(p[1]),
                "dir": "OUTSIDE_FAR",
                "move_mm": 0.0,
                "clearance_m": float(clearance[gi]),
                "dist_to_offset_m": float(d0[local_i]),
                "dist_to_outer_m": float(dist_to_outer[gi]),
                "origin_inside": int(bool(inside_flag[gi])),
                "keep": 0,
                "reason": "outside_far_from_part",
                "state": "outside_far"
            }

    infos = [row for row in infos if row is not None]
    orig_ok_flag = inside_flag & (clearance >= need)
    return {
        "outside_far_flag": outside_far_flag,
        "movable_candidate_flag": movable_candidate_flag,
        "inner_fixed_flag": inner_fixed_flag,
        "near_edge_flag": near_edge_flag,
        "dist_to_outer": dist_to_outer,
        "orig_ok_flag": orig_ok_flag,
        "moved2d": moved2d,
        "keep_flag": keep_flag,
        "infos": infos,
    }


def estimate_nn_distance_xy(points_xy: np.ndarray, sample_n: int = 2000) -> float:
    n = len(points_xy)
    if n < 10:
        return 0.01
    idx = np.random.choice(n, size=min(sample_n, n), replace=False)
    pts = points_xy[idx]
    tree = KDTree(points_xy)
    dists, _ = tree.query(pts, k=2)
    nn = dists[:, 1]
    return float(np.median(nn))


def lift_points_to_surface(nodes_xyz: np.ndarray, pts2d: np.ndarray, k: int = 12):
    tree = KDTree(nodes_xyz[:, :2])
    _, idx = tree.query(pts2d, k=k)
    z = nodes_xyz[idx, 2].mean(axis=1) if k > 1 else nodes_xyz[idx, 2]
    return np.column_stack([pts2d, z])


def export_points_csv(path: str, pts2d: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, pts2d, delimiter=",", header="x,y", comments="")
    print(f"💾 已导出: {path}  (N={len(pts2d)})")


def save_top_view_png(path: str,
                      nodes_xyz: np.ndarray,
                      outer3d: np.ndarray,
                      offset3d: np.ndarray,
                      grid3d_all: np.ndarray,
                      inner_fixed3d: np.ndarray | None = None,
                      movable_valid3d: np.ndarray | None = None,
                      invalid3d: np.ndarray | None = None,
                      ring3d: np.ndarray | None = None,
                      title: str = ""):
    """
    保存俯视图 PNG，便于对扫描结果做人工对比。
    颜色约定：
      - 橙色：内部不可移动点（inner_fixed）
      - 青蓝色：边缘邻域中可移动且成功进入可行位置的点（movable_candidate & keep）
      - 黑色：无效点（主要是可移动但无法通过移动满足条件的点）
      - 红色：最终 ring 点
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    pl = pv.Plotter(off_screen=True, window_size=VIEW_IMAGE_SIZE)
    pl.set_background("white")

    cloud = pv.PolyData(nodes_xyz)
    pl.add_mesh(
        cloud,
        color="lightgray",
        point_size=2,
        render_points_as_spheres=True
    )

    if outer3d is not None and len(outer3d) > 1:
        pl.add_mesh(pv.lines_from_points(outer3d, close=True), color="black", line_width=3)

    if offset3d is not None and len(offset3d) > 1:
        pl.add_mesh(pv.lines_from_points(offset3d, close=True), color="green", line_width=3)

    if grid3d_all is not None and len(grid3d_all) > 0:
        pl.add_mesh(pv.PolyData(grid3d_all), color="lightgray", point_size=5, render_points_as_spheres=True)

    if inner_fixed3d is not None and len(inner_fixed3d) > 0:
        pl.add_mesh(pv.PolyData(inner_fixed3d), color="orange", point_size=10, render_points_as_spheres=True)

    if movable_valid3d is not None and len(movable_valid3d) > 0:
        pl.add_mesh(pv.PolyData(movable_valid3d), color="deepskyblue", point_size=11, render_points_as_spheres=True)

    if invalid3d is not None and len(invalid3d) > 0:
        pl.add_mesh(pv.PolyData(invalid3d), color="black", point_size=10, render_points_as_spheres=True)

    if ring3d is not None and len(ring3d) > 0:
        pl.add_mesh(pv.PolyData(ring3d), color="red", point_size=16, render_points_as_spheres=True)

    text_lines = [
        "Top View | Orange:inner-fixed | Cyan:near-edge movable valid | Black:invalid | Red:ring | Green:offset70"
    ]
    if title:
        text_lines.append(title)
    pl.add_text("\n".join(text_lines), font_size=11)

    bounds = pv.PolyData(nodes_xyz).bounds
    xc = 0.5 * (bounds[0] + bounds[1])
    yc = 0.5 * (bounds[2] + bounds[3])
    zmax = bounds[5]
    span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], 1e-3)
    cam_z = zmax + 3.0 * span + 1.0
    pl.camera_position = [(xc, yc, cam_z), (xc, yc, zmax), (0, 1, 0)]
    pl.enable_parallel_projection()

    pl.screenshot(path)
    pl.close()


# =========================
# 3.5) 平台孔阵偏移扫描：在 [0, spacing) 内遍历 dx,dy 以最大化可吸附点数
# =========================
def _count_objective(objective: str,
                     ok_flag: np.ndarray,
                     keep_flag: np.ndarray,
                     ring_n: int):
    ok_n = int(ok_flag.sum())
    kept_n = int(keep_flag.sum()) if keep_flag is not None else 0
    if objective == "ok":
        return ok_n
    if objective == "kept":
        return kept_n
    if objective == "ring":
        return int(ring_n)
    raise ValueError("OFFSET_SCAN_OBJECTIVE 必须是 ok/kept/ring")


def scan_grid_offsets(min_xy, max_xy, spacing,
                      mask, dist_in, xs, ys,
                      offset2d,
                      outer2d,
                      nodes_xyz,
                      outer3d,
                      offset3d,
                      objective: str,
                      step_m: float,
                      tiebreak=("kept", "ring", "ok"),
                      save_views: bool = False,
                      view_dir: str | None = None):
    """
    在 dx,dy ∈ [0,spacing) 范围内扫描平台孔阵相对薄板的平移偏置。
    返回：best_dx, best_dy, scan_rows(list[dict])
    """
    n_steps = int(math.floor(spacing / step_m))
    d_vals = [i * step_m for i in range(n_steps)]

    if save_views and view_dir:
        os.makedirs(view_dir, exist_ok=True)

    best = None
    rows = []

    for dx in d_vals:
        for dy in d_vals:
            grid2d_all, (x0, y0), _, _ = generate_grid_points(
                min_xy=min_xy, max_xy=max_xy, spacing=spacing,
                origin_mode=GRID_ORIGIN_MODE, origin_xy=GRID_ORIGIN_CUSTOM,
                origin_offset_xy=(dx, dy)
            )

            inside_flag, ok_flag, clearance, need = classify_grid_points(
                grid2d_all, mask, dist_in, xs, ys,
                fixture_radius_m=FIXTURE_RADIUS_M,
                safety_margin_m=SAFETY_MARGIN_M
            )
            dyn = process_support_points_dynamic(
                grid2d_all=grid2d_all,
                inside_flag=inside_flag,
                clearance=clearance,
                need=need,
                offset2d=offset2d,
                outer2d=outer2d,
                mask=mask, dist_in=dist_in, xs=xs, ys=ys,
                edge_near_band_m=EDGE_NEAR_BAND_M
            )

            moved2d = dyn["moved2d"]
            keep_flag = dyn["keep_flag"]
            infos = dyn["infos"]
            movable_candidate_flag = dyn["movable_candidate_flag"]
            inner_fixed_flag = dyn["inner_fixed_flag"]

            moved2d_kept = moved2d[keep_flag]
            chosen_idx = np.empty((0,), dtype=int)
            if keep_flag.sum() == 0:
                ring_n = 0
            else:
                dist_to_offset = np.array([row["dist_to_offset_m"] for row in infos], dtype=float)
                dist_kept = dist_to_offset[keep_flag]
                chosen_idx = select_ring_supports(
                    moved2d=moved2d_kept,
                    dist_to_offset=dist_kept,
                    offset2d=offset2d,
                    spacing_along_offset_m=RING_SPACING_ALONG_OFFSET_M
                )
                ring_n = int(len(chosen_idx))

            ok_n = int(ok_flag.sum())
            kept_n = int(keep_flag.sum())
            edge_n = int(movable_candidate_flag.sum())
            inner_n = int(inner_fixed_flag.sum())

            score = _count_objective(objective, ok_flag, keep_flag, ring_n)

            row = {
                "dx_m": float(dx), "dy_m": float(dy),
                "x0_m": float(x0), "y0_m": float(y0),
                "ok_n": ok_n, "kept_n": kept_n, "ring_n": int(ring_n),
                "edge_n": edge_n, "inner_n": inner_n,
                "score": int(score),
            }
            rows.append(row)

            if save_views and view_dir:
                grid3d_all = lift_points_to_surface(nodes_xyz, grid2d_all, k=12)
                inner_fixed3d = lift_points_to_surface(nodes_xyz, grid2d_all[inner_fixed_flag], k=12) if inner_fixed_flag.any() else np.empty((0, 3))
                movable_valid3d = lift_points_to_surface(nodes_xyz, moved2d[movable_candidate_flag & keep_flag], k=12) if np.any(movable_candidate_flag & keep_flag) else np.empty((0, 3))
                invalid3d = lift_points_to_surface(nodes_xyz, grid2d_all[movable_candidate_flag & (~keep_flag)], k=12) if np.any(movable_candidate_flag & (~keep_flag)) else np.empty((0, 3))
                ring3d = lift_points_to_surface(nodes_xyz, moved2d_kept[chosen_idx], k=12) if ring_n > 0 else np.empty((0, 3))

                if SCAN_VIEW_FILENAME_WITH_SCORE:
                    filename = f"view_dx{dx:.3f}_dy{dy:.3f}_ok{ok_n:03d}_kept{kept_n:03d}_ring{ring_n:03d}_edge{edge_n:03d}_inner{inner_n:03d}.png"
                else:
                    filename = f"view_dx{dx:.3f}_dy{dy:.3f}.png"

                title = f"dx={dx*1000:.0f}mm  dy={dy*1000:.0f}mm | ok={ok_n} kept={kept_n} ring={ring_n} edge={edge_n} inner={inner_n} score={score}"
                save_top_view_png(
                    path=os.path.join(view_dir, filename),
                    nodes_xyz=nodes_xyz,
                    outer3d=outer3d,
                    offset3d=offset3d,
                    grid3d_all=grid3d_all,
                    inner_fixed3d=inner_fixed3d,
                    movable_valid3d=movable_valid3d,
                    invalid3d=invalid3d,
                    ring3d=ring3d,
                    title=title
                )

            def tb_value(name: str):
                if name == "ok":
                    return ok_n
                if name == "kept":
                    return kept_n
                if name == "ring":
                    return int(ring_n)
                return 0

            key = (score,) + tuple(tb_value(n) for n in tiebreak)
            if best is None or key > best["key"]:
                best = {"dx": dx, "dy": dy, "key": key, "row": row}

    return float(best["dx"]), float(best["dy"]), rows

def export_offset_scan_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["dx_m", "dy_m", "x0_m", "y0_m", "ok_n", "kept_n", "ring_n", "edge_n", "inner_n", "score"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"💾 已导出孔阵偏移扫描汇总: {path}  (N={len(rows)})")


# =========================
# 4) 离散移动 + 按优先级
# =========================
def _query_mask_and_clearance(points2d: np.ndarray, mask, dist_in, xs, ys):
    xi = np.clip(np.searchsorted(xs, points2d[:, 0]) - 1, 0, len(xs) - 1)
    yi = np.clip(np.searchsorted(ys, points2d[:, 1]) - 1, 0, len(ys) - 1)
    inside = mask[yi, xi]
    clearance = dist_in[yi, xi]
    return inside, clearance



def move_points_towards_offset_with_priority(base_points2d: np.ndarray,
                                             offset_curve2d: np.ndarray,
                                             mask, dist_in, xs, ys,
                                             require_clearance: bool,
                                             clearance_need: float,
                                             max_dist_to_offset_m: float,
                                             no_move_if_within_m: float,
                                             distances_m=None):
    """
    规则（按最新需求）：
      0) 候选必须能“完整进来”：inside=True 且 clearance>=need（如果 require_clearance）
      1) 若原点本身已经可行，且本身就足够贴近 offset70，则优先不移动
      2) 否则在离散候选集合里找 dist_to_offset 最小的可行点
      3) 只要存在可行候选就 keep；不再使用“离 offset70 太远”作为淘汰条件
    """
    if distances_m is None:
        distances_m = MOVE_DISTANCES_M

    offset_tree = KDTree(offset_curve2d)

    moved2d = np.zeros_like(base_points2d)
    ok_keep = np.zeros((len(base_points2d),), dtype=bool)
    infos = []

    for i, p in enumerate(base_points2d):
        p = np.asarray(p, dtype=float)

        inside0, clear0 = _query_mask_and_clearance(p.reshape(1, 2), mask, dist_in, xs, ys)
        feasible0 = bool(inside0[0]) and (not require_clearance or float(clear0[0]) >= clearance_need)

        dist0, _ = offset_tree.query(p, k=1)
        dist0 = float(dist0)

        if feasible0 and dist0 <= no_move_if_within_m:
            moved2d[i] = p
            ok_keep[i] = True
            infos.append({
                "index": i,
                "x0": float(p[0]), "y0": float(p[1]),
                "x1": float(p[0]), "y1": float(p[1]),
                "dir": "KEEP0",
                "move_mm": 0.0,
                "clearance_m": float(clear0[0]),
                "dist_to_offset_m": dist0,
                "keep": 1,
                "reason": "origin_feasible_keep0"
            })
            continue

        best_q = None
        best_dist = float("inf")
        best_meta = None

        for d in distances_m:
            for name, (ux, uy) in MOVE_DIRECTIONS.items():
                q = np.array([p[0] + d * ux, p[1] + d * uy], dtype=float)

                inside, clearance = _query_mask_and_clearance(q.reshape(1, 2), mask, dist_in, xs, ys)
                feasible = bool(inside[0]) and (not require_clearance or float(clearance[0]) >= clearance_need)
                if not feasible:
                    continue

                dist_to_offset, _ = offset_tree.query(q, k=1)
                dist_to_offset = float(dist_to_offset)

                if dist_to_offset < best_dist:
                    best_dist = dist_to_offset
                    best_q = q
                    best_meta = (name, d, float(clearance[0]), dist_to_offset)

        if best_q is None:
            moved2d[i] = p
            ok_keep[i] = False
            infos.append({
                "index": i,
                "x0": float(p[0]), "y0": float(p[1]),
                "x1": float(p[0]), "y1": float(p[1]),
                "dir": "NONE",
                "move_mm": 0.0,
                "clearance_m": float(clear0[0]) if np.isfinite(clear0[0]) else float("nan"),
                "dist_to_offset_m": dist0,
                "keep": 0,
                "reason": "no_feasible_candidate_full_clearance"
            })
            continue

        moved2d[i] = best_q
        ok_keep[i] = True

        infos.append({
            "index": i,
            "x0": float(p[0]), "y0": float(p[1]),
            "x1": float(best_q[0]), "y1": float(best_q[1]),
            "dir": best_meta[0],
            "move_mm": float(best_meta[1] * 1000.0),
            "clearance_m": float(best_meta[2]),
            "dist_to_offset_m": float(best_meta[3]),
            "keep": 1,
            "reason": "moved_to_feasible_candidate"
        })

    return moved2d, ok_keep, infos


def export_support_moves_csv(path: str, infos: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["index", "state", "origin_inside", "x0", "y0", "x1", "y1", "dir", "move_mm",
            "clearance_m", "dist_to_outer_m", "dist_to_offset_m", "keep", "reason"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in infos:
            f.write(",".join(str(row.get(c, "")) for c in cols) + "\n")
    print(f"💾 已导出移动结果: {path}  (N={len(infos)})")


# =========================
# 5) 沿绿线“选一圈”：弧长分箱
# =========================
def polyline_arclength(points2d: np.ndarray):
    d = np.linalg.norm(np.diff(points2d, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def select_ring_supports(moved2d: np.ndarray,
                         dist_to_offset: np.ndarray,
                         offset2d: np.ndarray,
                         spacing_along_offset_m: float):
    offset_tree = KDTree(offset2d)
    _, idx = offset_tree.query(moved2d, k=1)

    s = polyline_arclength(offset2d)
    t = s[idx]
    bin_id = np.floor(t / spacing_along_offset_m).astype(int)

    best_in_bin = {}
    for i, b in enumerate(bin_id):
        if b not in best_in_bin:
            best_in_bin[b] = i
        else:
            j = best_in_bin[b]
            if dist_to_offset[i] < dist_to_offset[j]:
                best_in_bin[b] = i

    chosen_idx = np.array(sorted(best_in_bin.values()), dtype=int)
    return chosen_idx


# =========================
# 6) 可视化线段（每段独立）
# =========================
def make_segment_polydata(starts3d: np.ndarray, ends3d: np.ndarray) -> pv.PolyData:
    n = len(starts3d)
    if n == 0:
        return pv.PolyData()

    pts = np.empty((2 * n, 3), dtype=float)
    pts[0::2] = starts3d
    pts[1::2] = ends3d

    lines = np.empty((n, 3), dtype=np.int64)
    for i in range(n):
        lines[i] = (2, 2 * i, 2 * i + 1)
    lines = lines.ravel()

    poly = pv.PolyData(pts)
    poly.lines = lines
    return poly


def compute_nodes_covered_by_spheres(nodes_xyz: np.ndarray, centers_xyz: np.ndarray, radius: float):
    if centers_xyz is None or len(centers_xyz) == 0:
        return np.zeros((len(nodes_xyz),), dtype=bool)

    tree = KDTree(nodes_xyz)
    covered = np.zeros((len(nodes_xyz),), dtype=bool)

    for c in centers_xyz:
        idx = tree.query_ball_point(c, r=radius)
        covered[idx] = True

    return covered


def add_world_axes_actor(pl: pv.Plotter,
                         origin=(0.0, 0.0, 0.0),
                         axis_length=0.5,
                         shaft_radius=0.01,
                         tip_length=0.08,
                         tip_radius=0.02,
                         show_labels=True):
    pl.add_mesh(pv.Sphere(radius=shaft_radius * 1.8, center=origin), color="black")

    dirs = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
    }
    colors = {"X": "red", "Y": "green", "Z": "blue"}

    for name, d in dirs.items():
        start = np.array(origin, dtype=float)
        end = start + d * axis_length

        arrow = pv.Arrow(
            start=start,
            direction=d,
            tip_length=tip_length,
            tip_radius=tip_radius,
            shaft_radius=shaft_radius,
            scale=axis_length
        )
        pl.add_mesh(arrow, color=colors[name])

        if show_labels:
            pl.add_point_labels(
                np.array([end]),
                [name],
                font_size=18,
                point_size=0,
                shape_opacity=0.15
            )


def visualize_scene(nodes_xyz: np.ndarray,
                    outer3d: np.ndarray,
                    offset3d: np.ndarray,
                    grid3d_all: np.ndarray,
                    kept_base3d: np.ndarray,
                    kept_moved3d: np.ndarray,
                    invalid_base3d: np.ndarray,
                    kept_label_pts3d: np.ndarray,
                    kept_labels: list,
                    seg_poly: pv.PolyData,
                    fixture_radius_m: float = FIXTURE_RADIUS_M,
                    show_fixture_spheres: bool = True,
                    sphere_opacity: float = 0.18,
                    show_move_labels: bool = True):
    cloud = pv.PolyData(nodes_xyz)
    pl = pv.Plotter()
    pl.set_background("white")

    pl.add_mesh(
        cloud,
        scalars=nodes_xyz[:, 2],
        cmap="jet",
        point_size=5,
        render_points_as_spheres=True,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Z Height (m)"}
    )

    pl.add_mesh(pv.lines_from_points(outer3d, close=True), color="black", line_width=4)
    pl.add_mesh(pv.lines_from_points(offset3d, close=True), color="green", line_width=4)

    pl.add_mesh(pv.PolyData(grid3d_all), color="lightgray", point_size=5, render_points_as_spheres=True)

    # 无效点：黑色
    if invalid_base3d is not None and len(invalid_base3d) > 0:
        pl.add_mesh(pv.PolyData(invalid_base3d), color="black", point_size=11, render_points_as_spheres=True)

    # 所有有效点（kept）：紫色（洋红色）
    if kept_moved3d is not None and len(kept_moved3d) > 0:
        pl.add_mesh(pv.PolyData(kept_moved3d), color="magenta", point_size=18, render_points_as_spheres=True)

    # 有移动的点：显示红色原位置
    if kept_base3d is not None and len(kept_base3d) > 0:
        pl.add_mesh(pv.PolyData(kept_base3d), color="red", point_size=14, render_points_as_spheres=True)

    # 移动连线（灰色）
    if seg_poly is not None and seg_poly.n_points > 0:
        pl.add_mesh(seg_poly, color="dimgray", line_width=2)

    # 有效点的夹具球体覆盖范围（黄色）
    if show_fixture_spheres and len(kept_moved3d) > 0:
        sphere = pv.Sphere(radius=fixture_radius_m, theta_resolution=24, phi_resolution=24)
        spheres = pv.MultiBlock()
        for c in kept_moved3d:
            s = sphere.copy()
            s.translate(c, inplace=True)
            spheres.append(s)
        pl.add_mesh(spheres.combine(), opacity=sphere_opacity)

        covered = compute_nodes_covered_by_spheres(nodes_xyz, kept_moved3d, fixture_radius_m)
        covered_pts = pv.PolyData(nodes_xyz[covered])
        if covered_pts.n_points > 0:
            pl.add_mesh(covered_pts, color="yellow", point_size=7, render_points_as_spheres=True)

    # 显示移动距离标签
    if show_move_labels and kept_label_pts3d is not None and len(kept_label_pts3d) > 0 and kept_labels:
        pl.add_point_labels(
            kept_label_pts3d,
            kept_labels,
            font_size=12,
            point_size=0,
            shape_opacity=0.25
        )

    pl.add_text(
        "LightGray:grid | Black:invalid | Magenta:valid(moved) | Red:original position | Yellow:nodes covered\n"
        "NOTE: XY axes are PCA-aligned (not original CAD global axes). Grid coordinates shown are PCA-XY.",
        font_size=11
    )

    bounds = pv.PolyData(nodes_xyz).bounds
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    L = 0.15 * max(dx, dy, dz)

    add_world_axes_actor(
        pl,
        origin=(0.0, 0.0, 0.0),
        axis_length=L,
        shaft_radius=0.01 * L,
        tip_length=0.18,
        tip_radius=0.02 * L,
        show_labels=True
    )

    pl.add_axes()
    pl.show_grid()
    pl.camera_position = "iso"
    pl.show()


def main():
    nodes = extract_or_load_nodes(force_rebuild=False, do_pca_align=True)

    # ===== 自适应 mask 参数（关键修复）=====
    pts_xy = nodes[:, :2]
    d_nn = estimate_nn_distance_xy(pts_xy)

    MASK_GRID_STEP_M = 0.5 * d_nn
    MASK_INSIDE_RADIUS_M = 1.5 * d_nn
    close_px = max(1, int(MASK_INSIDE_RADIUS_M / MASK_GRID_STEP_M * 0.6))

    print(
        f"🧠 Auto mask params: d_nn={d_nn:.5f} | "
        f"grid_step={MASK_GRID_STEP_M:.5f} | "
        f"inside_radius={MASK_INSIDE_RADIUS_M:.5f} | "
        f"close_px={close_px}"
    )

    mask, dist_in, xs, ys = build_part_mask_and_distance(
        nodes_xyz=nodes,
        grid_step_m=MASK_GRID_STEP_M,
        inside_radius_m=MASK_INSIDE_RADIUS_M,
        close_px=close_px
    )

    outer2d, offset2d = compute_outer_and_offset_curves(mask, dist_in, xs, ys, offset_m=OFFSET_M)
    outer2d = _downsample_polyline(outer2d, max_points=1500)
    offset2d = _downsample_polyline(offset2d, max_points=1500)

    export_points_csv(EXPORT_CSV_OUTER, outer2d)
    export_points_csv(EXPORT_CSV_OFFSET70, offset2d)

    outer3d = lift_points_to_surface(nodes, outer2d, k=12)
    offset3d = lift_points_to_surface(nodes, offset2d, k=12)

    # --- grid (建系坐标下的网格点)
    min_xy = pts_xy.min(axis=0)
    max_xy = pts_xy.max(axis=0)

    # ===== 平台孔阵相对薄板的 (dx,dy) 扫描：在一个 spacing 周期内寻找可吸附数量最大的位置 =====
    best_dx, best_dy = 0.0, 0.0
    if OFFSET_SCAN_ENABLE:
        print(f"🔍 开始孔阵偏移扫描: dx,dy∈[0,{GRID_SPACING_M}) 步长={OFFSET_SCAN_STEP_M}m | objective={OFFSET_SCAN_OBJECTIVE}")
        best_dx, best_dy, scan_rows = scan_grid_offsets(
            min_xy=min_xy, max_xy=max_xy, spacing=GRID_SPACING_M,
            mask=mask, dist_in=dist_in, xs=xs, ys=ys,
            offset2d=offset2d,
            outer2d=outer2d,
            nodes_xyz=nodes,
            outer3d=outer3d,
            offset3d=offset3d,
            objective=OFFSET_SCAN_OBJECTIVE,
            step_m=OFFSET_SCAN_STEP_M,
            tiebreak=OFFSET_SCAN_TIEBREAK,
            save_views=SAVE_SCAN_VIEWS,
            view_dir=EXPORT_SCAN_VIEW_DIR
        )
        export_offset_scan_csv(EXPORT_CSV_OFFSET_SCAN, scan_rows)
        if SAVE_SCAN_VIEWS:
            print(f"🖼️ 已保存扫描俯视图目录: {EXPORT_SCAN_VIEW_DIR}")
        print(f"🏁 扫描完成，最优偏移: dx={best_dx:.3f}m dy={best_dy:.3f}m  (详见 {EXPORT_CSV_OFFSET_SCAN})")
    else:
        print("ℹ️ 未启用孔阵偏移扫描（OFFSET_SCAN_ENABLE=False），将使用默认孔阵原点。")

    grid2d_all, (x0, y0), gx, gy = generate_grid_points(
        min_xy=min_xy,
        max_xy=max_xy,
        spacing=GRID_SPACING_M,
        origin_mode=GRID_ORIGIN_MODE,
        origin_xy=GRID_ORIGIN_CUSTOM,
        origin_offset_xy=(best_dx, best_dy)
    )
    print(f"🧭 网格原点 (x0,y0)=({x0:.3f},{y0:.3f}) spacing={GRID_SPACING_M}m mode={GRID_ORIGIN_MODE}")
    print(f"🔢 All grid points: {len(grid2d_all)} | gx={len(gx)} gy={len(gy)} | fixed hardware={GRID_ROWS}x{GRID_COLS}")

    inside_flag, ok_flag, clearance, need = classify_grid_points(
        grid2d_all, mask, dist_in, xs, ys,
        fixture_radius_m=FIXTURE_RADIUS_M,
        safety_margin_m=SAFETY_MARGIN_M
    )
    dist_to_outer = compute_distance_to_outer(grid2d_all, outer2d)
    outside_far_flag, movable_candidate_flag, inner_fixed_flag, near_edge_flag = split_grid_states(inside_flag, dist_to_outer, edge_near_band_m=EDGE_NEAR_BAND_M)

    export_points_csv(EXPORT_CSV_ALL, grid2d_all)
    export_points_csv(EXPORT_CSV_INSIDE, grid2d_all[inside_flag])
    export_points_csv(EXPORT_CSV_OK, grid2d_all[ok_flag])

    dyn = process_support_points_dynamic(
        grid2d_all=grid2d_all,
        inside_flag=inside_flag,
        clearance=clearance,
        need=need,
        offset2d=offset2d,
        outer2d=outer2d,
        mask=mask, dist_in=dist_in, xs=xs, ys=ys,
        edge_near_band_m=EDGE_NEAR_BAND_M
    )
    moved2d = dyn["moved2d"]
    keep_flag = dyn["keep_flag"]
    infos = dyn["infos"]
    export_support_moves_csv(EXPORT_CSV_SUPPORT_MOVED, infos)

    dist_to_offset = np.array([row["dist_to_offset_m"] for row in infos], dtype=float)

    moved2d_kept = moved2d[keep_flag]
    base2d_kept = grid2d_all[keep_flag]
    dist_kept = dist_to_offset[keep_flag]

    removed_flag = ~keep_flag
    base2d_removed = grid2d_all[removed_flag]
    invalid_flag = movable_candidate_flag & (~keep_flag)
    print(f"🧩 Fixed hardware grid: rows={GRID_ROWS} cols={GRID_COLS} total={GRID_ROWS*GRID_COLS}")
    print(f"🟦 Inside covered: {int(inside_flag.sum())} | near-edge movable(dynamic): {int(movable_candidate_flag.sum())} | inner-fixed(dynamic): {int(inner_fixed_flag.sum())} | outside-far: {int(outside_far_flag.sum())}")
    print(f"✅ Kept supports (dynamic edge rule): {len(moved2d_kept)} / {len(grid2d_all)}")
    print(f"⚫ Invalid movable points (black): {int(invalid_flag.sum())} / {len(grid2d_all)} | inner-fixed shown separately")

    if len(moved2d_kept) == 0:
        raise ValueError("没有任何点能通过当前规则形成有效支撑：请增大 EDGE_NEAR_BAND_M 或允许更大的 MOVE_DISTANCES_M")

    # 不再使用ring逻辑，直接使用所有kept点
    print(f"🎯 Total valid supports (kept): {len(moved2d_kept)}")

    # --- lift to 3D
    grid3d_all = lift_points_to_surface(nodes, grid2d_all, k=12)

    # 只区分无效点和有效点
    invalid_flag = ~keep_flag
    invalid_base3d = lift_points_to_surface(nodes, grid2d_all[invalid_flag], k=12) if invalid_flag.any() else np.empty((0, 3))

    # 所有有效点的3D坐标
    kept_base3d = lift_points_to_surface(nodes, base2d_kept, k=12)
    kept_moved3d = lift_points_to_surface(nodes, moved2d_kept, k=12)

    # 为所有有效点生成标签: dir + move_mm
    kept_global_indices = np.where(keep_flag)[0]
    kept_labels = []
    kept_label_pts3d = []
    for gi, p3 in zip(kept_global_indices, kept_moved3d):
        row = infos[int(gi)]
        kept_labels.append(f"{row['dir']} {row['move_mm']:.0f}mm")
        kept_label_pts3d.append(p3)
    kept_label_pts3d = np.asarray(kept_label_pts3d)

    # 保存“最优偏移”的最终俯视图
    save_top_view_png(
        path=EXPORT_BEST_VIEW,
        nodes_xyz=nodes,
        outer3d=outer3d,
        offset3d=offset3d,
        grid3d_all=grid3d_all,
        inner_fixed3d=np.empty((0, 3)),
        movable_valid3d=np.empty((0, 3)),
        invalid3d=invalid_base3d,
        ring3d=kept_moved3d,
        title=f"BEST | dx={best_dx*1000:.0f}mm dy={best_dy*1000:.0f}mm | kept={int(keep_flag.sum())} invalid={int(invalid_flag.sum())}"
    )
    print(f"💾 已导出最佳方案俯视图: {EXPORT_BEST_VIEW}")

    # 生成移动连线
    seg_poly = make_segment_polydata(kept_base3d, kept_moved3d)

    visualize_scene(
        nodes_xyz=nodes,
        outer3d=outer3d,
        offset3d=offset3d,
        grid3d_all=grid3d_all,
        kept_base3d=kept_base3d,
        kept_moved3d=kept_moved3d,
        invalid_base3d=invalid_base3d,
        kept_label_pts3d=kept_label_pts3d,
        kept_labels=kept_labels,
        seg_poly=seg_poly,
        fixture_radius_m=FIXTURE_RADIUS_M,
        show_fixture_spheres=True,
        sphere_opacity=0.18,
        show_move_labels=SHOW_MOVE_LABELS
    )

if __name__ == "__main__":
    main()
