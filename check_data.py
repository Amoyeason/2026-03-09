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

# ANSYS网格参数（仅首次提取点云）
THICKNESS = 0.005
MESH_SIZE = 0.01

# 网格参数
GRID_SPACING_M = 0.25  # 250mm

# 夹具参数
FIXTURE_RADIUS_M = 0.0625
SAFETY_MARGIN_M = 0.0

# 偏移线
OFFSET_M = 0.07

# 网格原点
GRID_ORIGIN_MODE = "aligned"
GRID_ORIGIN_CUSTOM = (0.0, 0.0)

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

# 关键：剔除“怎么挪都远”的点（移动后到绿线的距离阈值）
MAX_DIST_TO_OFFSET_M = 0.06  # 60mm

# 新增：如果原点本身已足够贴近绿线（<=10mm），就不移动（前提：它本身也要满足半径可放下）
NO_MOVE_IF_WITHIN_M = 0.01  # 10mm

# 最终“一圈支撑”沿绿线的间距（弧长分箱）
RING_SPACING_ALONG_OFFSET_M = 0.1  # 100mm

# 可视化：为了不太乱，只画“最终一圈支撑”的移动灰线
DRAW_MOVE_SEGMENTS_FOR = "ring"  # "all" / "ring"

# 是否显示“每个 ring 点的 move 标签”
SHOW_RING_MOVE_LABELS = True


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
def generate_grid_points(min_xy, max_xy, spacing, origin_mode="aligned", origin_xy=None):
    min_x, min_y = float(min_xy[0]), float(min_xy[1])
    max_x, max_y = float(max_xy[0]), float(max_xy[1])

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

    gx = np.arange(x0, max_x + spacing, spacing)
    gy = np.arange(y0, max_y + spacing, spacing)
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
    你的优先级（严格实现）：
      0) 候选必须能“完整进来”：inside=True 且 clearance>=need（如果 require_clearance）
      1) 若原点本身到绿线<=no_move_if_within_m 且原点本身可行 -> 不移动（最高优先级）
      2) 否则在离散候选集合里找 dist_to_offset 最小的可行点
      3) 最后 keep 判定：best_dist_to_offset <= max_dist_to_offset_m
    """
    if distances_m is None:
        distances_m = MOVE_DISTANCES_M

    offset_tree = KDTree(offset_curve2d)

    moved2d = np.zeros_like(base_points2d)
    ok_keep = np.zeros((len(base_points2d),), dtype=bool)
    infos = []

    for i, p in enumerate(base_points2d):
        p = np.asarray(p, dtype=float)

        # ---- 先检查原点本身是否“可行”（完整放下夹具）
        inside0, clear0 = _query_mask_and_clearance(p.reshape(1, 2), mask, dist_in, xs, ys)
        feasible0 = bool(inside0[0]) and (not require_clearance or float(clear0[0]) >= clearance_need)

        dist0, _ = offset_tree.query(p, k=1)
        dist0 = float(dist0)

        # ---- 规则1：若原点已足够近且可行 -> 不移动
        if feasible0 and dist0 <= no_move_if_within_m:
            moved2d[i] = p
            keep = dist0 <= max_dist_to_offset_m
            ok_keep[i] = keep
            infos.append({
                "index": i,
                "x0": float(p[0]), "y0": float(p[1]),
                "x1": float(p[0]), "y1": float(p[1]),
                "dir": "KEEP0",
                "move_mm": 0.0,
                "clearance_m": float(clear0[0]),
                "dist_to_offset_m": dist0,
                "keep": int(keep),
                "reason": "origin_within_tol_no_move"
            })
            continue

        # ---- 否则：搜索离散候选（必须可行，且选 dist_to_offset 最小）
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

                # 选离绿线最近
                if dist_to_offset < best_dist:
                    best_dist = dist_to_offset
                    best_q = q
                    best_meta = (name, d, float(clearance[0]), dist_to_offset)

        if best_q is None:
            # 没有任何“完整进来”的候选
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
        keep = best_dist <= max_dist_to_offset_m
        ok_keep[i] = keep

        infos.append({
            "index": i,
            "x0": float(p[0]), "y0": float(p[1]),
            "x1": float(best_q[0]), "y1": float(best_q[1]),
            "dir": best_meta[0],
            "move_mm": float(best_meta[1] * 1000.0),
            "clearance_m": float(best_meta[2]),
            "dist_to_offset_m": float(best_meta[3]),
            "keep": int(keep),
            "reason": "moved_best_to_offset_under_clearance_constraint"
        })

    return moved2d, ok_keep, infos


def export_support_moves_csv(path: str, infos: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["index", "x0", "y0", "x1", "y1", "dir", "move_mm",
            "clearance_m", "dist_to_offset_m", "keep", "reason"]
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
                    moved3d_kept: np.ndarray,
                    removed_base3d: np.ndarray,   # <-- 改：removed 显示 base（建系网格点）
                    ring_base3d: np.ndarray,
                    ring_moved3d: np.ndarray,
                    ring_moved3d_bad: np.ndarray,
                    ring_label_pts3d: np.ndarray,
                    ring_labels: list,
                    seg_poly: pv.PolyData,
                    fixture_radius_m: float = 0.06,
                    show_fixture_spheres: bool = True,
                    sphere_opacity: float = 0.18,
                    show_ring_labels: bool = True):
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

    pl.add_mesh(pv.PolyData(grid3d_all), color="gray", point_size=5, render_points_as_spheres=True)

    if moved3d_kept is not None and len(moved3d_kept) > 0:
        pl.add_mesh(pv.PolyData(moved3d_kept), color="purple", point_size=10, render_points_as_spheres=True)

    # removed：显示“建系时原始网格点”而不是 moved 后位置
    if removed_base3d is not None and len(removed_base3d) > 0:
        pl.add_mesh(pv.PolyData(removed_base3d), color="black", point_size=10, render_points_as_spheres=True)

    pl.add_mesh(pv.PolyData(ring_moved3d), color="magenta", point_size=18, render_points_as_spheres=True)
    pl.add_mesh(pv.PolyData(ring_base3d), color="red", point_size=14, render_points_as_spheres=True)

    if ring_moved3d_bad is not None and len(ring_moved3d_bad) > 0:
        pl.add_mesh(pv.PolyData(ring_moved3d_bad), color="black", point_size=26, render_points_as_spheres=True)

    if seg_poly is not None and seg_poly.n_points > 0:
        pl.add_mesh(seg_poly, color="dimgray", line_width=2)

    if show_fixture_spheres and len(ring_moved3d) > 0:
        sphere = pv.Sphere(radius=fixture_radius_m, theta_resolution=24, phi_resolution=24)
        spheres = pv.MultiBlock()
        for c in ring_moved3d:
            s = sphere.copy()
            s.translate(c, inplace=True)
            spheres.append(s)
        pl.add_mesh(spheres.combine(), opacity=sphere_opacity)

        covered = compute_nodes_covered_by_spheres(nodes_xyz, ring_moved3d, fixture_radius_m)
        covered_pts = pv.PolyData(nodes_xyz[covered])
        if covered_pts.n_points > 0:
            pl.add_mesh(covered_pts, color="yellow", point_size=7, render_points_as_spheres=True)

    if show_ring_labels and ring_label_pts3d is not None and len(ring_label_pts3d) > 0 and ring_labels:
        pl.add_point_labels(
            ring_label_pts3d,
            ring_labels,
            font_size=12,
            point_size=0,
            shape_opacity=0.25
        )

    pl.add_text(
        "Gray:grid | Green:offset70 | Purple:kept(moved) | Black:removed(base-grid) | Magenta:ring(moved) | Red:ring(base) | Yellow:nodes covered\n"
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

    grid2d_all, (x0, y0), gx, gy = generate_grid_points(
        min_xy=min_xy,
        max_xy=max_xy,
        spacing=GRID_SPACING_M,
        origin_mode=GRID_ORIGIN_MODE,
        origin_xy=GRID_ORIGIN_CUSTOM
    )
    print(f"🧭 网格原点 (x0,y0)=({x0:.3f},{y0:.3f}) spacing={GRID_SPACING_M}m mode={GRID_ORIGIN_MODE}")
    print(f"🔢 All grid points: {len(grid2d_all)} | gx={len(gx)} gy={len(gy)}")

    inside_flag, ok_flag, clearance, need = classify_grid_points(
        grid2d_all, mask, dist_in, xs, ys,
        fixture_radius_m=FIXTURE_RADIUS_M,
        safety_margin_m=SAFETY_MARGIN_M
    )
    export_points_csv(EXPORT_CSV_ALL, grid2d_all)
    export_points_csv(EXPORT_CSV_INSIDE, grid2d_all[inside_flag])
    export_points_csv(EXPORT_CSV_OK, grid2d_all[ok_flag])

    # --- move with your priorities
    clearance_need = FIXTURE_RADIUS_M + SAFETY_MARGIN_M

    moved2d, keep_flag, infos = move_points_towards_offset_with_priority(
        base_points2d=grid2d_all,
        offset_curve2d=offset2d,
        mask=mask, dist_in=dist_in, xs=xs, ys=ys,
        require_clearance=SUPPORT_REQUIRE_CLEARANCE,
        clearance_need=clearance_need,
        max_dist_to_offset_m=MAX_DIST_TO_OFFSET_M,
        no_move_if_within_m=NO_MOVE_IF_WITHIN_M,
        distances_m=MOVE_DISTANCES_M
    )
    export_support_moves_csv(EXPORT_CSV_SUPPORT_MOVED, infos)

    dist_to_offset = np.array([row["dist_to_offset_m"] for row in infos], dtype=float)

    moved2d_kept = moved2d[keep_flag]
    base2d_kept = grid2d_all[keep_flag]
    dist_kept = dist_to_offset[keep_flag]

    removed_flag = ~keep_flag
    base2d_removed = grid2d_all[removed_flag]   # <-- 关键：removed 用 base（建系坐标）
    print(f"✅ Moved&Kept (dist<= {MAX_DIST_TO_OFFSET_M*1000:.0f}mm): {len(moved2d_kept)} / {len(grid2d_all)}")
    print(f"🧹 Removed (black, base-grid): {len(base2d_removed)} / {len(grid2d_all)}")

    if len(moved2d_kept) == 0:
        raise ValueError("没有任何点能在允许移动范围内贴近绿线：请增大 MAX_DIST_TO_OFFSET_M 或允许更大的 MOVE_DISTANCES_M")

    chosen_idx = select_ring_supports(
        moved2d=moved2d_kept,
        dist_to_offset=dist_kept,
        offset2d=offset2d,
        spacing_along_offset_m=RING_SPACING_ALONG_OFFSET_M
    )

    ring_base2d = base2d_kept[chosen_idx]
    ring_moved2d = moved2d_kept[chosen_idx]
    ring_dist = dist_kept[chosen_idx]

    print(f"🎯 Ring supports: {len(ring_moved2d)} | spacing_along_offset={RING_SPACING_ALONG_OFFSET_M}m")
    print(f"📌 Ring dist_to_offset: min={ring_dist.min():.4f}m  mean={ring_dist.mean():.4f}m  max={ring_dist.max():.4f}m")

    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(EXPORT_CSV_RING_SUPPORTS, "w", encoding="utf-8") as f:
        f.write("x0,y0,x1,y1,dist_to_offset_m\n")
        for (x0_, y0_), (x1_, y1_), d_ in zip(ring_base2d, ring_moved2d, ring_dist):
            f.write(f"{x0_},{y0_},{x1_},{y1_},{d_}\n")
    print(f"💾 已导出一圈支撑: {EXPORT_CSV_RING_SUPPORTS}")

    # --- lift to 3D
    grid3d_all = lift_points_to_surface(nodes, grid2d_all, k=12)

    moved3d_kept = lift_points_to_surface(nodes, moved2d_kept, k=12) if len(moved2d_kept) else np.empty((0, 3))
    removed_base3d = lift_points_to_surface(nodes, base2d_removed, k=12) if len(base2d_removed) else np.empty((0, 3))

    ring_base3d = lift_points_to_surface(nodes, ring_base2d, k=12)
    ring_moved3d = lift_points_to_surface(nodes, ring_moved2d, k=12)

    ring_inside, ring_clear = _query_mask_and_clearance(ring_moved2d, mask, dist_in, xs, ys)
    ring_need = FIXTURE_RADIUS_M + SAFETY_MARGIN_M
    ring_bad = ~(ring_inside & (ring_clear >= ring_need))
    print(f"🧪 Ring clearance check: bad={int(ring_bad.sum())} / {len(ring_bad)}  (need>={ring_need:.3f}m)")

    ring_moved3d_bad = lift_points_to_surface(nodes, ring_moved2d[ring_bad], k=12) if ring_bad.any() else np.empty((0, 3))

    # ring labels: dir + move_mm
    kept_global_indices = np.where(keep_flag)[0]
    ring_global_indices = kept_global_indices[chosen_idx]

    ring_labels = []
    ring_label_pts3d = []
    for gi, p3 in zip(ring_global_indices, ring_moved3d):
        row = infos[int(gi)]
        ring_labels.append(f"{row['dir']} {row['move_mm']:.0f}mm")
        ring_label_pts3d.append(p3)
    ring_label_pts3d = np.asarray(ring_label_pts3d)

    if DRAW_MOVE_SEGMENTS_FOR == "all":
        base3d_for_seg = lift_points_to_surface(nodes, base2d_kept, k=12)
        seg_poly = make_segment_polydata(base3d_for_seg, moved3d_kept)
    else:
        seg_poly = make_segment_polydata(ring_base3d, ring_moved3d)

    visualize_scene(
        nodes_xyz=nodes,
        outer3d=outer3d,
        offset3d=offset3d,
        grid3d_all=grid3d_all,
        moved3d_kept=moved3d_kept,
        removed_base3d=removed_base3d,
        ring_base3d=ring_base3d,
        ring_moved3d=ring_moved3d,
        ring_moved3d_bad=ring_moved3d_bad,
        ring_label_pts3d=ring_label_pts3d,
        ring_labels=ring_labels,
        seg_poly=seg_poly,
        fixture_radius_m=FIXTURE_RADIUS_M,
        show_fixture_spheres=True,
        sphere_opacity=0.18,
        show_ring_labels=SHOW_RING_MOVE_LABELS
    )


if __name__ == "__main__":
    main()
