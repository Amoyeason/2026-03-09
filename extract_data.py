import numpy as np
from ansys.mapdl.core import launch_mapdl
import os
import glob
from scipy import sparse
from scipy.sparse import kron, eye, coo_matrix

# ================= 配置 =================
WORK_DIR = "E:\\ansys_data_final"
if not os.path.exists(WORK_DIR): os.makedirs(WORK_DIR)

CAD_PATH = "E:\\ZJU\\Learning baogao\\112_1.igs"

# 🟢 [恢复] 物理参数 (来自您的 extract_data_2)
THICKNESS = 0.005  # 5mm
MESH_SIZE = 0.01  # 40mm 网格

# 姿态微调
MANUAL_FLIP_Z = False
MANUAL_FLIP_X = False
MANUAL_ROT_90 = False


def parse_mapping_file(mapping_path):
    print(f"📖 解析映射文件: {os.path.basename(mapping_path)} ...")
    eq_map = {}
    with open(mapping_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3: continue
            try:
                if not parts[0].isdigit(): continue
                eq_id = int(parts[0]) - 1
                node_id = int(parts[1])
                dof_label = parts[2].strip().upper()
                eq_map[eq_id] = (node_id, dof_label)
            except:
                continue
    return eq_map


def main():
    # 增加 -smp 避免并行卡死
    mapdl = launch_mapdl(run_location=WORK_DIR, nproc=1, additional_switches="-smp", override=True,
                         cleanup_on_exit=True)
    mapdl.ignore_errors = False

    try:
        print("🚀 启动 V22 (T800材料 + 映射修复版)...", flush=True)

        # ================= 1. 导入 =================
        print("🏗️ 导入 IGES...", flush=True)
        mapdl.clear()
        mapdl.aux15()
        # 保持默认公差，防止过度合并导致几何损坏
        mapdl.ioptn("GTOL", 0.05)
        mapdl.ioptn("IGES", "SMOOTH")  # 您之前的代码用了 SMOOTH
        mapdl.igesin(CAD_PATH)
        mapdl.finish()
        mapdl.prep7()

        # ================= 2. 尺寸检查与缩放 =================
        print("📏 尺寸检查...", flush=True)
        mapdl.allsel()
        # 使用 KP 获取尺寸 (更准)
        xmin = mapdl.get_value("KP", 0, "MNLOC", "X")
        xmax = mapdl.get_value("KP", 0, "MXLOC", "X")
        raw_len = xmax - xmin

        print(f"   -> 原始长度: {raw_len:.2f}")

        # 恢复您的缩放逻辑: 如果 > 10，说明是 mm，缩放为 m
        if raw_len > 10.0:
            print(f"   ⚠️ 检测到单位疑似 mm，缩放为米 (x0.001)...")
            mapdl.arscale("ALL", "", "", 0.001, 0.001, 0.001, "", "", 1)
        else:
            print(f"   ✅ 单位看起来是米，保持不变。")

        # ================= 3. 材料与网格 =================
        print("⚙️ 定义 T800 材料...", flush=True)
        mapdl.et(1, "SHELL181")
        mapdl.sectype(1, "SHELL")
        mapdl.secdata(THICKNESS)

        # 🟢 [恢复] T800 碳纤维属性 (单位: Pa, kg/m^3)
        mapdl.mp("EX", 1, 157.7e9)
        mapdl.mp("EY", 1, 9.05e9)
        mapdl.mp("EZ", 1, 9.05e9)
        mapdl.mp("GXY", 1, 4.69e9)
        mapdl.mp("GXZ", 1, 4.69e9)
        mapdl.mp("GYZ", 1, 3.24e9)
        mapdl.mp("PRXY", 1, 0.3)
        mapdl.mp("DENS", 1, 1600)

        print(f"🕸️ 网格划分 (Size={MESH_SIZE}m)...", flush=True)
        mapdl.esize(MESH_SIZE)

        # 🟢 使用三角网格 (1) 保证通过率
        mapdl.mshape(1, "2D")
        mapdl.mshkey(0)
        mapdl.shpp("OFF")
        mapdl.amesh("ALL")

        total_nodes = int(mapdl.mesh.n_node)
        print(f"✅ 网格节点数: {total_nodes}", flush=True)

        final_nnum = mapdl.mesh.nnum
        node_id_to_logical = {nid: i for i, nid in enumerate(final_nnum)}

        # ================= 4. 提取矩阵 =================
        print("5️⃣  求解并提取数据...", flush=True)
        mapdl.run("/SOLU")
        mapdl.antype("STATIC")
        # 重力 9.8 m/s^2
        mapdl.acel(0, 0, 9.8)
        mapdl.run("WRFULL, 1")
        mapdl.solve()
        mapdl.finish()

        mm = mapdl.math
        k_mat = mm.stiff(fname="file.full")
        f_vec = mm.rhs(fname="file.full")
        K_raw = k_mat.asarray()
        F_raw = f_vec.asarray()

        # 检查载荷大小
        print(f"   🔎 F Total Norm: {np.linalg.norm(F_raw):.4e}")

        # ================= 5. 映射解析与重排序 =================
        # 🟢 这是修复 "0变形" 的关键：正确读取映射 + 重排序矩阵
        mapdl.run("/AUX2")
        mapdl.file("file", "full")
        mapdl.run("HBMAT, export_job, txt, , ASCII, STIFF, YES, YES")

        map_files = glob.glob(os.path.join(WORK_DIR, "*.mapping"))
        if not map_files:
            print("❌ 未找到 .mapping 文件！")
            return
        eq_map = parse_mapping_file(map_files[0])

        print("🔄 执行矩阵重排序 (Re-ordering)...", flush=True)
        dof_order = {'UX': 0, 'UY': 1, 'UZ': 2, 'ROTX': 3, 'ROTY': 4, 'ROTZ': 5}
        ndof_per_node = 6
        target_dim = total_nodes * ndof_per_node

        P_rows, P_cols, P_vals = [], [], []

        valid_eq_count = 0
        for eq_id, (nid, dof) in eq_map.items():
            if nid in node_id_to_logical and dof in dof_order:
                logical_node_idx = node_id_to_logical[nid]
                dof_idx = dof_order[dof]
                target_col = logical_node_idx * ndof_per_node + dof_idx
                P_rows.append(eq_id)
                P_cols.append(target_col)
                P_vals.append(1.0)
                valid_eq_count += 1

        P_mat = coo_matrix((P_vals, (P_rows, P_cols)), shape=(K_raw.shape[0], target_dim)).tocsr()

        K_ordered = P_mat.T @ K_raw @ P_mat
        F_ordered = P_mat.T @ F_raw

        # ================= 6. PCA 旋转 =================
        print("📐 计算姿态 (PCA)...", flush=True)
        raw_nodes = mapdl.mesh.nodes
        center = np.mean(raw_nodes, axis=0)
        centered = raw_nodes - center
        cov = np.cov(centered.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        new_z = eig_vecs[:, 0]
        new_x = eig_vecs[:, 2]
        new_y = np.cross(new_z, new_x)
        R = np.column_stack((new_x, new_y, new_z))

        R_6x6 = np.zeros((6, 6))
        R_6x6[:3, :3] = R
        R_6x6[3:, 3:] = R

        rotated_nodes = np.dot(centered, R)
        if MANUAL_FLIP_Z: rotated_nodes[:, 2] *= -1; R_6x6[:, 2] *= -1

        T_rot = kron(eye(total_nodes), R_6x6, format='csr')

        K_final = T_rot.T @ K_ordered @ T_rot
        F_final = T_rot.T @ F_ordered

        final_nodes = rotated_nodes - np.min(rotated_nodes, axis=0)

        # ================= 7. 保存 =================
        print("📝 保存数据...", flush=True)
        # 生成标准映射表 (因为矩阵已经重排整齐了)
        ux_map = np.arange(0, total_nodes * 6, 6, dtype=np.int32)
        uy_map = np.arange(1, total_nodes * 6, 6, dtype=np.int32)
        uz_map = np.arange(2, total_nodes * 6, 6, dtype=np.int32)

        sparse.save_npz(os.path.join(WORK_DIR, "K_pure.npz"), K_final)
        np.savez(
            os.path.join(WORK_DIR, "digital_twin_data.npz"),
            F=F_final,
            nodes=final_nodes,
            ux_map=ux_map, uy_map=uy_map, uz_map=uz_map,
            n_ids=final_nnum
        )
        print(f"✅ V22 完成！T800材料 + 映射修复。")

    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        mapdl.exit()


if __name__ == "__main__":
    main()