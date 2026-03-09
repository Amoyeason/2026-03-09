import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# ================= 配置 =================
DATA_DIR = "E:\\ansys_data_final"
FILE_PATH = os.path.join(DATA_DIR, "digital_twin_data.npz")


def main():
    print(f"📂 正在读取数据: {FILE_PATH} ...")
    if not os.path.exists(FILE_PATH):
        print("❌ 文件不存在！请先运行 extract_data_3d.py")
        return

    data = np.load(FILE_PATH)
    nodes = data['nodes']  # (N, 3) 数组: [x, y, z]
    F = data['F']

    # 过滤掉非 Z 向的力 (F 包含 UX, UY, UZ 的所有自由度)
    # 我们的 F_numpy 是按自由度排列的，这里我们主要验证几何，
    # 只要节点是对的，矩阵通常就是对的。

    print(f"✅ 加载成功！")
    print(f"   - 节点数量: {len(nodes)}")
    print(f"   - 坐标范围 X: {nodes[:, 0].min():.4f} ~ {nodes[:, 0].max():.4f} m")
    print(f"   - 坐标范围 Y: {nodes[:, 1].min():.4f} ~ {nodes[:, 1].max():.4f} m")
    print(f"   - 坐标范围 Z: {nodes[:, 2].min():.4f} ~ {nodes[:, 2].max():.4f} m")

    # ================= 3D 绘图 =================
    print("🎨 正在绘制 3D 节点云...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制节点 (使用 Z 坐标着色，方便看曲率)
    p = ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2],
                   c=nodes[:, 2], cmap='viridis', s=5, alpha=0.8)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f"Extracted 3D Geometry ({len(nodes)} Nodes)\nCheck if this looks like your SW part!", fontsize=14)

    # 设置比例尺一致 (避免变形)
    # Create cubic bounding box to simulate equal aspect ratio
    max_range = np.array([nodes[:, 0].max() - nodes[:, 0].min(),
                          nodes[:, 1].max() - nodes[:, 1].min(),
                          nodes[:, 2].max() - nodes[:, 2].min()]).max() / 2.0

    mid_x = (nodes[:, 0].max() + nodes[:, 0].min()) * 0.5
    mid_y = (nodes[:, 1].max() + nodes[:, 1].min()) * 0.5
    mid_z = (nodes[:, 2].max() + nodes[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.colorbar(p, label='Z Height (m)', shrink=0.7)
    plt.show()


if __name__ == "__main__":
    main()