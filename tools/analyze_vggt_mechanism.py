"""
VGGT 点云构建模式分析 + 相机运动分析
核心问题：相机在移动时，VGGT是如何建立世界坐标系的？
"""
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/232'

print("=" * 80)
print("VGGT 点云构建模式分析 + 相机运动分析")
print("=" * 80)

ext_dir = os.path.join(OUTPUT_DIR, 'extrinsics')
depth_dir = os.path.join(OUTPUT_DIR, 'depth')
color_dir = os.path.join(OUTPUT_DIR, 'color')
intrinsic = np.loadtxt(os.path.join(OUTPUT_DIR, 'intrinsic.txt'))
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]

# ============================================================
# 分析1: 相机运动轨迹
# ============================================================
print("\n" + "=" * 60)
print("分析1: 相机运动轨迹")
print("=" * 60)

ext_files = sorted([f for f in os.listdir(ext_dir) if f.endswith('.txt')])
cam_positions = []
cam_lookats = []

for ef in ext_files:
    ext = np.loadtxt(os.path.join(ext_dir, ef))
    R = ext[:3, :3]
    t = ext[:3, 3]
    cam_pos = -R.T @ t  # world space camera position
    cam_lookat = R.T @ np.array([0, 0, 1])  # camera forward direction in world
    cam_positions.append(cam_pos)
    cam_lookats.append(cam_lookat)

cam_positions = np.array(cam_positions)
cam_lookats = np.array(cam_lookats)

total_displacement = np.linalg.norm(cam_positions[-1] - cam_positions[0])
total_path = np.linalg.norm(np.diff(cam_positions, axis=0), axis=1).sum()

print(f"\n  相机位置范围:")
print(f"    X: [{cam_positions[:,0].min():.4f}, {cam_positions[:,0].max():.4f}]")
print(f"    Y: [{cam_positions[:,1].min():.4f}, {cam_positions[:,1].max():.4f}]")
print(f"    Z: [{cam_positions[:,2].min():.4f}, {cam_positions[:,2].max():.4f}]")
print(f"  首帧→末帧总位移: {total_displacement:.4f}m")
print(f"  累计路径长度: {total_path:.4f}m")
print(f"  相邻帧平均位移: {np.linalg.norm(np.diff(cam_positions, axis=0), axis=1).mean():.4f}m")

# 判断相机运动模式
pos_std = cam_positions.std(axis=0)
print(f"\n  相机位置标准差: X={pos_std[0]:.4f}, Y={pos_std[1]:.4f}, Z={pos_std[2]:.4f}")

if total_displacement < 0.05:
    print(f"  📌 相机模式: 基本固定（三脚架/固定机位）")
elif total_displacement < 0.3:
    print(f"  📌 相机模式: 小幅移动（手持微晃）")
elif total_displacement < 1.0:
    print(f"  📌 相机模式: 中等移动")
else:
    print(f"  📌 相机模式: 大幅移动")

# 查看相机朝向变化
lookat_angles = []
for i in range(len(cam_lookats)):
    angle = np.degrees(np.arccos(np.clip(np.dot(cam_lookats[i], cam_lookats[0]), 0, 1)))
    lookat_angles.append(angle)

print(f"\n  相机朝向变化:")
print(f"    首帧朝向: {cam_lookats[0].round(4)}")
print(f"    末帧朝向: {cam_lookats[-1].round(4)}")
print(f"    最大偏转角度: {max(lookat_angles):.2f}°")

# ============================================================
# 分析2: VGGT 的点云构建模式
# ============================================================
print("\n" + "=" * 60)
print("分析2: VGGT 点云构建模式")
print("=" * 60)
print("""
VGGT (Visual Geometry Grounded Transformer) 是一种基于Transformer的视频3D重建模型。
它的工作机制与传统的SfM/SLAM有本质不同：

传统SLAM:
  帧→特征提取→匹配→三角化→BA优化→稀疏/稠密点云
  逐帧递增构建，有漂移累积

VGGT:
  多帧输入→Transformer编码→联合预测(depth + extrinsic + world_points)
  所有帧同时处理，全局优化，无漂移累积

VGGT输出三类数据的关系:
  world_points[i] = f(image[i], extrinsics[i], depths[i])
  其中:
    - depths[i]: 第i帧每个像素的相机坐标系Z值（沿光轴深度）
    - extrinsics[i]: 第i帧的c2w变换矩阵，把相机坐标转到世界坐标
    - world_points[i]: 第i帧每个像素的3D世界坐标

  反投影公式:
    相机坐标: (x_cam, y_cam, z_cam) = ((u-cx)*z/fx, (v-cy)*z/fy, z)
    世界坐标: (x_w, y_w, z_w, 1) = extrinsic × (x_cam, y_cam, z_cam, 1)

关键点:
  - world_points 和 extrinsics 共享同一个世界坐标系
  - 如果VGGT重建完美，同一物理点在不同帧的world_points应该重合
  - 动态物体破坏了这个假设
""")

# ============================================================
# 分析3: 验证世界坐标系一致性
# ============================================================
print("=" * 60)
print("分析3: 验证世界坐标系一致性")
print("=" * 60)
print("""
验证方法: 从两帧分别反投影"同一个像素"，看它们的world_points是否接近。
如果世界坐标系一致，且该像素对应静态场景，两帧的world_points应该重合。
如果差异很大（>10cm），说明:
  a) 该像素对应动态物体 → VGGT重建被破坏
  b) 或者VGGT的世界坐标系本身就不一致
""")

# 取几个"应该静态"的像素（图像角落/边缘，不太可能有动态物体）
static_test_pixels = [
    (10, 10, "左上角(可能静态)"),
    (w_im := cv2.imread(os.path.join(color_dir, '0.jpg')).shape[1], 
     10, "右上角(可能静态)"),
    (10, cv2.imread(os.path.join(color_dir, '0.jpg')).shape[0] - 10, "左下角(可能静态)"),
    (w_im - 10, cv2.imread(os.path.join(color_dir, '0.jpg')).shape[0] - 10, "右下角(可能静态)"),
]

# 对每对帧，反投影这些像素，比较3D坐标
test_frames = [(0, 30), (0, 60), (0, 90), (0, 119), (30, 60), (90, 101)]

print("\n  逐帧对比较同一像素的世界坐标:")
print(f"  {'像素位置':<20} {'帧对':<15} {'帧A世界Z':>10} {'帧B世界Z':>10} {'差异':>10} {'判定'}")
print(f"  {'-'*20} {'-'*15} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

all_static_diffs = []

for px_x, px_y, px_name in static_test_pixels:
    if px_x >= w_im or px_y >= cv2.imread(os.path.join(color_dir, '0.jpg')).shape[0]:
        continue
    for fa, fb in test_frames:
        # Frame A
        ext_a = np.loadtxt(os.path.join(ext_dir, f'{fa}.txt'))
        Ra, ta = ext_a[:3, :3], ext_a[:3, 3]
        depth_a = cv2.imread(os.path.join(depth_dir, f'{fa}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        z_a = depth_a[px_y, px_x]
        if z_a <= 0:
            continue
        x_a = (px_x - cx) * z_a / fx
        y_a = (px_y - cy) * z_a / fy
        pt_world_a = Ra @ np.array([x_a, y_a, z_a]) + ta
        
        # Frame B
        ext_b = np.loadtxt(os.path.join(ext_dir, f'{fb}.txt'))
        Rb, tb = ext_b[:3, :3], ext_b[:3, 3]
        depth_b = cv2.imread(os.path.join(depth_dir, f'{fb}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        z_b = depth_b[px_y, px_x]
        if z_b <= 0:
            continue
        x_b = (px_x - cx) * z_b / fx
        y_b = (px_y - cy) * z_b / fy
        pt_world_b = Rb @ np.array([x_b, y_b, z_b]) + tb
        
        diff = np.linalg.norm(pt_world_a - pt_world_b)
        all_static_diffs.append(diff)
        
        status = "✅" if diff < 0.1 else ("⚠️" if diff < 0.3 else "❌")
        print(f"  {px_name:<20} {fa}-{fb:<11} {pt_world_a[2]:>10.3f} {pt_world_b[2]:>10.3f} {diff:>10.3f} {status}")

if all_static_diffs:
    print(f"\n  汇总:")
    print(f"    平均差异: {np.mean(all_static_diffs):.4f}m")
    print(f"    最大差异: {np.max(all_static_diffs):.4f}m")
    good_ratio = np.mean(np.array(all_static_diffs) < 0.1)
    bad_ratio = np.mean(np.array(all_static_diffs) > 0.3)
    print(f"    差异<10cm占比: {good_ratio:.2%}")
    print(f"    差异>30cm占比: {bad_ratio:.2%}")
    if bad_ratio > 0.2:
        print(f"    ❌ 大量静态像素的世界坐标不一致 → VGGT的世界坐标系本身不稳定")
    elif good_ratio > 0.8:
        print(f"    ✅ 大部分静态像素的世界坐标一致 → 世界坐标系基本稳定")
    else:
        print(f"    ⚠️ 部分一致部分不一致 → 中等质量")

# ============================================================
# 分析4: 检查深度一致性 vs 相机移动的关系
# ============================================================
print("\n" + "=" * 60)
print("分析4: 深度变化 vs 相机移动关系")
print("=" * 60)

# 如果相机在移动，同一像素在不同帧对应不同物体是正常的
# 但如果相机几乎不动，深度差异就说明VGGT不稳定
print(f"\n  总位移: {total_displacement:.4f}m")
print(f"  平均帧间位移: {np.linalg.norm(np.diff(cam_positions, axis=0), axis=1).mean():.4f}m")

# 采样几对相邻帧，看深度变化
print(f"\n  相邻帧深度对比（整图有效区域）:")
for i in [0, 20, 40, 60, 80, 100]:
    if i + 1 >= len(ext_files):
        break
    depth_a = cv2.imread(os.path.join(depth_dir, f'{i}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    depth_b = cv2.imread(os.path.join(depth_dir, f'{i+1}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    
    valid = (depth_a > 0) & (depth_b > 0)
    if valid.sum() > 0:
        diff = np.abs(depth_a[valid] - depth_b[valid])
        cam_move = np.linalg.norm(cam_positions[i+1] - cam_positions[i])
        print(f"  Frame {i}→{i+1}: 相机位移={cam_move:.5f}m, "
              f"深度平均变化={diff.mean():.4f}m, "
              f"深度中位变化={np.median(diff):.4f}m")

# ============================================================
# 分析5: VGGT世界坐标系构建的具体模式
# ============================================================
print("\n" + "=" * 60)
print("分析5: VGGT世界坐标系的定义方式")
print("=" * 60)

# 查看第一帧的相机位置和外参
ext0 = np.loadtxt(os.path.join(ext_dir, '0.txt'))
R0, t0 = ext0[:3, :3], ext0[:3, 3]
cam_pos0 = -R0.T @ t0

print(f"\n  Frame 0:")
print(f"    外参 R:\n{R0.round(4)}")
print(f"    外参 t: {t0.round(4)}")
print(f"    相机位置: {cam_pos0.round(4)}")
print(f"    det(R) = {np.linalg.det(R0):.6f}")

# 通常VGGT用第一帧定义世界坐标系
# 第一帧的相机位置是原点，或者某个标准位置
print(f"\n  世界坐标系定义:")
print(f"    首帧相机位置: {cam_pos0.round(4)}")
print(f"    (VGGT通常以首帧或场景中心定义世界坐标系)")

# 检查所有相机位置是否合理
# 相机应该都在场景的"观察者"一侧
print(f"\n  相机位置分布:")
print(f"    所有相机平均位置: {cam_positions.mean(axis=0).round(4)}")

# ============================================================
# 分析6: 点云中动态物体的证据
# ============================================================
print("\n" + "=" * 60)
print("分析6: 点云中动态物体的直接证据")
print("=" * 60)

# 加载点云
import trimesh
pcd = trimesh.load(os.path.join(OUTPUT_DIR, 'point_cloud.ply'))
vertices = pcd.vertices

# 看点云密度分布 - 动态物体通常表现为"ghost trail"
# 在XZ平面上看点云投影
print(f"\n  点云总点数: {len(vertices)}")

# 分析点云的Z分层 - 如果有多层"地板-like"结构，可能是动态物体留下的残影
z_hist, z_edges = np.histogram(vertices[:, 2], bins=200)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2

from scipy.signal import find_peaks
peaks, props = find_peaks(z_hist, height=z_hist.max()*0.02, distance=10)

print(f"\n  点云Z方向峰值（水平面）:")
for pi in peaks:
    # 计算该Z层的XY范围
    mask = (vertices[:, 2] >= z_edges[pi]) & (vertices[:, 2] < z_edges[pi+1])
    layer_pts = vertices[mask]
    print(f"    Z={z_centers[pi]:.3f}m, 点数={z_hist[pi]}, "
          f"XY范围=[{layer_pts[:,0].min():.3f},{layer_pts[:,0].max():.3f}]×"
          f"[{layer_pts[:,1].min():.3f},{layer_pts[:,1].max():.3f}]")

# 检查是否有"模糊"的地板平面（多点云层叠）
near_floor = vertices[vertices[:, 2] < 0.3]
if len(near_floor) > 0:
    z_floor_std = near_floor[:, 2].std()
    print(f"\n  Z<0.3m区域的Z标准差: {z_floor_std:.4f}m")
    if z_floor_std > 0.05:
        print(f"  ⚠️ 地板区域Z分布很宽（std>5cm），可能有动态物体残影叠加")
    else:
        print(f"  ✅ 地板区域Z分布集中")

# ============================================================
# 总览
# ============================================================
print("\n" + "=" * 60)
print("总结: VGGT 点云构建机制")
print("=" * 60)
print("""
VGGT的点云构建方式:
  1. VGGT接收多帧RGB图像作为输入
  2. 通过Transformer联合预测:
     - 每帧的深度图(depth) → 决定每个像素到相机的距离
     - 每帧的相机外参(extrinsic) → 决定相机在世界坐标系中的位置和朝向
     - 每帧的世界坐标点云(world_points) → 每个像素的3D世界坐标
  3. 所有帧共享同一个世界坐标系（通过extrinsic关联）
  4. world_points = extrinsic × backproject(depth, intrinsic)
  
  不是传统SLAM那种"逐帧增量"模式，
  而是"所有帧一次性全局预测"模式。

同一物理点在不同帧的world_points应该是重合的（这是理想情况）。
动态物体破坏了VGGT的多视角一致性假设，
导致深度预测不稳定、外参估计可能出现跳变。
""")