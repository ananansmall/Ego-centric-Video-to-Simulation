"""
232 场景核心问题分析：
1. 桌子悬浮为什么没被 refine 修复
2. 点云中桌子的实际位置 vs mesh 位置
3. 到底是点云混乱导致位置错，还是 VGGT extrinsic 本身就差
"""
import numpy as np
import trimesh
import os
import cv2
import json
import sys

sys.path.insert(0, '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene')

OUTPUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/232'

print("=" * 80)
print("232 场景核心问题深度分析")
print("=" * 80)

# ============================================================
# 问题1: 桌子悬浮0.47m，为什么 refine_supported_by_floor_object 没修复？
# ============================================================
print("\n" + "=" * 60)
print("问题1: refine_supported_by_floor_object 为什么没修复桌子？")
print("=" * 60)

# 加载精修后场景
glb_path = os.path.join(OUTPUT_DIR, 'final_scene_refined.glb')
scene = trimesh.load(glb_path)

# 找到 table 对应的 geometry (从 optimal_frames 知道 table 是 inst0_frame101)
# geometry_5 是 table (从之前的分析知道 z_min=0.470)
geometries = list(scene.geometry.items())
table_geom = None
table_name = None
for name, geom in geometries:
    if hasattr(geom, 'bounds'):
        bmin, bmax = geom.bounds
        # table 的 z_min 应该是 0.470
        if abs(bmin[2] - 0.470) < 0.01:
            table_geom = geom
            table_name = name
            break

if table_geom is None:
    print("  未找到 table geometry，尝试通过尺寸查找...")
    for name, geom in geometries:
        if hasattr(geom, 'bounds'):
            bmin, bmax = geom.bounds
            extents = bmax - bmin
            # table 大概 0.096 x 0.066 x 0.086
            if abs(extents[0] - 0.096) < 0.01 and abs(extents[2] - 0.086) < 0.01:
                table_geom = geom
                table_name = name
                break

if table_geom is not None:
    print(f"  找到 table: {table_name}")
    bmin, bmax = table_geom.bounds
    print(f"  bounds: [{bmin.round(3)}, {bmax.round(3)}]")
    print(f"  z_min = {bmin[2]:.3f}m")
    
    # 模拟 refine_supported_by_floor_object 的逻辑
    print(f"\n  --- 模拟 refine_supported_by_floor_object 逻辑 ---")
    
    # 注意：final_scene_refined.glb 已经是 y-up（经过 z-up to y-up 变换）
    # 但 refine 是在 z-up 坐标系下执行的
    # 所以我们需要从原始 T 矩阵分析，而不是从 glb 分析
    
    print("  ⚠️ 注意：glb 文件已经过 z-up → y-up 变换")
    print("  需要从原始数据（z-up 坐标系）分析 refine 逻辑")
else:
    print("  ❌ 未找到 table geometry")

# ============================================================
# 直接从代码逻辑分析 refine 为什么失败
# ============================================================
print(f"\n  --- refine_supported_by_floor_object 代码逻辑分析 ---")
print(f"  代码路径: src/sp_refinement.py:61-98")
print()
print(f"  Step 1: 检查物体朝上方向")
print(f"    upper_transformed_vector = T[:3,1] / norm(T[:3,1])")
print(f"    theta_gravity = angle(upper_transformed_vector, [0,0,1])")
print(f"    如果 theta_gravity < 10° 或 > 170° → 对齐到重力方向")
print(f"    否则 → 不对齐（保持原样）")
print()
print(f"  Step 2: 对齐底部到地板")
print(f"    z_min = transformed_mesh.bounds[0, 2]")
print(f"    如果 abs(z_min) < 0.3 → 平移使 z_min = 0")
print(f"    否则 → 不平移（保持原样）")
print()
print(f"  ❌ 关键问题：table 的 z_min = 0.470m > 0.3m")
print(f"     代码认为 '如果物体底部离地超过0.3m，就不应该强制吸附到地板'")
print(f"     这个 0.3m 阈值的设计意图是：避免把本来就在桌上/墙上的物体错误地吸附到地板")
print(f"     但在坐标系对齐错误的情况下，桌子本身就被放高了，0.3m 阈值反而阻止了修复")
print()
print(f"  💡 根本原因：refine 的设计假设坐标系对齐是正确的")
print(f"     它只做微调（0.3m以内），不做大范围位置修正")
print(f"     当坐标系本身就错了，refine 无法修复")

# ============================================================
# 问题2: 点云中桌子的实际位置 vs mesh 位置
# ============================================================
print("\n" + "=" * 60)
print("问题2: 点云中桌子的实际位置 vs 生成的 mesh 位置")
print("=" * 60)

# 加载点云
pcd = trimesh.load(os.path.join(OUTPUT_DIR, 'point_cloud.ply'))
vertices = pcd.vertices

# 加载颜色图 frame 101 (table 的最优帧)
color_path = os.path.join(OUTPUT_DIR, 'color', '101.jpg')
img_101 = cv2.imread(color_path)
h, w = img_101.shape[:2]
print(f"  Frame 101 图像尺寸: {w}x{h}")

# 加载深度图 frame 101
depth_path = os.path.join(OUTPUT_DIR, 'depth', '101.png')
depth_101 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
print(f"  Frame 101 深度范围: [{depth_101[depth_101>0].min():.3f}, {depth_101[depth_101>0].max():.3f}]m")

# 加载外参 frame 101
ext_101 = np.loadtxt(os.path.join(OUTPUT_DIR, 'extrinsics', '101.txt'))
R_101 = ext_101[:3, :3]
t_101 = ext_101[:3, 3]
cam_pos_101 = -R_101.T @ t_101
print(f"  Frame 101 相机位置: {cam_pos_101.round(4)}")

# 加载内参
intrinsic = np.loadtxt(os.path.join(OUTPUT_DIR, 'intrinsic.txt'))
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]

# 从深度图反投影桌子区域到3D空间
# 先看看图像中间区域（桌子大概在图像中间）
# 采样图像中心 1/3 区域
roi_y1, roi_y2 = h // 3, 2 * h // 3
roi_x1, roi_x2 = w // 3, 2 * w // 3
roi_depth = depth_101[roi_y1:roi_y2, roi_x1:roi_x2]

print(f"\n  图像中心区域深度统计 (大概桌子区域):")
print(f"    深度范围: [{roi_depth[roi_depth>0].min():.3f}, {roi_depth[roi_depth>0].max():.3f}]m")
print(f"    平均深度: {roi_depth[roi_depth>0].mean():.3f}m")

# 反投影到相机坐标系
v_coords, u_coords = np.where(depth_101 > 0)
z_cam = depth_101[v_coords, u_coords]
x_cam = (u_coords - cx) * z_cam / fx
y_cam = (v_coords - cy) * z_cam / fy
pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

# 转换到世界坐标系
# VGGT 的 extrinsic 是 c2w 格式
pts_world = (R_101 @ pts_cam.T + t_101.reshape(3, 1)).T

print(f"\n  反投影3D点统计:")
print(f"    总点数: {len(pts_world)}")
print(f"    X 范围: [{pts_world[:,0].min():.3f}, {pts_world[:,0].max():.3f}]")
print(f"    Y 范围: [{pts_world[:,1].min():.3f}, {pts_world[:,1].max():.3f}]")
print(f"    Z 范围: [{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}]")

# 分析桌子区域（中心偏下，深度约0.5-0.8m的区域）
# 从深度图看，桌子大概在深度0.5-0.8m的区域
table_depth_mask = (depth_101 > 0.4) & (depth_101 < 0.9)
# 限制在图像下半部分（桌子在图像下方）
table_depth_mask[:h//3, :] = False

v_t, u_t = np.where(table_depth_mask)
z_t = depth_101[v_t, u_t]
x_t = (u_t - cx) * z_t / fx
y_t = (v_t - cy) * z_t / fy
pts_cam_t = np.stack([x_t, y_t, z_t], axis=-1)
pts_world_t = (R_101 @ pts_cam_t.T + t_101.reshape(3, 1)).T

print(f"\n  桌子候选区域3D点 (深度0.4-0.9m, 图像下方2/3):")
print(f"    点数: {len(pts_world_t)}")
print(f"    X 范围: [{pts_world_t[:,0].min():.3f}, {pts_world_t[:,0].max():.3f}]")
print(f"    Y 范围: [{pts_world_t[:,1].min():.3f}, {pts_world_t[:,1].max():.3f}]")
print(f"    Z 范围: [{pts_world_t[:,2].min():.3f}, {pts_world_t[:,2].max():.3f}]")
print(f"    Z 均值: {pts_world_t[:,2].mean():.3f}m")
print(f"    Z 最小值: {pts_world_t[:,2].min():.3f}m")

# ============================================================
# 问题3: VGGT extrinsic 定位质量分析
# ============================================================
print("\n" + "=" * 60)
print("问题3: VGGT extrinsic 定位质量 — 到底是点云差还是 extrinsic 差？")
print("=" * 60)

# 加载所有外参
ext_dir = os.path.join(OUTPUT_DIR, 'extrinsics')
ext_files = sorted([f for f in os.listdir(ext_dir) if f.endswith('.txt')])

cam_positions = []
cam_rotations = []
for ef in ext_files:
    ext = np.loadtxt(os.path.join(ext_dir, ef))
    R = ext[:3, :3]
    t = ext[:3, 3]
    cam_pos = -R.T @ t
    cam_positions.append(cam_pos)
    cam_rotations.append(R)

cam_positions = np.array(cam_positions)

# 分析相机轨迹的合理性
print(f"\n  相机轨迹分析:")
print(f"    相机位置范围:")
print(f"      X: [{cam_positions[:,0].min():.4f}, {cam_positions[:,0].max():.4f}]")
print(f"      Y: [{cam_positions[:,1].min():.4f}, {cam_positions[:,1].max():.4f}]")
print(f"      Z: [{cam_positions[:,2].min():.4f}, {cam_positions[:,2].max():.4f}]")
print(f"    相机位置标准差: {cam_positions.std(axis=0).round(4)}")

# 如果相机几乎不动（标准差很小），说明是固定摄像头
# 如果相机移动范围很大，说明是手持/移动摄像头
pos_range = cam_positions.max(axis=0) - cam_positions.min(axis=0)
print(f"    相机移动范围: {pos_range.round(4)}")

if pos_range.max() < 0.1:
    print(f"    ✅ 相机基本固定（移动 < 0.1m）")
    print(f"    → VGGT 的 SLAM 应该比较稳定")
elif pos_range.max() < 0.5:
    print(f"    ⚠️ 相机有小幅移动（0.1-0.5m）")
else:
    print(f"    ❌ 相机有大幅移动（> 0.5m）")

# 分析 VGGT extrinsic 对桌子定位的影响
# 关键：T = matrix_ext_inv @ matrix_adjust @ matrix_l2c @ matrix_y2z
# 其中 matrix_ext_inv = inv(extrinsic)
# 如果 extrinsic 的平移部分有误，T 的平移也会错
print(f"\n  Frame 101 外参分析 (table 最优帧):")
print(f"    R = \n{R_101.round(4)}")
print(f"    t = {t_101.round(4)}")
print(f"    相机位置 = {cam_pos_101.round(4)}")

# 检查 det(R) 是否为1
det_R = np.linalg.det(R_101)
print(f"    det(R) = {det_R:.6f} (应该为1.0)")

# ============================================================
# 核心分析：mesh 位置 vs 点云位置 vs 图像位置
# ============================================================
print("\n" + "=" * 60)
print("核心分析: mesh 位置 vs 点云位置 — 重定位到底准不准？")
print("=" * 60)

# 从点云中找桌子区域
# 点云中 z < 0.3 的点可能是地板/桌面
# 桌面高度大约在 0.7-0.8m（正常桌子高度）
# 但如果坐标系对齐有问题，这个高度可能不对

# 先看点云的 Z 分布直方图
z_vals = vertices[:, 2]
z_hist, z_edges = np.histogram(z_vals, bins=100)
z_centers = (z_edges[:-1] + z_edges[1:]) / 2

# 找 Z 方向的峰值（代表不同的水平面）
from scipy.signal import find_peaks
peaks, properties = find_peaks(z_hist, height=z_hist.max()*0.1, distance=5)

print(f"\n  点云 Z 方向分布峰值（代表水平面）:")
for peak_idx in peaks:
    z_val = z_centers[peak_idx]
    point_count = z_hist[peak_idx]
    print(f"    Z = {z_val:.3f}m, 点数 = {point_count}")

# 分析每个峰值可能代表什么
print(f"\n  峰值解读:")
for peak_idx in peaks:
    z_val = z_centers[peak_idx]
    if z_val < 0.1:
        interpretation = "地板"
    elif z_val < 0.3:
        interpretation = "地板/低矮物体"
    elif z_val < 0.6:
        interpretation = "桌子中部/椅子"
    elif z_val < 0.9:
        interpretation = "桌面高度"
    elif z_val < 1.2:
        interpretation = "桌面上物体/墙壁中部"
    else:
        interpretation = "墙壁上部/天花板"
    print(f"    Z = {z_val:.3f}m → 可能是: {interpretation}")

# ============================================================
# 关键对比：如果点云定位是对的，那 mesh 应该在哪？
# ============================================================
print("\n" + "=" * 60)
print("关键对比: 如果点云定位正确，mesh 应该放在哪里？")
print("=" * 60)

# 从点云中提取桌子区域
# 假设桌子在 X: [-0.3, 0.0], Y: [0.1, 0.3] 区域（从 mesh 位置推断）
# mesh 中心: (-0.181, 0.197, 0.513)，尺寸: (0.096, 0.066, 0.086)
# 所以桌子大概在 X: [-0.23, -0.13], Y: [0.16, 0.23], Z: [0.47, 0.56]

table_x_range = (-0.3, 0.0)
table_y_range = (0.1, 0.3)

# 在点云中找这个区域
table_region_mask = (
    (vertices[:, 0] >= table_x_range[0]) & (vertices[:, 0] <= table_x_range[1]) &
    (vertices[:, 1] >= table_y_range[0]) & (vertices[:, 1] <= table_y_range[1])
)
table_region_pts = vertices[table_region_mask]

print(f"\n  点云中桌子区域 (X∈{table_x_range}, Y∈{table_y_range}):")
print(f"    点数: {len(table_region_pts)}")
if len(table_region_pts) > 0:
    print(f"    Z 范围: [{table_region_pts[:,2].min():.3f}, {table_region_pts[:,2].max():.3f}]")
    print(f"    Z 均值: {table_region_pts[:,2].mean():.3f}m")
    
    # 找这个区域的 Z 峰值
    z_table_hist, z_table_edges = np.histogram(table_region_pts[:, 2], bins=50)
    z_table_centers = (z_table_edges[:-1] + z_table_edges[1:]) / 2
    peaks_table, _ = find_peaks(z_table_hist, height=z_table_hist.max()*0.2, distance=3)
    print(f"    Z 方向峰值:")
    for pi in peaks_table:
        print(f"      Z = {z_table_centers[pi]:.3f}m (点数={z_table_hist[pi]})")

print(f"\n  生成的 table mesh:")
print(f"    Z 范围: [0.470, 0.555]m")
print(f"    Z 中心: 0.513m")

# ============================================================
# 更精确的分析：用 VGGT 的 world_points 直接看桌子区域
# ============================================================
print("\n" + "=" * 60)
print("精确分析: 用深度图+外参反投影桌子区域")
print("=" * 60)

# 从 frame 101 的图像中，桌子大概在图像下半部分
# 更精确地：看整个深度图的 Z 分布
print(f"\n  Frame 101 深度图 Z 分布 (反投影到世界坐标):")
print(f"    全图 Z 范围: [{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}]")

# 按 Z 分层统计
z_layers = [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.2), (1.2, 2.0)]
for z_lo, z_hi in z_layers:
    mask = (pts_world[:, 2] >= z_lo) & (pts_world[:, 2] < z_hi)
    count = mask.sum()
    if count > 0:
        xy_center = pts_world[mask][:, :2].mean(axis=0)
        print(f"    Z ∈ [{z_lo:.1f}, {z_hi:.1f}): {count} 点, XY中心=({xy_center[0]:.3f}, {xy_center[1]:.3f})")

# ============================================================
# 最终结论
# ============================================================
print("\n" + "=" * 60)
print("最终结论")
print("=" * 60)

print("""
问题1: 桌子悬浮0.47m为什么没被refine修复？

  refine_supported_by_floor_object 的逻辑：
  1. 先对齐朝上方向（theta_gravity < 10° 才对齐）
  2. 再吸附底部到地板（abs(z_min) < 0.3m 才吸附）
  
  桌子 z_min = 0.470m > 0.3m → 不满足吸附条件 → refine 跳过
  
  设计意图：0.3m 阈值是为了避免把桌上的物体错误吸附到地板
  但当坐标系本身就错了时，这个阈值反而阻止了修复
  
  ❌ 根本原因：refine 只做微调，假设坐标系对齐是正确的

问题2: 点云的作用是重定位3D资产，那混乱的点云到底有没有影响？

  需要区分两个概念：
  a) 点云 → 坐标系对齐(R,t) → 影响所有物体的全局位置
  b) extrinsic → T矩阵 → 影响单个物体的局部位置
  
  如果坐标系对齐是错的，所有物体都会偏移
  如果 extrinsic 是错的，只有对应帧的物体会偏移
  
  从232的数据看：
  - 有些物体在正确位置（donut_2 z_min=0.006, toy_0 z_min=0.068）
  - 有些物体悬浮（table z_min=0.470, bowl z_min=0.548）
  
  这种不一致性说明：不是坐标系全局偏移，而是**单个物体的 T 矩阵有问题**
  
  T = inv(extrinsic) @ adjust @ l2c @ y2z
  其中 l2c 来自 SAM3D，adjust 和 y2z 是固定变换
  所以 T 的准确性取决于：
  1. VGGT 的 extrinsic 是否准确
  2. SAM3D 的 l2c 变换是否准确
  
  点云对 T 的影响是间接的：
  - 点云 → R,t → 修正后的 extrinsic → T
  - 如果 R,t 错了，所有 extrinsic 都会偏移，所有 T 都会偏移

问题3: 到底是点云差还是VGGT定位差？

  从相机轨迹看：相机几乎不动（移动范围 < 0.06m），VGGT SLAM 应该稳定
  从旋转矩阵看：正交性完美，数学计算没问题
  
  但从深度帧间差异看：存在大量动态物体（53%像素变化>10cm）
  
  💡 关键洞察：
  VGGT 的问题不是"定位差"，而是"动态物体导致3D重建不一致"
  - 静态区域（地板、墙壁）的3D点可能是准确的
  - 动态区域（被移动的物体）的3D点是混乱的
  - 当用动态区域的点云来定位物体时，位置就会错
  
  更具体地说：
  - SAM3D 生成 mesh 时使用了 pointmap 作为几何条件
  - 如果 pointmap 中该物体区域的3D点是混乱的
  - SAM3D 生成的 mesh 形状可能变形
  - l2c 变换（SAM3D输出）可能不准确
  - 最终 T 矩阵就不准
  
  所以答案是：**两者都有问题**
  1. 点云混乱 → SAM3D 输入差 → mesh 形状和 l2c 不准
  2. VGGT 对动态物体的 extrinsic 估计可能不准 → T 不准
  3. 两者叠加，导致物体位置错误
""")
