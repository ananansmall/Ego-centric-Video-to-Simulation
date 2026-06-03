"""
232 场景深入分析：坐标系对齐 + 物体摆放 + 点云分区质量
"""
import numpy as np
import trimesh
import os
import cv2
import json

OUTPUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/232'

print("=" * 80)
print("232 场景深入分析：坐标系对齐 + 物体摆放 + 点云分区质量")
print("=" * 80)

# 加载点云
pcd = trimesh.load(os.path.join(OUTPUT_DIR, 'point_cloud.ply'))
vertices = pcd.vertices

# ============================================================
# 分析1: Z轴是否真的代表"上"方向？地板是否在Z=0？
# ============================================================
print("\n" + "=" * 60)
print("分析1: 坐标系对齐验证 — Z轴是否朝上？地板是否在Z≈0？")
print("=" * 60)

# 检查Z轴分布
z_vals = vertices[:, 2]
z_hist, z_edges = np.histogram(z_vals, bins=50)
z_densities = z_hist / len(z_vals)

# 找Z值最密集的区域（应该是地板）
peak_idx = np.argmax(z_hist)
z_peak = (z_edges[peak_idx] + z_edges[peak_idx + 1]) / 2
print(f"  Z轴最密集区域: z ≈ {z_peak:.3f}m (占比 {z_densities[peak_idx]:.2%})")

# 检查Z=0附近有没有大量点（地板应该在Z=0）
z_near_zero = np.sum((z_vals > -0.05) & (z_vals < 0.05))
z_near_zero_ratio = z_near_zero / len(z_vals)
print(f"  Z ∈ [-0.05, 0.05] 的点数: {z_near_zero} ({z_near_zero_ratio:.2%})")

# 检查Z最小值
z_min = z_vals.min()
print(f"  Z最小值: {z_min:.3f}m")
if z_min > 0.1:
    print(f"  ❌ Z最小值 > 0.1m → 地板不在Z=0，坐标系对齐有问题！")
    print(f"     可能原因：VGGT点云差导致房间坐标系对齐失败")
else:
    print(f"  ✅ Z最小值接近0，地板在Z≈0")

# 检查Z轴方向是否朝上
# 如果Z轴朝上，那么大部分点应该在Z>0的区域
z_positive_ratio = np.sum(z_vals > 0) / len(z_vals)
print(f"  Z > 0 的点占比: {z_positive_ratio:.2%}")
if z_positive_ratio > 0.9:
    print(f"  ✅ 大部分点在Z>0，Z轴方向合理")
else:
    print(f"  ⚠️ Z>0的点占比不够高，Z轴方向可能有问题")

# ============================================================
# 分析2: 地板平面拟合质量
# ============================================================
print("\n" + "=" * 60)
print("分析2: 地板平面拟合质量")
print("=" * 60)

# 假设Z值最低5%的点属于地板
z_floor_thresh = np.percentile(z_vals, 5)
floor_mask = z_vals < z_floor_thresh
floor_pts = vertices[floor_mask]
print(f"  地板候选点数: {len(floor_pts)} (Z < {z_floor_thresh:.3f})")

# PCA拟合地板平面
floor_centered = floor_pts - floor_pts.mean(axis=0)
floor_cov = np.dot(floor_centered.T, floor_centered) / len(floor_centered)
floor_eigenvalues, floor_eigenvectors = np.linalg.eigh(floor_cov)
floor_eigenvalues = floor_eigenvalues[::-1]
floor_eigenvectors = floor_eigenvectors[:, ::-1]

floor_normal = floor_eigenvectors[:, 2]  # 最小特征值对应的向量
print(f"  地板法向量: {floor_normal}")
print(f"  地板法向量与Z轴夹角: {np.degrees(np.arccos(np.clip(abs(np.dot(floor_normal, [0,0,1])), 0, 1))):.1f}°")

# 计算地板平面拟合误差
d = -np.dot(floor_normal, floor_pts.mean(axis=0))
distances = np.abs(np.dot(floor_pts, floor_normal) + d) / np.linalg.norm(floor_normal)
print(f"  地板平面拟合: mean_dist={distances.mean():.4f}m, max_dist={distances.max():.4f}m, std={distances.std():.4f}m")
if distances.mean() > 0.02:
    print(f"  ❌ 地板平面拟合误差 > 0.02m → 地板点云混乱，坐标系对齐不可靠")
else:
    print(f"  ✅ 地板平面拟合误差 < 0.02m → 地板点云质量OK")

# ============================================================
# 分析3: 墙壁平面检测
# ============================================================
print("\n" + "=" * 60)
print("分析3: 墙壁平面检测")
print("=" * 60)

# 检查X和Y方向的边界区域是否有平面结构
for axis_name, axis_idx in [('X', 0), ('Y', 1)]:
    axis_vals = vertices[:, axis_idx]
    # 取最外侧5%的点
    low_thresh = np.percentile(axis_vals, 5)
    high_thresh = np.percentile(axis_vals, 95)
    
    low_wall_pts = vertices[axis_vals < low_thresh]
    high_wall_pts = vertices[axis_vals > high_thresh]
    
    for side, wall_pts in [("低侧", low_wall_pts), ("高侧", high_wall_pts)]:
        if len(wall_pts) < 100:
            continue
        wall_centered = wall_pts - wall_pts.mean(axis=0)
        wall_cov = np.dot(wall_centered.T, wall_centered) / len(wall_centered)
        wall_eigenvalues, wall_eigenvectors = np.linalg.eigh(wall_cov)
        wall_eigenvalues = wall_eigenvalues[::-1]
        
        wall_normal = wall_eigenvectors[:, 2]
        # 检查法向量是否接近坐标轴
        axis_alignment = max(abs(wall_normal[axis_idx]), abs(wall_normal[0]), abs(wall_normal[1]), abs(wall_normal[2]))
        is_aligned_to_axis = abs(wall_normal[axis_idx]) > 0.9
        
        d = -np.dot(wall_normal, wall_pts.mean(axis=0))
        distances = np.abs(np.dot(wall_pts, wall_normal) + d) / np.linalg.norm(wall_normal)
        
        print(f"  {axis_name}轴{side}: {len(wall_pts)}点, "
              f"法向量={wall_normal.round(3)}, "
              f"对齐坐标轴={'✅' if is_aligned_to_axis else '❌'}, "
              f"拟合误差={distances.mean():.4f}m")

# ============================================================
# 分析4: 物体摆放位置分析
# ============================================================
print("\n" + "=" * 60)
print("分析4: 物体摆放位置分析")
print("=" * 60)

# 加载精修后场景
glb_path = os.path.join(OUTPUT_DIR, 'final_scene_refined.glb')
scene = trimesh.load(glb_path)

# 加载关系JSON
json_path = os.path.join(OUTPUT_DIR, 'relations_refined.json')
with open(json_path, 'r') as f:
    relations = json.load(f)

# 分析每个物体
geometries = list(scene.geometry.values())
geom_names = list(scene.geometry.keys())

# optimal_frames 中的物体信息
optimal_dir = os.path.join(OUTPUT_DIR, 'optimal_frames')
optimal_files = sorted(os.listdir(optimal_dir))

print(f"\n  物体总数: {len(geometries)}")
print(f"  关系定义: {len(relations)} 个")

for i, (name, geom) in enumerate(scene.geometry.items()):
    if not hasattr(geom, 'bounds'):
        continue
    bmin, bmax = geom.bounds
    center = (bmin + bmax) / 2
    extents = bmax - bmin
    z_min = bmin[2]
    z_max = bmax[2]
    
    # 从 optimal_frames 推断物体名
    obj_name = optimal_files[i] if i < len(optimal_files) else f"geometry_{i}"
    
    # 查找关系
    relation = "未知"
    for key, rel in relations.items():
        if key.startswith(obj_name.split('_inst')[0]) or key == f"geometry_{i}":
            relation = rel
            break
    
    # 判断摆放是否合理
    issues = []
    if z_min > 0.15:
        issues.append(f"底部离地{z_min:.3f}m（悬浮）")
    if z_min < -0.05:
        issues.append(f"底部在地面以下{z_min:.3f}m（穿地）")
    if extents[2] > 1.0:
        issues.append(f"Z方向尺寸{extents[2]:.3f}m（异常大）")
    
    status = "✅" if not issues else "⚠️"
    print(f"\n  {status} {obj_name}")
    print(f"     中心: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"     尺寸: ({extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f})")
    print(f"     Z范围: [{z_min:.3f}, {z_max:.3f}]")
    print(f"     关系: {relation}")
    if issues:
        for issue in issues:
            print(f"     ❌ {issue}")

# ============================================================
# 分析5: 点云分区质量 — 桌子区域 vs 其他区域
# ============================================================
print("\n" + "=" * 60)
print("分析5: 点云分区质量 — 桌面区域 vs 地板区域")
print("=" * 60)

# 从optimal_frames知道桌子在frame 101
# 加载对应帧的深度图和颜色图
color_dir = os.path.join(OUTPUT_DIR, 'color')
depth_dir = os.path.join(OUTPUT_DIR, 'depth')

# 分析几帧的深度一致性
print("\n  深度图帧间差异分析（检测动态物体）:")
prev_depth = None
for frame_idx in [0, 20, 40, 60, 80, 100, 119]:
    depth_path = os.path.join(depth_dir, f'{frame_idx}.png')
    if not os.path.exists(depth_path):
        continue
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    
    if prev_depth is not None:
        # 计算与前一帧的深度差异
        valid = (depth > 0) & (prev_depth > 0)
        if valid.any():
            diff = np.abs(depth[valid] - prev_depth[valid])
            large_change = np.sum(diff > 0.1)  # 变化超过10cm
            large_change_ratio = large_change / np.sum(valid)
            print(f"    Frame {frame_idx-20}->{frame_idx}: "
                  f"平均深度变化={diff.mean():.4f}m, "
                  f"大变化(>10cm)占比={large_change_ratio:.2%}")
    prev_depth = depth

# ============================================================
# 分析6: 相机外参一致性 — 检查VGGT的SLAM质量
# ============================================================
print("\n" + "=" * 60)
print("分析6: 相机外参一致性 — VGGT SLAM质量")
print("=" * 60)

ext_dir = os.path.join(OUTPUT_DIR, 'extrinsics')
ext_files = sorted([f for f in os.listdir(ext_dir) if f.endswith('.txt')])

rotations = []
translations = []
for ef in ext_files:
    ext = np.loadtxt(os.path.join(ext_dir, ef))
    R = ext[:3, :3]
    t = ext[:3, 3]
    rotations.append(R)
    translations.append(t)

rotations = np.array(rotations)
translations = np.array(translations)

# 检查旋转矩阵是否是正交矩阵
orth_errors = []
for i in range(len(rotations)):
    R = rotations[i]
    orth_error = np.linalg.norm(R @ R.T - np.eye(3))
    orth_errors.append(orth_error)

print(f"  旋转矩阵正交性误差: mean={np.mean(orth_errors):.6f}, max={np.max(orth_errors):.6f}")
if np.max(orth_errors) > 0.01:
    print(f"  ❌ 旋转矩阵不正交 → VGGT外参质量差")
else:
    print(f"  ✅ 旋转矩阵正交性OK")

# 检查相机轨迹是否平滑
cam_positions = []
for i in range(len(rotations)):
    R = rotations[i]
    t = translations[i]
    cam_pos = -R.T @ t
    cam_positions.append(cam_pos)
cam_positions = np.array(cam_positions)

# 计算轨迹平滑度（二阶差分）
if len(cam_positions) > 2:
    accel = np.diff(cam_positions, n=2, axis=0)
    accel_mag = np.linalg.norm(accel, axis=1)
    print(f"  相机加速度: mean={accel_mag.mean():.6f}m, max={accel_mag.max():.6f}m")
    if accel_mag.max() > 0.01:
        print(f"  ⚠️ 相机轨迹有突变 → VGGT SLAM不稳定")
    else:
        print(f"  ✅ 相机轨迹平滑")

# ============================================================
# 分析7: 最终结论
# ============================================================
print("\n" + "=" * 60)
print("分析7: 综合结论")
print("=" * 60)

# 汇总所有问题
problems = []

# 检查1: Z轴对齐
if z_min > 0.1:
    problems.append("地板不在Z=0，坐标系对齐失败")
if z_positive_ratio < 0.9:
    problems.append("Z轴方向可能不正确")

# 检查2: 地板拟合
if distances.mean() > 0.02:
    problems.append("地板平面拟合误差大，点云混乱")

# 检查3: 物体悬浮/穿地
for name, geom in scene.geometry.items():
    if hasattr(geom, 'bounds'):
        bmin = geom.bounds[0]
        if bmin[2] > 0.15:
            problems.append(f"{name} 悬浮 (z_min={bmin[2]:.3f})")
        if bmin[2] < -0.05:
            problems.append(f"{name} 穿地 (z_min={bmin[2]:.3f})")

# 检查4: 相机轨迹
if np.max(orth_errors) > 0.01:
    problems.append("VGGT外参不正交")

if problems:
    print("\n  ❌ 发现以下问题:")
    for p in problems:
        print(f"    - {p}")
else:
    print("\n  ✅ 未发现明显问题")

print("\n" + "=" * 80)
print("深入分析完成")
print("=" * 80)
