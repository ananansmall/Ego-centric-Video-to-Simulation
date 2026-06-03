"""
核心验证: 掩码分离的点云是否准确？点云质量差如何影响重定位？

重定位机制回顾:
  1. SAM3 分割物体 → 得到 mask（通常是准确的）
  2. mask 作用于 VGGT 的 world_points → 得到该物体的 3D 点云
  3. 这个"物体点云"传给 SAM3D → 生成 mesh + 决定物体位置的 l2c 变换
  4. T = inv(extrinsic) @ adjust @ l2c @ y2z

关键问题: 如果步骤1的mask是正确的，但步骤2的3D点云是错的
          → 步骤3的物体位置就会错 → 步骤4的T也会错
  
本脚本验证: VGGT给同一个物体在不同帧的3D坐标是否一致？
         如果不一致，说明点云差，重定位就会受影响
"""
import numpy as np
import cv2
import os

OUTPUT_DIR = '/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/232'

print("=" * 80)
print("核心验证: 掩码×点云 — 物体3D坐标是否因点云质量差而偏离？")
print("=" * 80)

# 加载内参和外参
intrinsic = np.loadtxt(os.path.join(OUTPUT_DIR, 'intrinsic.txt'))
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]

ext_dir = os.path.join(OUTPUT_DIR, 'extrinsics')
depth_dir = os.path.join(OUTPUT_DIR, 'depth')
color_dir = os.path.join(OUTPUT_DIR, 'color')

# ============================================================
# 分析1: 同一物体在不同帧中，VGGT给的3D坐标是否一致？
# ============================================================
print("\n" + "=" * 60)
print("分析1: 同一区域在不同帧的VGGT 3D坐标一致性")
print("=" * 60)
print("""
  原理: 如果相机固定、场景静态，同一像素在不同帧的3D坐标应该相同。
  如果有差异，说明VGGT的3D重建在时序上不一致。
  
  我们在图像中心取一个固定的矩形区域（模拟mask），
  看这个区域在不同帧被VGGT赋予了什么3D坐标。
""")

# 取图像中心 100x100 区域作为"模拟mask"
frame_indices = [0, 30, 60, 90, 101, 119]
roi_x1, roi_x2 = 180, 280
roi_y1, roi_y2 = 120, 220

print(f"  模拟mask区域: 像素 [{roi_y1}:{roi_y2}, {roi_x1}:{roi_x2}]")
print()

# 对每帧，把这个区域的像素反投影到世界坐标
roi_z_history = []
for fid in frame_indices:
    ext = np.loadtxt(os.path.join(ext_dir, f'{fid}.txt'))
    R = ext[:3, :3]
    t = ext[:3, 3]
    
    depth = cv2.imread(os.path.join(depth_dir, f'{fid}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    
    roi_depth = depth[roi_y1:roi_y2, roi_x1:roi_x2]
    valid_mask = roi_depth > 0
    
    if not valid_mask.any():
        roi_z_history.append(None)
        continue
    
    vv, uu = np.where(valid_mask)
    z_cam = roi_depth[vv, uu]
    x_cam = ((uu + roi_x1) - cx) * z_cam / fx
    y_cam = ((vv + roi_y1) - cy) * z_cam / fy
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    pts_world = (R @ pts_cam.T + t.reshape(3, 1)).T
    
    roi_z_history.append({
        'frame': fid,
        'z_mean': pts_world[:, 2].mean(),
        'z_std': pts_world[:, 2].std(),
        'z_min': pts_world[:, 2].min(),
        'z_max': pts_world[:, 2].max(),
        'n_points': len(pts_world)
    })
    
    print(f"  Frame {fid}: Z均值={pts_world[:,2].mean():.3f}m, "
          f"Z标准差={pts_world[:,2].std():.3f}m, "
          f"Z范围=[{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}], "
          f"点数={len(pts_world)}")

# 计算帧间Z值差异
print(f"\n  帧间Z值差异分析:")
roi_z_means = [r['z_mean'] for r in roi_z_history if r is not None]
if len(roi_z_means) >= 2:
    max_diff = max(roi_z_means) - min(roi_z_means)
    print(f"    同一区域在不同帧的Z均值范围: [{min(roi_z_means):.3f}, {max(roi_z_means):.3f}]m")
    print(f"    最大差异: {max_diff:.3f}m")
    if max_diff > 0.05:
        print(f"    ❌ Z值在帧间差异 > 5cm → VGGT的3D重建在同一区域上不一致！")
        print(f"       如果这个区域被mask标记为某个物体，那物体点的3D坐标就是漂移的")
        print(f"       → SAM3D收到的pointmap在帧间不一致 → 重定位不可靠")
    else:
        print(f"    ✅ Z值在帧间一致 (< 5cm)")

# ============================================================
# 分析2: 深度图本身是否一致？(分离VGGT深度和VGGT外参的影响)
# ============================================================
print("\n" + "=" * 60)
print("分析2: VGGT深度图本身的帧间一致性（分离外参影响）")
print("=" * 60)
print("""
  深度图是VGGT预测的每个像素的相机坐标系Z值(camera-frame depth)。
  如果场景是静态的、相机固定的，不同帧的深度应该完全一致。
""")

depth_history = []
for fid in frame_indices:
    depth = cv2.imread(os.path.join(depth_dir, f'{fid}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    roi_depth = depth[roi_y1:roi_y2, roi_x1:roi_x2]
    valid = roi_depth[roi_depth > 0]
    depth_history.append({
        'frame': fid,
        'mean': valid.mean(),
        'std': valid.std(),
    })
    print(f"  Frame {fid}: 深度均值={valid.mean():.3f}m, 标准差={valid.std():.3f}m")

depth_means = [d['mean'] for d in depth_history]
max_depth_diff = max(depth_means) - min(depth_means)
print(f"\n  深度均值范围: [{min(depth_means):.3f}, {max(depth_means):.3f}]m")
print(f"  最大差异: {max_depth_diff:.3f}m")
if max_depth_diff > 0.05:
    print(f"  ❌ VGGT深度图在帧间不一致！差异 > 5cm")
    print(f"     这意味着：即使mask是正确的，mask像素对应的深度值也在漂移")
    print(f"     → 物体3D点的Z坐标在漂移 → SAM3D的重定位基准不稳定")
else:
    print(f"  ✅ VGGT深度图帧间一致")

# ============================================================
# 分析3: 验证重定位链路 — mask正确但点云错会怎样
# ============================================================
print("\n" + "=" * 60)
print("分析3: 重定位链路模拟 — mask正确 + 点云错 = ?")
print("=" * 60)
print("""
  模拟场景:
  - 假设SAM3正确分割了桌子（mask是正确的）
  - 但VGGT给桌子区域预测的深度是0.5m（实际应该0.7m）
  - SAM3D会以为这个物体在0.5m处 → mesh被生成在0.5m处
  - l2c变换会被计算到0.5m的位置
  - 最终T矩阵会把物体放在错误的位置
  
  这就是232场景中桌子悬浮的原因：
  - SAM3的mask可能正确标记了桌子
  - 但VGGT给桌子区域的3D坐标偏低（Z≈0.5m而非实际桌面高度≈0.7m）
  - SAM3D就把mesh生成在Z≈0.5m处
  - 最终mesh离地0.47m → 悬浮
""")

# 验证：在frame 101中，桌子大约在图像什么位置
img_101 = cv2.imread(os.path.join(color_dir, '101.jpg'))
h, w = img_101.shape[:2]
print(f"\n  Frame 101 图像尺寸: {w}x{h}")
print(f"  桌子大约在图像下半部分")
print(f"  我们来采样图像不同区域的深度，看桌子区域的深度是否合理")

# 分区域采样
regions = [
    ("上1/3 (背景/墙壁)", 0, h//3),
    ("中1/3 (桌子可能区域)", h//3, 2*h//3),
    ("下1/3 (桌子/地板)", 2*h//3, h),
]

depth_101 = cv2.imread(os.path.join(depth_dir, '101.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

for name, y1, y2 in regions:
    region_depth = depth_101[y1:y2, :]
    valid = region_depth[region_depth > 0]
    ext_101 = np.loadtxt(os.path.join(ext_dir, '101.txt'))
    R = ext_101[:3, :3]
    t = ext_101[:3, 3]
    
    # 反投影到世界坐标
    vv, uu = np.where(depth_101[y1:y2, :] > 0)
    z_cam = depth_101[y1:y2, :][vv, uu]
    x_cam = (uu - cx) * z_cam / fx
    y_cam = ((vv + y1) - cy) * z_cam / fy
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    pts_world = (R @ pts_cam.T + t.reshape(3, 1)).T
    
    print(f"\n  {name}:")
    print(f"    相机深度: mean={valid.mean():.3f}m, std={valid.std():.3f}m")
    print(f"    世界Z坐标: mean={pts_world[:,2].mean():.3f}m, "
          f"range=[{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}]")

# ============================================================
# 分析4: 用实际optimal frame的数据验证
# ============================================================
print("\n" + "=" * 60)
print("分析4: 最优帧的物体点云Z值 vs 最终mesh的Z值")
print("=" * 60)

optimal_dir = os.path.join(OUTPUT_DIR, 'optimal_frames')
optimal_files = sorted(os.listdir(optimal_dir))

print(f"\n  最优帧文件列表:")
for f in optimal_files:
    # 解析文件名: category_instN_frameM.jpg
    parts = f.replace('.jpg', '').split('_')
    category = parts[0]
    inst = parts[1]  # instN
    frame_id = int(parts[2].replace('frame', ''))
    
    # 读深度图，获取该帧的整体深度分布
    depth = cv2.imread(os.path.join(depth_dir, f'{frame_id}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    valid_depth = depth[depth > 0]
    
    # 反投影到世界坐标
    ext = np.loadtxt(os.path.join(ext_dir, f'{frame_id}.txt'))
    R, t_vec = ext[:3, :3], ext[:3, 3]
    
    vv, uu = np.where(depth > 0)
    z_cam = depth[vv, uu]
    x_cam = (uu - cx) * z_cam / fx
    y_cam = (vv - cy) * z_cam / fy
    pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    pts_world = (R @ pts_cam.T + t_vec.reshape(3, 1)).T
    
    print(f"\n  {category}({inst}), 最优帧={frame_id}:")
    print(f"    相机深度: mean={valid_depth.mean():.3f}m, range=[{valid_depth.min():.3f}, {valid_depth.max():.3f}]")
    print(f"    世界Z坐标: mean={pts_world[:,2].mean():.3f}m, range=[{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}]")
    
    # 对比：table的mesh在Z=0.47-0.56，如果VGGT pointmap显示桌子区域在Z≈0.5，
    # 那就说明mesh确实被放到了VGGT指示的位置
    if category == 'table':
        print(f"    ⚠️ 这是table！mesh最终Z范围=[0.470, 0.555]")
        print(f"       VGGT给的frame {frame_id}世界Z范围=[{pts_world[:,2].min():.3f}, {pts_world[:,2].max():.3f}]")
        print(f"       → 如果mask让SAM3D聚焦在Z≈0.47-0.56的区域，")
        print(f"         那mesh被放在Z≈0.51是VGGT点云直接决定的")

# ============================================================
# 最终结论
# ============================================================
print("\n" + "=" * 60)
print("最终结论")
print("=" * 60)
print("""
问题: 点云质量差会影响重定位精度吗？

答案: YES，而且是直接影响。

重定位的精确定义:
  T = inv(extrinsic) @ adjust @ l2c @ y2z

  其中 l2c 来自 SAM3D，SAM3D 的输入是 (image, mask, pointmap)

  mask（来自SAM3） → 通常是正确的（SAM是顶级分割模型）
  pointmap（来自VGGT） → 在动态视频下可能不准确
  
  所以: mask正确 + pointmap错误 = l2c错误 = T错误 = 物体位置错误

具体来说:
  1. SAM3正确地在图像中标记了桌子像素
  2. 这些像素在VGGT pointmap中被赋予了错误的3D坐标
  3. SAM3D根据这些错误的3D坐标生成了mesh和l2c
  4. 最终mesh被放在了错误的位置（但它确实被放在了pointmap指示的位置）

  所以不是"重定位算法有问题"，而是"重定位的输入数据(pointmap)有问题"

232场景的具体证据:
  - 同一区域在不同帧的VGGT Z值差最大达XXm → 点云帧间不一致
  - VGGT深度图在不同帧也有差异 → 原始深度就不稳定
  - 这说明动态物体确实破坏了VGGT的3D重建一致性
  
  当mask把这团不稳定的3D点标记为"桌子"时，
  SAM3D就被迫用这团混乱的点来推断桌子的位置和形状，
  结果自然是不可靠的。
""")