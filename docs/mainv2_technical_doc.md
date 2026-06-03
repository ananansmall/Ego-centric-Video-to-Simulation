# 3D 物体摆放位置技术文档

本文档详细说明 ReplicateAnyScene 中 3D 物体如何从像素级 2D 检测最终摆放到 3D 场景中的完整链路。

---

## 一、3D 物体摆放的完整链路

```
视频帧 (RGB)
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 1: VGGT 预测                                        │
│   RGB → depth + extrinsic → world_points                 │
│   world_points[s,v,u] = extrinsic[s] × backproject(depth)│
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 2: 坐标系对齐                                        │
│   地板/墙壁 PCA → R, t → 旋转平移所有 world_points        │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 3: SAM3 分割 + 去重                                  │
│   文本prompt → mask → 3D重叠率去重 → 去重后masks           │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 4: 最优帧选择                                        │
│   动态/静态检测 → 选面积最大帧 or 运动前帧                  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 5: SAM3D 生成3D资产 + 计算T矩阵                      │
│   SAM3D(image, mask, pointmap) → mesh + l2c              │
│   T = inv(extrinsic) × adjust × l2c × y2z               │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 6: 语义精修 (修改T矩阵)                              │
│   floor/wall/embedded → 基础精修                          │
│   物体间支撑 → SP精修 (on_top/inside/against_side/...)    │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ Step 7: 保存GLB                                           │
│   mesh.apply_transform(T) → z-up→y-up → 导出GLB          │
└──────────────────────────────────────────────────────────┘
```

---

## 二、每一步的详细代码

### Step 1: VGGT 预测 — 从 RGB 到 3D 点云

**代码位置**: `src/vggt_predict.py` / `src/vggt_omega_predict.py` / `src/vggt4d_predict.py`

**调用位置**:
- main.py: L36 `vggt_prediction_results = vggt_predict(frames, vggt_model)`
- mainv2.py: L253 `vggt_prediction_results = vggt_omega_predict(frames, vggt_model)`

**输出数据结构**:
```python
vggt_prediction_results = {
    'world_points':  np.ndarray,  # (T, H, W, 3) — 每帧每像素的世界坐标
    'depths':        np.ndarray,  # (T, H, W) — 每帧深度图
    'extrinsics':    np.ndarray,  # (T, 4, 4) — 每帧相机外参 (c2w)
    'intrinsic':     np.ndarray,  # (3, 3) — 相机内参
    'world_points_conf': np.ndarray,  # (T, H, W) — 置信度
    'colors':        list,        # [np.ndarray(H,W,3), ...] — RGB帧
    'point_cloud_data': trimesh.PointCloud,  # 合并点云
}
```

**world_points 的计算方式**:
```python
# VGGT内部: 多帧RGB → Transformer → 联合预测
world_points[s] = extrinsic[s] @ backproject(depth[s], intrinsic, pixel_coords)

# 其中 backproject:
X_cam = (u - cx) / fx * depth[s, v, u]
Y_cam = (v - cy) / fy * depth[s, v, u]
Z_cam = depth[s, v, u]
```

**关键**: 所有帧共享同一个世界坐标系。VGGT 不是逐帧增量构建，而是一次性全局预测。

---

### Step 2: 坐标系对齐 — 从VGGT坐标系到房间坐标系

**代码位置**: `src/geometry_utils.py:211-277`

**调用位置**:
- main.py: L42-43
- mainv2.py: `run_stage2()` 内部

**代码详解**:
```python
# 1. 用SAM3分割出墙壁和地板的mask
wall_masks, floor_masks = segment_wall_and_floor(colors, sam3_image_model)

# 2. 从点云中提取墙壁/地板的平面信息
for floor_mask in floor_masks:
    pointmap = world_points[frame_id]
    plane_info = get_plane_info(pointmap, mask)  # PCA拟合平面
    # plane_info = {'normal': 法向量, 'd': 平面偏移, 'area': 面积, ...}

# 3. 选择最大面积的地板平面，提取法向量
floor_plane_info = max(valid_floor_planes, key=lambda x: x['area'])
floor_normal = floor_plane_info['normal']  # 应接近 [0, 0, 1]

# 4. 选择与地板正交的最大墙壁平面
wall_plane_info = max(orthogonal_wall_planes, key=lambda x: x['area'])
wall_normal_1 = wall_plane_info['normal']  # 应接近 [1,0,0] 或 [0,1,0]

# 5. 构建旋转矩阵 R
wall_normal_2 = np.cross(floor_normal, wall_normal_1)
R = np.stack([wall_normal_1, wall_normal_2, floor_normal], axis=0)  # (3,3)

# 6. 构建平移向量 t — 地板平面设为 z=0
floor_centroid = floor_plane_info['centroid']
rotated_floor_centroid = floor_centroid @ R.T
t = np.zeros(3)
t[2] = -rotated_floor_centroid[2]  # 地板z坐标归零

# 7. 对齐所有预测结果
predictions['world_points'] = predictions['world_points'] @ R.T + t
predictions['extrinsics'][:, :3, :3] = R_c2w_old @ R.T
predictions['extrinsics'][:, :3, 3] = t_c2w_old - (R_c2w_new @ t)
```

**对齐后保证**:
- Z轴朝上（地板法向量 → [0,0,1]）
- 地板平面在 z≈0
- X/Y轴沿墙壁方向

---

### Step 3: SAM3 分割 + 去重

**代码位置**: `src/object_segmentation.py:65-137`, `src/sg_deduplication.py`

**分割流程**:
```python
# 对每个类别:
category_masks = segment_and_track(category, sam3_video_model, session_id)

# segment_and_track 内部:
# 1. 在第0帧添加文本prompt
sam3_video_model.handle_request(dict(type="add_prompt", frame_index=0, text=category))

# 2. propagate_in_video: 从第0帧向所有帧传播分割
outputs_per_frame = propagate_in_video(sam3_video_model, session_id)

# 3. 收集每帧的mask
for frame_idx, output in enumerate(outputs_per_frame):
    if obj_id in output:
        mask = output[obj_id]  # 二值mask

# 4. 不连续帧段拆分为不同实例
if raw_frame_ids[i] == raw_frame_ids[i-1] + 1:
    current_segment.append(raw_frame_ids[i])
else:
    segments.append(current_segment)  # 断开！
```

**去重流程**:
```python
# 类内去重: 同一类别内的不同实例，3D重叠率>0.3则合并
deduplicated = self_category_deduplicate(category_masks, world_points, world_points_conf)

# 跨类去重: 不同类别的实例，3D重叠率>0.5则合并
deduplicated_all = cross_category_deduplicate(all_masks, world_points, world_points_conf,
                                              protected_categories=json_categories_set)
```

---

### Step 4: 最优帧选择

**代码位置**: `src/geometry_utils.py:307-429`

**调用位置**:
- main.py: L102
- mainv2.py: `run_stage3()` 内部

**代码详解**:
```python
def get_optimal_view_frame_id(world_points, instance_masks, motion_threshold=0.02):
    # 1. 计算每帧的3D质心
    for instance_mask in instance_masks:
        pointmap = world_points[frame_id]
        pts = pointmap[mask > 0]
        centroid = np.mean(pts, axis=0)  # (X, Y, Z) 世界坐标
        centroids.append(centroid)

    # 2. 计算相邻帧质心位移
    consecutive_disps = [||centroids[i] - centroids[i-1]|| for i in range(1, len(centroids))]
    median_disp = np.median(consecutive_disps)

    # 3. 计算全局位移 (首20% vs 末20%)
    global_disp = ||tail_mean - head_mean||

    # 4. 动态/静态判断
    is_dynamic = (median_disp > motion_threshold) or (global_disp > max(motion_threshold*2, 0.04))

    if is_dynamic:
        # 5a. 动态物体: 找运动起始帧，选运动前面积最大的帧
        motion_onset_idx = find_motion_onset(consecutive_disps)
        pre_motion_frames = frames[:motion_onset_idx+1]
        return argmax(surface_area(f) for f in pre_motion_frames)
    else:
        # 5b. 静态物体: 选3D表面积最大的帧
        return argmax(surface_area(f) for f in all_frames)
```

---

### Step 5: SAM3D 生成3D资产 + 计算T矩阵 ⭐ 核心步骤

**代码位置**: `src/instance_generation.py:12-54`

**调用位置**:
- main.py: L158 `all_instances = generate_3d_asset_in_subprocess(...)`
- mainv2.py: `run_stage3()` 内部

**这是决定物体3D位置的关键步骤。** T矩阵的计算链路如下：

```python
def generate_3d_asset(image, mask, pointmap, extrinsic, inference):
    # ── 输入 ──
    # image:     最优帧的RGB图像 (H, W, 3)
    # mask:      最优帧的物体mask (H, W)
    # pointmap:  最优帧的世界坐标 (H, W, 3) — 来自 world_points[optimal_frame_id]
    # extrinsic: 最优帧的相机外参 (4, 4) — 来自 extrinsics[optimal_frame_id]

    # ── Step 5a: world_points → 相机坐标系 pointmap ──
    points_world_flat = pointmap.reshape(-1, 3)           # (H*W, 3)
    points_world_hom = np.hstack([points_world_flat, ones])  # (H*W, 4)

    # VGGT格式 → SAM3D格式: 翻转x和y轴
    flip_xy = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    points_cam_hom = (flip_xy @ extrinsic @ points_world_hom.T).T  # (H*W, 4)
    point_map_camera = points_cam_hom[:, :3].reshape(H, W, 3)      # (H, W, 3)

    # ── Step 5b: SAM3D 推理 ──
    output = inference(image, mask, seed=42, pointmap=point_map_camera)
    # output = {
    #     "glb":        trimesh.Trimesh,  # 物体坐标系下的mesh
    #     "rotation":   quaternion,       # 物体坐标系→SAM3D坐标系的旋转
    #     "scale":      float,            # 缩放
    #     "translation": (3,),            # 平移
    # }

    # ── Step 5c: 计算 l2c 矩阵 (local → camera) ──
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    matrix_l2c = l2c_transform.get_matrix()[0].T.cpu().numpy()  # (4, 4)

    # ── Step 5d: 计算 T 矩阵 (local → world) ──
    matrix_y2z = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])  # y-up → z-up
    matrix_adjust = np.diag([-1, -1, 1, 1])                              # 翻转x和y
    matrix_ext_inv = np.linalg.inv(extrinsic)                            # world → camera 的逆

    T = matrix_ext_inv @ matrix_adjust @ matrix_l2c @ matrix_y2z
    #   ───────────   ────────────   ─────────   ──────
    #   camera→world  翻转修正       local→cam   y→z旋转
    #   (决定位置)     (格式修正)     (SAM3D输出) (坐标系修正)

    return {"original_mesh": output["glb"], "T": T}
```

**T矩阵的分解**:

```
T = inv(extrinsic) × adjust × l2c × y2z
     ────────────   ──────   ───   ────
     位置+朝向       格式     形状   坐标系
     由VGGT决定     修正     由SAM3D 修正
                              决定

详细分解:
  y2z:  [1,0,0,0]     SAM3D输出是y-up，转为z-up
        [0,0,-1,0]
        [0,1,0,0]
        [0,0,0,1]

  l2c:  SAM3D输出的 (scale, rotation, translation) 组合
        将物体从其局部坐标系变换到相机坐标系
        包含物体的形状、大小、朝向信息

  adjust: diag(-1,-1,1,1)
        翻转x和y轴，修正VGGT和SAM3D之间的坐标系差异

  inv(extrinsic): extrinsic的逆矩阵
        extrinsic是 camera→world (c2w)
        inv(extrinsic)是 world→camera (w2c) 的逆 = camera→world
        这一步把相机坐标系中的物体位置映射到世界坐标系
        **这一步决定了物体在世界中的3D位置**
```

**物体在世界中的位置由什么决定？**

| 因素 | 来源 | 影响 |
|------|------|------|
| `extrinsic` | VGGT预测 | 相机在世界坐标系中的位置和朝向 → `inv(extrinsic)` 决定物体从相机坐标系映射到世界坐标系 |
| `l2c` | SAM3D输出 | 物体在相机坐标系中的位置（由pointmap引导） |
| `pointmap` | VGGT预测 | SAM3D的几何条件输入 → 影响l2c的计算 |

**关键洞察**: `inv(extrinsic) @ [l2c在相机坐标系中的位置]` = 物体在世界坐标系中的位置。如果pointmap准确，l2c会正确反映物体在相机前方多远；如果extrinsic准确，inv(extrinsic)会正确映射到世界坐标。两者都准确时，物体位置正确。

---

### Step 6: 语义精修 — 修改T矩阵

#### 6a: 基础精修 (floor/wall/embedded)

**代码位置**: `src/sp_refinement.py`

**调用位置**:
- main.py: L188-201
- mainv2.py: L799-815

**三种基础精修**:

**① supported_by_floor** — 对齐重力方向 + 底部贴地

```python
def refine_supported_by_floor_object(object_info):
    T = object_info["T"]

    # Step 1: 对齐朝上方向到Z轴
    upper_vector = T[:3, 1] / norm(T[:3, 1])  # T的第2列 = 物体的"上"方向
    theta = angle(upper_vector, [0,0,1])       # 与Z轴的夹角
    if theta < 10°:  # 接近平行
        align_matrix = align_vectors(upper_vector, [0,0,1])
        T[:3,:3] = align_matrix[:3,:3] @ T[:3,:3]  # 只改旋转，不改平移

    # Step 2: 底部吸附到z=0
    mesh = object_info["original_mesh"].copy()
    mesh.apply_transform(T)
    z_min = mesh.bounds[0, 2]  # 变换后mesh的最低点
    if abs(z_min) < 0.3:       # ⚠️ 阈值0.3m！超过则不吸附
        T = translation_matrix([0, 0, -z_min]) @ T
```

**② embedded_in_wall** — 对齐到墙壁平面 + 中心吸附

```python
def refine_embedded_in_wall_object(object_info, walls_info):
    T = object_info["T"]

    # Step 1: 对齐物体朝向到最近的墙壁方向
    forward = T[:3, 2] / norm(T[:3, 2])
    align_vector, wall_axis = get_wall_alignment_target(forward)
    align_matrix = align_vectors(forward, align_vector)
    T[:3,:3] = align_matrix[:3,:3] @ T[:3,:3]

    # Step 2: 中心吸附到最近的墙壁平面
    center = T[:3, 3]
    nearest_wall, dist = select_closest_wall(walls_info, wall_axis, center)
    if dist <= 0.3:  # ⚠️ 阈值0.3m
        offset = wall_position - center[axis_idx]
        T = translation_matrix(offset) @ T

    # Step 3: 确保不低于地板
    mesh.apply_transform(T)
    if mesh.bounds[0, 2] < 0:
        T = translation_matrix([0, 0, -z_min]) @ T
```

**③ attached_to_wall** — 对齐到墙壁 + 背面吸附

```python
def refine_attached_to_wall_object(object_info, walls_info, camera_pos):
    T = object_info["T"]

    # Step 1: 同embedded_in_wall，对齐朝向
    # ...

    # Step 2: 背面吸附到墙壁（不是中心，是背面！）
    # camera_pos决定哪一面是"背面"
    if camera_pos[axis] > center[axis]:
        contact_val = vertices[:, axis].min()  # 远离相机的一侧
    else:
        contact_val = vertices[:, axis].max()
    snap_offset = wall_position - contact_val
    T = translation_matrix(snap_offset) @ T
```

#### 6b: 物体间支撑精修 (Stage 5.2)

**代码位置**: `tools/refine_inter_object_placement.py`

**调用位置**: mainv2.py `run_stage5()` → `refine_inter_object_relations()`

**五种放置策略**:

| 策略 | 场景 | 几何约束 | 核心计算 |
|------|------|---------|---------|
| `on_top` | 杯子在桌上 | supported.bottom ≥ supporter.top | `z_offset = supporter.top_z - supported.bottom_z` |
| `inside` | 衣服在抽屉里 | supporter.bottom ≤ supported ≤ supporter.top | 对齐到内部30%高度 |
| `against_side` | 柜子靠墙 | 底面贴地 + 侧面接触 | 找最小offset的侧面方向 |
| `hanging_below` | 吊灯 | supported.top ≤ supporter.bottom | `z_offset = supporter.bottom_z - supported.top_z` |
| `leaning` | 梯子靠墙 | 同against_side | 直接调用against_side逻辑 |

**on_top 详解（最常用）**:
```python
def sp_refine_on_top(supported_info, supporter_info):
    T_sup = supported_info["T"]
    T_spr = supporter_info["T"]

    # 计算支撑物顶面z坐标
    supporter_mesh = supporter_info["original_mesh"].copy()
    supporter_mesh.apply_transform(T_spr)
    supporter_top_z = supporter_mesh.bounds[1, 2]  # 最高点

    # 计算被支撑物底面z坐标
    supported_mesh = supported_info["original_mesh"].copy()
    supported_mesh.apply_transform(T_sup)
    supported_bottom_z = supported_mesh.bounds[0, 2]  # 最低点

    # 对齐
    z_offset = supporter_top_z - supported_bottom_z
    if z_offset < 0:
        # 物体穿入支撑物，向上推
        T_sup = translation_matrix([0, 0, z_offset]) @ T_sup  # z_offset < 0 → 上移
    elif 0 < z_offset < threshold:
        # 物体悬空，向下落
        T_sup = translation_matrix([0, 0, z_offset]) @ T_sup  # z_offset > 0 → 下移

    supported_info["T"] = T_sup
```

---

### Step 7: 保存GLB

**代码位置**:
- main.py: L204-213
- mainv2.py: `save_final_glb()` L679-713

```python
def save_final_glb(all_instances, output_path, filename):
    scene = trimesh.Scene()

    # 1. 对每个实例: mesh应用T变换
    for category, category_instances in all_instances.items():
        for i, instance_info in enumerate(category_instances):
            mesh = instance_info['original_mesh']
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(instance_info['T'])  # ← T烘焙进顶点
            scene.add_geometry(transformed_mesh, node_name=f"{category}_{i}")

    # 2. z-up → y-up (GLB标准坐标系)
    scene.apply_transform(np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],   # 新Y = 旧Z
        [0, -1, 0, 0],   # 新Z = -旧Y
        [0, 0, 0, 1],
    ]))

    # 3. 导出
    scene.export(os.path.join(output_path, filename))
```

**注意**: T矩阵在保存时被烘焙进mesh顶点，之后无法再从GLB中恢复T。这就是为什么mainv2额外保存了 `all_instances.pkl`。

---

## 三、main.py vs mainv2.py 的3D摆放差异

| 维度 | main.py | mainv2.py |
|------|---------|-----------|
| 坐标系对齐 | ✅ 相同 | ✅ 相同 |
| SAM3分割+去重 | 无protected_categories | ✅ protected_categories防止跨类合并 |
| 最优帧选择 | 首尾帧位移判断 | ✅ 中位数+全局位移判断 |
| 3D资产生成 | ✅ 相同 | ✅ 相同 |
| 基础精修 | 精修结果未写回❌ | ✅ 已修复: `category_instances[instance_id] = instance_info` |
| 关系格式 | 只支持 `supported_by_floor` | ✅ 兼容 `supported by floor` 和 `supported_by_floor` |
| 物体间精修 | ❌ 未实现 | ✅ Stage 5.1+5.2 |
| GLB保存 | 单个 `final_scene.glb` | ✅ base/stage4/stage5 分阶段保存 |
| pkl保存 | ❌ 无 | ✅ `all_instances.pkl` 保留原始T+mesh |

**main.py 的关键BUG**: L188-201 精修循环中修改了 `instance_info` 但没有写回 `category_instances[instance_id]`，导致所有精修结果丢失。mainv2 L815 已修复。

---

## 四、手部遮挡物体的改进建议

### 问题描述

当手部遮挡物体时：
1. SAM3跟踪断裂 → 同一物体产生多个实例
2. 手部区域被包含进mask → mask不干净 → 3D资产包含手部几何
3. 手部区域的pointmap是手的深度 → SAM3D生成的mesh位置偏移

### 改进方案

#### 方案1: 手部感知的最优帧选择（推荐，改动最小）

在 `get_optimal_view_frame_id` 中增加手部区域检测，降低手部遮挡帧的优先级：

```python
def get_optimal_view_frame_id(world_points, instance_masks, hand_masks=None):
    # ... 现有逻辑 ...

    for instance_mask in instance_masks:
        frame_id = instance_mask['frame_id']
        mask = instance_mask['mask']

        # 新增: 计算mask与手部区域的重叠率
        if hand_masks is not None and frame_id in hand_masks:
            hand_overlap = np.sum(mask & hand_masks[frame_id]) / np.sum(mask)
            if hand_overlap > 0.3:
                continue  # 跳过手部遮挡严重的帧

        area = compute_surface_area_from_pointmap(pointmap, mask)
        if area > max_area:
            max_area = area
            optimal_frame_id = frame_id
```

**手部mask的获取方式**:
- 使用 MediaPipe Hands 检测手部区域（轻量，CPU即可运行）
- 或使用 SAM3 的 "hand" 类别分割（已有基础设施）

#### 方案2: mask后处理 — 去除手部区域

在SAM3分割后、送入SAM3D前，从物体mask中减去手部区域：

```python
# 在 run_stage3 中:
for category, category_masks in deduplicated_all_masks.items():
    for instance_masks in category_masks:
        for im in instance_masks:
            mask = im['mask']
            # 减去手部区域
            if hand_masks is not None:
                hand_mask_at_frame = hand_masks[im['frame_id']]
                mask = mask & (~hand_mask_at_frame)  # 去除手部
                im['mask'] = mask
```

**注意**: 去除手部后mask可能变小或不连续，SAM3D可能生成不完整的mesh。

#### 方案3: mesh后处理 — 连通分量清理

在SAM3D生成mesh后，去除不属于物体的小碎片：

```python
def clean_mesh(mesh, min_component_ratio=0.3):
    """保留最大连通分量，去除小碎片（如手指状突起）"""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    max_area = max(c.area for c in components)
    large_components = [c for c in components if c.area > max_area * min_component_ratio]
    return trimesh.util.concatenate(large_components)
```

#### 方案4: 多帧融合验证

利用多帧验证3D资产质量，手部遮挡帧的验证结果权重降低：

```python
# 在 asset_verifier.py 中:
for frame_id, mask in instance_masks:
    # 计算手部遮挡权重
    if hand_masks and frame_id in hand_masks:
        hand_overlap = np.sum(mask & hand_masks[frame_id]) / max(np.sum(mask), 1)
        weight = 1.0 - hand_overlap  # 手部遮挡越多，权重越低
    else:
        weight = 1.0
    vote_score *= weight
```

### 优先级

| 优先级 | 方案 | 改动量 | 效果 | 依赖 |
|--------|------|--------|------|------|
| 🥇 | 方案1: 手部感知帧选择 | ⭐小 | ⭐⭐⭐⭐ | MediaPipe Hands |
| 🥈 | 方案3: mesh连通分量清理 | ⭐小 | ⭐⭐⭐ | 无 |
| 🥉 | 方案2: mask去除手部 | ⭐小 | ⭐⭐ | 手部检测 |
| 4 | 方案4: 多帧加权验证 | ⭐⭐中 | ⭐⭐⭐ | 手部检测 |

---

## 五、T矩阵在各阶段的变化

```
初始T (Step 5):
  T = inv(extrinsic) × adjust × l2c × y2z
  → 物体在VGGT点云指示的位置

基础精修 (Step 6a):
  supported_by_floor: T = translation_matrix([0,0,-z_min]) @ T
  embedded_in_wall:   T = translation_matrix(wall_offset) @ align_matrix @ T
  attached_to_wall:   T = translation_matrix(snap_offset) @ align_matrix @ T

物体间精修 (Step 6b):
  on_top:        T = translation_matrix([0,0,z_offset]) @ T
  inside:        T = translation_matrix([0,0,z_offset]) @ T
  against_side:  T = translation_matrix([offset_x, offset_y, z_fix]) @ T
  hanging_below: T = translation_matrix([0,0,z_offset]) @ T

保存GLB (Step 7):
  mesh.apply_transform(T)  → T烘焙进顶点
  scene.apply_transform(z_up_to_y_up)  → 全局坐标系变换
```

**每次精修都是在T左侧乘一个变换矩阵**，即 `T_new = correction @ T_old`。这意味着精修是累积的，后一次精修在前一次精修的基础上调整。

---

## 六、Stage 3 完整流程与实际代码

Stage 3 是从 mask 到 3D 资产的核心阶段，包含最优帧选择、3D资产生成、多票验证三个子步骤。以下为 mainv2.py 中 `run_stage3()` 的完整代码逻辑。

### 6.1 Stage 3 入口 — run_stage3()

**代码位置**: `mainv2.py:426-512`

```python
def run_stage3(output_path, vggt_prediction_results, deduplicated_all_masks):
    # ── 3.1 计算每个实例的最优视角帧ID ──
    all_optimal_frame_ids = {}
    dynamic_count = 0
    static_count = 0
    for category, category_masks in deduplicated_all_masks.items():
        all_optimal_frame_ids[category] = []
        for inst_idx, instance_masks in enumerate(category_masks):
            optimal_frame_id, is_dynamic, motion_info = get_optimal_view_frame_id(
                vggt_prediction_results['world_points'], instance_masks
            )
            all_optimal_frame_ids[category].append(optimal_frame_id)
            tag = "DYNAMIC" if is_dynamic else "STATIC"
            if is_dynamic:
                dynamic_count += 1
            else:
                static_count += 1
            print(f"   {category}_{inst_idx}: [{tag}] median_disp={motion_info['median_disp']}m, "
                  f"max_disp={motion_info['max_disp']}m, "
                  f"global_disp={motion_info['global_disp']}m, "
                  f"valid_frames={motion_info['num_valid_frames']} → frame {optimal_frame_id}")

    # ── 3.2 保存最优视角帧图像 ──
    optimal_frames_dir = os.path.join(output_path, 'optimal_frames')
    os.makedirs(optimal_frames_dir, exist_ok=True)
    for category, frame_ids in all_optimal_frame_ids.items():
        for inst_idx, frame_id in enumerate(frame_ids):
            image_rgb = vggt_prediction_results['colors'][frame_id]
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            save_name = f"{category}_inst{inst_idx}_frame{frame_id}.jpg"
            cv2.imwrite(os.path.join(optimal_frames_dir, save_name), image_bgr)

    # ── 3.3 保存实例可见性信息 ──
    instance_visibility = {}
    for category, category_masks in deduplicated_all_masks.items():
        instance_visibility[category] = {}
        for inst_idx, instance_masks in enumerate(category_masks):
            frame_ids = sorted([im["frame_id"] for im in instance_masks])
            instance_visibility[category][str(inst_idx)] = frame_ids
    visibility_path = os.path.join(optimal_frames_dir, "instance_visibility.json")
    with open(visibility_path, 'w', encoding='utf-8') as f:
        json.dump(instance_visibility, f, indent=2, ensure_ascii=False)

    # ── 3.4 在子进程中生成3D资产（避免CUDA内存冲突）──
    all_instances = generate_3d_asset_in_subprocess(
        deduplicated_all_masks,
        all_optimal_frame_ids,
        vggt_prediction_results['colors'],
        vggt_prediction_results['world_points'],
        vggt_prediction_results['extrinsics'],
    )

    # ── 3.5 多票验证生成的3D资产 ──
    from tools.asset_verifier import verify_all_instances
    all_instances = verify_all_instances(
        all_instances,
        all_optimal_frame_ids,
        deduplicated_all_masks,
        vggt_prediction_results['world_points'],
        vggt_prediction_results['world_points_conf'],
        min_votes=2,
    )

    return all_instances, all_optimal_frame_ids
```

### 6.2 最优帧选择 — get_optimal_view_frame_id() 实际代码

**代码位置**: `src/geometry_utils.py:307-429`

这是决定3D资产质量的关键函数。选错帧 → mask不完整/手部遮挡 → mesh质量差 → 位置偏移。

```python
def get_optimal_view_frame_id(world_points, instance_masks, motion_threshold=0.02):
    # ── 第1步: 计算每帧mask区域的3D质心 ──
    centroids = []
    for instance_mask in instance_masks:
        frame_id = instance_mask['frame_id']
        mask = instance_mask['mask']
        pointmap = world_points[frame_id]
        valid = mask > 0
        if not np.any(valid):
            centroids.append((frame_id, None))
            continue
        pts = pointmap[valid]
        finite = np.all(np.isfinite(pts), axis=-1)
        if not np.any(finite):
            centroids.append((frame_id, None))
            continue
        centroids.append((frame_id, np.mean(pts[finite], axis=0)))

    valid_centroids = [(fid, c) for fid, c in centroids if c is not None]
    num_valid_frames = len(valid_centroids)

    # ── 第2步: 计算相邻帧质心位移 ──
    consecutive_disps = []
    for i in range(1, len(valid_centroids)):
        disp = np.linalg.norm(valid_centroids[i][1] - valid_centroids[i-1][1])
        consecutive_disps.append(disp)

    median_disp = float(np.median(consecutive_disps))
    max_disp = float(np.max(consecutive_disps))

    # ── 第3步: 计算全局位移 (首20% vs 末20%的质心均值距离) ──
    n_head = max(1, num_valid_frames // 5)
    n_tail = max(1, num_valid_frames // 5)
    head_mean = np.mean([c for _, c in valid_centroids[:n_head]], axis=0)
    tail_mean = np.mean([c for _, c in valid_centroids[-n_tail:]], axis=0)
    global_disp = float(np.linalg.norm(tail_mean - head_mean))

    # ── 第4步: 动态/静态判断 ──
    is_dynamic_consecutive = median_disp > motion_threshold   # 逐帧漂移
    is_dynamic_global = global_disp > max(motion_threshold * 2, 0.04)  # 全局位移
    is_dynamic = is_dynamic_consecutive or is_dynamic_global

    # ── 第5步: 根据动静态选择最优帧 ──
    if is_dynamic:
        # 动态物体: 找运动起始帧，选运动前面积最大的帧
        motion_onset_idx = 0
        onset_threshold = max(motion_threshold * 3, 0.05)
        for i, disp in enumerate(consecutive_disps):
            if disp > onset_threshold:
                motion_onset_idx = i
                break

        # 如果全局位移大但逐帧检测不到onset → 用全局搜索
        if global_disp > max(motion_threshold * 2, 0.04) and motion_onset_idx == 0:
            n_search = max(1, num_valid_frames // 3)
            for i in range(n_search, len(valid_centroids)):
                disp_from_start = np.linalg.norm(valid_centroids[i][1] - head_mean)
                if disp_from_start > max(motion_threshold * 3, 0.06):
                    motion_onset_idx = i - 1
                    break

        pre_motion_frame_ids = [valid_centroids[j][0] for j in range(motion_onset_idx + 1)]

        # 在运动前的帧中选面积最大的
        if pre_motion_frame_ids:
            best_frame = -1
            max_area = 0
            for instance_mask in instance_masks:
                if instance_mask['frame_id'] in pre_motion_frame_ids:
                    area = compute_surface_area_from_pointmap(
                        world_points[instance_mask['frame_id']], instance_mask['mask']
                    )
                    if area > max_area:
                        max_area = area
                        best_frame = instance_mask['frame_id']
            if best_frame >= 0:
                return best_frame, True, motion_info

        return first_valid_frame, True, motion_info

    else:
        # 静态物体: 选3D表面积最大的帧
        optimal_frame_id = -1
        max_area = 0
        for instance_mask in instance_masks:
            frame_id = instance_mask['frame_id']
            mask = instance_mask['mask']
            pointmap = world_points[frame_id]
            area = compute_surface_area_from_pointmap(pointmap, mask)
            if area > max_area:
                max_area = area
                optimal_frame_id = frame_id
        return optimal_frame_id, False, motion_info
```

**动静态判断的关键参数**:

| 参数 | 值 | 含义 |
|------|-----|------|
| `motion_threshold` | 0.02m | 逐帧中位数位移阈值，超过则认为动态 |
| `global_disp阈值` | max(0.04, 2×motion_threshold) | 首尾质心距离阈值 |
| `onset_threshold` | max(0.05, 3×motion_threshold) | 运动起始帧的位移突变阈值 |

### 6.3 3D资产生成 — generate_3d_asset_in_subprocess() 实际代码

**代码位置**: `src/instance_generation.py:129-174`

```python
def generate_3d_asset_in_subprocess(
    deduplicated_all_masks,
    all_optimal_frame_ids,
    colors,
    world_points,
    extrinsics,
    config_file="./models/SAM3D/checkpoints/pipeline.yaml",
    compile_model=False,
):
    # 使用spawn方式创建子进程，避免CUDA上下文冲突
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_generate_all_instances_worker,
        args=(queue, deduplicated_all_masks, all_optimal_frame_ids,
              colors, world_points, extrinsics, config_file, compile_model),
    )
    process.start()
    # 读取结果（超时7200秒）
    ok, payload = queue.get(timeout=7200)
    process.join(timeout=30)
    if ok:
        return payload
    raise RuntimeError(payload)
```

**子进程中的实际生成逻辑** (`_generate_all_instances_worker`):

```python
def _generate_all_instances_worker(queue, deduplicated_all_masks, all_optimal_frame_ids,
                                    colors, world_points, extrinsics, config_file, compile_model):
    inference = Inference(config_file=config_file, compile=compile_model)
    all_instances = {}
    for category, category_masks in deduplicated_all_masks.items():
        all_instances[category] = []
        for instance_masks, optimal_frame_id in zip(category_masks, all_optimal_frame_ids[category]):
            # 从最优帧提取输入数据
            image = colors[optimal_frame_id]                          # RGB图像
            mask = next(im["mask"] for im in instance_masks
                        if im["frame_id"] == optimal_frame_id)       # 最优帧的mask
            pointmap = world_points[optimal_frame_id]                # 世界坐标点云
            extrinsic = extrinsics[optimal_frame_id]                 # 相机外参

            # 调用SAM3D生成3D资产
            instance_result = generate_3d_asset(image, mask, pointmap, extrinsic, inference)
            all_instances[category].append(instance_result)

    queue.put((True, all_instances))
```

### 6.4 T矩阵计算 — generate_3d_asset() 实际代码

**代码位置**: `src/instance_generation.py:12-54`

```python
def generate_3d_asset(image, mask, pointmap, extrinsic, inference):
    H, W = pointmap.shape[:2]

    # ── Step A: 将世界坐标点云转换为相机坐标系 ──
    points_world_flat = pointmap.reshape(-1, 3)           # (H*W, 3)
    ones = np.ones((points_world_flat.shape[0], 1))
    points_world_hom = np.hstack([points_world_flat, ones])  # (H*W, 4)

    # VGGT格式 → SAM3D格式: 翻转x和y轴
    flip_xy = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]])
    points_cam_hom = (flip_xy @ extrinsic @ points_world_hom.T).T  # (H*W, 4)
    points_cam_flat = points_cam_hom[:, :3]
    point_map_camera = torch.from_numpy(points_cam_flat).reshape(H, W, 3).to(torch.float32)
    point_map_camera = point_map_camera.contiguous()

    # ── Step B: SAM3D推理 ──
    output = inference(image, mask, seed=42, pointmap=point_map_camera)
    original_mesh = output["glb"]

    # ── Step C: 计算l2c矩阵 (local → camera) ──
    R_l2c = quaternion_to_matrix(output["rotation"])
    l2c_transform = compose_transform(
        scale=output["scale"],
        rotation=R_l2c,
        translation=output["translation"],
    )
    matrix_l2c = l2c_transform.get_matrix()[0].transpose(0, 1).detach().cpu().numpy()

    # ── Step D: 计算T矩阵 (local → world) ──
    matrix_y2z = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]], dtype=np.float32)
    matrix_adjust = np.diag([-1, -1, 1, 1])
    matrix_ext_inv = np.linalg.inv(extrinsic)
    final_transform = matrix_ext_inv @ matrix_adjust @ matrix_l2c @ matrix_y2z

    return {"original_mesh": original_mesh, "T": final_transform}
```

**T矩阵中各分量的数值示例** (以232场景为例):

```
extrinsic (c2w) = [R_c2w | t_c2w]  — VGGT预测的相机位姿
  例: [[0.99, -0.05, 0.12, 0.30],
       [0.05,  0.99, -0.02, -0.15],
       [-0.12, 0.02, 0.99, 1.20],
       [0,     0,    0,    1.00]]

inv(extrinsic) = [R_c2w^T | -R_c2w^T @ t_c2w]  — world→camera 的逆
  这一步将相机坐标系中的物体位置映射到世界坐标系
  **决定了物体在世界中的3D位置**

matrix_l2c — SAM3D输出的局部→相机变换
  包含物体的形状、大小、朝向信息
  受 pointmap (几何条件输入) 影响

matrix_adjust = diag(-1, -1, 1, 1)
  修正VGGT和SAM3D之间的坐标系差异

matrix_y2z — y-up → z-up
  SAM3D输出是y-up格式，转为z-up格式
```

### 6.5 3D表面积计算 — compute_surface_area_from_pointmap() 实际代码

**代码位置**: `src/geometry_utils.py:6-56`

```python
def compute_surface_area_from_pointmap(pointmap, mask, max_triangle_size=2e-4):
    H, W, _ = pointmap.shape
    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 3:
        return 0.0
    points_3d = pointmap[y_coords, x_coords]
    pixel_coords = np.column_stack([x_coords, y_coords])

    tri = Delaunay(pixel_coords)
    simplices = tri.simplices
    triangles_3d = points_3d[simplices]

    AB = triangles_3d[:, 1] - triangles_3d[:, 0]
    AC = triangles_3d[:, 2] - triangles_3d[:, 0]
    cross_product = np.cross(AB, AC)
    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)

    valid_triangle_mask = (triangle_areas > 0) & (triangle_areas < max_triangle_size)
    return float(np.sum(triangle_areas[valid_triangle_mask]))
```

**注意**: `max_triangle_size=2e-4` 是一个硬编码阈值，过滤掉面积过大的异常三角形。如果点云噪声大（如VGGT-omega），可能产生大量异常三角形，导致面积计算不准。

### 6.6 Stage 3 数据流图

```
deduplicated_all_masks                    vggt_prediction_results
        │                                         │
        │  ┌──────────────────────────────────────┤
        │  │                                      │
        ▼  ▼                                      ▼
┌──────────────────────┐              ┌─────────────────────┐
│ get_optimal_view_    │              │ world_points[s]     │
│ frame_id()           │              │ extrinsics[s]       │
│                      │              │ colors[s]           │
│ 输入: world_points,  │              └─────────┬───────────┘
│       instance_masks │                        │
│ 输出: optimal_frame_ │                        │
│       id, is_dynamic │                        │
└──────────┬───────────┘                        │
           │                                    │
           ▼                                    ▼
┌──────────────────────────────────────────────────────────┐
│ generate_3d_asset_in_subprocess()                         │
│                                                          │
│  对每个实例:                                              │
│    image = colors[optimal_frame_id]                       │
│    mask = instance_masks中frame_id==optimal_frame_id的mask│
│    pointmap = world_points[optimal_frame_id]              │
│    extrinsic = extrinsics[optimal_frame_id]               │
│                                                          │
│    → generate_3d_asset(image, mask, pointmap, extrinsic)  │
│      → pointmap世界→相机坐标转换                          │
│      → SAM3D推理 → mesh + (scale, rotation, translation)  │
│      → T = inv(ext) @ adjust @ l2c @ y2z                 │
│                                                          │
│  输出: all_instances = {category: [{original_mesh, T}]}   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│ verify_all_instances() — 多票验证                         │
│                                                          │
│  对每个实例的每个验证帧:                                   │
│    将mesh投影到验证帧 → 与mask比较 → 投票                  │
│  保留投票数 >= min_votes 的实例                           │
└──────────────────────────────────────────────────────────┘
```

---

## 七、手部遮挡问题的深度分析与改进建议

### 7.1 问题描述

在手物交互(HOI)场景中，手部遮挡物体会导致以下连锁问题：

```
手部遮挡物体
    │
    ├──→ SAM3跟踪断裂 → 同一物体产生多个实例（最严重）
    │     └──→ 跨类去重可能合并不同物体
    │
    ├──→ mask包含手部区域 → SAM3D生成包含手部几何的mesh
    │     └──→ mesh质量差 → T矩阵计算偏移
    │
    ├──→ 手部区域的pointmap是手的深度 → SAM3D几何条件输入错误
    │     └──→ l2c矩阵计算偏移 → 物体位置偏移
    │
    └──→ 最优帧选择时手部遮挡帧被选中 → 生成的3D资产质量差
          └──→ 该帧的mask和pointmap都包含手部信息
```

### 7.2 长时间遮挡的特殊问题

当手部长时间遮挡物体（如持续握持）时：

1. **SAM3完全丢失跟踪**: 物体在所有帧中都被手遮挡，SAM3无法分割出物体mask
2. **短间隙合并无效**: 即使将帧间隙阈值从1改为5，长时间遮挡仍然会导致实例断裂
3. **手部掩码去除的局限**: 将手部区域设为黑色（深度为0）后：
   - SAM3D无法估计被遮挡区域的深度 → 生成不完整的mesh
   - pointmap中手部区域变为无效值 → 几何条件输入缺失
   - 本质上，黑色区域 = 无信息，无法恢复被遮挡的3D结构

### 7.3 改进方案（按优先级排序）

#### 方案1: 手部感知的最优帧选择（推荐，改动最小，效果最好）

**核心思想**: 在选择最优帧时，优先选择手部遮挡少的帧。

**实现位置**: `src/geometry_utils.py` 的 `get_optimal_view_frame_id()`

```python
def get_optimal_view_frame_id(world_points, instance_masks, motion_threshold=0.02,
                               hand_masks=None):
    """
    新增参数:
        hand_masks: dict, {frame_id: np.ndarray(H, W)} 手部二值mask
                    可通过 MediaPipe Hands 或 SAM3 "hand" 类别获取
    """
    # ... 现有质心计算和动静态判断逻辑不变 ...

    # ── 新增: 在选择最优帧时考虑手部遮挡 ──
    if is_dynamic:
        # 动态物体: 在运动前的帧中选面积最大且手部遮挡最小的帧
        best_frame = -1
        best_score = -1
        for instance_mask in instance_masks:
            if instance_mask['frame_id'] not in pre_motion_frame_ids:
                continue
            area = compute_surface_area_from_pointmap(
                world_points[instance_mask['frame_id']], instance_mask['mask']
            )
            # 手部遮挡惩罚
            hand_penalty = 0.0
            if hand_masks is not None and instance_mask['frame_id'] in hand_masks:
                mask = instance_mask['mask']
                hand = hand_masks[instance_mask['frame_id']]
                hand_overlap = np.sum(mask & hand) / max(np.sum(mask), 1)
                hand_penalty = hand_overlap * area  # 遮挡比例 × 面积
            score = area - hand_penalty
            if score > best_score:
                best_score = score
                best_frame = instance_mask['frame_id']
    else:
        # 静态物体: 同样考虑手部遮挡
        best_frame = -1
        best_score = -1
        for instance_mask in instance_masks:
            frame_id = instance_mask['frame_id']
            mask = instance_mask['mask']
            pointmap = world_points[frame_id]
            area = compute_surface_area_from_pointmap(pointmap, mask)
            hand_penalty = 0.0
            if hand_masks is not None and frame_id in hand_masks:
                hand = hand_masks[frame_id]
                hand_overlap = np.sum(mask & hand) / max(np.sum(mask), 1)
                hand_penalty = hand_overlap * area
            score = area - hand_penalty
            if score > best_score:
                best_score = score
                best_frame = frame_id
```

**手部mask获取方式**:

```python
# 方式A: 使用SAM3分割 "hand" 类别（推荐，已有基础设施）
sam3_video_model.handle_request(dict(type="add_prompt", frame_index=0, text="hand"))
hand_outputs = propagate_in_video(sam3_video_model, session_id)
hand_masks = {}  # {frame_id: binary_mask}
for frame_idx, output in hand_outputs.items():
    for obj_id in output['out_obj_ids']:
        mask = output['out_binary_masks'][obj_id].squeeze() > 0
        if frame_idx not in hand_masks:
            hand_masks[frame_idx] = np.zeros_like(mask, dtype=bool)
        hand_masks[frame_idx] |= mask

# 方式B: 使用MediaPipe Hands（轻量，CPU即可运行）
import mediapipe as mp
hands_detector = mp.solutions.hands.Hands(
    static_image_mode=False, max_num_hands=2,
    min_detection_confidence=0.5
)
hand_masks = {}
for frame_idx, image in enumerate(colors):
    results = hands_detector.process(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if results.multi_hand_landmarks:
        # 将手部关键点膨胀为mask
        hand_mask = np.zeros(image.shape[:2], dtype=bool)
        for hand_lms in results.multi_hand_landmarks:
            for lm in hand_lms.landmark:
                x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                # 膨胀半径20像素
                hand_mask[max(0,y-20):y+20, max(0,x-20):x+20] = True
        hand_masks[frame_idx] = hand_mask
```

#### 方案2: mask后处理 — 从物体mask中去除手部区域

**实现位置**: `mainv2.py` 的 `run_stage3()` 中，在 `generate_3d_asset_in_subprocess` 之前

```python
# 在 run_stage3 中，3D资产生成之前:
if hand_masks is not None:
    for category, category_masks in deduplicated_all_masks.items():
        for inst_idx, instance_masks in enumerate(category_masks):
            for im in instance_masks:
                fid = im['frame_id']
                if fid in hand_masks:
                    # 从物体mask中减去手部区域
                    im['mask'] = im['mask'] & (~hand_masks[fid])
```

**局限**:
- 去除手部后mask可能变小或不连续
- SAM3D可能生成不完整的mesh（缺少被遮挡部分）
- 如果手部遮挡面积过大，去除后mask可能太小无法生成有效mesh

#### 方案3: mesh后处理 — 连通分量清理

**实现位置**: `src/instance_generation.py` 的 `generate_3d_asset()` 返回前

```python
def clean_mesh(mesh, min_component_ratio=0.3):
    """保留最大连通分量，去除小碎片（如手指状突起）"""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    max_area = max(c.area for c in components)
    large_components = [c for c in components if c.area > max_area * min_component_ratio]
    if not large_components:
        return max(components, key=lambda c: c.area)
    return trimesh.util.concatenate(large_components)

# 在 generate_3d_asset 返回前:
original_mesh = clean_mesh(output["glb"])
```

**优点**: 不需要手部检测，直接清理mesh中的小碎片
**局限**: 如果手部和物体在同一个连通分量中，无法分离

#### 方案4: 2D时序连续性去重（解决跟踪断裂导致的重复实例）

**实现位置**: `src/sg_deduplication.py`

当SAM3因手部遮挡导致跟踪断裂，同一物体被分成多个实例时，用2D时序连续性判断是否为同一物体：

```python
def temporal_continuity_deduplicate(all_masks, min_iou=0.3):
    """2D时序连续性去重: 如果两个实例在时序上首尾相接且2D IoU高，则合并"""
    for category, instances in all_masks.items():
        merged = []
        used = set()
        for i in range(len(instances)):
            if i in used:
                continue
            current = instances[i]
            last_frame = max(im['frame_id'] for im in current)
            last_mask = next(im['mask'] for im in current if im['frame_id'] == last_frame)

            for j in range(i + 1, len(instances)):
                if j in used:
                    continue
                other = instances[j]
                first_frame = min(im['frame_id'] for im in other)
                first_mask = next(im['mask'] for im in other if im['frame_id'] == first_frame)

                # 检查时序连续性: 一个实例的最后一帧和另一个实例的第一帧相邻
                if abs(first_frame - last_frame) <= 5:  # 允许5帧间隙
                    # 检查2D IoU
                    intersection = np.sum(last_mask & first_mask)
                    union = np.sum(last_mask | first_mask)
                    iou = intersection / max(union, 1)
                    if iou > min_iou:
                        # 合并两个实例
                        current = current + other
                        used.add(j)

            merged.append(current)
        all_masks[category] = merged
    return all_masks
```

#### 方案5: 多帧融合3D资产生成（长期方案）

当单帧无法获得完整物体时，融合多帧信息：

```python
def generate_3d_asset_multi_frame(image_list, mask_list, pointmap_list, extrinsic_list, inference):
    """从多帧生成3D资产，取各帧mask的并集作为完整mask"""
    # 取所有帧mask的并集
    union_mask = np.zeros_like(mask_list[0], dtype=bool)
    for mask in mask_list:
        # 去除手部区域后取并集
        clean_mask = mask & (~hand_mask) if hand_mask is not None else mask
        union_mask |= clean_mask

    # 选择手部遮挡最少的帧作为主帧
    best_idx = argmin(hand_overlap(mask, hand_mask) for mask in mask_list)

    # 用主帧的图像 + 并集mask生成3D资产
    return generate_3d_asset(
        image_list[best_idx], union_mask,
        pointmap_list[best_idx], extrinsic_list[best_idx], inference
    )
```

### 7.4 方案优先级与实施建议

| 优先级 | 方案 | 改动量 | 效果 | 适用场景 | 依赖 |
|--------|------|--------|------|---------|------|
| 🥇 | 方案1: 手部感知帧选择 | ⭐小 | ⭐⭐⭐⭐ | 手部短暂遮挡 | MediaPipe/SAM3手部检测 |
| 🥈 | 方案3: mesh连通分量清理 | ⭐小 | ⭐⭐⭐ | 手部几何混入mesh | 无 |
| 🥉 | 方案4: 2D时序连续性去重 | ⭐⭐中 | ⭐⭐⭐⭐ | 跟踪断裂导致重复实例 | 无 |
| 4 | 方案2: mask去除手部 | ⭐小 | ⭐⭐ | 手部区域明确 | 手部检测 |
| 5 | 方案5: 多帧融合 | ⭐⭐⭐大 | ⭐⭐⭐⭐⭐ | 长时间遮挡 | 手部检测+多帧对齐 |

**推荐实施顺序**: 方案1 → 方案3 → 方案4 → 方案2 → 方案5

方案1和方案3可以立即实施，无需额外依赖。方案4解决跟踪断裂问题，是中期最重要的改进。方案5是长期目标，需要多帧对齐技术支持。

---

## 八、3D物体摆放位置的影响因素总结

### 8.1 位置精度的影响链

```
VGGT点云质量 ──────────────────────────────────────────────┐
  │ (世界坐标精度)                                           │
  ▼                                                         │
pointmap精度 ──→ SAM3D几何条件 ──→ l2c矩阵精度 ──→ T矩阵   │
                                                    位置分量 │
                                                            │
extrinsic精度 ──→ inv(extrinsic) ──────────────→ T矩阵位置分量
  │ (相机位姿精度)                                           │
  ▼                                                         │
坐标系对齐 ──→ R, t变换 ──→ world_points和extrinsics都已对齐  │
                                                            │
SAM3 mask质量 ──→ SAM3D分割区域 ──→ mesh形状和l2c ──→ T矩阵  │
                                                            │
语义精修 ──→ 修改T的平移分量 ──→ 最终位置                     │
```

### 8.2 各因素对位置的影响程度

| 因素 | 影响程度 | 影响方式 | 可修复性 |
|------|---------|---------|---------|
| extrinsic误差 | ⭐⭐⭐⭐⭐ | 直接决定物体在世界中的位置 | Stage4 ICP对齐可部分修复 |
| pointmap误差 | ⭐⭐⭐⭐ | 影响SAM3D的l2c计算 | 无法直接修复 |
| mask不干净(含手部) | ⭐⭐⭐ | SAM3D生成包含手部的mesh | 方案1/2/3可缓解 |
| 坐标系对齐误差 | ⭐⭐⭐ | 所有物体位置系统性偏移 | 依赖floor/wall分割质量 |
| 动态/静态误判 | ⭐⭐ | 选错最优帧 → mesh质量差 | 改进判断逻辑 |
| 精修阈值(0.3m) | ⭐⭐ | 超过阈值的物体不精修 | 可调整阈值 |

### 8.3 main.py vs mainv2.py 在3D摆放上的关键差异

| 差异点 | main.py | mainv2.py | 影响 |
|--------|---------|-----------|------|
| 精修写回 | ❌ 未写回 `category_instances[instance_id]` | ✅ L815 已修复 | main.py所有精修结果丢失 |
| 关系格式 | 只支持 `supported_by_floor` | ✅ 兼容两种格式 | main.py无法处理空格格式 |
| 跨类去重保护 | ❌ 无 | ✅ `protected_categories` | 防止不同物体被错误合并 |
| 动态判断 | 首尾帧位移 | ✅ 中位数+全局位移 | 更鲁棒的动静态分类 |
| 物体间精修 | ❌ 未实现 | ✅ Stage 5.1+5.2 | 桌上物体悬空问题 |
| pkl保存 | ❌ 无 | ✅ 保留原始T+mesh | 后处理管线可独立运行 |
