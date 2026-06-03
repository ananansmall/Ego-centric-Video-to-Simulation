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
