# mainv2.py 完整技术文档

> 整合时间: 2026-06-02
> 最近更新: 2026-06-18 (同步默认值变更、修复状态更新)
> 涵盖: mainv2 vs main 差异、新增模块逻辑、常见问题与解答

---

## 目录

1. [mainv2 vs main: 架构差异总览](#1-架构差异)
2. [mainv2 vs main: 逐阶段对比](#2-逐阶段对比)
3. [新增模块详解](#3-新增模块详解)
4. [GLB文件体系](#4-glb文件体系)
4.1. [位姿变化记录 pose_changes.json](#41-位姿变化记录-pose_changesjson)
5. [动态物体检测](#5-动态物体检测)
6. [SP精修逻辑](#6-sp精修逻辑)
7. [后处理管线 run_post_pipeline.py](#7-后处理管线)
8. [常见问题与解答](#8-常见问题)
9. [命令行参数完整参考](#9-命令行参数)
10. [修改记录](#10-修改记录)

---

## 1. 架构差异

| 维度 | main.py | mainv2.py |
|------|---------|-----------|
| 函数结构 | 1个平铺函数 `main(args)` | 模块化: run_stage1/2/3/4 + vlm_dynamic_detection + save_final_glb |
| Stage1 | 手动加载 `--category_path` JSON | 自动发现 (subprocess调用 `generate_scene_json_stage1`) |
| VGGT模型 | 仅 vggt | vggt / vggt_omega / vggt4d |
| Stage4 | 注释 "代码暂不公开" | 完整实现 (可选 `--enable_stage4`) |
| Stage5 | 仅基础精修 (3种关系) | 基础精修(始终执行) + 5.1关系细化 + 5.2物体间精修 (可选) |
| VLM动态检测 | 无 | VLM+点云加权投票 (可选 `--enable_vlm_dynamic`) |
| 日志系统 | 无 | 双输出 (文件+控制台) |
| 输出目录 | 固定 | 含模型名 (`hoi4d_vggt_omega`) |

### 核心原则

**mainv2 不指定 stage4/5 参数时，功能与 main.py 完全一致**（除Stage1自动发现外）。

```
mainv2.py (无 --enable_stage4/5):
  Stage1 → Stage2 → Stage3 → 基础精修 → final_scene.glb
  等价于 main.py 的完整流程

mainv2.py (--enable_stage4 --enable_stage5):
  Stage1 → Stage2 → Stage3 → 基础精修 → Stage4 → Stage5 → final_scene_stage4_5.glb
  (生成 5 个 GLB, 详见 §4 GLB文件体系)
```

---

## 2. 逐阶段对比

### Stage 1: 物体发现

| | main.py | mainv2.py |
|---|---------|-----------|
| 方式 | 手动 `--category_path` | 自动 subprocess 调用 `generate_scene_json_stage1` |
| VLM | 不需要 | Qwen3.5-9B / Qwen2.5-VL-3B (自动回退) |
| 关键帧 | 无 | 贪心采样12帧 + `keyframes_metadata.json` |
| 关系格式 | `supported_by_floor` (下划线) | 兼容下划线和空格 |
| 代码位置 | `with open(args.category_path, 'r') as f:` | `run_stage1()` 函数，subprocess调用 |
| 输入参数 | `--category_path` (必须) | `--input_video` (必须), `--max_frames_stage1` (默认12) |
| 输出文件 | 无 | `scene_{id}_stage1.json` + `keyframes/` |

### Stage 2: 3D重建 + 去重

| | main.py | mainv2.py |
|---|---------|-----------|
| VGGT模型 | 仅 `vggt_predict` | 按 `--vggt_model` 选择 (vggt/vggt_omega/vggt4d) |
| 模型加载 | `load_vggt_model()` | `load_vggt_model()` / `load_vggt_omega_model()` / `load_vggt4d_model()` |
| 帧加载 | `load_video_frames(video, max_frames)` | vggt/vggt4d: `load_video_frames`; vggt_omega: `load_vggt_omega_frames` |
| 图像分辨率 | 518 (patch_size=14) | vggt/vggt4d: 518; vggt_omega: 512 (patch_size=16) |
| 预测函数 | `vggt_predict(frames, model)` | `vggt_predict` / `vggt_omega_predict` / `vggt4d_predict` |
| protected_categories | **无** | **白名单过滤**: 跨类去重后，只保留Stage1 JSON中存在的类别 (`json_categories_set`)，过滤掉SAM3误检的类别 |
| 跨类去重调用 | `cross_category_deduplicate(all_masks, pts, conf)` | `cross_category_deduplicate(all_masks, pts, conf)` + 白名单过滤 |
| 代码位置 | 平铺在main()中 | `run_stage2()` 函数 |

**关键差异: VGGT模型选择**

```python
# main.py: 固定使用VGGT
frames = load_video_frames(args.input_video, args.max_frames).to(device)
vggt_model = load_vggt_model().to(device)
vggt_prediction_results = vggt_predict(frames, vggt_model)

# mainv2.py: 按 --vggt_model 参数选择
if vggt_model_type == "vggt":
    frames = load_video_frames(input_video, max_frames).to(device)
    vggt_model = load_vggt_model().to(device)
    vggt_prediction_results = vggt_predict(frames, vggt_model)
elif vggt_model_type == "vggt_omega":
    from src.vggt_omega_predict import load_vggt_omega_frames
    frames = load_vggt_omega_frames(input_video, max_frames).to(device)
    vggt_model = load_vggt_omega_model().to(device)
    vggt_prediction_results = vggt_omega_predict(frames, vggt_model)
elif vggt_model_type == "vggt4d":
    from src.vggt4d_predict import load_vggt4d_frames
    frames = load_vggt4d_frames(input_video, max_frames).to(device)
    vggt_model = load_vggt4d_model().to(device)
    vggt_prediction_results = vggt4d_predict(frames, vggt_model)
```

**关键差异: VGGT-Omega 分辨率不同**

VGGT-Omega 使用 patch_size=16, image_resolution=512，而 VGGT/VGGT4D 使用 patch_size=14, image_resolution=518。每个模型必须用自己的 `load_*_frames` 函数，不能用 `ReplicateAnyScene` 的 `load_video_frames`。

**关键差异: Stage 2 四阶段 Z 轴对齐 (新增)**

`run_stage2()` 在 SAM3 分割 floor/wall 后，按以下优先级逐级尝试对齐到房间坐标系，只有当前阶段失败才进入下一阶段：

| 阶段 | 函数 | 输入 | 成功条件 |
|------|------|------|---------|
| 1 | `align_to_room_coordinate_system` | SAM3 `floor`/`wall` 文本提示 mask | 同时存在有效 floor 和正交 wall 平面 |
| 2 | `align_via_objects` | 放宽阈值的 floor (+ wall 或点云 PCA) | 存在有效 floor 平面 |
| 2.5 | `align_via_vlm_floor_points` | VLM 地面参考点 + SAM3 `box` prompt | VLM 返回有效点且 SAM3 分割出 floor |
| 3 | `align_via_large_plane` | SAM3 大平面 mask (`flat surface`/`ground`/`horizontal surface`) | 存在有效大平面 |
| 4 | `align_via_geocalib` | GeoCalib 重力估计 | 至少一帧重力估计成功且内点足够 |

```python
R, t = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
if _is_identity_alignment(R, t):
    R, t, info = align_via_objects(world_points, wall_masks, floor_masks)
    if _is_identity_alignment(R, t):
        # 2.5 VLM + SAM3 box prompt
        R, t, info = align_via_vlm_floor_points(...)
        if _is_identity_alignment(R, t):
            large_plane_masks = segment_large_flat_surfaces(...)
            R, t, info = align_via_large_plane(world_points, large_plane_masks)
            if _is_identity_alignment(R, t):
                R, t, info = align_via_geocalib(images, world_points)
```

**关于 SAM3 点提示**: `sam3/model/sam3_image_processor.py` 中的 `Sam3Processor` 只暴露了 `add_geometric_prompt(box=..., label=...)`，没有公开的点提示 API。底层 `FindStage` 虽然预留了 `input_points`/`input_points_mask` 字段，但图像推理封装未开放。因此阶段 2.5 采用 **VLM 生成地面参考点 + 围绕点构造小 box** 的方式来近似点提示。

### Stage 3: 资产生成

| | main.py | mainv2.py |
|---|---------|-----------|
| `get_optimal_view_frame_id` | 返回3个值 (frame_id, is_dynamic, motion_info) | 一致 |
| 动态检测 | 中值位移 + 全局位移 | 一致 |
| instance_visibility | 无 | 新增，供Stage5.1使用 |
| 代码位置 | 平铺在main()中 | `run_stage3()` 函数 |

**关键差异: 动态检测返回值**

main.py 和 mainv2.py 都正确接收3个返回值:
```python
optimal_frame_id, is_dynamic, motion_info = get_optimal_view_frame_id(
    vggt_prediction_results['world_points'], instance_masks
)
```

### 基础精修 (Stage 5.0)

| | main.py | mainv2.py |
|---|---------|-----------|
| 关系匹配 | 仅 `supported_by_floor` (下划线) | 兼容 `supported_by_floor` 和 `supported by floor` (空格) |
| 精修写回 | **BUG: 未写回** | **已修复: `category_instances[instance_id] = instance_info`** |
| 代码位置 | 平铺在main()中 | main()中，Stage4之前 |
| camera_pos计算 | `- extrinsic[:3,:3].T @ extrinsic[:3,3]` | `-extrinsic[:3, :3].T @ extrinsic[:3, 3]` (仅格式差异) |

**关键BUG修复: 精修结果未写回**

```python
# main.py: BUG - 精修结果未写回
for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(...)):
    if relationship == "supported_by_floor":
        instance_info = refine_supported_by_floor_object(instance_info)
    # ...
    else:
        continue
    # ← 缺少: category_instances[instance_id] = instance_info

# mainv2.py: 已修复
for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(...)):
    if relationship == "supported_by_floor" or relationship == "supported by floor":
        instance_info = refine_supported_by_floor_object(instance_info)
    # ...
    else:
        continue
    category_instances[instance_id] = instance_info  # ← 修复: 正确写回
```

### Stage 4: 视觉-空间对齐 (mainv2新增)

| | main.py | mainv2.py |
|---|---------|-----------|
| 状态 | 注释 "代码暂不公开" | 完整实现 (需 `--enable_stage4`) |
| 功能 | — | MASt3R/深度匹配 → Umeyama变换 → ICP精调 |
| 穿模解决 | — | Stage4后执行 `resolve_penetrations` |
| 参数 | — | `--stage4_iterations`, `--stage4_temporal_radius`, `--stage4_use_mast3r` |

### Stage 5: 语义精修 (mainv2新增)

| | main.py | mainv2.py |
|---|---------|-----------|
| 基础精修 | Stage5内 (3种关系) | **独立为5.0**，始终执行 |
| 5.1 关系细化 | 无 | **新增**: VLM投票细化 "supported by other objects" → "supported by {name}" |
| 5.2 空间精修 | 无 | **新增**: 物体间支撑关系几何精修 (on_top/inside/against_side等) |

### GLB保存

| | main.py | mainv2.py |
|---|---------|-----------|
| 保存方式 | 平铺在main()中 | `save_final_glb()` 函数 |
| 输出文件 | `final_scene.glb` | `final_scene_base.glb` (固定起点) + `final_scene.glb` (最新结果) |
| z-up→y-up变换 | `[[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]` | 一致 |
| node_name | `{category}_{i}` | 一致 |

### 其他差异

| | main.py | mainv2.py |
|---|---------|-----------|
| 日志系统 | 无 | 双输出 (文件+控制台)，`builtins.print` 劫持 |
| 异常处理 | 无 | 顶层 try/except + 日志记录 |
| 耗时统计 | 无 | 总耗时统计 (`time.time()`) |
| 输出目录 | 固定 `./outputs/hallway` | 自动生成 `./output_v2/{video_stem}_{vggt_model}` |
| 最终关系JSON | 无 | `final_relations.json` |
| 实例数据pkl | 无 | `all_instances.pkl` (含all_instances, all_optimal_frame_ids, categories_and_relations, walls_info) |
| VLM动态检测 | 无 | `vlm_dynamic_detection()` + `--enable_vlm_dynamic` |
| 删除的导入 | `generate_3d_asset` (直接调用) | 仅保留 `generate_3d_asset_in_subprocess` |
| `--category_path` | 必须指定 (默认 `./assets/example/hallway.json`) | **已移除**，Stage1自动发现 |
| `--input_images` | 无 | 新增，与 `--input_video` 互斥 |
| `--input_video` 默认值 | `'./assets/example/hallway.mp4'` | None (必须指定) |
| `--output_path` 默认值 | `'./outputs/hallway'` | None (自动生成) |
| 打印语言 | 英文 | 中文 |
| 注释掉的代码 | 含大段注释的替代3D资产生成代码 (Inference类) | 已移除 |
| VLM模型查找 | 无 | `_resolve_vlm_checkpoint()` 自动查找 (Qwen3.5-9B → Qwen2.5-VL-3B) |
| Stage5逻辑 | 注释占位 | `run_stage5()` 函数已定义并在 `main()` 中调用，`main()` 内只负责 GLB 命名与 pose 记录 |
| 输入验证 | 仅检查文件存在 | 互斥验证 + 目录检查 + 自动统一为input_video |

**关键差异: `--category_path` 已移除**

main.py 需要 `--category_path` 手动指定场景JSON，mainv2 通过 Stage1 自动发现，不再支持此参数。

**关键差异: `--input_images` 与 `--input_video` 互斥**

```python
# mainv2.py 输入验证
if args.input_images and args.input_video:
    parser.error("--input_video 和 --input_images 不能同时使用")
if not args.input_images and not args.input_video:
    parser.error("必须指定 --input_video 或 --input_images")
if args.input_images:
    if not os.path.isdir(args.input_images):
        parser.error(f"图片目录不存在: {args.input_images}")
    args.input_video = args.input_images  # 统一为 input_video
```

**关键差异: 输出路径自动生成**

```python
# main.py: 固定默认值
--output_path default='./outputs/hallway'

# mainv2.py: 动态生成
if args.output_path is None:
    video_stem = os.path.splitext(os.path.basename(args.input_video))[0]
    args.output_path = os.path.join(".", "output_v2", f"{video_stem}_{args.vggt_model}")
```

**关键差异: VLM模型自动查找**

```python
def _resolve_vlm_checkpoint():
    """按优先级自动查找VLM模型"""
    # 1. 用户指定路径
    # 2. /mnt/data/lza/models/Qwen3.5-9B
    # 3. /mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/{hash}
    # 找不到时 sys.exit(1)
```

**关键差异: Stage5 逻辑统一**

`run_stage5()` 函数已统一封装 5.1 关系细化与 5.2 SP 精修逻辑，`main()` 直接调用并负责最终 GLB 命名与 `pose_changes.json` 记录。

---

## 3. 新增模块详解

### 3.1 Stage 4: 迭代视觉-空间对齐

**目的**: 用多帧视觉匹配修正单帧点云对齐的位姿误差。

**流程**:
```
对每个物体:
  for i in range(iterations):
    1. 用当前 T 渲染物体 → 渲染图 I_ren
    2. MASt3R 匹配 渲染图 vs 真实RGB图 → 2D对应点对
    3. 2D对应点 → 反投影到3D → 3D对应点对
    4. Umeyama 算法求解最优 s,R,t
  选择 IoU 最高的迭代结果作为最终 T*
```

**坐标系**: Stage4在z-up空间操作，输出T矩阵直接应用到all_instances。

**穿模解决**: Stage4后执行 `resolve_penetrations`，用AABB碰撞检测分离穿模物体。

### 3.2 Stage 5.1: 关系细化

**目的**: 将 "supported by other objects" 细化为 "supported by {具体物体名}"。

**流程**:
```
对每个 "supported by other objects" 的物体:
  1. 从 instance_visibility 获取可见帧 (最多5帧)
  2. 裁剪物体区域图像
  3. VLM 判断: "What is this object resting on or supported by?"
  4. 多帧投票: 出现次数最多的支撑物名称
  5. 更新关系: "supported by other objects" → "supported by table"
```

### 3.2 Stage 5.2: 物体间SP精修

**目的**: 根据细化后的关系，对物体间支撑关系做几何精修。

**5种放置策略**:

| 关系 | 放置策略 | 物理约束 |
|------|---------|---------|
| supported by {name} | on_top | 底面贴顶面，不穿模不悬空 |
| inside {name} | inside | 放入容器内部，不穿出 |
| against_side of {name} | against_side | 底面贴地 + 侧面接触 |
| hanging_below {name} | hanging_below | 顶面贴支撑物底面 |
| leaning on {name} | leaning | 底面贴地 + 倾斜靠在支撑物上 |

**精修核心逻辑**: 详见[第6节](#6-sp精修逻辑)。

**依赖排序**: 拓扑排序处理物体间支撑关系，确保支撑物优先精修。

---

## 4. GLB文件体系

### GLB 生成流程图 (核心)

`all_instances` 是贯穿全流程的核心数据结构。每个阶段修改它，然后保存一个 GLB 快照。

```
Stage1 (物体发现)
    │  输出: categories_and_relations (JSON)
    ▼
Stage2 (3D重建 + Z轴对齐 + SAM3去重)
    │  修改: vggt_prediction_results (对齐R,t)
    │  输出: world_points, colors, extrinsics, point_cloud.ply
    ▼
Stage3 (最优视角资产生成)
    │  修改: all_instances = {category: [instance_info, ...]}
    │        每个 instance_info 含 original_mesh + T矩阵
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  save_final_glb(all_instances, "final_scene_initial.glb")       │
│  ► GLB #1: Stage3原始结果, 未做任何精修                          │
│  ► 代码: mainv2.py L1111                                        │
│  ► 数据来源: all_instances (Stage3刚生成)                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼  基础精修 (floor/wall/embedded, 始终执行)
       修改: all_instances 中每个物体的 T 矩阵
       (refine_supported_by_floor / refine_embedded_in_wall / refine_attached_to_wall)
┌─────────────────────────────────────────────────────────────────┐
│  save_final_glb(all_instances, "final_scene.glb")               │
│  ► GLB #2: 基础精修后结果 (固定起点)                             │
│  ► 代码: mainv2.py L1138                                        │
│  ► 数据来源: all_instances (基础精修后)                          │
│  ► 同时保存: all_instances.pkl (供 run_post_pipeline.py 使用)    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼  --enable_stage4?
    │
    ├── YES ──► Stage4 (迭代视觉-空间对齐)
    │           修改: all_instances (ICP对齐 + resolve_penetrations)
    │           ┌─────────────────────────────────────────────────┐
    │           │ save_final_glb(all_instances,                   │
    │           │   "final_scene_stage4.glb")                     │
    │           │ ► GLB #3: Stage4对齐后结果                      │
    │           │ ► 代码: mainv2.py L1167                         │
    │           │ ► 数据来源: all_instances (Stage4后)             │
    │           │ ► 同时保存: all_instances_stage4.pkl             │
    │           └─────────────────────────────────────────────────┘
    │
    ▼  --enable_stage5?
    │
    ├── YES ──► Stage5 (高级语义精修)
    │           │
    │           ├── 5.1: 关系推断 (scene_graph / per_object)
    │           │       修改: refined_relations (不修改 all_instances)
    │           │
    │           ├── 5.2: SP精修 (有物体间关系时)
    │           │       修改: all_instances (on_top/inside/against_side)
    │           │       + resolve_penetrations (穿模修复)
    │           │       ┌────────────────────────────────────────┐
    │           │       │ 保存: "final_scene_stage5_sp.glb"      │
    │           │       │ ► GLB #4: SP精修+穿模修复后中间结果     │
    │           │       │ ► 代码: refine_inter_object_placement  │
    │           │       │       .py L1793                        │
    │           │       │ ► 数据来源: all_instances (SP精修后)    │
    │           │       └────────────────────────────────────────┘
    │           │
    │           └── check_stability (稳定性检查)
    │               修改: all_instances (旋转对齐+z贴合+悬空修复)
    │               ┌────────────────────────────────────────────┐
    │               │ 保存: final_glb_name                       │
    │               │   有stage4 → "final_scene_stage4_5.glb"    │
    │               │   无stage4 → "final_scene_stage5.glb"      │
    │               │ ► GLB #5 (或#4无stage4时): 最终结果        │
    │               │ ► 代码: refine_inter_object_placement      │
    │               │       .py L1822                            │
    │               │ ► 数据来源: all_instances (check_stability)│
    │               └────────────────────────────────────────────┘
    │
    │           (无物体间关系时, 5.2跳过, mainv2.py L1199 兜底保存)
    │
    ▼
结束: 输出最终GLB路径 + pose_changes.json + z_axis_alignment.json
```

### GLB 文件数量 (按启用阶段)

| 启用阶段 | 生成文件 | 总数 |
|---------|---------|------|
| 默认 (无 stage4/5) | `final_scene_initial.glb`, `final_scene.glb` | 2 |
| 仅 `--enable_stage5` | + `final_scene_stage5_sp.glb`*, `final_scene_stage5.glb` | 4* |
| 仅 `--enable_stage4` | + `final_scene_stage4.glb` | 3 |
| `--enable_stage4 --enable_stage5` | + `final_scene_stage4.glb`, `final_scene_stage5_sp.glb`*, `final_scene_stage4_5.glb` | 5* |

> *`final_scene_stage5_sp.glb` 仅在有物体间支撑关系时生成。无物体间关系时 5.2 跳过，总数减 1。

### 各 GLB 详细说明

| GLB | 保存位置 | 保存时机 | 数据来源 | 坐标系 |
|-----|---------|---------|---------|--------|
| `final_scene_initial.glb` | mainv2.py L1111 | Stage3 完成后、基础精修前 | all_instances (Stage3原始) | y-up |
| `final_scene.glb` | mainv2.py L1138 | 基础精修 (floor/wall/embedded) 后 | all_instances (基础精修后) | y-up |
| `final_scene_stage4.glb` | mainv2.py L1167 | Stage4 视觉-空间对齐后 | all_instances (Stage4后) | y-up |
| `final_scene_stage5_sp.glb` | refine_inter_object_placement.py L1793 | Stage5 SP精修+穿模修复后 | all_instances (SP精修后) | y-up |
| `final_scene_stage5.glb` | refine_inter_object_placement.py L1822 / mainv2.py L1199 | 仅Stage5、check_stability后 | all_instances (check_stability后) | y-up |
| `final_scene_stage4_5.glb` | refine_inter_object_placement.py L1822 / mainv2.py L1199 | Stage4+Stage5全部后处理后 | all_instances (check_stability后) | y-up |

### GLB 传递关系 (谁根据谁生成)

```
all_instances (内存中的核心数据)
    │
    ├─ Stage3完成 ──► final_scene_initial.glb  (快照#1)
    │
    ├─ 基础精修 ────► final_scene.glb          (快照#2, 固定起点)
    │                 + all_instances.pkl      (pkl快照, 供独立后处理)
    │
    ├─ Stage4 ──────► final_scene_stage4.glb   (快照#3)
    │                 + all_instances_stage4.pkl
    │
    └─ Stage5 ──────► final_scene_stage5_sp.glb  (快照#4, 中间结果)
                      final_scene_stage5.glb      (快照#4最终, 无stage4)
                      final_scene_stage4_5.glb    (快照#5最终, 有stage4)
```

**关键点**:
- 每个 GLB 都是 `all_instances` 在某个时间点的**快照**
- `all_instances` 在内存中被各阶段**就地修改** (T矩阵变化)
- GLB 之间**不相互依赖** — 都直接从 `all_instances` 生成
- `all_instances.pkl` 是 `all_instances` 的 pickle 快照, 供 `run_post_pipeline.py` 独立重跑 Stage4/5

### pkl数据结构

```python
{
    'all_instances': all_instances,           # {category: [instance_info, ...]}
    'all_optimal_frame_ids': all_optimal_frame_ids,  # {category: [frame_id, ...]}
    'categories_and_relations': categories_and_relations,  # {category: relation_str}
    'walls_info': walls_info,                 # 墙壁几何信息
}
```

### --cleanup 参数 (新增)

启用 `--cleanup` 后，流水线结束时自动删除中间文件：

| 删除的文件 | 说明 |
|-----------|------|
| `all_instances.pkl` | 基础精修后pkl (382M+) |
| `all_instances_stage4.pkl` | Stage4后pkl |
| `color/` | VGGT输入帧 (160张jpg) |
| `depth/` | VGGT深度图 (160张png) |
| `extrinsics/` | 相机外参 |
| `keyframes/` | Stage1关键帧 |
| `optimal_frames/` | Stage3最优视角帧 |

**保留的文件**: 所有 `.glb`, `.ply`, `.json`, `.log`, `intrinsic.txt`

### GLB坐标系

- **内部处理**: z-up (与`sp_refinement.py`、`geometry_utils.py`一致)
- **GLB输出**: y-up (trimesh标准)
- **变换**: 保存时执行 z-up → y-up 变换 (仅作用于临时 Scene 对象, 不回写 all_instances)

### Stage5 是否改变坐标系？

**不改变。** Stage5 的所有操作都是**单个物体的 T 矩阵微调**, 不存在全局坐标系变换:

| 操作 | 代码位置 | 类型 |
|------|---------|------|
| SP精修 (on_top/inside/against_side等) | refine_inter_object_placement.py L608-869 | 单个物体: 旋转对齐+z贴合 |
| `_align_upright` | refine_inter_object_placement.py L590-605 | 单个物体: 旋转对齐 (当前已禁用) |
| `resolve_penetrations` | refine_inter_object_placement.py L1110-1278 | 单个物体: 沿分离轴推开 |
| `check_stability` Phase 1-4 | refine_inter_object_placement.py L1281-1504 | 单个物体: 旋转对齐+z贴合+悬空修复 |

**唯一的全局坐标系变换发生在 Stage2** (`align_to_room_coordinate_system` + `align_vggt_predictions`), 将 VGGT 原始预测对齐到房间坐标系。Stage5 只在此基础上微调各物体的位姿。

**用户观察到"不同GLB之间z轴变化"的原因**: 不是坐标系变了, 而是各阶段精修改变了物体的 T 矩阵, 导致物体在场景中的朝向和位置不同。例如:
- `final_scene_initial.glb`: 物体可能严重倾斜 (Stage2对齐失败时)
- `final_scene.glb`: 基础精修做了旋转对齐, 物体更竖直
- `final_scene_stage5_sp.glb`: SP精修进一步调整了物体位置
- `final_scene_stage5.glb`: check_stability 确保物体稳定

---

## 4.1 位姿变化记录 `pose_changes.json`

每个物体在各阶段的位姿会被记录到 `pose_changes.json`，便于追溯和调试：

```json
{
  "table_0": {
    "category": "table",
    "instance_idx": 0,
    "relation": "supported by floor",
    "stages": {
      "initial": { "T_matrix": [...], "position": [...], "bounds_min": [...], ... },
      "basic_refinement": { "T_matrix": [...], "position": [...], "delta_from_initial": [...] },
      "stage4": { "T_matrix": [...], "position": [...], "delta_from_basic_refinement": [...] },
      "stage5": { "T_matrix": [...], "position": [...], "delta_from_stage4": [...] }
    }
  }
}
```

**记录规则**:
- `initial`: Stage3 完成后、基础精修前
- `basic_refinement`: 基础精修 (floor/wall/embedded) 后
- `stage4`: 仅当启用 `--enable_stage4` 时记录
- `stage5`: 仅当启用 `--enable_stage5` 时记录
- `physics`: 仅当启用 `--enable_physics_validation` 时记录

**向后兼容**: 不启用 Stage4 时，`pose_changes.json` 中不会出现 `stage4` 键，代码通过 `args.enable_stage4` 条件判断，不会出现 KeyError。

---

## 5. 动态物体检测

### 5.1 点云位移检测 (`geometry_utils.py`)

**两个信号**:

| 信号 | 计算方式 | 阈值 | 特点 |
|------|---------|------|------|
| 逐帧中值位移 | 相邻帧质心位移的中位数 | > 0.02m | 捕捉逐帧运动 |
| 全局位移 | 首20%帧 vs 末20%帧质心距离 | > 0.04m | 捕捉VGGT漂移掩盖的真实运动 |

**判定规则**: 任一信号超过阈值 → 动态 (OR逻辑)

**为什么需要全局位移？** VGGT的漂移是系统性的，逐帧位移小但累积大。真实运动物体如果移动缓慢，逐帧中值位移可能低于阈值，但首尾帧质心距离会显著偏大。

### 5.2 VLM辅助动态检测 (`--enable_vlm_dynamic`)

**加权投票**:

| 信号 | 权重 | 判定方式 |
|------|------|---------|
| 点云全局位移 | 0.6 | global_disp > 0.04m |
| VLM视觉判断 | 0.4 | 观察首末帧，判断物体是否移动 |

**最终判定**: 加权得分 > 0.5 → 动态

**VLM判断流程**:
```
1. 取物体首帧和末帧图像
2. VLM提示: "Is the highlighted object moving between these frames?"
3. 回答 "moving" → VLM动态信号
```

### 5.3 动态物体帧选择

**静态物体**: 选3D表面积最大的帧（最完整建模）

**动态物体**: 选运动前的最大可见帧
```
1. 检测运动起始点 (位移突增帧)
2. 在运动前的帧中选3D表面积最大的
3. 如果没有检测到运动起始点，用首20%帧搜索
```

---

## 6. SP精修逻辑

### 6.1 核心原则

**"supported by table 的精修等价于 supported by floor，只是 z=0 换成 supporter_top_z"**

这意味着:
1. 旋转对齐: 让物体上方向对齐z轴 (竖直) — 和floor完全一样
2. z轴平移: 底面对齐到支撑面 — floor对齐到z=0，table对齐到table_top_z

### 6.2 `_align_upright` 统一旋转对齐

所有SP精修函数共用 `_align_upright(info)` 做旋转对齐:

```python
def _align_upright(info):
    """旋转对齐: 让物体上方向对齐z轴 (与 refine_supported_by_floor_object 的旋转逻辑一致)"""
    transform_matrix = info["T"].copy()
    upper_real_vector = np.array([0, 0, 1])
    upper_transformed_vector = transform_matrix[:3, 1]
    # ... align_vectors 对齐 ...
    info["T"] = transform_matrix
    return info
```

这与 `sp_refinement.py` 中 `refine_supported_by_floor_object` 的旋转逻辑完全一致。

### 6.3 五种放置策略的实现

#### on_top (最常用)

```
Step 1: _align_upright(supported_info)     — 旋转对齐 (和floor一样)
Step 2: z_offset = supporter_top_z - supported_bottom_z
Step 3: 沿z轴平移 z_offset                — 底面贴顶面 (和floor对齐z=0一样)
```

#### inside

```
Step 1: _align_upright(supported_info)     — 旋转对齐
Step 2: supported底面对齐到supporter内部30%高度
Step 3: 穿模检查 (穿出顶部/底部则修正)
```

#### against_side

```
Step 1: _align_upright(supported_info)     — 旋转对齐
Step 2: z轴底面贴地
Step 3: x/y轴移动到侧面刚好接触
Step 4: 穿模检查
```

#### hanging_below

```
Step 1: _align_upright(supported_info)     — 旋转对齐
Step 2: supported顶面对齐到supporter底面
```

#### leaning

```
调用 against_side 后，允许物体有倾斜角度
```

### 6.4 与 main.py 基础精修的关系

| main.py 基础精修 | mainv2 SP精修 | 关系 |
|-----------------|--------------|------|
| `refine_supported_by_floor_object` | `sp_refine_on_top` | 旋转对齐逻辑相同，z对齐目标不同 (z=0 vs supporter_top_z) |
| `refine_embedded_in_wall_object` | — | mainv2基础精修中执行，SP精修不重复 |
| `refine_attached_to_wall_object` | — | mainv2基础精修中执行，SP精修不重复 |

**Stage5不处理floor/wall物体**: 这些已在基础精修中处理完毕。

---

## 7. 后处理管线

### `run_post_pipeline.py`

**目的**: 独立于mainv2，对已有输出目录执行stage4/5。

**文件自动发现**:
```
输入: 场景目录路径 (如 output_v2/hoi4d)
自动发现:
  ├── color/              → RGB帧
  ├── depth/              → 深度帧
  ├── extrinsics/         → 外参
  ├── optimal_frames/     → 最优帧图像
  ├── keyframes/          → 关键帧
  ├── intrinsic.txt       → 内参
  ├── final_scene.glb     → 基础精修GLB (固定起点)
  ├── all_instances.pkl   → 实例数据
  └── scene_*_stage1.json → Stage1 JSON
```

**GLB发现优先级**: `final_scene.glb` > `final_scene_base.glb` (兼容旧版)

**调用方式**:
```bash
# 只执行Stage4
python tools/run_post_pipeline.py output_v2/hoi4d --stage4

# 只执行Stage5
python tools/run_post_pipeline.py output_v2/hoi4d --stage5

# 同时执行Stage4+5
python tools/run_post_pipeline.py output_v2/hoi4d --stage4 --stage5
```

**数据流**:
```
all_instances.pkl (基础精修后快照)
  → Stage4 → final_scene_stage4.glb
  → Stage5 → final_scene_stage5.glb / final_scene_stage4_5.glb
```

---

## 8. 常见问题

### Q1: 为什么有不同数量的 GLB 文件？

GLB 数量取决于启用的阶段：

| 启用阶段 | 文件 | 数量 |
|---------|------|------|
| 默认 | `final_scene_initial.glb`, `final_scene.glb` | 2 |
| `--enable_stage5` | + `final_scene_stage5_sp.glb`, `final_scene_stage5.glb` | 4 |
| `--enable_stage4` | + `final_scene_stage4.glb` | 3 |
| `--enable_stage4 --enable_stage5` | + `final_scene_stage4.glb`, `final_scene_stage5_sp.glb`, `final_scene_stage4_5.glb` | 5 |

`final_scene_base.glb` 仍作为 `run_post_pipeline.py` 的固定起点保留，但 `mainv2.py` 默认流程中不再生成（由 `final_scene.glb` 替代）。

### Q2: 为什么所有物体都被判为静态？

**原因1**: `get_optimal_view_frame_id` 返回3个值，旧版mainv2只接收1个，导致数据类型错误。

**原因2**: 逐帧中值位移对VGGT漂移太敏感，真实运动物体的逐帧位移可能低于阈值。

**修复**: 
1. 正确接收3个返回值 `(optimal_frame_id, is_dynamic, motion_info)`
2. 新增全局位移检测 (首20%帧 vs 末20%帧)
3. 可选VLM加权投票 (`--enable_vlm_dynamic`)

### Q3: SP精修效果为什么不好？

**原因**: 旧版 `sp_refine_on_top` 重新实现了旋转对齐逻辑，与 `refine_supported_by_floor_object` 存在差异。

**修复**: 统一使用 `_align_upright()` 函数，逻辑与 `refine_supported_by_floor_object` 的旋转对齐完全一致。

**核心原则**: "supported by table = supported by floor，只是z=0换成supporter_top_z"

### Q4: mainv2不指定参数时和main.py一样吗？

**是的**，除了Stage1自动发现外，mainv2不指定 `--enable_stage4/5` 时:
- Stage2/3逻辑与main.py一致
- 基础精修逻辑与main.py一致 (额外修复了写回BUG)
- 只输出 `final_scene_base.glb` + `final_scene.glb` (内容相同)

### Q5: 为什么动态物体没有选到运动前的帧？

**旧逻辑**: 选第一个有效帧，但物体可能刚进入画面、遮挡严重。

**新逻辑**: 
1. 检测运动起始点 (位移突增帧)
2. 在运动前的帧中选3D表面积最大的
3. 如果全局位移大但逐帧未检测到运动起始点，额外搜索首1/3帧

### Q6: 精修后物体为什么还穿模？

**可能原因**:
1. 坐标系不对齐 — Stage5在z-up空间操作，如果GLB加载时未正确转换坐标系
2. 支撑物本身位置不准 — 如果支撑物(如table)位置偏移，on_top精修会把物体放到错误高度
3. AABB碰撞检测不够精确 — 对于非规则形状物体，AABB可能过于保守

### Q7: refine_supported_by_floor_object 的 0.3m 阈值是什么意思？

`refine_supported_by_floor_object` 只对 `z_min < 0.3m` 的物体做z轴平移。如果物体离地面超过0.3m（如桌子z_min=0.47m），精修会跳过z轴调整，只做旋转对齐。

这个阈值的设计意图是：只对"接近地面"的物体做吸附，避免把桌上物体错误吸附到地面。

但在SP精修中，`_align_upright` 只做旋转对齐，不做z轴判断，z轴平移由各策略函数自己控制。

### Q8: Stage1 能不能加快速度？用 vggt_omega 行吗？还是 VLM 判断慢？

**结论：VGGT 3D重建是绝对瓶颈（~45%时间），VLM推理是第二大耗时（~30%时间）。**

Stage1 各步骤时间占比：

| 步骤 | 描述 | 时间占比 | 说明 |
|------|------|---------|------|
| Step 0 | VGGT 3D重建 | **~45%** | 120帧 × VGGT-1B完整前向推理 |
| Step 3 | 第一次VLM：物体检测 | **~18%** | 12帧逐帧VLM推理 |
| Step 6 | SAM分割floor/wall | **~12%** | SAM3模型加载+推理 |
| Step 7 | 第二次VLM：关系判断 | **~12%** | 12帧逐帧VLM推理 |
| Step 5.5 | 点云补充检测 | **~6%** | DBSCAN + ≤5次VLM |
| 其他 | 采样/帧提取/去重/射线投射 | **~7%** | 纯CPU/IO操作 |

**能否用 vggt_omega 替代？**

当前 Stage1 硬编码使用原版 VGGT（`generate_scene_json_stage1.py` 没有 `--vggt_model` 参数）。VGGT-Omega 推理速度可能快10-20%，但存在关键风险：

- VGGT-Omega 的 `depth_conf` 分布太均匀，百分位阈值无法有效区分可靠/不可靠点，导致**点云缺块**
- Stage1 的射线投射和3D覆盖采样高度依赖点云质量
- 缺块的点云会导致射线投射失败率升高、采样覆盖率下降

**不建议替换**。更好的加速方案是减少 `--vggt_max_frames`（如从120降到80），能直接减少近一半VGGT推理时间。

**VLM调用详情**：Stage1 共3类VLM调用，总计最多29次：
- 物体检测(Step3): 12次，完整帧输入，短prompt
- 补充检测(Step5.5): ≤5次，裁剪图输入，短prompt
- 关系判断(Step7): ≤12次，完整帧输入，长prompt

### Q9: 坐标系问题——点云和GLB的原点/坐标轴会有明显偏差吗？为什么会出现倒立？找不到地面就用相机吗？

**1. 点云和GLB的原点/坐标轴**

点云 (`point_cloud.ply`) 和 GLB 确实有不同的坐标系：
- 点云在 Room World 坐标系下（z-up，地板z=0，场景中心为原点）
- GLB 在 glTF 标准坐标系下（y-up），导出时做了 `z-up → y-up` 变换

两者之间是确定的轴变换关系，不会产生"偏差"，只是表示方式不同。

**2. 为什么会出现倒立坐标轴？**

倒立的根因在 `get_plane_info()` 中法线方向的确定逻辑不够鲁棒：

```python
# geometry_utils.py:193
normal = -normal if normal[0] < 0 else normal  # 仅根据x分量正负翻转
```

这个逻辑只根据法线 x 分量的正负来决定翻转，而不是根据物理含义（地板法线应该朝上）。虽然后续有修正逻辑：

```python
# geometry_utils.py:255-257
floor_to_wall_vector = wall_plane_info['centroid'] - floor_plane_info['centroid']
if np.dot(floor_to_wall_vector, floor_normal) < 0:
    floor_normal = -floor_normal
```

但这个修正**依赖墙面质心在地板上方的假设**。如果墙面分割不准确，或墙面质心恰好在地板下方，修正就会失败，导致 z 轴朝下 → 整个场景倒立。

**3. 找不到地面时是否用相机？**

**不会。** `align_to_room_coordinate_system` 在找不到地面时返回恒等变换：

```python
# geometry_utils.py:240-241
if len(floor_plane_infos) == 0:
    return np.eye(3), np.zeros(3)  # 不做任何变换，保留VGGT原始坐标系
```

没有回退到相机坐标的机制。mainv2 新增了三级坐标系后备方案（Level 1 → Level 1.5 → Level 2+3），详见 §9.9。

### Q10: SAM3D放置的姿态对吗？动态物体帧生成和摆放位置分开了吗？

**1. SAM3D的T矩阵是否正确？**

T矩阵的变换链数学推导是正确的：

```
Local(SAM3D) ──y2z──→ Local(z-up) ──l2c──→ Camera(SAM3D) ──adjust──→ Camera(VGGT) ──ext⁻¹──→ World(VGGT)
```

但T矩阵的正确性**完全依赖于VGGT预测的extrinsic精度**。VGGT可能预测了正确的相对3D结构，但相机位置偏了，导致物体绝对位置偏移。

**注意**：`_align_upright` 当前已被**禁用**（`refine_inter_object_placement.py:576-591`），函数体直接 `return info`。禁用原因是VGGT重建的物体朝向可能本身就不准确，强制旋转可能导致更差的结果。

**2. 动态物体帧生成和摆放位置是否分开？**

**是的，已部分分离。** 代码位于 `mainv2.py:507-557`：

- **mesh生成帧**：`optimal_frame_id`（3D表面积最大的帧）→ SAM3D用该帧的image/mask/pointmap/extrinsic生成mesh
- **放置位置帧**：`first_visible_frame_id`（首次被SAM3检测到的帧）→ 动态物体的T矩阵平移分量被调整到该帧的质心位置

调整方式：
```python
offset = first_visible_centroid - optimal_centroid
T[:3, 3] += offset  # 只调整平移分量，旋转不变
```

**重要限制**：
1. 只修正了位置偏移，没有修正朝向偏移（如果动态物体在不同帧朝向不同）
2. offset基于world_points的质心差，VGGT在动态区域的点云质量较差，质心本身可能有误差
3. 理想情况下应该用first_visible_frame的extrinsic重新计算整个T矩阵，但这需要重新运行SAM3D（代价太大）

### Q11: mask的给出是去重前还是去重后的？同一位置生成多个物体是去重的问题吗？和遮挡有关系吗？

**1. 传给SAM3D的masks是去重后的**

流水线中的数据流：

```
segment_and_track → category_masks (去重前)
  ↓ self_category_deduplicate
deduplicated_category_masks (类内去重后)
  ↓ cross_category_deduplicate
deduplicated_all_masks (跨类去重后)
  ↓ 白名单过滤 (json_categories_set)
filtered_masks (只保留Stage1发现的类别)
  ↓ 传入 run_stage3 → generate_3d_asset_in_subprocess
SAM3D接收去重后的masks
  ↓ deduplicate_3d_assets
3D Mesh级别二次去重
```

SAM3D接收的是经过**三层去重+白名单过滤**后的masks。

**2. 同一位置生成多个物体——根因是遮挡导致的跟踪断裂**

因果链：

```
手部遮挡/物体移动
  → SAM3跟踪断裂（同一物体被分割为多个实例）
  → self_category_deduplicate 尝试合并（基于3D点云重叠率）
  → 如果物体移动了，原位置和新位置的3D点云重叠率可能不够高
  → 合并失败 → 同一物体保留多个实例
  → SAM3D为每个实例分别生成3D资产
  → 同一位置（或相近位置）出现多个物体
```

| 场景 | 根因 | 去重能否解决 |
|------|------|-------------|
| 静态物体被遮挡后重新出现 | SAM3跟踪断裂，但3D位置不变 | **能** — 重叠率高，类内去重会合并 |
| 动态物体移动后 | SAM3在新旧位置各检测一次 | **部分能** — 新旧位置重叠率≥0.3则合并 |
| 同一物体被分为不同类别 | SAM3分割不一致 | **能** — 跨类去重+白名单过滤会处理 |

**3. 和遮挡的关系**

遮挡是间接原因。直接原因是SAM3在遮挡发生时跟踪断裂，导致同一物体被拆分为多个实例。去重机制（3D点云重叠率）可以部分修复，但对于移动过的物体，新旧位置重叠率不够高时无法合并。

### Q12: SAM floor分割报错 "boolean index did not match" 是什么问题？

**根因**：SAM对原图(如1080×W)分割的mask尺寸与VGGT输出的pointmap尺寸(如518×W_vggt)不同。`get_plane_info(pointmap, mask)` 中 `pointmap[mask]` 直接用大mask索引小pointmap导致维度不匹配。

**修复**：新增 `_resize_mask_to_pointmap(mask, pointmap)` 辅助函数，在应用mask前将其resize到pointmap尺寸（PIL NEAREST插值保持bool语义）。修复了两处：
1. `get_plane_info()` 调用前
2. 点云补充检测排除floor区域时

### Q13: 点云补充检测提取点数过少（如25个点）是什么原因？

**根因**：置信度阈值比较用 `>` 严格大于。当某帧的中位数=最小值=1.000时，`conf_frame > 1.000` 排除了所有等于1.000的点（超过一半有效点）。

**修复**：
1. `>` 改为 `>=`，保留等于阈值的点
2. 新增保底逻辑：如果过滤后点数<100，自动降低到25%分位数

修复后，同样的帧（518个有效点，中位数=1.000）会保留约259个点而非25个。

---

## 9. 命令行参数

### 9.1 mainv2.py 参数完整参考

```bash
python mainv2.py --input_video <视频路径> [选项]
python mainv2.py --input_images <图片目录> [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_video` | str | None | 输入视频路径 (与--input_images二选一，必须指定其一) |
| `--input_images` | str | None | 输入图片目录路径 (与--input_video二选一，必须指定其一) |
| `--output_path` | str | None | 输出目录 (默认自动生成: `./output_v2/{video_stem}_{vggt_model}`) |
| `--vlm_checkpoint` | str | None | VLM模型路径 (默认自动查找: Qwen3.5-9B → Qwen2.5-VL-3B) |
| `--max_frames` | int | 120 | VGGT最大帧数 (Stage2) |
| `--vggt_model` | str | "vggt" | 3D重建模型选择: `vggt` / `vggt_omega` / `vggt4d` |
| `--max_frames_stage1` | int | 12 | Stage1采样关键帧数 |
| `--enable_stage4` | flag | False | 启用Stage4视觉-空间对齐 |
| `--enable_stage5` | flag | False | 启用Stage5语义感知场景精修 |
| `--enable_vlm_dynamic` | flag | False | 启用VLM辅助动态检测 (加权投票) |
| `--stage4_iterations` | int | 8 | Stage4 ICP迭代次数 |
| `--stage4_temporal_radius` | int | 5 | Stage4时序邻域半径 |
| `--stage4_use_mast3r` | flag | False | Stage4使用MASt3R匹配 (需要GPU) |

**已移除的参数**: `--category_path` (main.py必须指定，mainv2通过Stage1自动发现)

### 9.2 main.py 参数参考 (对比)

```bash
python main.py --input_video <视频路径> --category_path <JSON路径> [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_video` | str | `'./assets/example/hallway.mp4'` | 输入视频或图片目录路径 |
| `--output_path` | str | `'./outputs/hallway'` | 输出目录 |
| `--category_path` | str | `'./assets/example/hallway.json'` | 类别和关系JSON文件 (必须) |
| `--max_frames` | int | 160 | 从视频中处理的最大帧数 |

### 9.3 run_post_pipeline.py 参数

```bash
python tools/run_post_pipeline.py <scene_dir> [选项]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `scene_dir` | pos | — | 场景输出目录 (如 `output_v2/hoi4d_vggt_omega`) |
| `--stage4` | flag | False | 执行Stage4视觉-空间对齐 |
| `--stage5` | flag | False | 执行Stage5 (含5.1关系细化 + 5.2 SP精修) |
| `--vlm_checkpoint` | str | None | VLM模型路径 (Stage5.1需要) |
| `--stage4_iterations` | int | 8 | Stage4 ICP迭代次数 |
| `--stage4_temporal_radius` | int | 5 | Stage4时序邻域半径 |
| `--stage4_use_mast3r` | flag | False | Stage4使用MASt3R匹配 |

### 9.4 完整调用示例

#### 基础用法 (等价于main.py)

```bash
# main.py 方式 (需手动指定category_path)
python main.py \
    --input_video ./video.mp4 \
    --category_path ./video.json \
    --output_path ./outputs/video

# mainv2 等价方式 (Stage1自动发现，无需category_path)
python mainv2.py \
    --input_video ./video.mp4
# 输出: ./output_v2/video_vggt/final_scene.glb
```

#### 使用不同3D重建模型

```bash
# VGGT (默认，patch_size=14, image_resolution=518)
python mainv2.py --input_video ./video.mp4 --vggt_model vggt

# VGGT-Omega (patch_size=16, image_resolution=512)
python mainv2.py --input_video ./video.mp4 --vggt_model vggt_omega

# VGGT4D (patch_size=14, image_resolution=518)
python mainv2.py --input_video ./video.mp4 --vggt_model vggt4d
```

#### 启用Stage4 (视觉-空间对齐)

```bash
# 基本Stage4
python mainv2.py --input_video ./video.mp4 --enable_stage4

# Stage4 + MASt3R匹配 (更精确，需要GPU)
python mainv2.py --input_video ./video.mp4 --enable_stage4 --stage4_use_mast3r

# Stage4 自定义参数
python mainv2.py --input_video ./video.mp4 \
    --enable_stage4 \
    --stage4_iterations 12 \
    --stage4_temporal_radius 3 \
    --stage4_use_mast3r
```

#### 启用Stage5 (语义精修)

```bash
# 基本Stage5 (含5.1关系细化 + 5.2 SP精修)
python mainv2.py --input_video ./video.mp4 --enable_stage5

# Stage5 + 自定义VLM模型
python mainv2.py --input_video ./video.mp4 \
    --enable_stage5 \
    --vlm_checkpoint /path/to/Qwen3.5-9B
```

#### 全流水线 (Stage1-5)

```bash
# 完整流水线: VGGT-Omega + Stage4 + Stage5
python mainv2.py --input_video ./video.mp4 \
    --vggt_model vggt_omega \
    --enable_stage4 --stage4_use_mast3r \
    --enable_stage5

# 完整流水线 + VLM动态检测
python mainv2.py --input_video ./video.mp4 \
    --vggt_model vggt_omega \
    --enable_stage4 --stage4_iterations 10 \
    --enable_stage5 \
    --enable_vlm_dynamic
```

#### 使用图片目录代替视频

```bash
python mainv2.py --input_images ./my_images/
```

#### 自定义输出路径

```bash
python mainv2.py --input_video ./video.mp4 --output_path ./my_output
```

#### 自定义Stage1帧数和VGGT帧数

```bash
python mainv2.py --input_video ./video.mp4 \
    --max_frames_stage1 15 \
    --max_frames 200
```

#### 后处理管线 (独立调用Stage4/5)

```bash
# 对已有输出目录执行Stage4
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega --stage4

# 对已有输出目录执行Stage5
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega --stage5

# 同时执行Stage4+5
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega --stage4 --stage5

# 后处理 + 自定义参数
python tools/run_post_pipeline.py ./output_v2/video_vggt_omega \
    --stage4 --stage4_iterations 12 --stage4_use_mast3r \
    --stage5 --vlm_checkpoint /path/to/Qwen3.5-9B
```

### 9.5 参数组合速查

| 场景 | 命令 |
|------|------|
| 最快运行 (等价main.py) | `--input_video video.mp4` |
| 更好的3D重建 | `--input_video video.mp4 --vggt_model vggt_omega` |
| 精确位姿对齐 | `--input_video video.mp4 --enable_stage4 --stage4_use_mast3r` |
| 物体间关系精修 | `--input_video video.mp4 --enable_stage5` |
| 最佳质量 | `--input_video video.mp4 --vggt_model vggt_omega --enable_stage4 --stage4_use_mast3r --enable_stage5` |
| 动态场景 | `--input_video video.mp4 --enable_vlm_dynamic` |
| 图片输入 | `--input_images ./images/` |

---

## 10. 修改记录

| 日期 | 修改 | 文件 |
|------|------|------|
| 2026-05-29 | mainv2.py 初版 | mainv2.py |
| 2026-05-30 | 修复坐标系变换、GLB覆盖逻辑 | run_post_pipeline.py, mainv2.py |
| 2026-05-31 | 鲁棒运动检测+日志标记 | geometry_utils.py, mainv2.py, main.py |
| 2026-05-31 | protected_categories防止跨类合并 | sg_deduplication.py, mainv2.py |
| 2026-06-01 | SP精修复用refine_supported_by_floor_object | refine_inter_object_placement.py |
| 2026-06-02 | 修复get_optimal_view_frame_id返回值处理 | mainv2.py |
| 2026-06-02 | 新增全局位移检测+VLM加权投票 | geometry_utils.py, mainv2.py |
| 2026-06-02 | 统一_align_upright旋转对齐 | refine_inter_object_placement.py |
| 2026-06-02 | GLB输出简化: base(固定起点) + final(最新) | mainv2.py, run_post_pipeline.py |
| 2026-06-02 | pkl格式扩展为字典 | mainv2.py, run_post_pipeline.py |
| 2026-06-03 | 补充遗漏差异: --category_path移除、--input_images互斥、输出路径自动生成、VLM自动查找、Stage5逻辑重复等 | mainv2_technical_doc.md |
| 2026-06-03 | 补充完整参数调用示例: 各模型/Stage组合/后处理管线/参数组合速查 | mainv2_technical_doc.md |
| 2026-06-18 | 默认帧数更新: max_frames 160→120, max_frames_stage1 10→12 | mainv2.py |
| 2026-06-18 | 修复SAM floor分割mask维度不匹配: 新增_resize_mask_to_pointmap() | generate_scene_json_stage1.py |
| 2026-06-18 | 修复点云补充检测阈值: >改>=, 点数<100自动降低阈值 | generate_scene_json_stage1.py |
| 2026-06-18 | 更新技术文档: 同步默认值、更新protected_categories描述、新增Q8-Q13 | mainv2_technical_doc.md |
