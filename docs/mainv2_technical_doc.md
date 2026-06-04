# mainv2.py 完整技术文档

> 整合时间: 2026-06-02
> 涵盖: mainv2 vs main 差异、新增模块逻辑、常见问题与解答

---

## 目录

1. [mainv2 vs main: 架构差异总览](#1-架构差异)
2. [mainv2 vs main: 逐阶段对比](#2-逐阶段对比)
3. [新增模块详解](#3-新增模块详解)
4. [GLB文件体系](#4-glb文件体系)
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
  Stage1 → Stage2 → Stage3 → 基础精修 → Stage4 → Stage5 → final_scene.glb
```

---

## 2. 逐阶段对比

### Stage 1: 物体发现

| | main.py | mainv2.py |
|---|---------|-----------|
| 方式 | 手动 `--category_path` | 自动 subprocess 调用 `generate_scene_json_stage1` |
| VLM | 不需要 | Qwen3.5-9B / Qwen2.5-VL-3B (自动回退) |
| 关键帧 | 无 | 贪心采样10帧 + `keyframes_metadata.json` |
| 关系格式 | `supported_by_floor` (下划线) | 兼容下划线和空格 |
| 代码位置 | `with open(args.category_path, 'r') as f:` | `run_stage1()` 函数，subprocess调用 |
| 输入参数 | `--category_path` (必须) | `--input_video` (必须), `--max_frames_stage1` (默认10) |
| 输出文件 | 无 | `scene_{id}_stage1.json` + `keyframes/` |

### Stage 2: 3D重建 + 去重

| | main.py | mainv2.py |
|---|---------|-----------|
| VGGT模型 | 仅 `vggt_predict` | 按 `--vggt_model` 选择 (vggt/vggt_omega/vggt4d) |
| 模型加载 | `load_vggt_model()` | `load_vggt_model()` / `load_vggt_omega_model()` / `load_vggt4d_model()` |
| 帧加载 | `load_video_frames(video, max_frames)` | vggt/vggt4d: `load_video_frames`; vggt_omega: `load_vggt_omega_frames` |
| 图像分辨率 | 518 (patch_size=14) | vggt/vggt4d: 518; vggt_omega: 512 (patch_size=16) |
| 预测函数 | `vggt_predict(frames, model)` | `vggt_predict` / `vggt_omega_predict` / `vggt4d_predict` |
| protected_categories | **无** | **新增**: 传入Stage1 JSON类别名，防止跨类合并 |
| 跨类去重调用 | `cross_category_deduplicate(all_masks, pts, conf)` | `cross_category_deduplicate(all_masks, pts, conf, protected_categories=...)` |
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
| Stage5逻辑 | 注释占位 | `run_stage5()` 函数已定义，但 `main()` 中未调用，Stage5逻辑在 `main()` 内L843-885重新实现 |
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

**关键差异: Stage5 逻辑重复**

`run_stage5()` 函数 (L620-676) 已定义，但 `main()` 中未调用。Stage5 的实际逻辑在 `main()` 内 L843-885 重新实现。`run_stage5()` 可能供 `run_post_pipeline.py` 使用。

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

### 文件说明

| 文件 | 生成时机 | 说明 |
|------|---------|------|
| `final_scene_base.glb` | 基础精修后 | **固定起点**，供 `run_post_pipeline.py` 使用，不再更改 |
| `final_scene.glb` | 管线结束时 | **始终为最新结果**，不启用stage4/5时等同base |
| `all_instances.pkl` | 基础精修后 | 实例数据字典，供后处理管线独立调用 |

### pkl数据结构

```python
{
    'all_instances': all_instances,           # {category: [instance_info, ...]}
    'all_optimal_frame_ids': all_optimal_frame_ids,  # {category: [frame_id, ...]}
    'categories_and_relations': categories_and_relations,  # {category: relation_str}
    'walls_info': walls_info,                 # 墙壁几何信息
}
```

### 为什么需要 final_scene_base.glb？

`run_post_pipeline.py` 需要一个**固定起点**来执行stage4/5。如果直接用`final_scene.glb`，重复运行时可能已经包含了之前stage4/5的修改，导致叠加错误。`final_scene_base.glb`始终是基础精修的结果，不受stage4/5影响。

### GLB坐标系

- **内部处理**: z-up (与`sp_refinement.py`、`geometry_utils.py`一致)
- **GLB输出**: y-up (trimesh标准)
- **变换**: 保存时执行 z-up → y-up 变换

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
  ├── final_scene_base.glb → 基础精修GLB (固定起点)
  ├── all_instances.pkl   → 实例数据
  └── scene_*_stage1.json → Stage1 JSON
```

**GLB发现优先级**: `final_scene_base.glb` > `final_scene.glb`

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
final_scene_base.glb (固定起点)
  → Stage4 → final_scene.glb (覆盖)
  → Stage5 → final_scene.glb (覆盖)
```

---

## 8. 常见问题

### Q1: 为什么有3个GLB文件？

| 文件 | 说明 |
|------|------|
| `final_scene_base.glb` | 基础精修后保存，**固定起点**，不再更改 |
| `final_scene.glb` | 最终结果，始终为最新 |

旧版代码曾保存 `final_scene_stage4.glb` 等中间文件，已移除。现在只有2个GLB。

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
| `--max_frames` | int | 160 | VGGT最大帧数 (Stage2) |
| `--vggt_model` | str | "vggt" | 3D重建模型选择: `vggt` / `vggt_omega` / `vggt4d` |
| `--max_frames_stage1` | int | 10 | Stage1采样关键帧数 |
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
