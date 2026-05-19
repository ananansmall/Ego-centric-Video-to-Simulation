# ReplicateAnyScene Stage 2 管线汇总与使用示例

## 环境要求

- Python 3.11+
- PyTorch 2.0+
- 依赖项安装: `pip install -r requirements.txt`
- **关键模型要求**:
  - VLM模型: Qwen3.5-9B (默认路径 `/mnt/data/lza/models/Qwen3.5-9B`)
  - VGGT模型: 本地已有权重
  - SAM3模型: 可选，用于floor/wall分割 + floor重叠检测

---

## 命令行参数

| 参数                          | 必需 | 默认值                   | 说明                                                                    |
| ----------------------------- | ---- | ------------------------ | ----------------------------------------------------------------------- |
| `--input_video`             | ✅   | -                        | 输入视频路径                                                            |
| `--output_json`             | ❌   | 自动生成                 | 输出JSON路径，默认 `./assets/json_configs/scene_<视频名>_stage2.json` |
| `--vlm_checkpoint`          | ❌   | 自动查找                 | VLM模型路径，默认 `/mnt/data/lza/models/Qwen3.5-9B`                   |
| `--max_frames`              | ❌   | 10                       | VGGT采样最大关键帧数                                                    |
| `--temp_dir`                | ❌   | `./temp_frames_stage2` | 临时帧存储目录                                                          |
| `--centroid_dist_thre`      | ❌   | 0.15                     | 3D去重质心距离阈值(米)                                                  |
| `--use_sam`                 | ❌   | `auto`                 | SAM3分割:`auto`(自动判断)/`yes`(强制)/`no`(禁用)                  |
| `--no_supplementary_detect` | ❌   | `False`                | 禁用点云补充检测（默认启用）                                            |

---

## 完整运行示例: `beizi.mp4`

### 基本用法（使用默认参数）

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/generate_scene_json_stage2.py \
  --input_video assets/example/beizi.mp4
```

### 完整参数示例

```bash
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/generate_scene_json_stage2.py \
  --input_video assets/example/beizi.mp4 \
  --output_json my_result.json \
  --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B \
  --max_frames 10 \
  --use_sam auto
```

---

## 整体管线总览

```
视频输入
  │
  ▼
┌─────────────────────────────────┐
│ Step 0: VGGT 3D场景重建          │
│   输入: 完整视频                  │
│   输出: 3D点云 + 相机外参/内参     │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Step 1: SimRecon 关键帧采样       │
│   贪心最大覆盖算法选择关键帧        │
│   输出: 关键帧索引列表             │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Step 2: 提取关键帧图像            │
│   ffmpeg按索引提取帧图片           │
│   输出: [(vid_idx, frame_path)]  │
└──────────────┬──────────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────┐  ┌──────────────────────┐
│ 第一部分:     │  │ 辅助: SAM分割         │
│ 物体发现+去重 │  │ Step 6                │
│ Step 3-5     │  │ floor/wall检测        │
│              │  │ + 关键帧floor mask     │
│              │  │ (选择最像地板的mask)    │
└──────┬───────┘  └────────┬─────────────┘
       │                   │
       ▼                   ▼
┌─────────────────────────────────────────┐
│ Step 5.5: 点云补充检测                    │
│   分析VGGT点云中未被VLM覆盖的3D聚类       │
│   → DBSCAN聚类 → 过滤大小/形状            │
│   → 裁剪2D区域 → VLM识别                  │
│   → 用点云大小/位置决定是否采纳            │
│   (利用SAM floor mask排除floor区域)       │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ 第二部分: 关系判断 (Step 7)              │
│                                         │
│ 7.1 SAM floor预判断 (bbox与floor mask重叠)│
│     → 重叠>=30% → 直接判定"supported by  │
│       floor"，不经过VLM                  │
│                                         │
│ 7.2 Per-frame可见性映射                   │
│     → 物体在哪帧出现 → 只在该帧判断       │
│                                         │
│ 7.3 VLM多帧推理 (仅判断可见物体)          │
│     → 每帧独立prompt + 后处理纠错         │
│                                         │
│ 7.4 汇总: SAM预判断 + VLM投票            │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│ Step 8: 输出场景JSON             │
│   过滤floor/wall，保存最终结果     │
└─────────────────────────────────┘
```

---

## 处理流程详解

### Step 0: VGGT 3D场景重建

- 输入: 完整视频（最多160帧）
- 输出: 3D点云 `world_points`、置信度 `world_points_conf`、相机外参 `extrinsics`、内参 `intrinsic`
- 目的: 构建场景的3D几何结构，为后续射线投射和采样提供基础

### Step 1: SimRecon 关键帧采样

- **输入**: 相机外参 + 3D点云 + 置信度
- **输出**: 覆盖整个场景的关键帧索引列表
- **方法**: 贪心最大覆盖算法（`maximum_coverage_sampling`）

#### 1.1 体素化3D空间

将连续的3D点云离散化为体素网格，便于计算覆盖率：

```python
# 计算场景边界盒
x_min, y_min, z_min = valid_points.min(axis=0)
x_max, y_max, z_max = valid_points.max(axis=0)
scene_extent = max(x_max - x_min, y_max - y_min, z_max - z_min)

# 体素大小按场景最小维度/20计算
voxel_size = max(scene_extent / 20.0, 0.01)

# 将3D点映射到体素坐标
voxel_coords = np.floor((points - offset) / voxel_size).astype(int)
```

**关键设计**:
- **自适应体素大小**: 根据场景尺度自动调整，避免过大（丢失细节）或过小（计算量大）
- **坐标偏移量**: 使用 `(x_min, y_min, z_min)` 作为偏移，提高体素离散化精度
- **全局置信度阈值**: 筛选置信度 >= 50%分位数 且 > 0.1 的点，排除低质量点云

#### 1.2 贪心最大覆盖算法

**核心思想**: 每轮选择能覆盖最多**未覆盖体素**的帧，直到达到目标帧数。

**算法流程**:

```
初始化:
  selected = []          # 已选帧列表
  covered = {}           # 已覆盖的体素集合
  remaining = {0,1,...,T-1}  # 待选帧集合

循环 K 次 (K = max_frames):
  1. 对每个候选帧 f ∈ remaining:
     gain(f) = |voxels(f) - covered|  # 该帧能新增的体素数
  
  2. 选择增益最大的帧:
     best_frame = argmax(gain(f))
  
  3. 更新状态:
     selected.append(best_frame)
     covered.update(voxels(best_frame))
     remaining.remove(best_frame)
  
  4. 终止条件:
     - 达到K帧
     - 无新增覆盖 (gain=0)
     - 无剩余帧

返回 sorted(selected)
```

**示例执行过程**:

```
🎯 第1轮: 选择帧#15, 新增3250个体素, 累计覆盖3250个
🎯 第2轮: 选择帧#42, 新增2180个体素, 累计覆盖5430个
🎯 第3轮: 选择帧#8,  新增1520个体素, 累计覆盖6950个
...
📊 最终覆盖率: 8750/9500 体素 (92.1%)
```

#### 1.3 算法优势

| 特性 | 说明 |
|------|------|
| **最大化空间覆盖** | 确保选中的帧能从不同角度覆盖场景的3D空间 |
| **视角多样性** | 贪心策略自然倾向于选择视角差异大的帧 |
| **计算高效** | O(K×T) 复杂度，K为目标帧数，T为总帧数 |
| **无需预定义规则** | 不依赖启发式规则（如每隔N帧采样），自适应场景结构 |
| **可解释性强** | 每轮选择的增益清晰可见，便于调试和优化 |

#### 1.4 与SimRecon的关系

本实现参考 `SimRecon/coverage_sampling.py`，采用相同的:
- 体素化策略（场景维度/20）
- 置信度过滤（50%分位数 + >0.1）
- 坐标偏移量处理
- 贪心最大覆盖算法

**目标**: 在10帧内实现90%+的体素覆盖率，为后续物体检测提供充分的视角覆盖。

### Step 2: 提取关键帧

- 输入: 原始视频 + 关键帧索引
- 输出: 临时目录中的帧图片 `[(vid_idx, frame_path)]`
- 方法: ffmpeg 按时间戳精确提取

---

### 第一部分: 物体发现与去重 (Step 3-5)

#### Step 3: VLM 第一次调用 — 物体检测

- 输入: 关键帧图片
- 输出: 每帧的物体名称列表 + 2D中心坐标 + **bbox边界框**
- 提示词设计:
  - 只检测可见物体，输出 `{"objects": [{"name": "cup"}, ...]}`
  - 忽略 hands, body parts, walls, floors, ceilings
  - 每种物体每帧只列一次
- 兼容性处理:
  - VLM可能输出 `label` 而非 `name` → 自动转换
  - VLM可能输出 `bbox_2d` 而非 `center_x/center_y` → 从bbox计算中心点
  - **bbox信息保留**，用于后续SAM floor重叠检测
  - 无位置信息时默认使用图像中心
- **重要**: `all_detections` 的结果会传递到 Step 7，用于 per-frame 可见性过滤

#### Step 4: 射线投射 — 3D位置估计

- 输入: 2D检测坐标 + VGGT结果（点云、外参、内参）
- 输出: 每个物体实例的3D质心 `centroid`
- 方法: 射线投射 (`pixel_to_3d_position`)
  1. 从相机中心沿像素方向发射射线
  2. 计算射线到点云中所有点的距离
  3. 取距离最近的前K个点（`RAY_CAST_TOP_K=5`）
  4. 中值滤波得到3D位置

#### Step 5: 去重

- 输入: 所有帧的物体实例
- 输出: 唯一物体列表
- 方法: 名称匹配去重（同名物体直接合并为一个）
  - 先通过 `merge_synonyms` 合并同义词
  - 同名物体的3D质心取中值
  - 同义词映射示例:
    - `ground` / `flooring` / `carpet` / `rug` → `floor`
    - `walling` / `walls` → `wall`
    - `mug` / `glass` / `tumbler` → `cup`
    - `sofa` / `settee` → `couch`
    - `desk` / `dining table` → `table`
    - `television` / `monitor` / `display` → `tv`

---

### Step 5.5: 点云补充检测 — 发现VLM遗漏的远端大物体

**问题**: VLM对近处物体检测效果好，但可能遗漏远端的大物体（如远处的柜子、书架等）。

**方案**: 保留原有VLM检测，新增基于点云的补充检测，用点云大小/位置决定是否采纳。

#### 5.5.1 提取高置信度3D点

- 从VGGT点云中采样5帧，提取置信度 > 1.5 的3D点
- 体素降采样（5cm体素），减少计算量

#### 5.5.2 排除已知物体和floor/wall区域

- 排除距离已知物体质心 < 0.3m 的点（已被VLM检测到）
- 利用SAM关键帧floor mask，排除属于floor区域的3D点

#### 5.5.3 DBSCAN聚类

- 对剩余点做DBSCAN聚类（eps=0.15m, min_samples=30）
- 每个聚类代表一个潜在的未检测物体

#### 5.5.4 过滤聚类（用点云大小/位置决定）

| 过滤规则   | 条件                         | 原因                |
| ---------- | ---------------------------- | ------------------- |
| 过小       | 体积 < 0.005m³              | 噪声或小碎片        |
| 过大       | 体积 > 8m³                  | 可能是墙/地板       |
| 过扁       | 最薄维度 < 3cm 且最长 > 1.5m | 平面结构（墙/地板） |
| 长宽比过大 | max_dim/min_dim > 30         | 线状结构（管道等）  |
| 点数过少   | < 50个点                     | 稀疏噪声            |

#### 5.5.5 VLM识别

- 对每个候选聚类：
  1. 将3D质心投影到最佳关键帧的2D像素位置
  2. 裁剪 300×300 像素区域
  3. 用VLM识别: "What is the large object in the center?"
  4. 过滤: "none" → 跳过，FILTER_CATEGORIES → 跳过，已存在同名 → 跳过
- 最多识别5个候选聚类（按体积降序）

#### 5.5.6 结果合并

- 新检测到的物体添加到 `unique_objects`
- 后续进入 Step 7 关系判断流程

**示例输出**:

```
🔎 Step 5.5: 点云补充检测 — 发现遗漏的远端大物体
   已知物体: 4 个
   📊 降采样后点云: 12580 个点
   排除已知物体附近点后: 8320 个点
   排除floor区域点后: 5100 个点
   🔮 DBSCAN聚类: 12 个聚类
   ✅ 过滤后候选聚类: 3 个
      候选1: 质心=[1.2, -0.8, 0.5], 尺寸=[0.6, 0.4, 1.2], 体积=0.288, 点数=380
      候选2: 质心=[-1.5, 2.1, 0.3], 尺寸=[0.8, 0.3, 0.9], 体积=0.216, 点数=290
   [1/3] 识别候选聚类 (质心=[1.2, -0.8, 0.5])...
      ✅ 新发现: bookshelf (体积=0.288, 点数=380)
   🎉 补充检测发现 1 个新物体:
      - bookshelf (质心=[1.2, -0.8, 0.5])
```

---

### 辅助: SAM3 语义分割 (Step 6)

#### Step 6: SAM3 分割 floor 和 wall

- 输入: 关键帧图片 + VGGT结果
- 输出: `sam_results = {'has_floor', 'has_wall', 'floor_masks', 'wall_masks', 'keyframe_floor_masks'}`
- **新增**: `keyframe_floor_masks: {frame_idx: numpy(H,W)}` — 每个关键帧的floor mask
- 目的:
  1. 为关系判断提供 floor/wall 可见性信息（`has_floor`/`has_wall`）
  2. **为SAM floor重叠检测提供逐帧floor mask**（`keyframe_floor_masks`）
  3. **为点云补充检测提供floor区域排除信息**
- 实现细节:
  - 先对VGGT颜色帧做SAM分割（获取 `has_floor`/`has_wall`）
  - 再对每个关键帧单独做SAM floor分割（获取 `keyframe_floor_masks`）
  - **Floor mask选择逻辑**（与main.py `align_to_room_coordinate_system`一致）:
    1. 只有1个候选 → 直接使用
    2. 有多个候选 → 用 `get_plane_info` 计算3D平面信息:
       - 过滤: `mean_distance > 0.02` → 丢弃（不像平面）
       - 过滤: 法向量偏离平均法向量 > 30° → 丢弃（错误分割，如桌面）
       - 选择: **面积最大的** → 最像地板
    3. 无world_points时 → 降级合并所有mask
  - SAM模型在两步完成后才卸载，避免重复加载
- 降级策略:
  - SAM3不可用时 → 回退到默认值（floor和wall均视为可见，无keyframe floor mask）
  - `--use_sam no` → 跳过SAM分割

---

### 第二部分: 关系判断 (Step 7)

#### Step 7: 关系判断 — SAM预判断 + Per-frame VLM推理 + 多帧投票

这是整个管线的核心判断步骤，采用 **"SAM floor预判断 → Per-frame可见性过滤 → VLM多帧推理 → 物理常识后处理纠错 → 多帧投票"** 的架构。

##### 7.1 预处理: 过滤场景结构物体

在构建VLM prompt之前，先将 floor/wall/ceiling 等场景结构物体从物体列表中移除：

```
原始物体列表: [floor, chair, table, cup, wall, clock]
                    ↓ FILTER_CATEGORIES 过滤
待判断物体列表: [chair, table, cup, clock]
```

- **floor/wall 不参与VLM关系判断**，它们只作为场景上下文信息（`has_floor`/`has_wall`）传递给prompt
- 这确保了 floor 只有一个（通过 Step 5 的同义词合并已保证），且不浪费VLM判断能力
- `FILTER_CATEGORIES = {floor, wall, ground, ceiling, floor_area, wall_section, flooring, wall_surface, room, space}`

##### 7.2 SAM floor 预判断（几何方法，非VLM）

使用SAM分割得到的floor mask，通过**几何重叠检测**判断物体是否在地板上：

**检测逻辑** (`_check_object_on_floor`):

1. 取物体bbox的**底部30%区域**（靠近地面的部分）
2. 计算该底部区域与SAM floor mask的重叠像素比例
3. 若重叠比例 >= `FLOOR_OVERLAP_THRESHOLD`（默认30%），判定物体在地板上

```
物体bbox:  ┌──────────┐
           │          │
           │  上部70%  │  ← 不检查
           │──────────│
           │  底部30%  │  ← 检查与floor mask重叠
           └──────────┘
               ↕
         SAM floor mask (绿色区域)
```

**预判断流程** (`_sam_prejudge_floor`):

1. 遍历每个关键帧的检测结果
2. 对每个有bbox的物体，检查其底部与该帧floor mask的重叠
3. 若**任一帧**中重叠 >= 30%，标记该物体为 "supported by floor"
4. SAM预判断的物体**直接确定关系，不再经过VLM**

**示例输出**:

```
🏠 7.1 SAM floor预判断...
   chair: ✅ 在地板上 (最大重叠率: 85.2%, 检查帧数: 3)
   table: ✅ 在地板上 (最大重叠率: 72.1%, 检查帧数: 3)
   cup: ❌ 不在地板上 (最大重叠率: 5.3%, 检查帧数: 2)
   clock: ❌ 不在地板上 (最大重叠率: 0.0%, 检查帧数: 1)
🏠 SAM判定在地板上的物体: {'chair', 'table'}
```

**优势**:

- **几何方法比VLM更可靠**: 直接检测物体底部是否接触地板，而非依赖VLM的语义理解
- **减少VLM工作量**: SAM预判断的物体不再需要VLM推理
- **可解释性**: 重叠比例是可量化的，便于调试

##### 7.3 Per-frame 可见性映射

利用 Step 3 的检测结果，构建每帧的物体可见性映射：

```python
frame_visibility = {
    frame_idx_0: {'chair', 'table', 'cup'},
    frame_idx_1: {'chair', 'clock'},
    frame_idx_2: {'table', 'cup', 'clock'},
    ...
}
```

**关键规则**: 物体在某帧没出现 → 该帧不对该物体做VLM判断 → 不产生默认投票

**旧逻辑问题**:

```
旧: 所有帧都判断所有物体 → 物体在某帧不可见 → VLM可能乱猜或跳过 → 默认"supported by floor"
新: 只在该物体可见的帧做判断 → 不可见帧直接跳过 → 不会产生错误投票
```

##### 7.4 VLM 多帧推理（仅判断可见物体）

对每个关键帧，只对该帧**可见且未被SAM预判断**的物体调用VLM：

```python
for vid_idx, frame_path in frame_paths_with_indices:
    visible_in_frame = frame_visibility.get(vid_idx, set())
    frame_objects = [n for n in objects_to_query if n in visible_in_frame]

    if not frame_objects:
        continue  # 该帧无待判断物体，跳过

    frame_prompt = _build_relationship_prompt(frame_objects, has_floor, has_wall)
    output_text = _vlm_inference(image, model, processor, frame_prompt)
    ...
```

- **每帧使用独立的prompt**，只包含该帧可见的物体
- VLM从不同视角观察不同的物体子集，给出关系判断
- 每帧结果先经过后处理纠错，再计入投票

**4种空间关系**:

1. `supported by floor`: 直接放在地面/地板上（如桌子、椅子、柜子）
2. `supported by other objects`: 放在其他物体上（如桌上的杯子、椅上的枕头）
3. `attached to wall`: 挂在墙上（如画框、时钟、窗帘）
4. `embedded in wall`: 嵌入墙内（如门、窗、插座）

##### 7.5 物理常识后处理纠错

`_post_process_relationships(relationships, object_names)` 对每帧的VLM输出进行纠错：

| 规则               | 物体类别                                                                           | VLM错误判断                         | 纠正为             | 原因                    |
| ------------------ | ---------------------------------------------------------------------------------- | ----------------------------------- | ------------------ | ----------------------- |
| 家具必须在地面     | cabinet, table, chair, sofa, bed, desk, shelf, refrigerator, plant, box, carpet... | attached to wall / embedded in wall | supported by floor | 大型家具不可能挂在墙上  |
| 墙面装饰必须挂墙   | picture, painting, mirror, clock, poster, curtain...                               | supported by floor                  | attached to wall   | 画/镜子等不可能放在地上 |
| 墙面嵌入物必须嵌墙 | window, door, outlet, vent, socket...                                              | supported by floor                  | embedded in wall   | 门/窗等不可能放在地上   |
| 无效关系兜底       | 任何物体                                                                           | 非法关系字符串                      | supported by floor | 安全默认值              |
| 缺失物体兜底       | 该帧可见但VLM未输出的物体                                                          | 无                                  | supported by floor | 确保该帧完整性          |

##### 7.6 汇总结果: SAM预判断 + VLM投票

最终结果由两部分组成：

```
对每个物体:
  ├─ SAM预判断为floor → 直接 "supported by floor"（标记🏠）
  ├─ 有VLM投票 → 取多数票（标记📋）
  └─ 无SAM判断也无VLM投票 → 兜底 "supported by floor"（标记⚠️）
```

**示例输出**:

```
📊 7.4 汇总结果:
🏠 chair: supported by floor (SAM预判断, 最大重叠率: 85.2%)
🏠 table: supported by floor (SAM预判断, 最大重叠率: 72.1%)
📋 cup: supported by other objects (VLM投票: {'supported by other objects': 2, 'supported by floor': 1})
📋 clock: attached to wall (VLM投票: {'attached to wall': 2})
```

##### 7.7 最终输出过滤

在生成最终结果时，再次过滤掉 floor/wall/ceiling 等场景结构物体，确保它们不会出现在输出JSON中。

---

### Step 8: 保存场景JSON

- 输入: 最终关系判断结果（已过滤floor/wall）
- 输出: JSON文件
- 格式: `{"物体名": "空间关系", ...}`，按名称排序

---

## 输出格式示例

### `beizi_scene.json` (参考)

```json
{
  "chair": "supported by floor",
  "clock": "attached to wall",
  "cup": "supported by other objects",
  "door": "embedded in wall",
  "table": "supported by floor"
}
```

### JSON结构说明

- **键**: 物体类别名（标准化后，不含floor/wall/ceiling等场景结构）
- **值**: 4种空间关系之一
- **判断来源**:
  - SAM预判断: 物体bbox底部与floor mask重叠>=30%
  - VLM投票: 多帧VLM推理 + 后处理纠错 + 多数票
  - 兜底默认: 无SAM判断也无VLM投票时

---

## 关键设计决策

### 为什么用 SAM floor mask 做预判断，而不是纯VLM？

1. **几何方法更可靠**: VLM可能因视角问题误判（如俯视时把桌子判为"attached to wall"），但SAM的floor mask是几何分割，不受视角影响
2. **重叠检测可量化**: 30%的重叠阈值是可调节的，便于针对不同场景优化
3. **减少VLM工作量**: SAM预判断的物体不需要VLM推理，节省计算资源
4. **SAM和VLM互补**: SAM擅长判断"是否在地板上"（几何），VLM擅长判断"是否在墙上/其他物体上"（语义）

### 为什么需要 Per-frame 可见性过滤？

1. **物体不是每帧都出现**: 一个物体可能只在3个关键帧中的2个出现
2. **不可见帧不应产生投票**: 如果物体在某帧不可见，VLM对该物体的判断是不可靠的
3. **避免默认floor污染**: 旧逻辑中，不可见物体会被默认为"supported by floor"，导致投票偏差
4. **Step 3已提供可见性信息**: 检测结果天然包含per-frame物体列表，直接利用即可

### 为什么 floor/wall 不参与 VLM 关系判断？

1. **floor 是场景结构，不是物体**: floor 的"关系"是自指的（floor supported by floor？），没有意义
2. **同义词合并保证唯一性**: 通过 `SYNONYM_MAP`，ground/flooring/carpet/rug 全部映射为 "floor"，Step 5 去重后只有一个 "floor"
3. **floor/wall 作为场景上下文**: 通过 `has_floor`/`has_wall` 标志传递给VLM，帮助VLM判断其他物体的关系
4. **避免干扰VLM判断**: 如果把 floor 放在物体列表中，VLM可能浪费判断能力在无意义的 floor 关系上

### 为什么使用多帧投票？

- 单帧VLM判断可能因视角问题出错（如俯视时误判家具"attached to wall"）
- 多帧从不同角度观察，投票可以消除偶然错误
- 后处理纠错 + 多帧投票双重保障

### 为什么需要物理常识后处理？

- VLM有时会犯违反物理常识的错误（如判断柜子"attached to wall"）
- 后处理规则基于物体类别的物理属性进行确定性纠错
- 纠错在投票之前执行，确保每帧的投票都是物理合理的
