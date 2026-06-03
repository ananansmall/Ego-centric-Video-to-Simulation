# ReplicateAnyScene Stage 2 管线文档

> 代码文件: `tools/generate_scene_json_stage1.py`

## 环境要求

- Python 3.11+, PyTorch 2.0+
- VLM模型: Qwen3.5-9B (默认 `/mnt/data/lza/models/Qwen3.5-9B`)
- VGGT模型: 本地已有权重
- SAM3模型: 可选，用于floor/wall分割 + floor重叠检测
- sklearn: 可选，用于DBSCAN聚类（无sklearn时降级为体素聚类）

---

## 命令行参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--input_video` | ✅ | - | 输入视频路径 |
| `--output_json` | ❌ | 自动生成 | 输出JSON路径，默认 `./assets/json_configs/scene_<视频名>_stage2.json` |
| `--output_dir` | ❌ | 与output_json同目录 | 输出目录 |
| `--vlm_checkpoint` | ❌ | 自动查找 | VLM模型路径 |
| `--max_frames` | ❌ | 10 | VGGT采样最大关键帧数 |
| `--temp_dir` | ❌ | `./temp_frames_stage2` | 临时帧存储目录 |
| `--centroid_dist_thre` | ❌ | 0.15 | 3D去重质心距离阈值(米) |
| `--use_sam` | ❌ | `auto` | SAM3分割: `auto`/`yes`/`no` |
| `--no_supplementary_detect` | ❌ | False | 禁用点云补充检测（默认启用） |

---

## 运行示例

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene

# 基本用法
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/generate_scene_json_stage1.py \
  --input_video assets/example/beizi.mp4

# 禁用点云补充检测
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/generate_scene_json_stage1.py \
  --input_video assets/example/beizi.mp4 --no_supplementary_detect
```

---

## 整体管线总览

```
视频输入
  │
  ▼
┌──────────────────────────────┐
│ Step 0: VGGT 3D场景重建       │
│   输出: 3D点云 + 相机参数      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Step 1: SimRecon关键帧采样    │
│   贪心最大覆盖 → 关键帧索引    │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Step 2: ffmpeg提取关键帧图像   │
│   输出: [(vid_idx, path)]     │
└──────────────┬───────────────┘
               │
      ┌────────┴────────┐
      ▼                 ▼
┌──────────────┐  ┌───────────────────────┐
│ Step 3: VLM  │  │ Step 6: SAM分割       │
│ 物体检测      │  │ floor/wall检测        │
│ (名称+bbox)  │  │ + 关键帧floor mask    │
│      ▼        │  │ (选择最像地板的mask)   │
│ Step 4: 射线  │  └───────────┬───────────┘
│ 投射→3D位置   │              │
│      ▼        │              │
│ Step 5: 语义  │
│ 去重(SYN+CLIP)│              │
└──────┬────────┘              │
       │                       │
       ▼                       ▼
┌──────────────────────────────────────────┐
│ Step 5.5: 点云补充检测                     │
│   VGGT点云 → 排除已知物体+floor → DBSCAN  │
│   → 过滤大小/形状 → 裁剪2D → VLM识别       │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ Step 7: 关系判断                           │
│   7.1 SAM floor预判断 (bbox底部与floor重叠) │
│   7.2 Per-frame可见性映射                   │
│   7.3 VLM多帧推理 (仅可见物体)              │
│   7.4 物理常识后处理纠错                    │
│   7.5 汇总: SAM预判断 + VLM多帧投票         │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────┐
│ Step 8: 保存场景JSON          │
│ Step 9: 保存关键帧到          │
│   assets/key_frames/<视频名>/ │
└──────────────────────────────┘
```

**执行顺序**: Step 0→1→2→3→4→5→**6**→**5.5**→7→8→9

> 注意: Step 6 (SAM) 在 Step 5.5 之前执行，因为 Step 5.5 需要 SAM 的 floor mask 来排除 floor 区域。

---

## 处理流程详解

### Step 0: VGGT 3D场景重建

- 输入: 完整视频（最多160帧）
- 输出: `world_points`(3D点云), `world_points_conf`(置信度), `extrinsics`(外参), `intrinsic`(内参), `colors`(帧颜色)
- VGGT模型用完后立即卸载释放显存

### Step 1: SimRecon 关键帧采样

- 输入: 相机外参 + 3D点云 + 置信度
- 输出: 关键帧索引列表
- 方法: 贪心最大覆盖算法
  1. 体素化3D空间（体素大小 = 场景最大维度 / 20）
  2. 置信度过滤: >= 50%分位数 且 > 0.1
  3. 每轮选覆盖最多新体素的帧，直到达到 `max_frames` 或无新增覆盖

### Step 2: 提取关键帧

- 输入: 原始视频 + 关键帧索引
- 输出: 临时目录中的帧图片 `[(vid_idx, frame_path)]`
- 方法: ffmpeg 按时间戳精确提取

---

### 第一部分: 物体发现与去重 (Step 3-5)

#### Step 3: VLM 第一次调用 — 物体检测

- 输入: 关键帧图片
- 输出: `all_detections = [{"frame_idx", "frame_path", "objects": [{name, center_x, center_y, bbox}]}]`
- Prompt: `"List all visible objects in this image. Output JSON only."`
- 兼容性处理:
  - `label` → `name` 自动转换
  - `bbox_2d` → 计算 `center_x/center_y`，保留 `bbox`
  - 无位置信息 → 默认图像中心
- **重要**: `all_detections` 传递到 Step 7 用于 per-frame 可见性过滤

#### Step 4: 射线投射 — 3D位置估计

- 输入: 2D检测坐标 + VGGT点云
- 输出: 每个物体实例的3D质心 `centroid`
- 方法: 从相机中心沿像素方向发射射线，取距离最近的前5个点（`RAY_CAST_TOP_K=5`），中值滤波

#### Step 5: 语义去重（SYNONYM_MAP + CLIP）

- 输入: 所有帧的物体实例
- 输出: 唯一物体列表
- 方法: **三层联合去重**

**第一层: SYNONYM_MAP 精确匹配**（硬编码常见同义词）:
  - `ground`/`flooring`/`carpet`/`rug` → `floor`
  - `walling`/`walls` → `wall`
  - `mug`/`glass`/`tumbler` → `cup`
  - `sofa`/`settee` → `couch`
  - `desk`/`dining table` → `table`
  - `television`/`monitor`/`display` → `tv`
  - `potted plant`/`flowerpot` → `plant`

**第二层: CLIP 语义匹配**（发现 SYNONYM_MAP 未覆盖的同义词）:
  - 加载 `CLIPSemanticMatcher`（clip-vit-base-patch32）
  - 对所有去重后的候选名称两两计算文本相似度
  - 相似度 >= `CLIP_MERGE_THRESHOLD`（0.90）→ 视为同义，合并
  - 合并方向: SYNONYM_MAP 中的名称优先保留，否则保留实例数更多的
  - 示例: `"trash can"` ≈ `"garbage can"` (CLIP相似度=0.92) → 合并为 `"trash_can"`

**第三层: 名称标准化**（`normalize_category_name`）:
  - 小写、去复数（`chairs` → `chair`，`shelves` → `shelv` 等）

**降级策略**: CLIP 不可用时仅使用 SYNONYM_MAP + 名称标准化

---

### Step 6: SAM3 分割 floor 和 wall

- 输入: 关键帧图片 + VGGT结果
- 输出: `sam_results = {'has_floor', 'has_wall', 'floor_masks', 'wall_masks', 'keyframe_floor_masks'}`

**两阶段分割**:

**阶段1**: 对VGGT颜色帧做SAM分割 → 获取 `has_floor`/`has_wall`

**阶段2**: 对每个关键帧单独做SAM floor分割 → 获取 `keyframe_floor_masks`

**Floor mask 选择逻辑**（与 main.py `align_to_room_coordinate_system` 一致）:

当SAM对一帧返回多个floor mask候选时，用3D几何信息选择最像地板的那个：

```
多个 floor mask 候选
  │
  ├─ 只有1个 → 直接使用
  │
  ├─ 有多个 + 有world_points:
  │   1. get_plane_info: PCA拟合平面 → normal, area, mean_distance
  │   2. 过滤: mean_distance > 0.02 → 丢弃（不像平面）
  │   3. 过滤: 法向量偏离平均法向量 > 30° → 丢弃（如桌面误分）
  │   4. 选择: 面积最大的 → 最像地板
  │
  └─ 有多个 + 无world_points → 降级合并所有mask
```

**降级策略**: SAM3不可用 → `has_floor`/`has_wall` 默认 True，无 `keyframe_floor_masks`

---

### Step 5.5: 点云补充检测 — 发现VLM遗漏的远端大物体

**问题**: VLM对近处物体检测效果好，但可能遗漏远端的大物体（如远处的柜子、书架等）。

**方案**: 保留原有VLM检测，通过分析VGGT点云中未被检测物体覆盖的显著3D聚类，补充检测远端大物体。

#### 5.5.1 提取高置信度3D点

从VGGT点云中提取可靠3D点，作为后续聚类的基础：

```
VGGT world_points: shape (T, H, W, 3)  — T帧，每帧H×W个3D点
VGGT world_points_conf: shape (T, H, W) — 每个点的置信度

采样策略:
  - 从T帧中等间隔采样5帧 (sample_step = T // 5)
  - 对每帧:
    1. 统计置信度分布: min, max, 均值, 中位数
    2. 动态阈值: max(50%分位数, 0.1)  — 自适应不同场景的置信度分布
    3. 提取置信度 > 阈值的3D点
    4. 体素降采样 (5cm体素) — 减少计算量，保持空间均匀性

合并所有帧的降采样点 → all_points (float32)
```

**为什么用动态阈值而非固定阈值**: 不同视频的VGGT置信度分布差异很大，固定阈值（如1.5）在某些场景可能过滤掉太多或太少点。50%分位数自适应地选择"上半部分"高质量点。

#### 5.5.2 排除已知物体附近的点

```
已知物体: unique_objects 中每个物体的 3D centroid

排除方法:
  对 all_points 中的每个点:
    计算到所有已知物体质心的最小距离
    如果最小距离 < 0.3m → 排除（该点属于已被VLM检测到的物体）

结果: 只保留"远离已知物体"的点
```

**为什么是0.3m**: 一般物体的3D质心在其几何中心，0.3m足以覆盖常见物体的范围（椅子~0.4m，杯子~0.1m），同时不会误排除远处的独立物体。

#### 5.5.3 排除floor区域的点

利用Step 6的 `keyframe_floor_masks` 排除属于地板的3D点：

```
对每个关键帧的 floor_mask:
  1. 从 world_points[frame_idx] 中提取 mask 对应的3D点
  2. 体素降采样 (10cm体素) → floor_3d_points 集合
     (用体素坐标的tuple作为集合key，加速查找)

对 all_points 中的每个点:
  计算其10cm体素坐标
  如果该体素在 floor_3d_points 中 → 排除

结果: 只保留"不在地板上"的点
```

**为什么用体素集合而非逐点距离**: floor mask可能包含数十万个3D点，逐点计算距离太慢。体素化后用集合查找，O(1)复杂度。

#### 5.5.4 DBSCAN聚类

对剩余的"未知"3D点做聚类，找出独立的3D区域：

```
DBSCAN参数:
  eps = 0.15m    — 两个点距离<15cm视为邻居
  min_samples = 30 — 一个聚类至少30个点

聚类结果: labels 数组
  label = -1: 噪声点
  label >= 0: 属于某个聚类

降级: sklearn不可用时 → 简单体素聚类 (10cm体素，同体素的点归为一类)
```

**为什么选这些参数**: `eps=0.15m` 适合室内场景中物体的尺度（家具通常>15cm），`min_samples=30` 过滤掉零散噪声点。

#### 5.5.5 过滤聚类（用点云大小/形状决定）

对每个聚类计算几何特征，过滤不像物体的聚类：

```
对每个聚类:
  1. centroid = 所有点的均值 (3D质心)
  2. bbox_extent = 各轴的极差 (ptp) → [dx, dy, dz]
  3. volume = dx × dy × dz

过滤规则:
  ┌─────────────┬────────────────────────────┬──────────────────────┐
  │ 规则         │ 条件                        │ 原因                 │
  ├─────────────┼────────────────────────────┼──────────────────────┤
  │ 点数过少     │ < 50个点                    │ 稀疏噪声             │
  │ 体积过小     │ < 0.005m³                   │ 小碎片/噪声          │
  │ 体积过大     │ > 8m³                       │ 可能是墙/地板        │
  │ 过扁         │ 最薄<3cm 且 最长>1.5m       │ 平面结构(墙/地板)    │
  │ 长宽比过大   │ max_dim/min_dim > 30        │ 线状结构(管道等)     │
  └─────────────┴────────────────────────────┴──────────────────────┘

按体积降序排列，最多取前5个候选
```

**为什么这些过滤规则有效**:
- **体积过滤**: 真实物体通常在 0.01~5 m³ 范围内，超出范围的大概率是场景结构
- **扁平过滤**: 墙壁/地板的特征是"很薄但很宽"（如 2cm × 3m × 2m），真实物体不会这么扁
- **长宽比过滤**: 管道/栏杆等线状结构不是我们要检测的物体

#### 5.5.6 VLM识别

对每个候选聚类，投影到2D图像并裁剪，用VLM识别：

```
对每个候选聚类:
  1. 3D→2D投影: 找最佳关键帧
     - 将3D质心投影到每个关键帧的2D像素位置
       (在world_points中找距离质心最近的像素)
     - 选择条件:
       a. 投影位置不在图像边缘 (距边缘 > 150px)
       b. 投影位置的3D点与质心距离 < 0.5m (深度一致性)
     - 按 1/(距离+0.01) 打分，选最佳帧

  2. 裁剪图像: 以投影位置为中心，裁剪 300×300 像素区域

  3. VLM识别: "What is the large object in the center of this cropped image?"
     输出: {"name": "object_name"} 或 {"name": "none"}

  4. 过滤:
     - "none"/"null"/"unknown" → 跳过
     - 属于 FILTER_CATEGORIES (floor/wall/...) → 跳过
     - 与 unique_objects 中已有物体同名 → 跳过

  5. 采纳: 通过所有过滤 → 添加到 unique_objects
     记录: name, centroid, source='pointcloud_supplementary', volume, point_count
```

**为什么裁剪300×300**: 远端物体在原图中很小，裁剪放大后VLM能看清细节。300px足够包含一个物体的局部特征。

**为什么用专门的prompt而非复用检测prompt**: 裁剪图像中只有一个物体居中，用简单直接的识别prompt效果更好，避免VLM尝试检测多个物体。

**示例输出**:
```
🔎 Step 5.5: 点云补充检测 — 发现遗漏的远端大物体
   已知物体: 4 个
   🔍 采样帧索引: [0, 32, 64, 96, 128]
      帧#0: 置信度范围 [0.050, 3.200], 均值=1.420, 中位数=1.380, 有效点数=28900
      → 提取 1520 个点 (阈值=1.380)
      帧#32: 置信度范围 [0.030, 2.980], 均值=1.350, 中位数=1.310, 有效点数=27500
      → 提取 1380 个点 (阈值=1.310)
   📊 降采样后点云: 8500 个点
   排除已知物体附近点后: 5200 个点
   排除floor区域点后: 3100 个点
   🔮 DBSCAN聚类: 8 个聚类
   ✅ 过滤后候选聚类: 2 个
      候选1: 质心=[1.2, -0.8, 0.5], 尺寸=[0.6, 0.4, 1.2], 体积=0.288, 点数=380
   [1/2] 识别候选聚类 (质心=[1.2, -0.8, 0.5])...
      ✅ 新发现: bookshelf (体积=0.288, 点数=380)
   🎉 补充检测发现 1 个新物体:
      - bookshelf (质心=[1.2, -0.8, 0.5])
```

---

### 第二部分: 关系判断 (Step 7)

#### Step 7: 关系判断 — SAM预判断 + Per-frame VLM推理 + 多帧投票

##### 7.1 预处理: 过滤场景结构物体

```
原始物体列表: [floor, chair, table, cup, wall, clock]
                    ↓ FILTER_CATEGORIES 过滤
待判断物体列表: [chair, table, cup, clock]
```

- `FILTER_CATEGORIES = {floor, wall, ground, ceiling, floor_area, wall_section, flooring, wall_surface, room, space}`
- floor/wall 只作为场景上下文 (`has_floor`/`has_wall`) 传递给prompt
- 通过 `SYNONYM_MAP` + Step 5 去重，floor 只有一个

##### 7.2 SAM floor 预判断（几何方法）

用SAM的floor mask通过**几何重叠检测**判断物体是否在地板上：

**`_check_object_on_floor(bbox, floor_mask, threshold=0.3)`**:
1. 取物体bbox的**底部30%区域**
2. 计算该底部区域与floor mask的像素重叠比例
3. 重叠 >= 30% → 判定在地板上

```
物体bbox:  ┌──────────┐
           │  上部70%  │  ← 不检查
           │──────────│
           │  底部30%  │  ← 检查与floor mask重叠
           └──────────┘
               ↕
         SAM floor mask
```

**`_sam_prejudge_floor(all_detections, keyframe_floor_masks)`**:
1. 遍历每个关键帧的检测结果
2. 对每个有bbox的物体，检查底部与该帧floor mask的重叠
3. **任一帧**重叠 >= 30% → 标记为 "supported by floor"
4. SAM预判断的物体**直接确定关系，不再经过VLM**

##### 7.3 Per-frame 可见性映射

利用 Step 3 的 `all_detections` 构建每帧的物体可见性：

```python
frame_visibility = {
    frame_idx_0: {'chair', 'table', 'cup'},
    frame_idx_1: {'chair', 'clock'},
    ...
}
```

**关键规则**: 物体在某帧没出现 → 该帧不对该物体做VLM判断 → 不产生默认投票

##### 7.4 VLM 多帧推理

只对**可见且未被SAM预判断**的物体调用VLM：

```python
objects_to_query = [n for n in object_names if n not in sam_floor_objects]

for vid_idx, frame_path in frame_paths_with_indices:
    visible_in_frame = frame_visibility.get(vid_idx, set())
    frame_objects = [n for n in objects_to_query if n in visible_in_frame]
    if not frame_objects:
        continue  # 该帧无待判断物体，跳过
    frame_prompt = _build_relationship_prompt(frame_objects, has_floor, has_wall)
    ...
```

**4种空间关系**:
1. `supported by floor`: 直接放在地面上
2. `supported by other objects`: 放在其他物体上
3. `attached to wall`: 挂在墙上
4. `embedded in wall`: 嵌入墙内

##### 7.5 物理常识后处理纠错

`_post_process_relationships` 对每帧VLM输出纠错：

| 规则 | 物体类别 | VLM错误 | 纠正为 |
|------|---------|---------|--------|
| 家具必须在地面 | cabinet, table, chair, sofa, bed, shelf, refrigerator, plant, box, carpet... | attached/embedded in wall | supported by floor |
| 墙面装饰必须挂墙 | picture, painting, mirror, clock, poster, curtain... | supported by floor | attached to wall |
| 墙面嵌入物必须嵌墙 | window, door, outlet, vent, socket... | supported by floor | embedded in wall |
| 无效关系 | 任何物体 | 非法字符串 | supported by floor |
| 缺失物体兜底 | 该帧可见但VLM未输出 | 无 | supported by floor |

##### 7.6 汇总结果

```
对每个物体:
  ├─ SAM预判断为floor → "supported by floor" (🏠)
  ├─ 有VLM投票 → 取多数票 (📋)
  └─ 无SAM也无VLM → 兜底 "supported by floor" (⚠️)
```

---

### Step 8: 保存场景JSON

- 输出格式: `{"物体名": "空间关系", ...}`，按名称排序
- 不包含 floor/wall/ceiling 等场景结构

### Step 9: 保存关键帧

- 输出目录: `assets/key_frames/<视频名>/`
- 文件命名: `frame_vid{vid_idx}.jpg`
- 元数据: `keyframes_metadata.json`（包含关键帧索引、可见性映射、场景物体）

---

## 输出格式示例

```json
{
  "chair": "supported by floor",
  "clock": "attached to wall",
  "cup": "supported by other objects",
  "door": "embedded in wall",
  "table": "supported by floor"
}
```

---

## 关键设计决策

### 为什么用 SAM floor mask 做预判断而非纯VLM？

1. **几何方法更可靠**: VLM可能因视角误判（俯视时把桌子判为"attached to wall"），SAM的floor mask是几何分割，不受视角影响
2. **可量化**: 30%重叠阈值可调节，便于针对不同场景优化
3. **减少VLM工作量**: SAM预判断的物体不需要VLM推理
4. **SAM和VLM互补**: SAM擅长"是否在地板上"（几何），VLM擅长"是否在墙上/其他物体上"（语义）

### 为什么需要 Per-frame 可见性过滤？

1. 物体不是每帧都出现，不可见帧的VLM判断不可靠
2. 旧逻辑中不可见物体默认"supported by floor"，导致投票偏差
3. Step 3 的 `all_detections` 天然包含per-frame物体列表

### 为什么 floor/wall 不参与 VLM 关系判断？

1. floor是场景结构，不是物体，其"关系"是自指的
2. 同义词合并保证floor唯一（ground/flooring/carpet/rug → floor）
3. floor/wall作为场景上下文（`has_floor`/`has_wall`）帮助VLM判断其他物体

### 为什么 Step 5.5 在 Step 6 之后执行？

Step 5.5 需要 SAM 的 `keyframe_floor_masks` 来排除 floor 区域的3D点。如果不排除floor点，DBSCAN会把大片地板聚成一个大聚类，浪费VLM识别次数。

### 为什么 Step 5.5 用动态置信度阈值？

不同视频的VGGT置信度分布差异很大。固定阈值（如1.5）在某些场景可能过滤掉太多点（高质量重建但整体置信度偏低），在其他场景可能保留太多噪声。50%分位数自适应地选择"上半部分"高质量点。
