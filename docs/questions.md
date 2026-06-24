# ReplicateAnyScene 问题与解答汇总

本文档汇总项目开发过程中遇到的所有问题、分析过程和解决方案。包含原 `comprehensive_issues.md` 的内容。

---

## 一、main.py 与 mainv2.py 的区别

### Q1: main函数和src目录里的有什么区别？

**回答**: main.py 是原始的单文件流水线（212行），需要手动提供场景JSON。mainv2.py 是完整自动化流水线（881行），新增了 Stage 1 自动物体发现、Stage 5 语义精修等功能。src/ 目录包含被两者共用的核心模块。

### Q2: 主文件夹里的 ReplicateAnyScene 和当前目录有什么区别？

**回答**: 对比的是 `~/robot_world_ws/src/ReplicateAnyScene`（home目录）和 `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene`（工作目录）。两者可能有微小差异，需要对比确认。

### Q3: mainv2是不是没有全覆盖main函数的东西？鲁棒性过关吗？

**回答**: mainv2 完全覆盖了 main.py 的所有功能，并修复了 main.py 的关键BUG（精修结果未写回）。mainv2 的鲁棒性改进包括：
- 异常处理和日志系统
- `protected_categories` 防止跨类合并
- 兼容两种关系格式（`supported_by_floor` 和 `supported by floor`）
- pkl 保存保留原始 T+mesh 数据
- 分阶段 GLB 保存

---

## 二、最优视角帧与3D资产生成

### Q4: 最优视角帧有输出图片吗？

**回答**: 已添加保存功能。在 `run_stage3()` 中，最优视角帧保存到 `output_path/optimal_frames/` 目录，文件名格式为 `{category}_inst{idx}_frame{fid}.jpg`。

### Q5: 需要保存提取生成3D资产的帧

**回答**: 已实现。保存的是用于3D资产生成的最优帧，包含物体和帧数信息。

---

## 三、VGGT 点云质量与3D物体摆放

### Q6: VGGT输出的点云很差，会影响3D物体的摆放吗？

**回答**: 会影响，影响链路如下：

```
VGGT点云质量差
  → pointmap精度低 → SAM3D几何条件输入错误 → l2c矩阵偏移 → T矩阵位置偏移
  → extrinsic精度低 → inv(extrinsic)映射不准 → 物体世界坐标偏移
```

具体影响程度取决于点云质量差的原因：
- **VGGT-omega缺块**: 低纹理区域(墙壁)点云缺失 → 坐标系对齐可能不准 → 所有物体位置系统性偏移
- **VGGT手部云团**: 手部区域散乱3D点 → SAM3分割mask包含手部 → SAM3D生成包含手部的mesh → 位置偏移
- **VGGT漂移**: 相邻帧点云不一致 → 动态/静态判断不准 → 选错最优帧 → mesh质量差

### Q7: 点云质量差会影响重定位精度吗？它不是根据掩码和点云分割的吗？

**回答**: 点云确实有分物体的点云。SAM3分割出物体mask后，只取mask内的点云用于3D资产生成。但问题是：

1. **mask内的点云也可能不干净**: 如果手部遮挡，mask包含手部区域，手部的点云也会被包含进来
2. **点云是全局共享的**: VGGT输出的是每帧每像素的世界坐标 `world_points[T,H,W,3]`，所有物体共享同一个点云。mask只是选择哪些像素属于哪个物体
3. **SAM3D使用的是mask内的pointmap**: `generate_3d_asset()` 接收 `pointmap = world_points[optimal_frame_id]`，SAM3D根据mask区域内的pointmap生成3D资产

### Q8: 相机也有移动的，点云建立到底是什么模式？

**回答**: VGGT的点云建立模式是**一次性全局预测**，不是逐帧增量构建：

```
VGGT内部流程:
  160帧RGB → Transformer编码器 → 联合预测所有帧的depth + extrinsic
  → world_points[s] = extrinsic[s] @ backproject(depth[s])
  → 所有帧共享同一个世界坐标系
```

这意味着：
- 所有帧的点云是在同一个世界坐标系下预测的
- 相机移动是通过extrinsic（相机外参）体现的
- 点云质量取决于VGGT对整个视频序列的联合理解能力

### Q9: VGGT-omega也是动态共享机制吗？

**回答**: VGGT-omega 使用的是 DenseHead（密集预测头），不是动态共享机制。它预测的是每帧的深度图（depth map），然后通过 extrinsic 反投影到世界坐标系。与 VGGT 的区别在于：
- VGGT: PointHead 直接预测3D点坐标 → 点云质量高但推理慢
- VGGT-omega: DenseHead 预测深度图 → 推理快但点云质量差（缺块问题）
- VGGT4D: 在VGGT基础上增加动态mask预测 → 适合动态场景

### Q10: D²USt3R 是最好的动态分割点云吗？VGGT4D呢？

**回答**: 各模型定位不同：

| 模型 | 动态场景 | 点云质量 | 速度 | 适用场景 |
|------|---------|---------|------|---------|
| VGGT | ★★☆ | ★★★★★ | ★★☆ | 静态场景 |
| VGGT-omega | ★★☆ | ★★★☆☆ | ★★★★ | 快速推理 |
| VGGT4D | ★★★★☆ | ★★★★☆ | ★★☆ | 动态场景 |
| D²USt3R | ★★★★★ | ★★★☆☆ | ★★☆ | 纯动态场景 |
| MonST3R | ★★★★☆ | ★★★☆☆ | ★★★ | 动态场景 |

VGGT4D 在动态场景下整体可用性最高（★★★★☆），但存在 dyn_masks 过度标记问题。D²USt3R 在纯动态场景下表现最好，但点云质量不如 VGGT 系列。

---

## 四、物体关系投票与实例检测

### Q11: scissor_0 只有1帧投票，为什么没有补上？

**回答**: 根因是跨类去重将 scissor 合并到 toy，导致 `instance_visibility.json` 中无 scissor 数据，无法补充帧。因果链：

```
跨类去重合并 scissor → toy
  → instance_visibility 中无 scissor
  → 无法补充帧
  → 只有1帧投票
  → 无法精修
```

**修复**: 在 `cross_category_deduplicate()` 中新增 `protected_categories` 参数，Stage1 发现的不同类别互不合并。

### Q12: toy_1 平票（4/8 floor vs table），如何处理？

**回答**: 新增 `_resolve_tie_by_z()` 函数，用Z坐标高度判断平票关系：
- 比较物体Z中心到table顶面和floor的距离
- 取距离更近的作为支撑物
- 桌面和地面在Z轴上差距通常很大（>0.5m），容易区分

### Q13: scissor_0 未找到对应实例，为什么？

**回答**: 因为跨类去重已将 scissor 合并到 toy，`all_instances` 中不存在 scissor key。修复方式：
1. `protected_categories` 防止合并
2. `_find_supporter_instances()` 增加第4层匹配：去掉实例后缀再搜索

---

## 五、动态/静态物体检测

### Q14: 动态物体判断不准确，有些不动的被识别成动态

**回答**: 原始判断方法（首尾帧质心位移 > 0.10m）不够鲁棒。改进为双重判断：

1. **中位数位移**: `median_disp > 0.02m` → 逐帧漂移检测
2. **全局位移**: `global_disp > max(0.04, 2×motion_threshold)` → 首尾质心距离检测

任一信号超过阈值即判定为动态。这比单一首尾帧位移更鲁棒，因为VGGT漂移会平滑掉逐帧位移。

### Q15: 动态物体的最大生成帧应该调整到最开始物体出现的那一帧

**回答**: 当前逻辑已经实现了这个功能。对于动态物体：
1. 找到运动起始帧（motion onset）
2. 选择运动起始帧之前面积最大的帧
3. 这样选出的帧是物体在原始位置、形状最完整的时刻

### Q16: 首尾帧质心位移 > 0.10m → 动态，确定实现了吗？

**回答**: 已改为更鲁棒的双重判断（中位数+全局位移）。当前代码在 `src/geometry_utils.py:307-429`，使用 `motion_threshold=0.02` 作为默认阈值。log中已输出每个物体的 `[DYNAMIC]` 或 `[STATIC]` 标记及位移数据。

---

## 六、SAM3 分割与手部遮挡

### Q17: SAM3会把移动的物体因为点云过近在同一位置生成新物体

**回答**: 这是SAM3的一个已知问题。当动态物体移动后，原位置和新位置可能被分割为两个不同的实例。解决方案：

1. **3D空间重叠去重**: `self_category_deduplicate()` 和 `cross_category_deduplicate()` 通过3D点云重叠率合并同一物体
2. **2D时序连续性去重**: 如果两个实例在时序上首尾相接且2D IoU高，则合并（建议实现）
3. **protected_categories**: 防止不同类别被错误合并

### Q18: 手部遮挡时SAM3把手也分割进mask

**回答**: 这是手物交互场景的核心问题。当手部遮挡物体时：
1. SAM3跟踪断裂 → 同一物体产生多个实例
2. mask包含手部区域 → 3D资产包含手部几何
3. 手部区域的pointmap是手的深度 → mesh位置偏移

**改进方案**（按优先级）:
1. 手部感知的最优帧选择（推荐，改动最小）
2. mesh连通分量清理（去除手指状碎片）
3. 2D时序连续性去重（解决跟踪断裂）
4. mask后处理去除手部区域
5. 多帧融合3D资产生成（长期方案）

### Q19: 手部掩码去除也是黑色的，也没法估计深度

**回答**: 确实如此。将手部区域设为黑色（深度为0）后：
- SAM3D无法估计被遮挡区域的深度 → 生成不完整的mesh
- pointmap中手部区域变为无效值 → 几何条件输入缺失
- 本质上，黑色区域 = 无信息，无法恢复被遮挡的3D结构

**因此**: 手部掩码去除方案只适用于手部遮挡面积较小的情况。对于长时间遮挡，需要多帧融合方案：从不同帧中获取物体不同角度的完整信息。

---

## 七、VGGT 模型对比

### Q20: VGGT-omega没有比VGGT4D好吗？

**回答**: 各有优劣，取决于场景：

| 维度 | VGGT-omega | VGGT4D |
|------|-----------|--------|
| 点云质量 | ★★★☆☆（缺块） | ★★★★☆ |
| 动态场景 | ★★☆ | ★★★★☆ |
| 推理速度 | ★★★★ | ★★☆ |
| 动态mask | ❌ 无 | ✅ 有（但过度标记） |
| 深度图质量 | ★★★★ | ★★★ |

**结论**:
- **静态场景**: VGGT（原始）最好，点云质量最高
- **动态场景**: VGGT4D 最好，有动态mask且点云质量不错
- **快速推理**: VGGT-omega 最快，但点云质量差（缺块问题）

VGGT-omega 的核心问题是 DenseHead 的 depth_conf 分布太均匀，50%百分位阈值无法区分好坏点，导致低纹理区域（墙壁）被过滤掉。

### Q21: VGGT和VGGT-Omega同样砍50%置信度，为什么效果差异巨大？

**回答**: 数学定义相同（`conf = 1/(1+exp(-logit))`），但语义完全不同：
- VGGT的PointHead conf是3D点位置的置信度 → 分布双峰（好坏分明）→ 50%阈值有效分离
- VGGT-Omega的DenseHead conf是深度值的置信度 → 分布均匀（差异不大）→ 50%阈值≈随机砍半

这就是VGGT-Omega缺块的根因：depth_conf分布太均匀，百分位阈值无法区分好坏点。

---

## 八、Stage 3 流程与问题

### Q22: Stage 3的流程是什么？有哪些问题？

**回答**: Stage 3 流程：

```
1. 计算每个实例的最优视角帧ID（动态/静态检测）
2. 保存最优视角帧图像
3. 保存实例可见性信息
4. 在SAM3D子进程中生成3D资产
5. 多票验证生成的3D资产
```

**问题与解答**:

**问题1: 第0帧添加prompt就可以发现所有物体**
- 回答: 确实如此。SAM3的 `propagate_in_video` 从第0帧向所有帧传播分割，第0帧的prompt足以发现所有物体。多个prompt发现没有本质区别。

**问题2: 动态/静态判断不准**
- 回答: 已改进为双重判断（中位数+全局位移），并在log中输出每个物体的动静态标记。

**问题3: SAM3分割出动态物体后，点云在一起会导致生成多个物体**
- 回答: 这是3D空间重叠去重需要解决的问题。当动态物体移动后，原位置和新位置的点云重叠，可能导致不同物体被合并或同一物体被拆分。2D时序连续性去重可以部分解决。

---

## 九、综合问题分析（原 comprehensive_issues.md 内容）

### 核心问题1: VGGT点云质量差 → 3D物体摆放位置偏移

**根因**:
- VGGT-omega: DenseHead预测深度图，conf分布均匀，百分位阈值无法区分好坏点 → 低纹理区域缺块
- VGGT: 手部区域产生大量散乱3D点，PointHead对动态物体也输出高置信度 → 手部云团
- VGGT4D: dyn_masks过度标记，手扫过的背景区域也被标为动态

**影响链**:
```
点云质量差 → pointmap精度低 → SAM3D几何条件输入错误
  → l2c矩阵偏移 → T矩阵位置偏移 → 物体摆放位置偏移
```

**解决方案**:
- 降低置信度阈值（VGGT-omega）
- 过滤深度异常值（exp激活产生的离群点）
- 使用VGGT4D的动态mask过滤手部区域
- 时间持续性过滤（VGGT4D dyn_masks过度标记）

### 核心问题2: SAM3跟踪断裂 → 同一物体产生多个实例

**根因**:
- 手部遮挡导致SAM3丢失跟踪
- 物体被拿起后形状变化导致mask不连续
- 不连续帧段被拆分为不同实例

**影响链**:
```
跟踪断裂 → 同一物体多个实例 → 跨类去重可能合并不同物体
  → instance_visibility数据不完整 → 投票帧不足 → 无法精修
```

**解决方案**:
- 修改帧间隙阈值（从1改为5，允许短暂遮挡）
- 2D时序连续性去重（首尾帧IoU高则合并）
- protected_categories防止跨类合并

### 核心问题3: mask包含手部区域 → 3D资产质量差

**根因**:
- SAM3无法区分物体和手部
- OR操作把手部区域也保留在mask中

**影响链**:
```
mask包含手部 → SAM3D生成包含手部几何的mesh
  → mesh质量差 → T矩阵计算偏移 → 物体位置偏移
```

**解决方案**:
- 手部感知的最优帧选择（优先选手部遮挡少的帧）
- mask后处理去除手部区域
- mesh连通分量清理（去除手指状碎片）

### 核心问题4: 物体关系投票平票

**根因**:
- VLM投票时floor和table得票相同
- 没有坐标系判断机制

**解决方案**:
- `_resolve_tie_by_z()`: 用Z坐标高度判断
- 桌面和地面在Z轴上差距通常很大（>0.5m），容易区分

### 核心问题5: 跨类去重合并不同物体

**根因**:
- 3D空间重叠率超过阈值时合并
- 不同物体在3D空间中可能确实重叠（如桌上的碗和桌子）

**解决方案**:
- `protected_categories`: Stage1发现的不同类别互不合并

### 核心问题6: 桌子悬浮0.47m

**根因**:
- 代码中有 `refine_supported_by_floor_object()` 但阈值 `abs(z_min) < 0.3`
- 如果桌子底面z_min超过0.3m，精修不会执行
- 可能是坐标系对齐不准导致桌子z坐标偏移过大

**解决方案**:
- 检查坐标系对齐质量
- 调整精修阈值
- 确保精修结果正确写回（main.py的BUG已修复）

### 核心问题7: final_scene.glb 和 final_scene_base.glb 的区别

**回答**:
- `final_scene_base.glb`: 基础精修后（floor/wall/embedded）的固定起点，不再更改
- `final_scene.glb`: 最终场景，始终为最新结果
  - 不启用stage4/5时，等同base
  - 启用stage4后，包含Stage4的视觉-空间对齐结果
  - 启用stage5后，包含Stage5的语义精修结果

---

## 十、管线使用方式

### mainv2.py 参数

```
--input_video       输入视频路径 (与--input_images二选一)
--input_images      输入图片目录路径 (与--input_video二选一)
--output_path       输出目录 (默认: ./output_v2/{video_stem}_{模型名})
--category_path     手动指定场景JSON (跳过Stage1自动发现)
--vggt_model        3D重建模型: vggt(默认) | vggt_omega | vggt4d
--max_frames        VGGT最大帧数 (默认160)
--max_frames_stage1 Stage1采样关键帧数 (默认10)
--vlm_checkpoint    VLM模型路径 (默认自动查找)
--enable_stage4     启用Stage 4视觉-空间对齐
--enable_stage5     启用Stage 5高级语义精修 (5.1+5.2)
--stage4_iterations Stage4 ICP迭代次数 (默认8)
--stage4_temporal_radius Stage4时序邻域半径 (默认5)
--stage4_use_mast3r Stage4使用MASt3R匹配
```

### run_post_pipeline.py 参数

```
--scene_dir              场景输出目录 (必需)
--stage4                 启用Stage 4
--stage5                 启用Stage 5 (5.1+5.2)
--only_refine_relations  只运行5.1 (细化关系)
--only_sp_refinement     只运行5.2 (SP精修)
--relations_json         手动指定关系JSON (配合--only_sp_refinement)
--vlm_checkpoint         VLM模型路径
--stage4_iterations      Stage4 ICP迭代次数 (默认8)
--stage4_temporal_radius Stage4时序半径 (默认2)
--stage4_use_mast3r      Stage4使用MASt3R匹配
```

### 两者关系

```
mainv2:          Stage1 → Stage2 → Stage3 → 基础精修 → [Stage4] → [Stage5] → final_scene.glb
run_post_pipeline:                                      [Stage4] → [Stage5] → final_scene_stageX.glb
```

两者的 Stage4/5 逻辑完全一致，只是入口不同。mainv2 是完整流水线，run_post_pipeline 只做后处理。

---

## 十一、坐标系与点云

### Q23: VGGT的点云建立模式

**回答**: VGGT 是一次性全局预测，不是逐帧增量构建。所有帧共享同一个世界坐标系：

```python
# VGGT内部: 多帧RGB → Transformer → 联合预测
world_points[s] = extrinsic[s] @ backproject(depth[s], intrinsic, pixel_coords)
```

### Q24: 点云的作用是什么？

**回答**: 点云在管线中有两个关键作用：

1. **坐标系对齐**: 从地板/墙壁的点云中提取平面信息（PCA），构建旋转矩阵R和平移向量t，将VGGT坐标系对齐到房间坐标系（Z轴朝上，地板z=0）

2. **3D资产生成的几何条件**: `pointmap = world_points[optimal_frame_id]` 作为SAM3D的输入，引导SAM3D生成正确位置和朝向的3D mesh。SAM3D根据pointmap中物体的3D位置来确定mesh的l2c矩阵（局部→相机变换）

### Q25: 混乱的点云到底会不会对坐标对齐产生影响？

**回答**: 会产生影响，但程度取决于混乱的类型：

- **系统性偏移**（如VGGT漂移）: 对坐标对齐影响小，因为PCA拟合的是平面法向量，对个别点的偏移不敏感
- **局部缺失**（如VGGT-omega缺块）: 对坐标对齐影响中等，如果缺失的是墙壁/地板区域，可能导致平面拟合不准
- **散乱噪声**（如手部云团）: 对坐标对齐影响小，因为手部区域通常不是墙壁/地板，不会被用于平面拟合
- **对3D资产位置影响大**: pointmap中的噪声直接影响SAM3D的l2c计算，导致物体位置偏移

---

## 十二、mask遮掩与点云清晰度

### Q26: mask的遮掩有用吗？提取手部掩码让它变成黑白会让点云更清楚吗？

**回答**: 有限有用。将手部区域设为黑色（深度为0）后：

**有用的情况**:
- 手部遮挡面积较小（<30%物体区域）
- 物体在部分帧中未被遮挡 → 可以选择未遮挡帧作为最优帧

**没用的情况**:
- 手部遮挡面积过大 → 去除手部后mask太小，SAM3D无法生成有效mesh
- 物体在所有帧中都被遮挡 → 无法获得完整的物体信息
- 黑色区域 = 深度为0 = 无信息 → SAM3D无法估计被遮挡区域的3D结构

**更好的方案**: 手部感知的最优帧选择（方案1），优先选择手部遮挡少的帧，而不是去除手部区域。

---

## 十三、输出目录文件说明

### hoi4d_vggt_omega 输出目录

| 文件/目录 | 说明 |
|-----------|------|
| `final_scene_base.glb` | 基础精修后固定起点（不再更改） |
| `final_scene.glb` | 最终场景GLB（始终为最新结果） |
| `all_instances.pkl` | 实例数据（供后处理管线使用） |
| `point_cloud.ply` | 3D重建点云 |
| `intrinsic.txt` | 相机内参 |
| `scene_*_stage1.json` | Stage1自动发现的场景JSON |
| `scene_*_refined.json` | Stage5.1细化后的关系JSON |
| `final_relations.json` | 最终关系JSON |
| `color/` | RGB帧 |
| `depth/` | 深度图（mm uint16） |
| `extrinsics/` | 相机外参 |
| `optimal_frames/` | Stage3最优视角帧 |
| `keyframes/` | Stage1关键帧+元数据 |
| `instance_masks.mp4` | 分割mask可视化 |

---

## 十四、VGGT 点云使用方式对比

### Q27: main.py 和 mainv2.py 对 VGGT 点云的使用有区别吗？

**回答**: **基本无区别。** 核心 Stage 2-3 中对 `world_points` 的使用参数和调用顺序完全一致。详细对比见 [VGGT_models_and_analysis.md 第29章](VGGT_models_and_analysis.md)。

唯一参数差异：`self_category_deduplicate` 中 mainv2 多传了 `category_name=category`（仅影响日志显示）。

### Q28: 物体位置不准是代码差异导致的吗？

**回答**: **不是。** 物体位置不准的根因在 VGGT 预测的 extrinsic/depth 精度，而非代码差异。

T矩阵的位置由 `extrinsic` 的逆矩阵决定：
```python
# generate_3d_asset 第49行
final_transform = matrix_ext_inv @ matrix_adjust @ matrix_l2c @ matrix_y2z
# matrix_ext_inv = np.linalg.inv(extrinsic)
```

如果 VGGT 预测的相机位置不准，所有物体的世界坐标都会偏移。这是上游问题，下游精修（floor/wall/Stage5）只能修正部分偏差。

### Q29: mainv2.py 有哪些 main.py 没有的点云使用步骤？

**回答**: 3处新增：

1. **动态物体位置调整**（第508-557行）：用 `world_points[fid]` 计算质心偏移，将动态物体从最优帧位置移到首次可见帧位置
2. **3D Mesh 去重后重算最优帧**（第568-571行）：`deduplicate_3d_assets` 移除重复实例后，重新调用 `get_optimal_view_frame_id`
3. **Stage 4 重建 world_points**（第620-631行）：从 `depths + extrinsics + intrinsic` 重新计算 `world_points`，置信度设为全1

### Q30: main.py 有什么 bug？

**回答**: 2个潜在问题：

1. **关系字符串不兼容空格格式**：main.py 只匹配 `"supported_by_floor"`（下划线），不匹配 `"supported by floor"`（空格）。如果 Stage1 输出空格格式，物体不会被精修
2. **精修结果可能未写回**：main.py 第192-199行修改了 `instance_info` 但没有 `category_instances[instance_id] = instance_info`。由于 Python 字典是可变对象且 `refine_*` 函数是原地修改，目前实际生效，但如果函数改为返回新字典，修改会丢失

### Q31: 点云质量高但物体摆放不准，问题出在哪里？

**回答**: 问题出在 **T矩阵的计算链路**，而非点云使用方式：

```
VGGT预测 depth + extrinsic
  → world_points = extrinsic @ backproject(depth)  ← 点云质量高
  → align_to_room_coordinate_system(world_points)  ← 坐标系对齐
  → align_vggt_predictions(R, t)                    ← 更新 extrinsics
  → generate_3d_asset:
       T = inv(extrinsic) @ adjust @ l2c @ y2z      ← T矩阵由extrinsic决定
```

**点云质量高 ≠ 物体位置准**。因为：
- 点云质量反映的是3D点的**相对位置**精度
- 物体位置由 `inv(extrinsic)` 决定，反映的是**绝对位置**精度
- VGGT 可能预测了正确的相对3D结构（点云质量高），但相机位置偏了（extrinsic 不准），导致物体绝对位置偏移

**验证方法**：检查 `point_cloud.ply` 中物体的位置是否与 `final_scene.glb` 中一致。如果 PLY 中位置正确但 GLB 中偏了，说明问题在 `generate_3d_asset` 的 T 矩阵计算；如果 PLY 中位置也偏了，说明问题在 VGGT 的 extrinsic 预测。

---

## 十七、动态物体分类：手抓物体 vs 手戴物品

### Q39: 手抓的物体和手戴的东西如何区别？

**背景**: 在 HOI（手物交互）场景中，物体与手的关系有两种本质不同的类型：
- **手抓物体 (Grasped Object)**: 杯子、锤子、手机等，被手抓取后可以随时放下，放下后物体独立存在于支撑面上
- **手戴物品 (Worn Item)**: 手套、手表、手环等，与手形成固定绑定关系，不会独立放置在支撑面上

**为什么需要区分**:
- 手抓物体放下后应该在支撑面上（桌子/地板），精修时需要做 z 轴对齐
- 手戴物品永远不会独立放在支撑面上，不应该做 floor/table 对齐
- 当前代码统一回退为 `"supported by other objects"`，对手抓物体不做精修，导致其悬浮

### 方法一：基于物体语义类别判断（最简单，推荐优先实现）

**原理**: 利用手物交互的先验知识，根据物体类别直接判断。

**实现**:
```python
WORN_ITEMS = {
    'glove', 'watch', 'ring', 'bracelet', 'wristband',
    'armband', 'sleeve', 'mittens',
}

def is_worn_item(category: str) -> bool:
    cat_lower = category.lower().strip()
    return any(w in cat_lower for w in WORN_ITEMS)
```

**优点**: 实现简单，无需额外模型，准确率高（类别语义明确）
**缺点**: 需要维护列表，对未知类别无法判断

### 方法二：基于手-物相对运动模式判断（最可靠）

**原理**: 手戴物品与手保持刚性绑定，运动轨迹高度一致；手抓物体在抓取/释放时有明显的相对运动变化。

**特征**:
- **手戴物品**: 全程与手同步运动，相对位移方差 ≈ 0
- **手抓物体**: 有"抓取-保持-释放"三阶段，释放后物体与手分离

**实现**:
```python
def classify_by_motion_correlation(hand_trajectory, object_trajectory, frames):
    """通过手-物运动相关性判断"""
    relative_displacements = []
    for fid in frames:
        hand_pos = hand_trajectory[fid]
        obj_pos = object_trajectory[fid]
        relative_displacements.append(np.linalg.norm(hand_pos - obj_pos))
    
    variance = np.var(relative_displacements)
    
    if variance < threshold_low:
        return "worn"      # 相对距离几乎不变 → 手戴物品
    elif variance > threshold_high:
        return "grasped"   # 相对距离变化大 → 手抓物体
    else:
        return "unknown"   # 需要其他方法辅助
```

**优点**: 不依赖类别先验，适用于任意物体
**缺点**: 需要手部轨迹追踪（当前管线没有），需要足够帧数

### 方法三：基于 VLM 视觉判断（最灵活）

**原理**: 让 VLM 直接判断物体是"被手抓着"还是"戴在手上"。

**实现**: 在 `infer_relations_scene_graph` 的 VLM prompt 中增加判断：
```
For each object that appears to be held by a hand:
- Is it WORN on the hand (glove, watch, ring)? → relation="worn", parent=0
- Is it GRASPED by the hand (cup, hammer, phone)? → relation="grasped", parent=0
```

**优点**: 可以处理复杂情况（如手机壳 vs 手机）
**缺点**: VLM 判断不稳定，增加 prompt 复杂度

### 方法四：基于物体-手接触面积比例判断

**原理**: 手戴物品与手的接触面积占物体表面积比例大（>50%），手抓物体接触面积比例小（<30%）。

**实现**: 利用 SAM3 的 mask 计算手-物重叠比例：
```python
def classify_by_contact_ratio(hand_mask, object_mask):
    overlap = np.sum(hand_mask & object_mask)
    object_area = np.sum(object_mask)
    contact_ratio = overlap / max(object_area, 1)
    
    if contact_ratio > 0.5:
        return "worn"
    elif contact_ratio > 0.1:
        return "grasped"
    else:
        return "independent"
```

**优点**: 利用已有 mask 数据，无需额外模型
**缺点**: 2D mask 的接触面积不一定反映真实 3D 接触关系，遮挡时不可靠

### 推荐策略

| 阶段 | 方法 | 理由 |
|------|------|------|
| 短期 | 方法一（语义类别） | 实现简单，覆盖常见场景 |
| 中期 | 方法一 + 方法三（VLM辅助） | 处理语义列表外的物体 |
| 长期 | 方法二（运动模式） | 最可靠，但需要手部追踪 |

**当前处理**: 暂不区分，统一回退为 `"supported by other objects"` 不做精修。待实现方法一后再调整。

---

## 十八、laptop_vggt_omega 输出分析

### Q40: laptop_vggt_omega 的输出有什么问题？

**场景**: laptop 场景，2个 table、1个 laptop

**发现的问题**:

1. **table 悬浮**: table_0 的 z_min=0.83m，table_1 的 z_min=0.76m，但基础精修 delta_from_initial=[0,0,0]，说明 `refine_supported_by_floor_object` 没有生效。需要检查代码。

2. **laptop 被推到 1.59m 高**: Stage5 将 laptop 从 z=0.945 推到 z=1.593（delta=0.647m），而 table_1 的 top_z=1.51m。laptop 被放在了 table_1 上面，但高度异常。

3. **instance-level 支撑物 bug**: `supported by table_1` 中的 `table_1` 被 `_build_instance_frame_map` 当作 category 名，生成了 `table_1_0` 这个不存在的实例键。**已修复**：收集 `all_categories` 时用 `rsplit('_', 1)` 解析回 category 名。

4. **`refined_relations` 格式**: 当前输出 `"laptop": "supported by table_1"`，这是 category-level 键但值是 instance-level 支撑物名。这是合理的——laptop 只有1个实例，所以不需要 instance-level 键，但支撑物 table 有2个实例，需要指定是哪一个。

---

## 十九、坐标系 Bug 与 Stage5 精修问题

### Q41: `final_relations.json` 为什么没有 table_0/table_1？

**回答**: 因为 `convert_scene_graph_to_relations` **只对原始关系为 `"supported by other objects"` 的类别生成 instance-level 键**。

laptop 场景中：
- `table` 的原始关系是 `"supported by floor"` → 已确定的关系，不需要 instance-level 键
- `laptop` 的原始关系是 `"supported by other objects"` → VLM 推断后变为 `"supported by table_1"`

所以 `final_relations.json` 只有：
```json
{
  "laptop": "supported by table_1",
  "table": "supported by floor",
  "laptop_0": "supported by table_1"
}
```

没有 `table_0` 和 `table_1` 键，因为两个 table 的关系相同（都是 floor），不需要区分。**`"table"` 是一个总结性键，代表该类别所有实例的共同关系。**

### Q42: `id_scene_mapping.json` 为什么 ID 从3开始？1和2去哪了？

**回答**: **1和2是保留ID**，分别代表 floor 和 wall：

| display_id | 含义 | 用途 |
|-----------|------|------|
| 0 | 无效/手持 | VLM 输出中 parent=0 表示无法确定关系 |
| 1 | 地板 (floor) | VLM 输出中 parent=1 表示被地板支撑 |
| 2 | 墙壁 (wall) | VLM 输出中 parent=2 表示附着在墙上 |
| 3+ | 物体 | 从3开始编号，每个物体实例一个 ID |

**为什么需要保留**: VLM 在推断关系时，需要引用 floor 和 wall 作为 parent。如果 floor 的 ID 是 1，VLM 就可以输出 `{"id": 3, "relation": "support", "parent": 1}` 表示"物体3被地板支撑"。

**`id_scene_mapping.json` 的用途**: 记录 `{category: [display_id, ...]}` 映射，用于将 VLM 输出的 display_id 反向映射回类别名。例如 VLM 输出 `parent=5`，查映射表发现 `table: [4, 5]`，所以 parent 是 table 的第2个实例（索引1）。

### Q43: Stage5 精修坐标系 Bug —— `refine_supported_by_floor_object` 取错了轴！

**回答**: **已发现并修复。** 这是一个严重的坐标系 bug。

**Bug 位置**: [sp_refinement.py:73](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/src/sp_refinement.py)

**Bug 内容**:
```python
# 错误: 取的是 T 矩阵的第1列 (y轴)
upper_transformed_vector = transform_matrix[:3,1]
```

**正确应该是**:
```python
# 正确: T 矩阵在 z-up 坐标系下，"上"方向是第2列 (z轴)
upper_transformed_vector = transform_matrix[:3,2]
```

**为什么这是 bug**:
- T 矩阵在 z-up 坐标系下（代码注释和 `instance_generation.py:38` 都明确说明）
- z-up 坐标系中，"上"方向是 z 轴，对应 T 矩阵旋转部分的**第2列**
- 原代码取了第1列（y轴），导致重力对齐方向完全错误
- 结果：物体的"上"方向被错误地对齐到 z 轴，导致物体被旋转到错误朝向

**对比其他函数**:
- `refine_embedded_in_wall_object`: 使用 `transform_matrix[:3, 2]` ✅ 正确
- `refine_attached_to_wall_object`: 使用 `transform_matrix[:3, 2]` ✅ 正确
- `refine_supported_by_floor_object`: 使用 `transform_matrix[:3, 1]` ❌ **错误**

**影响**:
1. 物体重力方向判断错误 → 旋转对齐方向错误 → 物体被旋转到错误朝向
2. 旋转错误后 z_min 计算也错误 → z 轴平移也错误
3. 这是 Stage5 精修"越调越差"的根本原因之一

### Q44: `refine_supported_by_floor_object` 的 z_min 阈值问题

**回答**: **已修复。** 原代码有 `abs(z_min) < 0.3` 阈值，导致悬浮物体不做对齐。

**Bug 内容**:
```python
# 错误: z_min > 0.3m 时不做对齐，导致悬浮
if abs(z_min) < 0.3:
    translation_vector = np.array([0, 0, -z_min])
```

**修复后**:
```python
# 正确: 始终做 z 轴对齐（只要 z_min 不为0）
if abs(z_min) > 1e-6:
    translation_vector = np.array([0, 0, -z_min])
```

**影响**: table 的 z_min=0.83m > 0.3m，所以原代码不做对齐，导致 table 悬浮。修复后 table 会被对齐到 z=0。

### Q45: 地板网格和物体坐标系是否一致？

**回答**: **一致。** 地板网格和物体在同一个坐标系下。

`save_final_glb` 的流程：
1. 在 z-up 坐标系下创建地板网格（z=0 平面上的线段）
2. 在 z-up 坐标系下应用 T 矩阵变换物体
3. 整个 scene 一起做 z-up → y-up 变换

所以地板网格和物体在 GLB 中的相对位置是正确的。**如果看起来不一致，是因为物体被错误精修（坐标系 bug）导致位置偏移，而非地板网格本身有问题。**

---

## 二十、Stage3 帧选择与多图生成

### Q46: Stage3 挑选最大帧有没有进行正则化约束？

**回答**: **没有。** 当前 Stage3 的最优帧选择逻辑是贪心策略——选择物体面积最大的帧，没有空间覆盖约束。

**当前逻辑** (`get_optimal_view_frame_id`):
- 遍历所有可见帧，计算 mask 面积
- 选择面积最大的帧作为最优帧
- 没有考虑空间覆盖完整性

**对比 SimRecon 的 `sa_sampling.py`**:
- 使用 **voxel 覆盖率** 作为选择标准
- 将 3D 空间划分为体素网格
- 贪心选择覆盖最多新体素的帧
- 保证选出的帧集合最大化 3D 空间覆盖

**问题**: 当前只选面积最大的帧，可能选到物体被遮挡或只看到部分视角的帧，导致 SAM3D 生成的 3D 资产不完整。

**改进建议**:
1. **短期**: 在面积最大基础上，增加遮挡率约束（手部遮挡面积占比 < 阈值）
2. **中期**: 引入 SimRecon 的 voxel 覆盖率方法，从多个候选帧中选择覆盖最完整的
3. **长期**: 多帧融合生成 3D 资产（见 Q47）

### Q47: 能否给与多个图片生成 3D 资产？

**回答**: **当前不支持，但技术上可行。**

**当前限制**: SAM3D 只接受单张图片 + pointmap 作为输入，生成单个 3D 资产。`generate_3d_asset` 的接口设计为单帧输入。

**多图生成的可行性方案**:

| 方案 | 原理 | 难度 | 效果 |
|------|------|------|------|
| **A: 多帧投票选最佳** | 从多个视角分别生成 3D 资产，选择最完整的 | 低 | 中等 |
| **B: 多视角融合** | 将多个视角的 pointmap 融合，输入 SAM3D | 中 | 较好 |
| **C: 多图重建** | 使用多图 3D 重建模型（如 InstantMesh, TripoSR 多视角版） | 高 | 最好 |

**方案 A 的具体实现思路**:
1. 选择 3-5 个候选帧（面积大 + 遮挡少 + 视角多样）
2. 对每个候选帧分别调用 `generate_3d_asset`
3. 选择 mesh 完整度最高的（顶点数/体积/对称性评分）
4. 用最佳 mesh 替换当前结果

**方案 B 的具体实现思路**:
1. 选择 3-5 个候选帧
2. 将多个帧的 pointmap 在世界坐标系下融合（加权平均，权重=置信度）
3. 用融合后的 pointmap 作为 SAM3D 的几何条件
4. 仍然用单张图片作为纹理条件

**推荐**: 先实现方案 A（改动最小），验证效果后再考虑方案 B。

---

## 二十一、坐标系适应问题

### Q48: sp_refinement.py 的坐标系与 Stage5 的坐标系是否一致？

**回答**: **需要 Stage5 适应 sp_refinement.py 的坐标系约定。**

**关键事实**:
- `sp_refinement.py` 是原始项目的代码，使用 `transform_matrix[:3,1]`（y轴）作为物体的"上"方向
- `refine_embedded_in_wall_object` 和 `refine_attached_to_wall_object` 使用 `transform_matrix[:3,2]`（z轴）
- 这意味着 `sp_refinement.py` 的坐标系约定中，物体的"上"方向对应 T 矩阵的 y 轴列

**Stage5 需要做的适应**:
- Stage5 的 `sp_refine_on_top` 直接操作 z 轴（`bounds[0,2]`、`bounds[1,2]`），这在 z-up 坐标系下是正确的
- 但 Stage5 调用 `refine_supported_by_floor_object` 时，T 矩阵必须与 sp_refinement 期望的坐标系一致
- 如果 Stage5 引入了额外的坐标变换（如 Stage4 的 ICP 对齐），需要在调用 sp_refinement 之前确保 T 矩阵的坐标系正确

**当前问题**: Stage5 的 `refine_full_scene` 和 `refine_inter_object_relations` 直接使用 `all_instances` 中的 T 矩阵调用 sp_refinement 函数，没有检查坐标系是否一致。如果 Stage4 修改了 T 矩阵的坐标系，Stage5 可能需要先做坐标变换。

**修复方向**: 不改 sp_refinement.py，而是在 Stage5 调用 sp_refinement 之前，确保 T 矩阵的坐标系与 sp_refinement 期望的一致。

---

## 十一、RAS + HaWoR 坐标原点与平移量

### Q49: 两者的坐标原点都是怎么指定的？如何查看原点差距和平移量？

**回答**:

**1. RAS 原点指定方式**

RAS 的原点由 `align_to_room_coordinate_system()` 决定（`geometry_utils.py:264-276`）：

```python
# 地板设为 z=0
t[2] = -rotated_floor_centroid[2]

# 场景 x-y 中心设为原点
center = (min_coords + max_coords) / 2
t[:2] = -center[:2]
```

原点 = **地板平面上场景包围盒的中心**。

实测数据：`cam_pos[0] ≈ [0.00004, -0.00003, 0.00008]`，接近原点是因为相机恰好在场景中心附近。但 `cam_pos[0][2] ≈ 0.0001` 而非 1.0-1.7m，说明**相机也在地板附近**——这不对，应该是 VGGT 的尺度问题或相机模型差异。

**2. HaWoR 原点指定方式**

HaWoR 的原点由 DROID-SLAM 初始化决定：

```python
# hawor_slam.py:103
droid, traj = run_slam(imgfiles, masks=masks, calib=calib)

# custom_utils.py:133
t_c2w_sla = torch.tensor(pred_traj[:, :3]) * pred_cam['scale']
```

DROID-SLAM 初始化时，**第一帧相机位置设为原点** (0,0,0)，后续帧增量估计。

实测数据：`cam_original[0] ≈ [0.004, -0.004, -0.001]`，接近但不完全为零（有微小数值误差）。

**3. 原点差距的计算**

两个系统的原点不同，但它们处理同一个视频，第一帧相机位置是同一个物理点。因此：

```
RAS 相机[0] = s * R_total @ HaWoR 相机[0] + t

当 s=1, R_residual=I 时:
t = RAS_cam[0] - R_axis @ HaWoR_cam_original[0]
```

实测：`t = [-0.0042, 0.0009, -0.0037]`（约 5.6mm）

**4. 为什么差距这么小？**

因为两个系统的第一帧外参都接近单位矩阵：
- RAS: `||R_w2c[0] - I|| = 0.0005`
- HaWoR: `||R_c2w_original[0] - I|| ≈ 0.004`

这意味着两个系统的世界坐标系原点都接近第一帧相机位置，差距主要来自：
- VGGT 的房间对齐把原点从相机位置移到了场景中心
- 但在这个数据中，相机恰好在场景中心附近

**5. 如何查看原点差距**

```python
# 加载 RAS 相机位置
ext = np.loadtxt('extrinsics/0.txt')
ras_cam0 = -ext[:3,:3].T @ ext[:3,3]

# 加载 HaWoR 相机位置（恢复原始 SLAM World）
h = np.load('hawor_results_0_113.npz', allow_pickle=True)
Rx = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
hawor_cam0 = Rx @ h['t_c2w'][0]  # 逆 R_x

# 计算平移量
R_axis = np.array([[1,0,0],[0,0,1],[0,-1,0]])
t = ras_cam0 - R_axis @ hawor_cam0
print(f"原点差距 t = {t}")
print(f"原点距离 = {np.linalg.norm(t):.4f} m")
```

---

## 二十二、Floor 检测与坐标系对齐

### Q50: 当前 floor 检测的完整流程是什么？

**回答**: Floor 检测分为3个阶段：mask 生成 → 平面拟合过滤 → 坐标系构建。

**阶段1: SAM3 分割 floor mask** (`object_segmentation.py:40-49`)
- SAM3 以 `"floor"` 为文本提示，逐帧分割
- 过滤掉像素数 < 500 的小 mask
- 输出: `floor_masks = [{'frame_id': i, 'mask': mask}, ...]`

**阶段2: PCA 平面拟合 + 阈值过滤** (`geometry_utils.py:233-239`)
- 对每个 floor mask 内的 3D 点做 PCA 拟合平面
- 计算 `mean_distance`（所有点到拟合平面的平均距离）
- **阈值**: `mean_distance < floor_mean_distance_thres` (默认 0.02m) 的才保留
- mainv2.py 未传参，仍使用默认值 0.02（设计建议 0.04 但未生效）

**阶段3: 坐标系构建** (`geometry_utils.py:240-277`)
- **无合格 floor** → 返回恒等变换 (R=I, t=0)，不做任何坐标系对齐
- **有 floor 但无正交 wall** → 也返回恒等变换
- **有 floor + wall** → floor 法向量 = Z 轴，wall 确定XY轴，floor 质心 z=0

**关键问题**:
1. `floor_mean_distance_thres=0.02` 过严，laptop(0.0286)、cup(0.0296) 等场景的 floor 被丢弃
2. floor 检测失败时**静默回退**到恒等变换，无日志警告
3. 没有 wall 时即使有合格 floor 也不做对齐（设计缺陷，应至少用 floor 做 z 轴对齐）
4. mainv2.py 顶部 docstring 中已标注这些为"已设计但未实现"的逻辑

### Q51: floor_mean_distance_thres=0.02 和 0.04 的区别是什么？

**回答**: `mean_distance` 是 floor mask 内所有 3D 点到 PCA 拟合平面的平均距离，反映分割质量。

| 阈值 | 含义 | 影响 |
|------|------|------|
| 0.02 (2cm) | 只接受非常贴合平面的 floor | laptop(0.0286)、cup(0.0296) 的 floor 被丢弃 → 坐标系回退恒等变换 → z 轴可能反转 |
| 0.04 (4cm) | 接受稍有噪声的 floor | laptop/cup 的 floor 可通过 → 坐标系正确对齐 |

**为什么 0.02 太严**: VGGT-omega 的点云本身就有噪声（DenseHead 预测精度有限），导致 floor 点到拟合平面的距离偏大。但方向是正确的，只是平面拟合质量稍差。0.04 是一个更合理的阈值。

**当前状态**: mainv2.py L1 已传入 `floor_mean_distance_thres=0.04`。L1.5=0.04, L2=0.06。

### Q53: 为什么 SAM3 无法识别剪刀 (scissor)？跨类去重合并的原因

**回答**: 分析 cup 场景日志 (092_C9_Cup):

```
Stage 1: 发现 3 个物体: eye, scissor, table
Stage 2: 跨类去重: 4 instances → 1 instances
  eye_0 + table_3 ← ov1=0.406>=0.3
  eye_1 + table_3 ← ov1=0.344>=0.3
  scissor_2 + table_3 ← ov1=0.327>=0.3
```

**根因**: 不是 SAM3 无法识别剪刀，而是**跨类去重把 eye/scissor 合并到了 table**。

原因分析:
1. **SAM3 video tracking 分割了 eye/scissor/table** — Stage 2 确实检测到了这些物体
2. **3D 空间重叠度 (overlap) 过高** — scissor_2 与 table_3 的 ov1=0.327 >= 0.3 阈值
3. **小物体被大物体"吞并"** — 放在桌上的小物体（eye, scissor）在 3D 空间中与桌面高度重叠

**可能的解决方案**:
- 降低跨类去重阈值 (当前 0.3)，或对小物体使用更严格的阈值
- 在跨类去重前，按物体大小排序，小物体优先保留
- 增加"面积比"判断：如果 A 面积远小于 B，即使 overlap 高也不合并

### Q54: "接触不足1个，需人工确认" 的产生原因

**回答**: 分析 laptop 场景日志:

```
⚠️ 稳定性检查: 2 个不稳定 (接触不足 1 个, 需人工确认)
```

**代码逻辑** ([refine_inter_object_placement.py:1256-1366](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/tools/refine_inter_object_placement.py#L1256-L1366)):

1. `check_stability()` 对每个"supported by other objects"的物体:
   - 计算物体底面与支撑物顶面的 2D 投影重叠面积
   - `support_ratio = overlap_area / supported_area`
   - 如果 `support_ratio < contact_threshold (0.2)` → "接触不足"

2. **laptop 场景的具体情况**:
   - laptop_0 被 table_1 支撑，但 on_top 精修被跳过 (z_offset=0.3586m > 0.3m)
   - laptop_0 没有被正确放在 table_1 上 → 2D 投影重叠面积很小
   - table_1 本身穿入地面 -0.573m (基础精修未修复)

3. **根因**: Stage 5.2 的 `sp_refine_on_top` 阈值太保守 (0.3m)，导致 laptop 没有被调整到 table 上

4. **解决方案**: 需要改进 sp_refine_on_top 的决策逻辑（使用 initial_offset 判断），或降低阈值

---

### Q55: Stage5 不应动基础精修过的物体

**问题**: hoi4d 场景中，`toy_0` 的关系是 "supported by floor"（已被基础精修处理），但 Stage5.2 仍然调整了它

**根因**:
1. `refined_relations` 中有 `toy ← table`（类别级别，不带编号）和 `toy_0 ← supported by floor`（实例级别）
2. `supported_pairs` 过滤条件只排除了 `"floor" in rel`，但 `toy ← table` 不含 "floor"，所以进入精修列表
3. SP 精修时 `_find_supporter_instances("toy")` 返回 toy 类别的**所有实例**（包括 toy_0）
4. toy_0 被当作 "supported by table" 处理，z 轴被调整 +0.64m

**修复**:
- SP 精修循环中，用实例级别 key (`toy_0`) 查找 `refined_relations`，如果关系包含 "floor"/"wall"/"embedded" 则跳过
- `check_stability` 中同样用实例级别 key 查找关系

---

### Q56: SAM3D 生成物体和地面对应不上的原因分析

**现象**: 桌子等物体在 GLB 中穿入地面或悬浮，不符合物理环境

**原因分析** (按影响程度排序):

1. **VGGT 点云尺度不一致** (最关键)
   - VGGT 从视频重建的 3D 点云是相对尺度，不同帧的深度估计可能有尺度漂移
   - 同一物体在不同帧中的 3D 位置可能不一致，导致提取的 mesh 尺寸和位置偏差
   - 表现: 桌面比实际高/低 0.1-0.3m，物体整体偏大/偏小

2. **SAM3D 实例分割的时间不一致性**
   - SAM3D video tracker 逐帧跟踪物体，但遮挡/运动导致 mask 在某些帧丢失或偏移
   - 从不同帧提取的 3D 点可能属于物体的不同部分，导致 mesh 不完整或偏移
   - 表现: 桌腿缺失、桌面倾斜、物体只包含正面没有背面

3. **坐标系对齐误差**
   - 如果 floor mask 的 PCA 平面拟合不准（mean_distance > 0.04），坐标系 z 轴偏斜
   - z 轴偏斜 5° 就会导致 1m 远处 0.09m 的高度误差
   - 表现: 桌子一侧高一侧低，整体倾斜

4. **最优帧选择偏差**
   - Stage3 选择 median_disp 最小的帧作为最优帧，但该帧可能不是物体最完整的视角
   - 从该帧重建的 mesh 可能缺少底部（被遮挡），导致 z_min 偏高
   - 表现: 桌子悬浮在地面之上

5. **基础精修的 theta_gravity 判断**
   - `refine_supported_by_floor_object` 用 theta_gravity 判断物体是否倾斜
   - 如果 theta_gravity < 60°（物体接近水平），不调整 z 位置
   - 但 VGGT 重建的物体可能 theta_gravity 正常但 z 位置不对
   - 表现: 桌子水平但悬浮/穿入地面

6. **跨帧 mesh 融合问题**
   - TSDF 融合多帧深度图时，如果相机位姿有误差，融合结果会有重影/偏移
   - 表现: 物体边缘模糊、双轮廓

### Q57: 桌面柜子位置正确但与地面不平行，原因是什么？

**现象**: 所有桌面/柜子类物体的位置(x,y)正确，但都与地面不平行（倾斜）

**根因**: `sp_refinement.py` 的 `refine_supported_by_floor_object` (L76) 只在 `theta_gravity < 10° or > 170°` 时对齐旋转。10°-170° 范围的物体直接设 `upper_align_matrix = np.eye(4)`（不做旋转对齐），但仍然做了 z 轴贴合（底面放到 z=0）。结果：物体底面在 z=0 但朝向仍然是倾斜的。

**修复**: 在 `check_stability` 的 Phase 1 中，对所有 `theta_gravity > 1° and < 179°` 的 floor 物体做旋转对齐（之前阈值是 10°-170°，现在放宽到 1°-179°），然后重新做 z 轴贴合。

**为什么基础精修不修改**: 用户要求不修改 `sp_refinement.py`，所以在 check_stability 阶段补上旋转对齐。

### Q58: 物体悬浮问题如何在 SP 后的第四阶段修复？

**现象**: 经过 SP 精修后，部分物体仍然悬浮在支撑面上方

**原因**:
1. SP 精修 (`sp_refine_on_top`) 有 `max_offset=0.3` 限制，超过 0.3m 的偏移不修复
2. 基础精修可能抬升了 supporter，但 supported 物体没有跟着调整
3. 旋转对齐后 bottom_z 发生变化，但之前的 z 贴合是基于旧的旋转

**修复**: 在 `check_stability` 中添加 Phase 4（最终 z 轴强制贴合）:
- 对 floor 物体: 确保 `bottom_z = 0`（无论 Phase 1 是否已修复）
- 对 supported 物体: 确保 `bottom_z = supporter_top_z`（无论 Phase 2/3 是否已修复）
- 阈值极低 (0.001m)，几乎任何间隙都会被修复

**check_stability 四阶段结构**:
- Phase 1: 地面物体旋转对齐 + z轴贴合
- Phase 2: 支撑物体间隙检测 + 悬空修复
- Phase 3: 接触不足检测 + z轴修复
- Phase 4: 最终 z 轴强制贴合（兜底）

---

## 二十三、参考论文分析

### Q59: "Do as I Do" (malik-group, arXiv:2606.19333) 对本项目的参考价值有哪些？

**论文概述**: "Do as I Do" (UC Berkeley, Jitendra Malik 组, 2026-06) 是一个从单目 RGB 人手-物体交互视频中重建+重定向到灵巧手机器人的完整管线。其 `reconstruction/` 子模块专注于手物交互重建和 6-DoF 物体位姿跟踪，`retargeting/` 子模块将重建结果重定向到机器人手。

**与本项目的关系**: 两个项目目标高度重叠——都从人手-物体交互视频出发，重建 3D 场景/物体，最终在仿真器中复现操作。但侧重点不同：本项目侧重**场景级重建**（多物体+空间关系），Do as I Do 侧重**手物交互级重建**（单物体+6DoF跟踪+retargeting）。

**参考价值分析（按优先级排序）**:

#### 1. 物体 6-DoF 位姿跟踪方法（最高价值，直接对应 Stage4 改造）

Do as I Do 的 reconstruction 管线核心创新是 **guided diffusion for 6-DoF tracking**：

| 阶段 | Do as I Do 方法 | 本项目当前方法 | 可借鉴点 |
|------|----------------|-------------|---------|
| 物体分割 | SAM3 (click + text) | SAM3 (text) | 已一致 |
| 3D mesh 生成 | SAM3D | SAM3D | 已一致 |
| 点图估计 | MoGe pointmaps | VGGT pointmaps | MoGe 更稳定但需额外模型 |
| 重力估计 | GeoCalib | SAM3 floor/wall PCA | GeoCalib 更鲁棒 |
| 速度跟踪 | TAPIR (TapNet) | VGGT4D TrackHead | TAPIR 更成熟，可替换 |
| 6DoF 跟踪 | **Fast-SAM3D guided diffusion** | ICP + Umeyama | **核心差异，见下文** |
| 平移/尺度优化 | 独立 optimize_translation_scale | Stage4 combined_alignment | 可参考其优化目标函数 |

**核心差异**: 本项目 Stage4 用 ICP + Umeyama 对齐 VGGT 点云，而 Do as I Do 用 **Fast-SAM3D 的 guided diffusion** 做物体跟踪。后者直接在视频帧上工作，天然与视频像素对齐，避免了 VGGT 点云噪声问题。

**具体可借鉴**:
- 用 TAPIR 替代 VGGT4D TrackHead 做点跟踪（更鲁棒，有独立 conda env）
- 用 Fast-SAM3D 的 guided diffusion 替代 ICP 做 6DoF 跟踪
- 用 GeoCalib 替代 PCA 做重力方向估计（更鲁棒，不依赖 wall mask）

#### 2. 手部重建与坐标系对齐（高价值，对应 HaWoR 集成）

Do as I Do 使用 HaWoR 做手部重建，与本项目相同。但其**手-物坐标系对齐**方法值得借鉴：

| 问题 | Do as I Do 解法 | 本项目当前状态 |
|------|----------------|-------------|
| 手-物深度对齐 | MoGe pointmaps 提供统一深度参考 | RAS 和 HaWoR 坐标系独立，需手动对齐 |
| 手-物相对位姿 | 优化 translation + scale 使手-物一致 | object_tracking 中有运动耦合检测但无深度对齐 |
| 重力对齐 | GeoCalib → camera-frame up direction | PCA 拟合 floor/wall 平面 |

**可借鉴**: 在 `object_tracking/` 管线中，用 MoGe pointmaps 统一 RAS 和 HaWoR 的深度参考，避免当前的手动坐标系对齐。

#### 3. Retargeting 管线设计（高价值，对应 EGO_VIDEO_TO_SIM_ROADMAP）

Do as I Do 的 `retargeting/` 管线与本项目路线图中的 Phase 2 高度对应：

| 步骤 | Do as I Do | 本项目路线图 |
|------|-----------|------------|
| 凸分解 | 凸分解物体 mesh | 未实现 |
| MJCF 场景生成 | 自动生成 MuJoCo XML | SAPIEN 场景构建 |
| IK | MuJoCo Warp IK | Galaxea bimanual_relaxed_ik |
| 运动规划 | Sampling-based MPC in MuJoCo Warp | 未实现 |
| 物理仿真 | MuJoCo Warp | SAPIEN |

**可借鉴**:
- **凸分解**: 物体 mesh 的凸分解是物理仿真的前提，本项目 `physics_validator.py` 直接用原始 mesh，碰撞检测不准确
- **Sampling-based MPC**: Do as I Do 用 MuJoCo Warp 做 sampling-based MPC 生成机器人轨迹，比纯 IK replay 更鲁棒
- **Warmup + force perturbation + transition reward**: 三个关键 trick 解决常见失败模式（项目页有可视化）

#### 4. 管线架构设计（中等价值）

Do as I Do 的管线架构值得参考：

| 设计决策 | Do as I Do | 本项目 | 评价 |
|---------|-----------|-------|------|
| 模块化 | 每个阶段独立 conda env + shell 脚本 | 单一 Python 进程 | Do as I Do 避免 CUDA 冲突更彻底 |
| 配置管理 | `config/paths.sh` 集中管理路径 | 硬编码路径 | Do as I Do 更灵活 |
| 子模块管理 | git submodules + fork | 直接集成 | Do as I Do 便于跟踪上游更新 |
| 中间产物 | `layout.json → layout_camera_frame.json → layout_camera_frame_optimized.json` | `all_instances.pkl` | Do as I Do 的渐进式 JSON 更透明 |

**可借鉴**: 将 Stage4 的中间结果也保存为渐进式 JSON（初始 → 对齐 → 优化），便于调试和断点续跑。

#### 5. 不适用的部分

| Do as I Do 特性 | 不适用原因 |
|----------------|----------|
| 单物体假设 | 本项目需要多物体场景重建 |
| Click-based SAM3 GUI | 本项目用 text prompt，更适合自动化 |
| MuJoCo Warp | 本项目用 SAPIEN，不需要换仿真器 |
| MANO 手部模型直接 retarget | 本项目目标是 Galaxea 二指夹爪，不是灵巧手 |
| 32GB VRAM 要求 | 本项目已有自己的 GPU 需求 |

#### 总结：最值得立即借鉴的 3 件事

1. **用 TAPIR 替代 VGGT4D TrackHead 做点跟踪** — TAPIR 是 Google DeepMind 出品，比 VGGT4D 的 TrackHead 更成熟稳定，且有独立 conda env 避免 CUDA 冲突
2. **用 GeoCalib 替代 PCA 做重力估计** — GeoCalib 不依赖 wall mask，对 VGGT 点云噪声更鲁棒
3. **物体 mesh 凸分解** — 在 `physics_validator.py` 和 `scene_builder.py` 中加入凸分解，提升物理仿真碰撞检测精度
