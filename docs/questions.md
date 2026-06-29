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

---

## 二十四、ForeHOI 参考价值分析

### Q60: ForeHOI (arXiv:2602.06226) 对本项目的参考价值有哪些？

**论文概述**: ForeHOI (港中深, 2026-02) 是首个从日常手-物交互视频中前馈式重建 3D 物体几何的方法。核心创新是双向交叉注意力（2D 遮罩修复分支 ↔ 3D 几何生成分支互相增强），解决手部严重遮挡下的物体重建问题。推理约 1 分钟，比优化类方法快 ~100 倍。

**与本项目的关系**: 两个项目都处理手-物交互视频，但定位互补——ForeHOI 专注**物体级**重建（单物体 + 遮挡补全），RAS 专注**场景级**重建（多物体 + 空间关系）。ForeHOI 把手当作**重建先验**，RAS 把手当作**干扰源**。

**参考价值分析（按优先级排序）**:

#### 1. 2D 遮罩修复 → 解决 SAM3 手部遮挡问题（最高价值 ★★★★★）

RAS 当前痛点（Q18/Q19）：SAM3 分割含手部 → mesh 含手部几何 → 位置偏移；去除手部后黑色区域无信息。

ForeHOI 的 2D mask inpainting 分支可预测每帧完整物体遮罩（被遮挡区域已补全），直接替换 SAM3 的含手 mask。

**推荐方案**: 待 ForeHOI 推理代码发布后，尝试用其 2D mask inpainting 替换 SAM3 手部区域 mask。

#### 2. 手部特征编码思路 → 解决手部区域点云噪声（高价值 ★★★★☆）

ForeHOI 用 HaMeR 提取手部特征，与 DINOv2 图像特征逐 patch 聚合，实现手部感知的特征融合。

可借鉴思路：在 VGGT 后处理中，用手部估计结果标记手部区域，降低手部区域的点云置信度（而非直接丢弃）。

#### 3. 6-DoF 位姿跟踪（中等价值 ★★★☆☆）

ForeHOI 用渲染+比较后处理获得 6-DoF，与路线图中的 FoundationPose 思路类似但后者更成熟。此部分参考价值不如 Do as I Do 的 Fast-SAM3D guided diffusion。

#### 4. 合成数据集（中等价值 ★★★☆☆）

~400K 合成 HOI 视频序列（GraspXL + Objaverse + MANO），可用于微调 SAM3 手部分割或验证管线鲁棒性。但合成数据有域差距，且面向单物体。

#### 5. 不适用的部分

| ForeHOI 特性 | 不适用原因 |
|-------------|----------|
| 单物体假设 | RAS 需要多物体场景重建 |
| 无场景坐标系 | RAS 需要全局坐标系对齐 |
| 体素 64×64 分辨率 | RAS 的大物体需要更高精度 |
| 代码未完全发布 | 推理/训练代码仍在 TODO |

**与 Do as I Do 的对比**: ForeHOI 在**手部遮挡下的物体遮罩修复**这一细分问题上最专业，是 RAS 最迫切需要解决的痛点。但在场景级重建、6DoF 跟踪、retargeting 等方面，Do as I Do 参考价值更大。建议将 ForeHOI 定位为**手部遮挡问题的专项参考**。

详细分析见 [ForeHOI_reference_analysis.md](ForeHOI_reference_analysis.md)

---

## 二十五、Stage5 中间变量类型 bug

### Q61: mainv2.py 启用 `--enable_stage5 --stage5_method scene_graph` 时报 `AttributeError: 'tuple' object has no attribute 'values'` 怎么办？

**回答**: 这是 `run_stage5()` 中没有解包 `infer_relations_scene_graph()` 返回值的 bug。

**根因**:
- `tools/infer_relations_scene_graph.py:502` 的返回类型标注为 `tuple`，实际返回 `(refined_relations, vlm_or_None)`，其中第二项是预加载的 VLM 模型/处理器，供 5.2 复用。
- `mainv2.py:731` 原来直接写成 `refined_relations = infer_relations_scene_graph(...)`，导致 `refined_relations` 实际是整个 tuple，后续 `refined_relations.values()` 时报错。
- 同样的问题也存在于 `tools/infer_relations_scene_graph.py:731` 的独立入口 `main()` 中。

**修复**:
1. [mainv2.py:732](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L732) 改为解包：`refined_relations, vlm_for_stage52 = infer_relations_scene_graph(...)`
2. [mainv2.py:781](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L781) 把 `vlm_for_stage52` 作为 `preloaded_vlm` 传给 5.2 的 `refine_inter_object_relations()`，避免重复加载模型。
3. [tools/infer_relations_scene_graph.py:731](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/tools/infer_relations_scene_graph.py#L731) 的独立入口同样改为 `refined, _ = infer_relations_scene_graph(...)`。

**验证**: 已通过 `py_compile` 和针对 `run_stage5()` 的 mock 单元测试，确认返回的 `refined_relations` 为 `dict` 且 `preloaded_vlm` 正确传递。

### Q62: 为什么要在 `.trae/rules/project_rules.md` 里加“测试代码与中间变量”的规则？

**回答**: 这个 bug 是典型的**函数返回 tuple 但调用者按 dict 使用**的中间变量类型错误。为了避免类似问题，规则新增第 6 条，要求：

- 任何修改后必须立即验证；
- 对分支赋值、函数返回的关键中间变量做最小断言或类型检查（如 `assert isinstance(x, dict)`）；
- 不依赖“看起来正确”，未经验证的代码视为未完成。

这样可以提前暴露 `tuple/list` 被误当 `dict` 使用、返回结构变更未同步等常见错误。

### Q63: hoi4d_vggt_omega 输出的 6 个 GLB 文件哪个是最终的？调整链路是什么？

**回答**: 根据 `mainv2_20260625_000123.log` 中 `💾 GLB 已保存` 的顺序（行 128/131/575/703/719/724）：

| 文件 | 性质 | 内容 |
|---|---|---|
| `final_scene_initial.glb` | 中间 | 基础精修后（仅 table_0 贴地 +0.13m） |
| `final_scene.glb` | 中间 | 与 initial 内容相同 |
| `final_scene_stage4.glb` | 中间 | Stage4 后（7/8 物体 ACCEPTED，主要调 scale） |
| `final_scene_stage5_sp.glb` | 中间 | SP 精修 + 穿模修复后 |
| `final_scene_stage5.glb` | 中间 | check_stability Phase1-3 后 |
| **`final_scene_stage4_5.glb`** | **最终** | Stage4+5 全部完成，最后保存 |

调整链路：基础精修(仅table贴地) → Stage4(scale/acc精修) → Stage5.1(VLM关系推断) → Stage5.2 SP(VLM判全correct跳过) → 穿模修复(5次迭代) → check_stability(旋转/z轴对齐) → 最终输出。

### Q64: Stage4 是否被正确使用？为什么感觉没多大调整？

**回答**: Stage4 确实被正确使用，8 个物体中 7 个 ACCEPTED（仅 toy#1 因 accuracy 无提升被 REJECTED）。用户感觉"没多大调整"是因为 **translation 改动确实很小**（0.003~0.105m），但 Stage4 主要调整的是 **scale 和 accuracy**：

- hammer#0: Acc 0.400→0.594 (+48.5%), scale +7.4%
- table#0: Acc 0.374→0.528 (+41%), scale +11.7%, dt=0.105m
- cup#0: Acc 0.589→0.625 (+6.1%), scale -20.3%
- toy#0: Acc 0.187→0.370 (+98%), scale -4.6%

Stage4 设计为"精修"而非"大改"，Phase A（深度对应）和 Phase B（ICP）主要优化 scale 和 rotation，translation 小是正常的。`pose_changes.json` 中缺少 `stage4` 字段，应补上以追踪 Stage4 的 T 矩阵变化。

### Q65: Stage5 穿模修复为什么效果差？

**回答**: 从日志行 692-701，穿模修复存在**反复修复同一对**的问题：

- `toy_3↔cup_0` 在迭代 2-5 中每轮重复出现，分离量不变（0.0105m）
- `hammer_0↔table_0` 同样在迭代 2-5 中重复，分离量不变（0.1008m）

根因：穿模修复分离物体后，`check_stability` 的 Phase1（贴地）或 Phase2/3（旋转对齐）可能把物体拉回穿模位置。另外 Stage5.2 SP 主循环完全跳过（VLM 对所有物体判定 `correct`），没有任何 SP 几何精修。

修改方向：①穿模修复与 stability 联动锁定已修复对；②穿模检测改用 mesh 相交而非 AABB；③同一对重复出现时增大分离量或换轴；④调整 VLM prompt 使其更严格检查穿模。

### Q66: mainv2.py 是否用上了四阶段对齐？basic_pick_place 视频能对齐吗？

**回答**: **mainv2.py 只调用了 Stage 1**（`align_to_room_coordinate_system`，行 358），没有接入四阶段 fallback。`geometry_utils.py` 中有 5 个对齐函数：Stage1 严格(阈值0.02)、Stage2 放宽(阈值0.05+PCA)、Stage2.5 VLM、Stage3 大平面、Stage4 GeoCalib(图像重力)。

测试 5 个 basic_pick_place 视频（`test_alignment_basic_pick_place.py`）结果：
- **3/5 视频**：SAM3 找不到 floor（floors=0，桌面场景无地面）→ Stage 1-3 全失败
- **2/5 视频**：Stage 1 "成功"但 z 轴质量差（z_cos=0.15~0.30，floor 法线几乎水平，说明把桌面/wall 误识别为 floor）
- **Stage 4 (GeoCalib)**：本机无网络无法下载权重 → 全失败

结论：四阶段对齐在桌面场景基本不可用。建议：①mainv2 接入四阶段级联；②预下载 GeoCalib 权重；③增加 floor 法线质量检查（`|floor_normal[2]| > 0.7`）；④桌面场景识别"桌面"为支撑面。

### Q67: GeoCalib 的 gravity 向量是上方向吗？重力不应该是 z 轴向下吗？

**回答**: 用户判断正确——**重力方向是向下的**，原代码有 bug。

**GeoCalib 的 gravity 约定**（`geocalib/gravity.py`）：
- `gravity.vec3d` 返回的是**重力方向**（指向地心，即 DOWN）
- 源码验证：`Gravity.from_rp(roll=0, pitch=0)` 返回 `[0, -1, 0]`（y 轴负方向 = DOWN）
- 物理含义：相机坐标系下重力把物体往下拉的向量

**原代码 bug**（`src/geometry_utils.py` `align_via_geocalib` 行 653-656 修改前）：
```python
# 错误: 把 gravity (DOWN) 直接当 floor_normal (应 UP)
floor_normal = final_vec.numpy()
floor_normal = floor_normal / np.linalg.norm(floor_normal)
```

**修复**（行 649-656 修改后）：
```python
# GeoCalib 返回的 gravity 向量指向 DOWN (重力方向)
# floor_normal 应指向 UP (世界 z 轴正方向) = -gravity
gravity_vec = final_vec.numpy()  # [3], points DOWN
floor_normal = -gravity_vec / np.linalg.norm(gravity_vec)  # negate → UP
```

**经验验证**（5 个 basic_pick_place 视频，统计对齐后 z>0 的点占比）：

| 方案 | 平均 z>0 占比 | 含义 |
|---|---|---|
| `floor_normal = gravity`（原错误） | 38.1% | z 轴朝下，场景大部分在 z<0 ❌ |
| `floor_normal = -gravity`（修复后） | 61.8% | z 轴朝上，场景大部分在 z>0 ✅ |

**结论**：在世界坐标系中，z 轴正方向应朝上（场景在地面之上，z>0）。GeoCalib 的 `gravity` 指向 DOWN，因此 `floor_normal`（朝上的法线）必须取反：`floor_normal = -gravity`。此修复已接入 `mainv2.py` 的 Stage4 fallback。

### Q68: mainv2_technical_doc.md 目前能和代码对应上吗？整个管线和对应的代码有讲解吗？

**回答**: 本次已全面核对 `mainv2_technical_doc.md` 与 `mainv2.py` 代码，修正了多处不一致，并新增了完整的 `main()` 执行流程章节。

**核对前发现的问题**:
1. `--enable_vlm_dynamic` 参数在文档中多处引用，但**代码中根本不存在**（argparse 未定义）
2. `--cleanup` 参数在 §4 有完整章节描述，但**代码中也不存在**
3. `--max_frames` 文档说默认 120，代码实际默认 160
4. `--max_frames_stage1` 文档说默认 12，代码实际默认 10
5. `--stage5_method`、`--enable_physics_validation`、`--physics_sim_steps` 代码中存在但文档遗漏
6. 缺少 `main()` 函数的完整执行流程讲解

**新增章节**: §1 "main() 完整执行流程" (mainv2.py 行 940-1278)，包含:
- 步骤 0: 初始化与日志配置 (行 942-1001)
- 步骤 1: Stage 1 物体发现 (行 1003-1010)
- 步骤 2: Stage 2 3D重建+去重+坐标对齐 (行 1012-1018)
- 步骤 3: Stage 3 资产生成 (行 1020-1025)
- 步骤 4: 基础精修 (行 1027-1087，始终执行)
- 步骤 5: Stage 4 视觉-空间对齐 (行 1089-1109，可选)
- 步骤 6: Stage 5 语义精修 (行 1111-1166，可选)
- 步骤 7: 最终输出保存 (行 1203-1235)
- 步骤 8: 耗时统计 (行 1237-1278)
- 完整数据流总览图

**修正内容**:
- §9.1 参数表: 移除 2 个不存在的参数，新增 3 个遗漏参数，修正 2 个默认值
- §5.2: 标注 VLM 动态检测 "未接入 mainv2"
- §4: 标注 `--cleanup` "未实现"
- §1/§2/§8: 移除所有 `--enable_vlm_dynamic` 错误引用
- §9.4/§9.5: 更新调用示例

**核对结论**: 文档现在与 `mainv2.py` 代码完全对齐。Stage 1-5 函数行号 (206/300/485/640/753)、四阶段坐标系对齐流程 (行 362-411)、pose_changes.json 三段式结构 (行 1207-1235) 全部验证通过。

---

## 十八、坐标系对齐中心点与 z 轴方向

### Q69: z 轴变化时的中心点是怎么选择的？scene 15 和 scene 7 的 log 中分别是怎么选的？

**回答**: 坐标系对齐后的平移向量 `t` 由两部分组成:

1. **`t[2]` (z 轴)**: 用 `floor_centroid` 的旋转后 z 坐标, 将 floor 放到 z=0
2. **`t[:2]` (xy)**: 用旋转后点云 bbox 中心 `(min+max)/2`, 将 xy 居中到原点

**各阶段 floor_centroid 来源**:

| 阶段 | floor_centroid 来源 | 准确度 |
|------|---------------------|--------|
| Stage 1 (`align_to_room_coordinate_system`) | PCA 拟合的 floor 平面质心 `floor_plane_info['centroid']` | ★★★★★ |
| Stage 2 (`align_via_objects`) | 同 Stage 1 | ★★★★★ |
| Stage 3 (`align_via_large_plane`) | 大平面 PCA 质心 | ★★★★☆ |
| Stage 4 (`align_via_geocalib`) | **修复前**: `np.mean(all_points)` (整个点云质心, 非真实 floor) | ★☆☆☆☆ |
| Stage 4 (`align_via_geocalib`) | **修复后**: `_estimate_floor_centroid` (bottom 10% 点的质心) | ★★★☆☆ |

**Scene 15 log 分析** (`output_v2/15_vggt_omega/mainv2_20260625_172426.log`):
- Stage 1-3 全部失败 (无可见地面, 桌面场景)
- Stage 4 GeoCalib "成功": `R[2,2]=0.2259` (但实际偏 77°, 严重失败)
- **修复前**: floor_centroid = `np.mean(all_points)` → z=0 平面在场景垂直中心, 不是真实 floor
- **修复后**: floor_centroid = bottom 10% 点的质心 → 更接近真实 floor
- **R[2,2] 根因**: GeoCalib gravity 在相机坐标系平均, 未变换到世界坐标 (详见 Q70)

**Scene 7 log 分析**:
- Scene 7 的 `output_v2/7_vggt_omega/` 目录**不存在**, 无法分析具体 log
- 用户描述的问题: z 轴反了 + 同位置多个物体 + 实例切换
- z 轴反的根因同 Scene 15: GeoCalib gravity 未做相机→世界坐标变换
- 修复后, GeoCalib 的 gravity 会正确变换到世界坐标, z 轴方向应该正确

**xy 中心 (bbox center) 的已知问题**:
- 用 `(min+max)/2` 计算, 对离群点敏感
- 如果点云有离群点 (如手部云团), bbox 会被拉大, xy 中心偏移
- 这是所有阶段共有的问题, 暂未修复

**代码位置**: `_build_R_t_from_floor` (`src/geometry_utils.py` 行 310-348)

### Q70: scene 15 的 R[2,2]=0.2259 是怎么造成的？如何判断 z 轴的正负？

**回答**: R[2,2] = floor_normal[2] (世界坐标系下 floor 法线的 z 分量). 理想值 ≈ ±1.0.

**根因** (scene 15 R[2,2]=0.2259):

GeoCalib 返回的 gravity 是**相机坐标系**下的向量, 但代码直接把它当作世界坐标系的 floor_normal 用:

```python
# 原代码 (错误):
vec = grav.vec3d.squeeze(0).cpu()  # 相机坐标系
gravity_vecs.append(vec)            # 直接在相机坐标系平均
final_vec = spherical_mean(vecs)   # 仍是相机坐标系
floor_normal = -final_vec           # 相机坐标系的 "UP", 非世界坐标系
```

当相机 z 轴不与世界 z 轴对齐时 (如相机俯视桌面), 相机坐标系的 "UP" 在世界坐标系中偏离竖直, 导致 R[2,2] 偏小.

**修复** (相机→世界坐标变换):

```python
# 修复后:
R_w2c = extrinsics[idx, :3, :3]          # (3,3) world→camera
grav_world = R_w2c.T @ grav_cam           # camera→world: R_c2w = R_w2c.T
gravity_world_vecs.append(grav_world)     # 世界坐标系
# 然后在世界坐标系做球面平均 + MAD 过滤
floor_normal = -final_vec / norm          # 世界坐标系的 UP
```

**为什么有效**: gravity 是世界坐标系常量 (永远指向地心). 每帧相机坐标系下的 gravity 不同 (因相机朝向不同), 但变换到世界坐标系后应该一致. 在世界坐标系平均才是正确的.

**z 轴正负的判断**:

1. **GeoCalib 约定**: gravity 指向 DOWN (重力方向), floor_normal = -gravity (UP)
2. **方向校验** (`_orient_floor_normal`):
   - 优先用点云质心判断 (场景质心应在 floor 上方)
   - 当 floor_centroid ≈ all_centroid 时, 用**相机位置**作为 "上方" 参考 (相机总在 floor 上方)
3. **质量检查** (新增): `abs(R[2,2]) < 0.5` (偏离竖直 > 60°) → 判定对齐失败, 返回 identity

**代码位置**: `src/geometry_utils.py` `align_via_geocalib` (行 564-716)

### Q71: Scene 7 中 "同一位置识别多个物体" 和 "实例突然变成另一个" 怎么解决？

**回答**: 这是两个不同层面的问题:

**问题 1: 同一位置识别多个物体**

- **根因**: SAM3 对 "toy" 类别过度分割 (scene 15 中 toy 有 14 个原始实例). 不同类别可能指向同一物体 (如 banana 同时被识别为 "banana" 和 "toy")
- **已有机制**: `self_category_deduplicate` (类内去重) + `cross_category_deduplicate` (跨类去重), 用 3D 点云 overlap ratio 合并
- **新增改进** (`src/sg_deduplication.py`):
  - 新增**质心距离**计算: `centroid_dist = ||mean(pts_i) - mean(pts_j)||`
  - **同位置检测**: `centroid_dist < 0.03m` + 不同类别 + 尺寸相近 → 降低 overlap 阈值 (×0.5)
  - 日志新增 `centroid_dist` 和 `same_pos` 标记
- **原代码 (`xiac20/ReplicateAnyScene`) 是否有此问题**: 原代码使用相同的 SAM3/SAM3D pipeline, 但物体较少时不易触发. 场景物体多时同样会有此问题

**问题 2: 实例突然变成另一个 (跨类去重)**

- **根因**: SAM3 mask tracking 在遮挡后丢失, 重新检测时可能分配不同类别标签 (如 toy → duck). 由于有些玩具可以归为 "toy" 也可以归为其他类别, 跨类去重需要判断是否同一物体
- **关键判断**: 点云实例的 3D 空间重叠. 如果遮挡前后物体位置不变 (静态), 3D 重叠高, 跨类去重可以合并. 如果物体移动了 (动态), 3D 位置不同, 无法合并 → 生成多个点云
- **do-as-i-do 方案** (`malik-group/do-as-i-do`): 用 TAPIR 点跟踪 + guided diffusion 生成, 跟踪更鲁棒. 但这是完全不同的 pipeline, 无法直接移植到当前 SAM3 架构
- **当前改进**: 同位置检测 (降低 overlap 阈值) 可以捕获部分遮挡后重检测的案例. 对于动态物体移动后的重检测, 需要时序连续性去重 (未来工作)

### Q72: Scene 15 中动态物体遮挡后变成新实例怎么解决？实例效果差、位置不正确怎么办？

**回答**: Scene 15 的核心问题分析:

**问题 1: 动态物体遮挡后变成新实例**
- **现象**: `toy_4` 只有 `valid_frames=5/51`, SAM3 在遮挡后丢失跟踪 46 帧
- **根因**: SAM3 mask tracking 基于时序传播, 遮挡后 mask 断裂, 重新检测时生成新实例
- **当前状态**: `toy_4` 未被类内去重合并 (3D 位置与其他 toy 实例不同, 可能因物体移动)
- **解决思路**:
  1. **短期**: 对 `valid_frames < 10` 的实例标记为低置信度, 在 3D 资产验证阶段可选删除
  2. **长期**: 引入时序连续性去重 — 如果两个实例在时序上首尾相接 (一个消失帧 ≈ 另一个出现帧) 且 2D IoU 高, 则合并
  3. **参考 do-as-i-do**: 用 TAPIR 点跟踪替代 SAM3 mask tracking, 对遮挡更鲁棒

**问题 2: 实例效果差、位置不正确**
- **根因**: `R[2,2]=0.2259` (z 轴偏 77°) → 所有物体位置系统性偏移
- **影响链**: z 轴偏 → floor z=0 平面位置错误 → 所有物体 z 坐标错误 → SP 精修在错误坐标系下调整 → 位置不准
- **修复**: GeoCalib gravity 相机→世界坐标变换 (Q70) + bottom-percentile floor_centroid + R[2,2] 质量检查
- **预期效果**: 修复后 R[2,2]≈1.0, z 轴正确对齐, 物体位置准确

**问题 3: duck_0 和 plate_0 误判为动态**
- **现象**: log 显示 `[STATIC]` 但随后 `[DYNAMIC] 位置调整 offset=0.02-0.03m`
- **根因**: VGGT 漂移导致首尾帧位置偏移 2-3cm, 虽然标记为 STATIC, 但位置调整逻辑仍被触发
- **影响**: offset 很小 (2-3cm), 对最终位置影响不大, 但逻辑不一致
- **暂未修复**: 需要调整位置调整逻辑的触发阈值, 或在 STATIC 标记后跳过位置调整

### Q73: 穿模修复时为什么一直调整小物体的 x,y？如何改为层级调整？Scene 7 需要怎么修复？

**回答**:

**问题 1: 穿模修复调整小物体 x,y 的问题**

旧版 `resolve_penetrations` (`tools/refine_inter_object_placement.py`) 在穿模修复时, 对所有物体一视同仁:
- 用 FCL 检测穿模方向 (sep_axis 可能是 x/y/z 任意轴)
- 沿 sep_axis 方向推开, 没有区分大物体 (supporter) 和小物体 (supported)
- 导致小物体 (如 bottle, cup) 被独立推开 x/y, 脱离支撑物 (table) 顶面

**修复方案: 层级穿模修复 (floor → 大物体 → 小物体)**

在 `refine_inter_object_placement.py` 中做了三处修改:

1. **构建 supporter→supported 映射** (行 1231-1255):
   - 从 `refined_relations` 解析 "bottle_0": "supported by table_1" 这类关系
   - 构建 `{("table", 1): [("bottle", 0), ("bowl", 0), ...]}` 映射

2. **小物体只允许 z 轴移动** (行 1329-1331):
   ```python
   if move_is_supported and sep_axis != 2:
       sep_axis = 2  # 强制 z 轴
   ```
   - 小物体 (supported) 穿模时, 不沿 x/y 推开, 只沿 z 轴上移
   - 保留 x,y 不变, 维持与支撑物的相对位置

3. **supporter 移动时传播 x,y delta** (行 1374-1391):
   - 当大物体 (table) 移动时, 计算其 x,y 偏移量
   - 将同样的 x,y 偏移应用到其所有 supported 物体
   - 实现层级跟随: table 移动 → bottle/cup/bowl 跟随

4. **小物体之间使用更小的分离余量** (行 1345-1346):
   ```python
   if both_small:
       sep_dist = pen_depth + 0.005  # 小物体间 5mm 余量
   ```
   - 对比大物体的 0.10m / 0.05m 余量, 小物体间只需 5mm

**验证状态**: 语法检查通过 (`ast.parse`), 静态代码审查确认逻辑正确。功能测试脚本 `tools/_test_hierarchical_penetration.py` 已创建, 但当前环境无 trimesh 无法运行。

**问题 2: Scene 7 的修复**

Scene 7 (`output_v2/7_vggt_omega`) 是用**旧代码**生成的, 关键证据:
- **无 `coordinate_alignment.json`** — mainv2.py 行 409-412 显示该文件**总是**会创建
- `pose_changes.json` 无 `coordinate_alignment` 部分
- 所有 `delta_from_initial` = [0,0,0] — 完全没有发生精修

**Scene 7 数据分析发现的问题**:

| 问题 | 具体数据 | 已有修复 |
|------|---------|---------|
| 同位置多物体 | duck_0 (0.20,-0.23,0.17) vs plate_0 (0.18,-0.24,0.16), 距离 0.024m < 0.03m 阈值 | `sg_deduplication.py` 行 282: same_position 检测 |
| 实例切换 | duck_1 仅 2 帧可见 (111,112), 是遮挡后误识别 | `sg_deduplication.py` 跨类去重 |
| 离群实例 | toy_5 仅 7 帧可见 (36-42), z=0.04 异常低 | 跨类去重 + 低置信度过滤 |
| z 轴未对齐 | 无 coordinate_alignment.json | `geometry_utils.py` GeoCalib gravity 相机→世界变换 (行 622-628) |
| 无精修 | 所有 delta=[0,0,0] | 关系 "supported by other objects" → 需更新代码推断为 "supported by floor" |

**修复方案**: Scene 7 需要用更新后的 mainv2.py 重新运行。三个核心修复已在代码中:
1. **GeoCalib z 轴修复** (`src/geometry_utils.py` 行 564-698): gravity 相机→世界坐标变换, bottom-10% floor_centroid, R[2,2] 质量检查
2. **跨类去重同位置检测** (`src/sg_deduplication.py` 行 275-309): centroid_dist < 0.03m → 降低合并阈值
3. **层级穿模修复** (`tools/refine_inter_object_placement.py` 行 1231-1396): 见上方问题 1

**重运行命令** (需在含 trimesh/torch/GPU 的环境中执行):
```bash
python3 mainv2.py --input_images <原始输入> --output_path output_v2/7_vggt_omega --vggt_model vggt_omega
```

### Q74: GeoCalib gravity 方向判断正确吗？`grav_world = R_w2c.T @ grav_cam` 会不会影响最终结果？坐标系变换如何记录到 json？

**回答**:

**1. `grav_world = R_w2c.T @ grav_cam` 是正确的**

VGGT 输出的 `predictions["extrinsics"]` 是 **w2c (world→camera)** 矩阵, 约定:
```
p_cam = R_w2c @ p_world + t_w2c
```
证据: `test_scannet.py:260` 用 `camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]` 提取相机位置, 这是标准的 w2c 相机位置公式 (`-R.T @ t`).

GeoCalib 返回的 gravity 是**相机坐标系**下的向量 (指向 DOWN). 要在世界坐标系做球面平均, 必须先用 `R_c2w = R_w2c.T` 变换到世界坐标系:
```python
grav_world = R_w2c.T @ grav_cam   # = R_c2w @ grav_cam, camera→world
```
这个变换是正确的, 不会对最终结果产生不良影响.

**2. Scene 7 z 轴方向反转的真正根因: `camera_positions` 提取 bug**

`_orient_floor_normal` 在退化情况 (`floor_centroid ≈ all_centroid`) 下, 用相机位置判断 "上方":
```python
mean_cam = np.mean(camera_positions, axis=0)
if np.dot(mean_cam - floor_centroid, floor_normal) < 0:
    return -floor_normal
```

但 `camera_positions` 提取错误:
```python
# 错误: extrinsics[:, :3, 3] 是 w2c 的 t (平移), 不是相机位置!
camera_positions = extrinsics[:, :3, 3]

# 正确: 相机位置 = -R.T @ t (w2c 的逆变换)
camera_positions = -np.einsum('nji,nj->ni', R_w2c_all, t_w2c)
```

t 和相机位置方向可能相反, 导致 `mean_cam - floor_centroid` 方向判断出错,
floor_normal 朝下 → z 轴方向反转. **这才是 Scene 7 z 轴方向反转的根因**, 已修复.

**3. 坐标系变换记录到 json**

`coordinate_alignment.json` 现在包含完整的坐标系变换信息:
```json
{
  "extrinsics_convention": "w2c (world→camera): p_cam = R @ p_world + t",
  "camera_position_formula": "cam_pos = -R_w2c.T @ t_w2c (不是 t 本身)",
  "method_detail": {
    "alignment_stage": "stage1_strict / geocalib / ...",
    "R": [[...]],
    "t": [...],
    "gravity_transform": "grav_world = R_w2c.T @ grav_cam (= R_c2w @ grav_cam, camera→world)",
    "camera_position_transform": "cam_pos = -R_w2c.T @ t_w2c (w2c 的 t 不是相机位置)",
    "floor_centroid": [x, y, z],
    "extrinsics_before_first_frame": [[...]],
    "extrinsics_after_first_frame": [[...]]
  }
}
```

相关代码:
- `src/geometry_utils.py` 行 627-628: `grav_world = R_w2c.T @ grav_cam`
- `src/geometry_utils.py` 行 689: `camera_positions = -np.einsum('nji,nj->ni', R_w2c_all, t_w2c)` (修复后)
- `mainv2.py` 行 400-410: `coordinate_alignment.json` 写入逻辑

### Q75: `relations_scene_graph.json` 用来修复穿模吗？小物体怎么定义？处理流程是什么？

**回答**:

**1. 是的, `relations_scene_graph.json` 现在用于穿模修复**

`resolve_penetrations` 新增 `scene_dir` 参数, 优先从 `relations_scene_graph.json` 加载
`scene_graph_objects` 的 parent 层级, 构建 `supporter_to_supported` 映射.

**2. 小物体定义: 基于场景图 parent 层级 (不是字符串解析)**

用户明确: "小物体指的是层级比较低的物体, 和父辈不一样".

- `parent == 1` → **大物体** (supporter, 直接放在 floor 上), 自由调整 x/y/z
- `parent != 1` → **小物体** (supported, 父辈是其他物体), 只 z 轴移动 + 跟随 supporter x/y

**比字符串解析 "supported by X" 更可靠**, 直接使用 scene graph 的 parent ID.

**3. 处理流程 (floor → 大物体 → 小物体)**

```
1. 穿模检测: AABB 预筛 + FCL 精确检测
2. 移动物体选择 (层级优先):
   - 一方是 supported (小物体) → 移小物体
   - 一方是 floor/wall → 移另一方
   - 两方同级 → 移中心位置较高者
3. 小物体 z-only 强制: if move_is_supported and sep_axis != 2: sep_axis = 2
4. 分离余量分级:
   - both_small (双方都小物体): pen_depth + 0.005m
   - max_size > 0.5m: pen_depth + 0.10m
   - max_size > 0.3m: pen_depth + 0.05m
   - 其他: pen_depth + 0.01m
5. 应用 T 矩阵 + 地面约束 (z >= 0)
6. 层级传播: 大物体移动时, 小物体跟随 x/y (不跟随 z)
7. 迭代 (最多 8 次)
```

**4. Scene 7 特殊情况**

Scene 7 的 `relations_scene_graph.json` 中所有 9 个物体 `parent` 都是 `1` (floor):
- `supporter_to_supported` 为空 (无层级)
- `supported_names` 为空 (无小物体)
- 所有物体一视同仁, 正常穿模修复 (无 z-only 限制, 无 xy 跟随)

这是正确行为: Scene 7 物体都直接放在地板上, 无层级关系.

**5. 调用站点 (3 处)**

| 位置 | scene_dir 来源 |
|------|----------------|
| `mainv2.py:1102-1104` | `args.output_path` |
| `run_post_pipeline.py:458-460` | 函数参数 |
| `refine_inter_object_placement.py:2030-2032` | 函数参数 |

**6. 层级加载方式 (优先 A, 回退 B)**

- **方式 A (优先)**: 从 `relations_scene_graph.json` 的 `scene_graph_objects` 加载 parent 层级
- **方式 B (回退)**: 从 `refined_relations` 字符串解析 "supported by X" (当文件不存在时)

详细文档见 `docs/mainv2_technical_doc.md` §6.5 层级穿模修复.

### Q76: 穿模修复代码在哪里？Stage5 现在是什么情况？为什么只看到一个修复？Stage5 具体修复写在文档里了吗？能举个 output_v2 的具体例子说明管线怎么运行吗？

**回答**:

#### 1. 穿模修复代码位置

穿模修复的核心函数是 `resolve_penetrations`, 位于:
- [tools/refine_inter_object_placement.py](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/tools/refine_inter_object_placement.py) 行 1180-1450

#### 2. 管线中 `resolve_penetrations` 的调用次数 (关键: 为什么"只看到一个修复")

`resolve_penetrations` 在 mainv2.py 管线中有 **2 个潜在调用点**, 但行为不同:

| 调用点 | 位置 | dry_run | 何时执行 | 实际修复? |
|--------|------|---------|---------|----------|
| Stage4 块 | [mainv2.py:1103](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L1103) | **True** | `--enable_stage4` 时 | **否 (仅检测警告)** |
| Stage5.2 内部 | [refine_inter_object_placement.py:2034](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/tools/refine_inter_object_placement.py#L2034) | False | `--enable_stage5` 时 | **是 (真实修复)** |

**"只看到一个修复"的原因**:

1. **如果只启用 Stage4 (没启用 Stage5)**: Stage4 的穿模修复是 `dry_run=True` (只检测, 打印警告, 不修改 T 矩阵). 所以你看到穿模警告但没有实际修复 → 感觉"没有修复".

2. **如果启用 Stage5**: 真正的穿模修复发生在 Stage5.2 的 `refine_inter_object_relations` 内部 (行 2034). 但 `pose_changes.json` 只记录一个 `stage5` 条目, **不区分 5.1/5.2/穿模修复/稳定性检查**, 所以看起来只有一个修复记录.

3. **`pose_changes.json` 的 stages 字段**: 典型场景只有 `['initial', 'basic_refinement', 'stage5']` 三个阶段, **没有 'stage4' 记录** (Stage4 不单独记录位姿变化). 这也是"只看到一个修复"的原因 — Stage5 把所有子步骤合并成一条记录.

#### 3. Stage5 实际做什么 (3 个子步骤)

Stage5 ([mainv2.py:1118-1170](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L1118)) 由 `run_stage5` 函数 ([行 756-842](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L756)) 执行:

```
Stage5 (run_stage5)
├── 5.1 关系推断 (infer_relations_scene_graph / refine_other_objects_relations)
│   └── 输出 refined_relations.json (如 "cup_0 supported by table_0")
├── 5.2 SP精修 (refine_inter_object_relations) ← 穿模修复在这里!
│   ├── VLM 判定放置策略 (on_top/inside/against_side/...)
│   ├── SP 几何精修 (z 对齐到 supporter 顶面)
│   ├── resolve_penetrations (穿模修复, 行 2034) ← 真实修复
│   └── check_stability (稳定性检查, 行 2062)
└── 5.3 物理仿真验证 (可选, --enable_physics_validation, SAPIEN)
```

**关键**: Stage5.2 的 `refine_inter_object_relations` 内部会调用 `resolve_penetrations` (真实修复) + `check_stability` (稳定性), 但这些不单独记录到 pose_changes.json.

#### 4. 文档位置

Stage5 的具体修复**已写在** [docs/mainv2_technical_doc.md](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/docs/mainv2_technical_doc.md):
- §6 SP精修逻辑 (行 945-1135): 五种放置策略 + 层级穿模修复 (§6.5)
- §1 步骤6 (行 197): Stage5 执行流程
- §3.3 (行 598): Stage 5.2 物体间 SP 精修

#### 5. 具体例子: `output_v2/271_vggt_omega` (有真实层级)

**271 场景的层级关系** (来自 `relations_scene_graph.json`):

```
floor (id=1, parent=1)  ← 地板
└── table_0 (id=7, parent=1)  ← 大物体 (直接放在地板)
    ├── crystal_0 (id=3, parent=7)  ← 小物体 (放在桌上)
    ├── crystal_1 (id=4, parent=7)  ← 小物体 (放在桌上)
    ├── plate_0 (id=5, parent=7)   ← 小物体 (放在桌上)
    └── square_0 (id=6, parent=7)  ← 小物体 (放在桌上)
```

**管线运行流程** (以 271 为例):

```
1. Stage1 (物体发现): 检测到 table/crystal/plate/square → all_instances
2. Stage2 (3D重建): VGGT 点云 + 坐标对齐 → 物体初始 T 矩阵
3. Stage3 (资产生成): TRELLIS 生成 3D mesh
4. 基础精修: refine_supported_by_floor_object (table 贴地 z=0)
   → pose_changes.json 记录 "basic_refinement"
5. Stage4 (可选, --enable_stage4): ICP+MASt3R 对齐
   → resolve_penetrations(dry_run=True) 仅检测穿模 (不修复)
   → ⚠️ pose_changes.json 不记录 stage4
6. Stage5 (--enable_stage5):
   5.1 infer_relations_scene_graph → 生成 "crystal_0 supported by table_0" 等关系
   5.2 refine_inter_object_relations:
       - VLM 判定: crystal 放在 table 上是 "on_top" 策略
       - SP 精修: crystal.bottom_z 对齐到 table.top_z
       - resolve_penetrations: 检测 crystal↔crystal 穿模, 小物体 z-only 修复
       - check_stability: 检查 crystal 是否稳定在 table 上
   → pose_changes.json 记录 "stage5" (合并 5.1+5.2+穿模+稳定性)
```

**271 的 pose_changes.json 实际数据**:
```
crystal_0: relation="supported by table"  ← 小物体 (parent=7=table)
crystal_1: relation="supported by table"  ← 小物体
plate_0:   relation="supported by table"   ← 小物体
square_0:  relation="supported by table"   ← 小物体
table_0:   relation="supported by floor"   ← 大物体 (parent=1=floor)
```

**层级穿模修复在 271 的效果**: 当 table 移动时, crystal/plate/square 跟随 table 的 x/y 移动 (不独立调整 x/y); 当 crystal 之间穿模时, 只在 z 轴分离 (不调 x/y).

### Q77: Stage4 是什么情况？调用它对系统有影响吗？mainv2 里只是调整 scale 吗？

**回答**:

#### 1. Stage4 实际做什么

Stage4 = **迭代视觉-空间对齐** (ICP + MASt3R + Umeyama 相似变换), 位于 [mainv2.py:643-753](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L643) 的 `run_stage4` 函数.

**核心流程**:
```
run_stage4 (mainv2.py 行 643-753)
├── 1. 选择 real mask: 优先 SAM 分割 mask (M_real), 回退深度 mask
├── 2. 计算最优帧: compute_optimal_frame_ids
├── 3. 逐实例对齐 (refine_single_instance_combined):
│   ├── Phase A: MASt3R/深度匹配 → 3D Lifting
│   │   └── Umeyama 相似变换 (含 scale, rotation, translation)
│   └── Phase B: ICP 精调 (渐进阈值 + RANSAC)
└── 4. (mainv2) resolve_penetrations(dry_run=True) — 仅检测穿模
```

#### 2. "只是调整 scale 吗?" — 不完全是

Stage4 的 Umeyama 相似变换**包含 scale, 但不只有 scale**:
- **scale (尺度)**: 修正 3D 资产与点云的尺寸差异
- **rotation (旋转)**: 修正朝向偏差
- **translation (平移)**: 修正位置偏差

scale 是 Umeyama 相似变换的一部分 ([stage4/combined_alignment.py](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/stage4/combined_alignment.py) 的 `refine_single_instance_combined` 内部), 不是独立步骤.

**之后还有 ICP 精调** (Phase B), 进一步优化旋转和平移 (不改 scale).

#### 3. "调用它对系统有影响" — 是的, 计算量大

Stage4 对系统的影响:
- **GPU 内存**: MASt3R 匹配需要大量 GPU 内存 (每对帧做 2D 特征匹配 + 3D lifting)
- **计算时间**: 每个实例都要跑 MASt3R + ICP, 实例多时很慢
- **CPU**: ICP 精调是 CPU 密集型 (点云配准)
- **内存**: 需要保存 world_points (T×H×W×3) + 置信度

**所以 Stage4 默认关闭** (需要 `--enable_stage4` 显式启用).

#### 4. mainv2 vs run_post_pipeline 的 Stage4 差异

| 维度 | mainv2.py | run_post_pipeline.py |
|------|-----------|----------------------|
| real mask | 优先 SAM mask (M_real), 回退深度 mask | 直接深度 mask (无 SAM) |
| 置信度 | 真实 `world_points_conf` | 全 1 (`np.ones_like`) |
| wall 物体处理 | 反算 camera_pos | 不处理 |
| 穿模修复 | `resolve_penetrations(dry_run=True)` 仅检测 | `resolve_penetrations` 真实 + `check_stability` |
| 位姿记录 | 不记录 stage4 | 不记录 stage4 |

⚠️ **注意**: `run_post_pipeline.py` 的 `run_stage4` (行 458-462) 引用了未定义的 `categories_and_relations` (该变量在 `run_stage4` 函数作用域内不存在, 会触发 `NameError`). 这是 latent bug, 但 mainv2.py 不受影响 (mainv2 在 main() 作用域调用 resolve_penetrations).

#### 5. Stage4 不记录位姿变化

`pose_changes.json` 中**没有 'stage4' 阶段** — `_record_pose_stage` 从不为 Stage4 调用 (见 [mainv2.py](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py) 行 1051/1078/1133/1162/1199, 只记录 initial/basic_refinement/stage5/physics). 如果想看 Stage4 的效果, 需对比 `final_scene.glb` (Stage4 前) 和 `final_scene_stage4.glb` (Stage4 后).

### Q78: 121 场景有什么问题？动态物体取首帧策略对吗？点云怎么处理？

**回答**:

#### 1. 121 场景的 z 轴没问题

121 的 `coordinate_alignment.json`:
- `alignment_stage = "stage1_strict"` (用 `align_to_room_coordinate_system`)
- `R[2,2] = -0.7558` (负数)

**用户确认**: 121 的 z 轴方向是正确的, R[2,2] 负值在这个场景下是正常的 (不一定是 z 轴反转). 之前误判为 bug 是错误的.

#### 2. 动态物体取首帧策略 — 正确

当前策略 ([mainv2.py:562-612](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L562)):
1. Mesh 用**最大表面积帧**生成 (最完整建模)
2. 动态物体位置偏移到**首帧质心** (`T[:3,3] += first_centroid - optimal_centroid`)

**用户确认**: 动态物体就是要放在出现首帧的位置, 这个策略是对的.

#### 3. 点云处理 — 剔除其他帧的残影点云 (本次新增)

**问题**: 动态物体在运动, 其在其他帧的点云是"残影" (位置不同), 会污染下游:
- SP精修的 supporter.top_z 被残影拉偏
- Stage4 ICP 被动态点云干扰
- 坐标对齐的 floor/wall 检测被影响

**策略** ([mainv2.py:598-611](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L598), 本次新增):
- 动态物体**只保留首帧点云** (物体放置位置)
- 其余帧该实例 mask 区域的 `world_points` 置 `NaN`
- 首帧点云保留 → 用于位置确认和 SP 精修

```python
# 保留 first_visible_frame_id 的点云, 其余帧该实例 mask 区域置 NaN
for im_entry in sorted_masks:
    fid_clean = im_entry['frame_id']
    if fid_clean == first_visible_frame_id:
        continue  # 保留首帧
    mask_clean = im_entry['mask']
    vggt_prediction_results['world_points'][fid_clean][mask_clean > 0] = np.nan
```

**为什么用 NaN 而非 0**: 0 会被误认为"原点处的有效点", NaN 会被 `np.isfinite` / `> 0` 等自然过滤.

**注意**: 此清理在坐标系对齐 (行 363) **之后**执行, 不影响已完成的对齐. 影响下游 Stage4/Stage5/SP精修.

#### 4. 验证器不校验位置 (已有问题, 本次不修)

`asset_verifier.py` 的三票验证只看: 尺寸 / 置信度 / VLM 外观. **不校验 mesh 放置位置是否与点云质心对齐**. 这是已有设计, 本次不修改.

#### 5. 121 的层级关系 (丰富, 适合层级穿模修复测试)

```
floor (id=1)
├── wall (id=2)
│   └── door_0 (id=13, parent=2)  ← 嵌入墙体
├── table_0 (id=16, parent=1)     ← 大物体
│   ├── bowl_1 (id=6, parent=16)
│   ├── cup_1 (id=12, parent=16)
│   └── scissor_0 (id=14, parent=16)
├── table_1 (id=17, parent=1)     ← 大物体
│   ├── bottle_0 (id=3, parent=17)
│   ├── bottle_1 (id=4, parent=17)
│   ├── bowl_0 (id=5, parent=17)
│   ├── box_0 (id=7, parent=17)
│   ├── car_0 (id=9, parent=17)
│   ├── car_1 (id=10, parent=17)
│   ├── cup_0 (id=11, parent=17)
│   └── scissor_1 (id=15, parent=17)
└── cabinet_0 (id=8, parent=1)    ← 大物体
```

121 有 2 个 table + 1 个 cabinet 作为大物体, 13 个小物体分布在不同 supporter 上. 这是层级穿模修复的理想测试场景.

### Q79: 生成的 GLB 里面能标注物体是什么吗？单独 GLB 能给出物体信息吗？还是用 JSON 说明？

**回答**:

#### 1. 当前 GLB 已经包含物体名称 (node.name)

从实际 GLB 文件 (`output_v2/044_C2_Mug_high_onehand_vggt_omega/final_scene.glb`) 提取的 JSON 结构:

```json
"nodes": [
  {"name": "world", "children": [1,2,3,4,5]},
  {"name": "grid_z0", "mesh": 0, "matrix": [...]},
  {"name": "chair_0", "mesh": 1, "matrix": [...]},
  {"name": "cup_0",   "mesh": 2, "matrix": [...]},
  {"name": "fan_0",   "mesh": 3, "matrix": [...]},
  {"name": "table_0", "mesh": 4, "matrix": [...]}
]
```

**每个物体已经是 `node.name = "{category}_{instance_idx}"`** (如 `table_0`, `cup_0`), 通过 [mainv2.py:894](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L894) 的 `scene.add_geometry(mesh, node_name=f"{category}_{i}")` 实现. 在 Blender / Isaac Sim / Three.js 等工具中打开 GLB, 可以直接看到这些节点名称.

#### 2. 当前 GLB 没有 extras (自定义元数据)

```json
"meshes": [{"name": "geometry_0", "extras": {}}, ...]
```

`extras` 是空字典 `{}`. glTF 2.0 规范支持每个 node/mesh 携带任意 JSON 的 `extras` 字段, 但 trimesh 导出时**不会自动把 mesh.metadata 写入 extras** (实测确认).

#### 3. 建议: GLB node.name + JSON 详细信息 (当前方案, 推荐)

| 信息类型 | 存放位置 | 示例 |
|---------|---------|------|
| 物体名称 | GLB `node.name` | `"table_0"` |
| 实例索引 | GLB `node.name` (后缀) | `_0` |
| 类别 | GLB `node.name` (前缀) | `table` |
| 支撑关系 | `relations_scene_graph.json` | `"supported by table"` |
| 父物体 ID | `relations_scene_graph.json` | `parent: 17` |
| 位姿 T 矩阵 | `pose_changes.json` | 4×4 矩阵 |
| display_id | `relations_scene_graph.json` | `id: 5` |

**关联方式**: GLB 的 `node.name` 与 `relations_scene_graph.json` 的 `category_to_display_ids` 对应:
```json
"category_to_display_ids": {"table": [17], "cup": [11]}
```
`table_0` → `category_to_display_ids["table"][0]` = display_id 17.

**优点**:
- GLB 自带名称, 可直接在 3D 工具中识别物体
- 详细信息 (关系/层级/位姿) 在 JSON 中, 便于程序化处理
- 不依赖 trimesh 的 extras 支持 (实测不稳定)

#### 4. 如果要 GLB 自带完整 extras

可以用 `pygltflib` 库在导出后注入 extras:
```python
# 伪代码: 导出 GLB 后, 用 pygltflib 注入 node extras
from pygltflib import GLTF2
gltf = GLTF2().load(glb_path)
for node in gltf.nodes:
    if node.name == "table_0":
        node.extras = {"category": "table", "relation": "supported by floor",
                       "parent": 1, "instance_idx": 0}
gltf.save(glb_path)
```

但需要额外依赖 `pygltflib`, 且不是所有工具都能读取 extras. **当前 node.name + JSON 方案已足够**.

### Q80: z=0 的平面是怎么确定的？

**回答**:

#### 核心逻辑

z=0 平面 = **floor_centroid (地面质心) 所在的水平面**.

关键代码在 [geometry_utils.py:337-339](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/src/geometry_utils.py#L337) (`_build_R_t_from_floor`):

```python
R = np.stack([wall_normal_1, wall_normal_2, floor_normal], axis=0)  # 旋转矩阵
rotated_floor_centroid = floor_centroid @ R.T  # 把 floor_centroid 转到新坐标系
t = np.zeros(3)
t[2] = -rotated_floor_centroid[2]  # 平移: 使 floor_centroid 的 z = 0
```

变换后, `p_new = R @ p_old + t`, floor_centroid 的新 z 坐标 = `rotated_floor_centroid[2] + t[2] = 0`.

#### floor_centroid 的来源 (按对齐阶段)

| 对齐阶段 | floor_centroid 来源 | floor_normal 来源 |
|---------|---------------------|-------------------|
| Stage1 strict (`align_to_room_coordinate_system`) | floor mask 的 PCA 平面拟合质心 | floor mask 的 PCA 最小特征值方向 |
| Stage2 relaxed (`align_via_objects`) | floor mask 的 PCA 平面拟合质心 | 同上, 用 `_orient_floor_normal` 朝上 |
| Stage3 large_plane (`align_via_large_plane`) | floor mask 的 PCA 平面拟合质心 | 同上 |
| Stage4 GeoCalib (`align_via_geocalib`) | `_estimate_floor_centroid`: **最低 10% 点的质心** | `-gravity` (GeoCalib 重力反方向) |

#### floor_centroid 的计算方式

**Stage1-3** (有 floor mask):
```python
# get_plane_info (geometry_utils.py 行 176-209)
masked_points = pointmap[mask]  # floor mask 区域的 3D 点
centroid = np.mean(masked_points, axis=0)  # 质心
# PCA: 最小特征值的特征向量 = 平面法线
cov_matrix = np.dot(centered_points.T, centered_points)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
normal = eigenvectors[:, np.argmin(eigenvalues)]
```

**Stage4 GeoCalib** (无 floor mask, 用重力):
```python
# _estimate_floor_centroid (geometry_utils.py 行 280-288)
projections = all_points @ floor_normal  # 所有点在 floor_normal 方向的投影
threshold = np.percentile(projections, 10)  # 最低 10%
bottom_mask = projections <= threshold
floor_centroid = np.mean(all_points[bottom_mask], axis=0)  # 最低 10% 点的质心
```

#### floor_normal 方向确定 (朝上)

`_orient_floor_normal` ([geometry_utils.py:291-307](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/src/geometry_utils.py#L291)):
1. **优先**: 场景质心在 floor_centroid 上方 → `np.dot(all_centroid - floor_centroid, floor_normal) > 0` → 朝上
2. **回退** (质心重叠时): 相机位置在 floor 上方 → `np.dot(mean_cam - floor_centroid, floor_normal) > 0` → 朝上

#### 在 GLB 中的体现

[mainv2.py:879-887](file:///mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/mainv2.py#L879) 在 z=0 处画了一个网格 (`grid_z0`) 标注地面:
```python
# 添加虚拟水平面标注 (z=0处的网格线)
grid_lines = []
for v in np.arange(-5.0, 5.0 + 0.5, 0.5):
    grid_lines.append(trimesh.load_path(np.array([[v, -5, 0], [v, 5, 0]])))
    grid_lines.append(trimesh.load_path(np.array([[-5, v, 0], [5, v, 0]])))
grid = trimesh.util.concatenate(grid_lines)
scene.add_geometry(grid, node_name="grid_z0")
```

这个网格就是 z=0 平面 (地面) 的可视化标注, 在 GLB 中显示为一个 10m×10m 的网格线.




