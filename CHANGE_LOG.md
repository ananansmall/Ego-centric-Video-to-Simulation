# Change Log

项目变更记录。每次 Agent 任务完成后自动追加。

---

## [2026-06-22] - 修复3个bug + Stage5坐标系审查 + 文档更新

### 核心问题
1. `motion_info['global_disp']` KeyError — motion_info 字典没有 global_disp key
2. `spherical_mean(weights=...)` TypeError — 参数名应为 w
3. `'refined_relations' in dir()` 脆弱判断 — 直接用 categories_and_relations
4. 用户质疑 Stage5 是否改变了 GLB 坐标系

### Changed

- **mainv2.py**: 删除 `motion_info['global_disp']` 引用 (L475)
- **mainv2.py**: `'refined_relations' in dir()` → 直接用 `categories_and_relations` (L997)
- **src/geometry_utils.py**: `spherical_mean(weights=inlier_confs)` → `spherical_mean(w=inlier_confs)` (L555)
- **docs/mainv2_technical_doc.md**: 新增"四个GLB文件"表 + "Stage5是否改变坐标系"章节 (基于代码审查, 确认Stage5不改变坐标系)

### 审查结论
- Stage5 的所有操作都是单个物体的 T 矩阵微调, 不存在全局坐标系变换
- 4个GLB文件之间没有坐标系差异 (全部 y-up), 区别仅在于各阶段精修导致的物体位姿不同
- 唯一的全局坐标系变换发生在 Stage2

### Docs to Review
- 无需额外同步

---

## [2026-06-18] - 修复3个运行时问题 + 重写mainv2技术文档

### 核心问题
1. mainv2.py默认帧数参数(10/160)未同步更新，导致print输出仍显示旧值
2. SAM floor分割mask(1080xW)与VGGT pointmap(518xW_vggt)维度不匹配，boolean index报错
3. 点云补充检测置信度阈值用`>`严格大于，当中位数=最小值时排除大量有效点

### Changed

- **mainv2.py**: 5处默认值同步更新
  - `run_stage1()` 参数: `max_frames_stage1=10->12`, `vggt_max_frames=160->120`
  - `run_stage2()` 参数: `max_frames=160->120`
  - argparse `--max_frames`: `default=160->120`
  - argparse `--max_frames_stage1`: `default=10->12`
  - 顶部帮助文本同步更新

- **tools/generate_scene_json_stage1.py**: 3处修改
  - 新增 `_resize_mask_to_pointmap(mask, pointmap)`: 将SAM mask resize到VGGT pointmap尺寸
  - `get_plane_info()` 调用前使用 `_resize_mask_to_pointmap()` 对齐mask维度
  - 点云补充检测排除floor区域时使用 `_resize_mask_to_pointmap()` 对齐mask维度
  - 置信度阈值比较 `>` 改为 `>=`，避免排除等于阈值的点
  - 新增保底逻辑：过滤后点数<100时自动降低到25%分位数

- **docs/mainv2_technical_doc.md**: 在原有文档基础上更新
  - 更新默认值: max_frames=120, max_frames_stage1=12
  - 更新protected_categories描述: 改为白名单过滤实现
  - 新增Q8-Q13: Stage1速度分析、坐标系问题、SAM3D姿态、mask去重流程、SAM维度修复、点云补充检测修复
  - 新增修改记录条目

### Docs to Review
- 无需额外同步的.md文件

---

## [2026-05-30] - 修复Stage5坐标系错误 + pkl加载 + floor/wall物体保护

### 核心问题
Stage5 (run_post_pipeline.py) 从GLB加载3D资产时存在两个严重bug:
1. **坐标系错误**: GLB是y-up格式, mainv2的save_final_glb已将T烘焙进mesh顶点并做z-up→y-up变换.
   run_post_pipeline加载GLB后做y-up→z-up反转, 但T已烘焙进顶点, 导致SP精修在错误的坐标系下调整位置.
2. **floor/wall物体被错误调整**: 穿模检测对所有物体对操作, 包括mainv2已精修好的floor/wall物体.

### Changed

- **mainv2.py**: 基础精修后保存 `all_instances.pkl`, 保留原始z-up空间的original_mesh+T数据结构
  - 原理: pkl保存的是mainv2内部使用的all_instances字典, original_mesh在z-up空间, T是独立变换矩阵.
    SP精修函数(sp_refine_on_top等)直接读取T和original_mesh计算bounds, 然后修改T. 这与mainv2内部逻辑完全一致.
  - 修复: Stage4中resolve_penetrations的参数从refined_relations改为categories_and_relations(前者在Stage4时未定义)

- **tools/run_post_pipeline.py**: 重写加载逻辑, 优先pkl, 回退GLB
  - 新增 `load_instances(scene_dir, prefer_stage4=False)`: 优先加载all_instances.pkl(z-up, T独立),
    回退到GLB加载(y-up→z-up, T=I已烘焙, 精度有损)
  - 新增 `_load_instances_from_glb()`: 原GLB加载逻辑, 作为回退方案
  - Stage4执行后保存 `all_instances_stage4.pkl`, Stage5可直接加载
  - discover_scene_files新增pkl_path字段
  - 必需文件检查: pkl_path或glb_path至少存在一个即可
  - 移除旧的_cleanup_stale_outputs删除逻辑(改为直接覆盖)
  - GLB命名按Stage参与决定: final_scene_stage4.glb / final_scene_stage5.glb / final_scene_stage4_5.glb

- **tools/refine_inter_object_placement.py**: resolve_penetrations增加floor/wall物体保护
  - 新增wall_names集合: 从refined_relations中提取"wall"相关物体
  - 两个floor/wall物体之间跳过穿模检测(已在mainv2基础精修中处理)
  - 选择移动对象时: floor/wall物体优先被保护(不移动)
  - z轴保护: floor/wall物体不允许被z轴推动(位置已由mainv2确定)
  - 原理: mainv2的基础精修已经处理了supported_by_floor/embedded_in_wall/attached_to_wall,
    Stage5的SP精修只应处理物体间支撑关系(supported by [other object]), 不应改变已精修的floor/wall物体位置

### Added

- **tools/__init__.py**: 空文件, 使tools目录可作为Python包导入, 修复ModuleNotFoundError

### ⚠️ Docs to Review
- 无需同步的.md文件(本次修改均为代码逻辑修复)

---

## [2026-05-31] - 修复hoi4d_vggt_omega三个核心问题: 投票帧不足/平票/实例丢失

### 核心问题
分析 `output_v2/hoi4d_vggt_omega` 日志发现三个关联问题:
1. **scissor_0 只有1帧投票** (应至少5帧): `instance_visibility.json` 中无 scissor, 无法补充帧
2. **toy_1 平票** (4/8 floor vs table): VLM投票平票时无坐标系判断机制
3. **scissor_0 未找到实例**: Stage2 跨类去重将 scissor 合并到 toy, `all_instances` 中无 scissor key

三个问题形成因果链: 跨类去重合并 → 无 visibility 数据 → 无法补充帧 → 无法精修

### Changed

- **tools/refine_other_objects_relations.py**: `build_object_to_frames()` 增加回退帧补充
  - 当 `instance_visibility` 中无某物体数据时, 不再跳过, 而是从 `color/` 目录均匀采样帧补充至 min_frames
  - 将 `existing_fids` 计算提前到 if/else 之前, 确保两个分支都能使用

- **tools/refine_inter_object_placement.py**: 新增 `_resolve_tie_by_z()` + 改进实例查找
  - 新增 `_resolve_tie_by_z()`: 用Z坐标高度判断平票关系 — 比较物体Z中心到table顶面和floor的距离, 取更近的
  - `_find_supporter_instances()` 增加第4层匹配: 去掉实例后缀再搜索 (scissor_0 → scissor)
  - 改进未找到实例时的错误信息: 打印可用类别列表, 帮助调试

- **src/sg_deduplication.py**: `cross_category_deduplicate()` 新增 `protected_categories` 参数
  - 当两个实例属于不同的 protected categories 时, 不合并 (即使3D重叠率超阈值)
  - 防止 Stage1 JSON 中明确识别的不同物体类别被错误合并

- **mainv2.py**: 传入 `protected_categories` 到跨类去重
  - `protected_categories=set(categories_and_relations.keys())` — Stage1 发现的所有类别互不合并

### Added

- **output_v2/hoi4d_vggt_omega/README.md**: 完整的输出目录说明文档
  - 文件说明 (最终输出/Stage1/2/3产物)
  - 物体实例清单 (含VLM投票和SP精修结果)
  - 问题分析及修复方案 (4个问题的因果链和修复方式)
  - 修复文件清单

### ⚠️ Docs to Review
- `TECHNICAL_DOCUMENTATION.md`: `cross_category_deduplicate` 新增了 `protected_categories` 参数, 文档中该函数的描述需同步更新
- `CHANGES.md`: 跨类去重保护机制是重要变更, 建议记录

---

## [2026-05-31] - mainv2 vs main 完整对比 + protected_categories修复确认

### 核心问题
1. mainv2.py 的 `protected_categories` 修改之前未正确写入 (run_stage2函数内的调用遗漏)
2. 用户要求 mainv2 vs main 的完整变更对照

### Changed

- **mainv2.py**: 修复 `run_stage2()` 中 `cross_category_deduplicate()` 调用 — 确认 `protected_categories=json_categories_set` 已正确传入

### Added

- **docs/mainv2_changelog.md**: mainv2.py vs main.py 完整变更对照文档
  - 架构差异 (212行 → 881行)
  - 函数定义变更 (1个 → 7个)
  - 命令行参数变更 (4个 → 12个)
  - 流水线阶段变更 (Stage1-5)
  - BUG修复记录 (精修写回BUG、关系格式不兼容)
  - 新增功能 (日志系统、中间结果、异常处理)
  - 鲁棒性分析

### ⚠️ Docs to Review
- `TECHNICAL_DOCUMENTATION.md`: mainv2的Stage1/4/5新功能需要补充描述

---

## [2026-05-31] - VGGT-Omega点云缺块 + VGGT手部云团 根因分析

### 核心内容
分析两个实际观察到的点云质量问题：VGGT-Omega碗后/墙壁缺块，VGGT手部呈云团状。

### Changed

- **docs/VGGT_Models_Comparison.md**: 新增 §13 实际点云质量分析
  - §13.1 现象描述
  - §13.2 VGGT-Omega缺块四层原因:
    1. 置信度百分位过滤(最直接): conf_thres=50砍掉50%低置信度点, 低纹理区域(墙壁)被过滤
    2. DenseHead pixel_shuffle上采样: 4x一步到位, 16通道预测能力有限, 低纹理区域细节丢失
    3. exp激活深度异常值: 离群点depth_conf不一定低, 挤占百分位空间, 抬高阈值
    4. CameraHead无迭代精修: 位姿误差大, 多帧对齐不准
  - §13.3 VGGT手部云团: 160帧×手部像素面积=大量散乱3D点, PointHead对动态物体也输出高置信度
  - §13.4 改进方案: 降低阈值/过滤深度异常值/使用VGGT4D
  - §13.5 三模型点云质量对比表

### ⚠️ Docs to Review
- 无需同步的.md文件

---

## [2026-05-31] - VGGT4D 时间持续性过滤实现

### 核心变更
在 vggt4d_predict.py 中实现 _temporal_filter_dyn_masks(), 解决 dyn_masks 过度标记问题。
手扫过的背景区域不再被误标为动态。

### Changed

- **src/vggt4d_predict.py**:
  - 新增 `_temporal_filter_dyn_masks(dyn_masks, min_dynamic_ratio=0.5)` 函数
    - 对每个像素统计动态帧比例, 低于阈值的恢复为静态(短暂遮挡)
    - min_dynamic_ratio=0.5: 至少一半帧中动态才保留
  - `vggt4d_predict()` 新增 `min_dynamic_ratio=0.5` 参数
  - 在 dyn_masks 转 numpy 后、filter_dynamic_points 前调用过滤
  - 更新函数注释: 新增 Stage 2 时间持续性过滤流程说明

- **docs/VGGT_Models_Comparison.md**:
  - §15.7 方案1标记为已实现, 更新代码示例和效果说明

### ⚠️ Docs to Review
- 无需同步的.md文件

---

## [2026-05-31] - 置信度分布差异分析 + VGGT4D过度标记问题

### 核心内容
1. VGGT和VGGT-Omega同样砍50%置信度，为什么效果差异巨大？
2. VGGT4D dyn_masks过度标记问题：手扫过的地方都被标为动态

### Changed

- **docs/VGGT_Models_Comparison.md**: 新增 §14-§15
  - §14 同样砍50%置信度为什么效果不同:
    - 数学定义相同(1+exp), 但语义完全不同(3D点位置 vs 深度值)
    - PointHead conf分布双峰(好坏分明), DenseHead conf分布均匀(差异不大)
    - 50%阈值对双峰分布有效分离, 对均匀分布≈随机砍半
    - VGGT-Omega缺块根因: depth_conf分布太均匀, 百分位阈值无法区分好坏点
    - VGGT-Omega设计取舍: 更高效推理但牺牲了3D点云质量
  - §15 VGGT4D dyn_masks过度标记:
    - dyn_map公式检测"运动线索"而非"真正动态物体", 短暂遮挡也被标记
    - 整个流程没有时间维度持续性判断
    - 几何精修"损失点太少→动态"规则加剧误标
    - 形态学膨胀扩大掩码范围
    - 改进方案: 时间持续性过滤/缩小膨胀/调整阈值/时间一致性后处理
  - §13.5 VGGT4D整体可用性从★★★★★降为★★★★☆(过度标记)

### ⚠️ Docs to Review
- 无需同步的.md文件

---

## [2026-06-23 16:30] - Stage2 四阶段Z轴对齐 + GLB 5个逻辑 + z_axis_alignment.json 记录

### 任务
1. 确认 SAM3 是否支持点提示 → 不支持，用 VLM+box prompt 近似
2. `--enable_stage4 --enable_stage5` 同时启用时生成 5 个 GLB
3. `pose_changes.json` 接入 Stage4 记录，不启用时无 bug
4. 分析 068/121 日志 Z 轴对齐效果
5. 添加 Z 轴对齐结果记录

### Changed

- **mainv2.py**:
  - 新增 import: `align_via_objects`, `align_via_vlm_floor_points`, `align_via_large_plane`, `align_via_geocalib`, `segment_large_flat_surfaces`
  - 新增 `_load_vlm_model()` 辅助函数 (Stage2.5 VLM 加载)
  - 新增 `_is_identity_alignment()` 判断对齐是否失败 (返回单位阵)
  - `run_stage2()` 新增 `vlm_checkpoint` 参数，实现四阶段 Z 轴对齐 fallback 链:
    - 阶段1: `align_to_room_coordinate_system` (SAM3 floor+wall 文本提示)
    - 阶段2: `align_via_objects` (放宽阈值 + floor/PCA)
    - 阶段2.5: `align_via_vlm_floor_points` (VLM 地面参考点 + SAM3 box prompt)
    - 阶段3: `align_via_large_plane` (SAM3 大平面 mask)
    - 阶段4: `align_via_geocalib` (GeoCalib 重力估计)
  - 新增 `z_axis_alignment.json` 保存: 记录 align_method, R_matrix, t_vector, is_identity, align_info, n_wall_masks, n_floor_masks
  - `main()` 中 `run_stage2()` 调用传入 `vlm_checkpoint`
  - Stage4 后记录 `pose_history` (仅 `args.enable_stage4` 时)
  - Stage5 最终 GLB 命名: 有 stage4 → `final_scene_stage4_5.glb`，无 stage4 → `final_scene_stage5.glb`

- **src/geometry_utils.py**:
  - 新增 `align_via_objects()` — 阶段2: 放宽阈值 + 只用 floor (+ wall 或 PCA)
  - 新增 `align_via_vlm_floor_points()` — 阶段2.5: VLM + SAM3 box prompt
  - 新增 `align_via_large_plane()` — 阶段3: SAM3 大平面 mask
  - 新增 `align_via_geocalib()` — 阶段4: GeoCalib 重力估计
  - 新增 `_orient_floor_normal()` — 确保 floor_normal 朝上
  - 新增 `_build_R_t_from_floor()` — 从 floor 平面构造 R, t

- **src/object_segmentation.py**:
  - 新增 `detect_floor_reference_points_with_vlm()` — VLM 识别地面参考点
  - 新增 `segment_floor_with_box_prompts()` — SAM3 box prompt 分割 floor
  - 新增 `segment_large_flat_surfaces()` — 阶段3 大平面分割

- **tools/refine_inter_object_placement.py**:
  - `refine_inter_object_relations()` 新增 `final_glb_name` 参数
  - 最终 GLB 保存路径使用 `final_glb_name` (支持 `final_scene_stage4_5.glb`)

- **docs/mainv2_technical_doc.md**:
  - 新增 "Stage 2 四阶段 Z 轴对齐" 章节 (含优先级表、代码示例、SAM3 点提示说明)
  - 更新 GLB 文件体系: 按启用阶段列出 GLB 数量 (2/3/4/5)
  - 新增 "位姿变化记录 pose_changes.json" 章节
  - 更新 Q1: GLB 文件数量说明
  - 修正 Stage5 逻辑描述

- **docs/questions.md**:
  - 新增 Q61-Q64: SAM3 点提示、GLB 数量、pose_changes.json Stage4 记录、068/121 日志分析

### Added

- **test_glb_alignment_logic.py**: 测试 GLB 命名逻辑 (6 种组合)、z_axis_alignment.json 格式、_is_identity_alignment 判断、068 日志分析

### 测试结果

```
GLB 命名逻辑: 6/6 通过
  stage4=OFF, stage5=ON, inter_obj=True → 4 个 GLB (068 现状)
  stage4=ON, stage5=ON, inter_obj=True → 5 个 GLB (新逻辑)
z_axis_alignment.json 格式: 通过
_is_identity_alignment: 通过
068 日志分析: 旧代码无 Z 轴对齐输出, theta_gravity 最大 167° (严重倾斜)
```

### 068/121 日志分析结论

- 两个日志均为**旧代码**运行，Stage2 无四阶段对齐输出
- 068: theta_gravity 135°-167° (严重倾斜，接近倒立)
- 121: cabinet_0 theta_gravity=171.9° (几乎倒立)
- pose_changes.json 只有 initial/basic_refinement/stage5，无 stage4
- 需用新代码重跑才能验证四阶段对齐效果

### ⚠️ Docs to Review
- 无需额外同步

---

## [2026-06-23 18:00] - 添加 --cleanup 参数 + GLB 流程图 + 修复文档

### 任务
1. 添加 `--cleanup` 参数，运行结束后自动清理中间文件
2. 梳理 GLB 传递流程，在技术文档中给出流程图
3. 验证 mainv2.py 当前能正常运行

### Changed

- **mainv2.py**:
  - 新增 `--cleanup` 参数 (L1366-1367): 运行结束后删除 `all_instances*.pkl`, `color/`, `depth/`, `extrinsics/`, `keyframes/`, `optimal_frames/`
  - 新增清理逻辑 (L1328-1356): 统计释放空间并打印

- **docs/mainv2_technical_doc.md**:
  - 重写 §4 GLB文件体系: 新增完整流程图 (Stage1→Stage2→Stage3→基础精修→Stage4→Stage5)
  - 新增 "GLB 传递关系" 图: 明确每个 GLB 的数据来源是 `all_instances` 快照
  - 新增 "各 GLB 详细说明" 表: 含保存位置(行号)、保存时机、数据来源
  - 新增 "--cleanup 参数" 说明表
  - 修正 `run_post_pipeline.py` 数据流: `all_instances.pkl` → Stage4/5 (不再用 `final_scene_base.glb`)
  - 修正 pipeline 概览: `--enable_stage4 --enable_stage5` 最终输出 `final_scene_stage4_5.glb`

### 验证

```
python3 -m py_compile mainv2.py → OK
GLB 流程逻辑: 5 个 GLB (stage4+stage5+inter_obj) → 验证通过
```

### ⚠️ Docs to Review
- 无需额外同步

---

## [2026-06-23 22:30] - 修复 SP精修 on_top 阈值 + 大物体穿模分离 + 重建工具

### 任务
1. 修复 Stage5 SP精修: on_top 策略 0.3m 阈值导致物体卡在桌子内部
2. 修复 resolve_penetrations: 大物体穿模分离距离不足
3. 确认 GLB 传递链正确性
4. 创建 tools/rebuild_glbs_from_json.py

### 根因分析

**121 日志关键发现**:
```
[on_top] z_offset=+0.5421m 超出阈值 (|z_offset|>0.3m), 保留原位
```
- 15 个物体中 14 个被拒绝移动 (z_offset > 0.3m)
- VLM 正确判定 on_top, 但 SP精修拒绝执行贴合
- 穿模修复只做 0.01m 分离, 对大物体无效

### Changed

- **tools/refine_inter_object_placement.py**:
  - `sp_refine_on_top()`: 移除 0.3m 阈值, VLM 判定 on_top 后始终执行 z 轴贴合
  - `resolve_penetrations()`: 大物体分离距离从 `pen_depth*1.5+0.01` 改为分级:
    - 超大物体 (>0.5m): `pen_depth + 0.10m`
    - 大物体 (>0.3m): `pen_depth + 0.05m`
    - 小物体: `pen_depth + 0.01m`

### Added

- **tools/rebuild_glbs_from_json.py**: 从 pose_changes.json 重建各阶段 GLB
  - 用法: `python tools/rebuild_glbs_from_json.py --scene_dir output_v2/xxx`
  - 输出: `rebuild_initial.glb`, `rebuild_basic_refinement.glb`, `rebuild_stage5.glb`

### GLB 传递链确认

```
all_instances (内存中就地修改)
  Stage3完成 → final_scene_initial.glb     (快照#1)
  基础精修   → final_scene.glb             (快照#2)
  Stage4    → final_scene_stage4.glb       (快照#3)
  Stage5 SP → final_scene_stage5_sp.glb    (快照#4)
  Stage5最终 → final_scene_stage4_5.glb     (快照#5)
```
每个 GLB 都是 all_instances 在该时间点的快照, 正确继承前一阶段结果。

### 验证
```
python3 -m py_compile mainv2.py → OK
python3 -m py_compile tools/refine_inter_object_placement.py → OK
python3 -m py_compile tools/rebuild_glbs_from_json.py → OK
pose_changes.json 结构验证: 15个物体, 3阶段(initial/basic_refinement/stage5), 4x4 T_matrix
```

### ⚠️ Docs to Review
- docs/mainv2_technical_doc.md: SP精修策略变更 (0.3m阈值移除)

---
