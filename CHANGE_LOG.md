# Change Log

项目变更记录。每次 Agent 任务完成后自动追加。

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
