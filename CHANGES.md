# ReplicateAnyScene 修改记录

## 修改日期: 2026-05-08

---

## 一、Stage 1: generate_scene_json_stage1_qwen36.py

**文件**: `tools/generate_scene_json_stage1_qwen36.py` (新建)

基于原 `generate_scene_json_stage1.py` 适配 Qwen3.6-27B-FP8 模型。

### 与原版的区别

| 项目 | 原版 (Qwen2.5-VL-3B) | 新版 (Qwen3.6-27B-FP8) |
|------|----------------------|------------------------|
| 默认模型 | Qwen2.5-VL-3B-Instruct | Qwen3.6-27B-FP8 |
| 图像预处理 | `qwen_vl_utils.process_vision_info` | 直接传 `images=` 给 processor |
| 图片 resize | 手动 `resize((512,512))` | processor 内部处理 |
| GPU 分配 | 固定 GPU 3 | `device_map="auto"` 自动分配 |
| 推理函数 | 分散在各处 | 统一 `_vlm_inference()` |

### 关键帧提取策略

- 基于 VGGT 相机位姿变化（位移 + 旋转）
- 累积位移 >= 0.1m 或累积转角 >= 5° 才选为新关键帧
- 不强制补充均匀采样帧，VGGT 选几帧就几帧
- 帧提取使用 ffmpeg 按时间戳提取，避免 cv2 seek 超时

### 运行命令

```bash
/mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python tools/generate_scene_json_stage1_qwen36.py \
  --input_video assets/basic_pick_place/7.mp4 \
  --num_frames 10
```

---

## 二、main.py 修改

### 2.1 VLM 幻觉验证（新增）

**问题**: SAM3 纯文本提示分割会产生幻觉，在视频里没有的物体也被分割出 mask。

**方案**: 在 `cross_category_deduplicate` 之后，加载 VLM 模型验证每个实例是否真实存在。

**流程**:
```
SAM3 分割出 mask
  → 裁剪 mask 区域的图像
  → VLM 问 "Does this image contain a '{category}'?"
  → 至少 2 帧回答 yes → 保留
  → 否则 → 丢弃（幻觉过滤）
```

**新增参数**: `--vlm_checkpoint`（默认自动检测 Qwen3.6-27B-FP8）

**新增函数**: `src/object_segmentation.py` 中的 `verify_instance_with_vlm()`

### 2.2 运动物体选首次出现帧

**问题**: 原逻辑选 3D 表面积最大的帧生成 GLB，但运动物体在被拿起/移动时面积最大，此时形状不完整。

**方案**: 修改 `get_optimal_view_frame_id()` 的选帧策略：

```
计算每帧 mask 的 3D 质心
  → 相邻帧质心位移 > 0.1m？ → 物体在运动 → 选首次出现的帧
  → 位移都小？ → 物体静止 → 选面积最大的帧（原逻辑）
```

**修改文件**: `src/geometry_utils.py`

**新增参数**: `motion_threshold=0.1`（质心位移阈值，单位：米）

### 2.3 cross_category_deduplicate 阈值调整

**问题**: 原阈值 `< 3` 导致只在 1~2 帧出现的物体被丢弃，无法生成 GLB。

**方案**: 阈值从 `< 3` 改为 `< 2`，只在 1 帧出现的实例才被丢弃。

**修改文件**: `src/sg_deduplication.py` 第 321 行

---

## 三、其他修改

### 3.1 ffmpeg 替代 cv2 提取帧

**问题**: cv2 的 `cap.set(CAP_PROP_POS_FRAMES)` 在某些视频上会超时，导致提取 0 帧。

**方案**: 全部改用 ffmpeg 按时间戳提取帧。

**修改文件**: `tools/generate_scene_json_stage1.py` 中的 `extract_specific_frames()` 和 `extract_frames_from_video()`

### 3.2 VLM_PROMPT_LOCATE 花括号转义

**问题**: Python `.format()` 把 JSON 中的 `{}` 当占位符，导致 KeyError。

**方案**: JSON 中的 `{` `}` 改为 `{{` `}}` 转义。

**修改文件**: `tools/generate_scene_json_stage1.py` 中的 `VLM_PROMPT_LOCATE`

---

## 四、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `tools/generate_scene_json_stage1_qwen36.py` | 新建 | 适配 Qwen3.6-27B-FP8 的 Stage 1 |
| `tools/test_qwen36.py` | 新建 | Qwen3.6-27B-FP8 加载测试脚本 |
| `main.py` | 修改 | 加 VLM 幻觉验证 + `--vlm_checkpoint` 参数 |
| `src/geometry_utils.py` | 修改 | `get_optimal_view_frame_id` 运动物体选首帧 |
| `src/sg_deduplication.py` | 修改 | 阈值 `< 3` → `< 2` |
| `src/object_segmentation.py` | 修改 | 新增 `verify_instance_with_vlm()` |
| `tools/generate_scene_json_stage1.py` | 修改 | ffmpeg 提取帧 + 旋转角度 + 花括号转义 |
