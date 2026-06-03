"""
物体间支撑关系空间位置精修 (refine_inter_object_placement.py)
============================================================

本文件负责: 已知物体间支撑关系后，判定放置策略，并用几何计算精修物体空间位置。

输入:
  - all_instances: 3D资产实例 (每个含 original_mesh + T变换矩阵)
  - refined_relations: 细化后的关系字典
      { "bowl_0": "supported by table", "table": "supported by floor", ... }
  - vlm_checkpoint + scene_dir: VLM模型路径 + 场景目录 (用于策略判定)

输出:
  - 更新后的 all_instances (T矩阵已精修)

================================================================================
  核心流程
================================================================================

  refined_relations
       │
       ▼
  ┌─────────────────────────────────────────┐
  │  Step 1: 策略判定                        │
  │  输入: 物体名称 + 场景图像               │
  │  VLM看图判断: bowl放在table上是什么方式?  │
  │  输出: "on_top" / "inside" / ...         │
  │                                          │
  │  帧来源: optimal_frames/ + keyframes/    │
  │  投票: 多帧VLM推理 → 取多数策略          │
  └─────────────┬───────────────────────────┘
                │
                ▼
  ┌─────────────────────────────────────────┐
  │  Step 2: SP几何精修                      │
  │  根据策略施加物理约束，修改T矩阵          │
  │  只做约束方向的最小移动，保留原始位置      │
  └─────────────────────────────────────────┘

================================================================================
  五种放置策略及物理约束实现
================================================================================

  ── on_top: 放在支撑物顶面 ──

    场景: 杯子在桌上、枕头在床上、书在架子上

    物理约束:
      supported.bottom_z >= supporter.top_z  (不穿入支撑物)
      supported.bottom_z <= supporter.top_z  (不悬空)

    实现:
      z_offset = supporter.top_z - supported.bottom_z

      z_offset < 0     → 物体穿入支撑物，向上推到顶面
      z_offset ≈ 0     → 已接触，无需调整
      0 < z_offset < 阈值 → 物体悬空，向下落到顶面
      z_offset 很大    → 物体远在上方，可能是误判，跳过

      supporter顶面z ───────┐
                           │ z_offset
      supported底面z ───────┘ → 对齐到顶面

  ── inside: 放在支撑物内部 ──

    场景: 衣服在抽屉里、碗在柜子里

    物理约束:
      supported.bottom_z >= supporter.bottom_z  (不穿出底部)
      supported.top_z <= supporter.top_z        (不穿出顶部)
      supported.bottom_z ≈ supporter内部30%高度  (合理放置)

    实现:
      1. 先把 supported 底面对齐到 supporter 内部30%高度
      2. 检查是否穿出顶部 → 如果穿出则上移到刚好不穿出
      3. 检查是否穿出底部 → 如果穿出则上移到刚好在内部

      supporter顶面z ───────┐
                           │ 70%
      target_z (30%) ──────┤ ← supported底面对齐到这里
                           │ 30%
      supporter底面z ───────┘

  ── against_side: 靠在支撑物侧面 ──

    场景: 柜子靠墙、画贴墙

    物理约束:
      1. 底面贴地: supported.bottom_z >= 0
      2. 侧面接触: supported与supporter在x/y轴刚好接触
      3. 不穿模: 侧面接触后，如果z轴穿入supporter内部，上移到顶面

    实现:
      # 约束1: 底面贴地
      if supported.bottom_z < 0:
          z_offset = -supported.bottom_z

      # 约束2: 侧面接触
      遍历 x/y 轴 (axis_idx=0,1) 和 正/负方向 (sign=1,-1):
          if sign > 0:  offset = supporter.min_x - supported.max_x
          else:         offset = supporter.max_x - supported.min_x
      选择 |offset| 最小的方向 (最近侧面)

      # 约束3: 穿模检查
      if 侧面移动后 AABB 在x/y上重叠 且 z轴穿入:
          z_fix = supporter.top_z - supported.bottom_z

      supporter ┃  supported
               ┃← offset →│
               ┃           │  ← 侧面接触

  ── hanging_below: 悬挂在支撑物下方 ──

    场景: 吊灯、吊扇

    物理约束:
      supported.top_z <= supporter.bottom_z  (不穿入支撑物)
      supported.top_z >= supporter.bottom_z  (不远离)

    实现:
      z_offset = supporter.bottom_z - supported.top_z

      z_offset > 0 → 物体在下方太远，上移到刚好接触
      z_offset < 0 → 物体穿入支撑物，下移到刚好接触
      z_offset ≈ 0 → 已接触，无需调整

      supporter底面z ───────┐
                           │ z_offset
      supported顶面z ───────┘ → 对齐到底面

  ── leaning: 斜靠在支撑物上 ──

    场景: 画靠墙、梯子靠墙

    物理约束: 同 against_side (底面贴地 + 侧面接触 + 穿模检查)

    实现: 直接调用 sp_refine_against_side

================================================================================
  使用方式
================================================================================

  Python API:

    from tools.refine_inter_object_placement import refine_inter_object_relations

    all_instances = refine_inter_object_relations(
        all_instances=all_instances,
        refined_relations=refined_relations,
        walls_info=walls_info,
        vlm_checkpoint="/mnt/data/lza/models/Qwen3.5-9B",
        scene_dir="outputs/232",
    )

  命令行:

    python3 -m tools.refine_inter_object_placement \
        --input_glb outputs/232/final_scene.glb \
        --relations_json assets/json_configs/232_refined.json \
        --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B

    输出: outputs/232/final_scene_refined.glb
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import trimesh
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PLACEMENT_STRATEGIES = {
    "on_top": "放在支撑物顶面",
    "inside": "放在支撑物内部",
    "against_side": "靠在支撑物侧面",
    "hanging_below": "悬挂在支撑物下方",
    "leaning": "斜靠在支撑物上",
}

VALID_STRATEGIES = set(PLACEMENT_STRATEGIES.keys())

SP_REFINE_MAP: Dict[str, callable] = {}

DEFAULT_VLM_CHECKPOINT = "/mnt/data/lza/models/Qwen3.5-9B"


def _resolve_vlm_checkpoint(checkpoint_arg):
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    if os.path.exists(DEFAULT_VLM_CHECKPOINT):
        return DEFAULT_VLM_CHECKPOINT
    return None


def _load_vlm_model(checkpoint):
    """加载VLM模型和处理器"""
    import torch
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
        from transformers import AutoProcessor

    print(f"   📥 加载VLM: {checkpoint}", flush=True)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def _vlm_inference(image, model, processor, prompt, max_new_tokens=512):
    """调用VLM进行单次推理"""
    import torch
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]}
    ]
    try:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False
        )
    except TypeError:
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]


def _extract_json_from_text(text):
    """从VLM输出文本中提取JSON"""
    if not text or not text.strip():
        return None

    for tag in ['</think_>', '</think >', '</think\n', '</think\r\n']:
        if tag in text:
            text = text.split(tag)[-1].strip()

    if '</think' in text:
        idx = text.rfind('</think')
        after = text[idx:]
        close_pos = after.find('>')
        if close_pos != -1:
            text = after[close_pos + 1:].strip()

    for pattern in [r'```json\s*([\s\S]*?)\s*```', r'```\s*({[\s\S]*?})\s*```']:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[-1]

    first_brace = text.find('{')
    if first_brace != -1:
        brace_count = 0
        for i in range(first_brace, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    candidate = text[first_brace:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        continue
        last_brace = text.rfind('}')
        if last_brace > first_brace:
            return text[first_brace:last_brace + 1]
    return None


def build_placement_prompt(supported_name, supporter_name, all_pairs):
    """构建VLM放置策略判定提示词

    参数:
        supported_name: 被支撑物体名称
        supporter_name: 支撑物名称
        all_pairs: 所有支撑关系对 [(supported, supporter), ...]
    返回:
        prompt文本
    """
    pairs_str = "\n".join([f'  - "{s}" is supported by "{t}"' for s, t in all_pairs])

    prompt = f"""Look at the image. Determine how "{supported_name}" is physically placed on/against "{supporter_name}".

**All support relationships in the scene**:
{pairs_str}

**Question**: How is "{supported_name}" placed relative to "{supporter_name}"?

**Choose ONE placement strategy**:
- "on_top" — {supported_name} rests ON TOP of {supporter_name} (e.g., cup on table, pillow on bed, book on shelf, toy on table)
- "inside" — {supported_name} is INSIDE {supporter_name} (e.g., clothes in drawer, dish in cabinet, shoe in shoe rack)
- "against_side" — {supported_name} is against the SIDE of {supporter_name} (e.g., cabinet against wall, picture on wall)
- "hanging_below" — {supported_name} hangs BELOW {supporter_name} (e.g., lamp below ceiling, chandelier)
- "leaning" — {supported_name} leans against {supporter_name} at an angle (e.g., painting leaning on wall, ladder against wall)

**Physical common sense**:
- Most small objects on furniture → "on_top"
- Objects stored in containers/drawers → "inside"
- Objects mounted flush on walls → "against_side"
- Suspended from ceiling/overhang → "hanging_below"
- Tilted/angled contact → "leaning"

JSON only, no explanation:
{{"strategy": "one_of_on_top_inside_against_side_hanging_below_leaning"}}"""

    return prompt


def _parse_optimal_frame_filename(filename):
    """解析optimal_frames文件名: {category}_inst{idx}_frame{fid}.jpg → (category, idx, fid)"""
    name = os.path.splitext(filename)[0]
    match = re.match(r'(.+)_inst(\d+)_frame(\d+)', name)
    if match:
        return match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None


def _fuzzy_match_name(query, candidates):
    """模糊匹配名称到候选列表"""
    if not query or not candidates:
        return None
    q = query.lower().strip().replace(' ', '_')
    for c in candidates:
        if c.lower().strip().replace(' ', '_') == q:
            return c
    for c in candidates:
        c_base = c.lower().split('_')[0].split(' ')[0]
        q_base = q.split('_')[0].split(' ')[0]
        if c_base == q_base:
            return c
    for c in candidates:
        if q in c.lower() or c.lower() in q:
            return c
    return None


def _load_optimal_frames_for_instances(optimal_frames_dir, target_names):
    """加载optimal_frames/，按实例索引区分帧映射

    参数:
        optimal_frames_dir: optimal_frames目录路径
        target_names: 目标物体名称列表（如 ["bowl", "table", "toy"]）
    返回:
        instance_frames: {instance_key: [(PIL.Image, source_str), ...]}
            instance_key 如 "bowl_0", "toy_1"
        instance_count: {category: int} 每个类别的实例数量
    """
    from PIL import Image as PILImage

    if not os.path.isdir(optimal_frames_dir):
        return {}, {}

    instance_frames = defaultdict(list)
    instance_count = defaultdict(int)
    seen = defaultdict(set)

    for fname in sorted(os.listdir(optimal_frames_dir)):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        obj_name, inst_idx, frame_id = _parse_optimal_frame_filename(fname)
        if obj_name is None:
            continue

        matched = _fuzzy_match_name(obj_name, target_names)
        cat = matched if matched else obj_name

        inst_key = f"{cat}_{inst_idx}"
        src = f"optimal_frames/{fname}"
        if src in seen[inst_key]:
            continue

        img_path = os.path.join(optimal_frames_dir, fname)
        try:
            img = PILImage.open(img_path).convert("RGB")
            instance_frames[inst_key].append((img, src))
            seen[inst_key].add(src)
            instance_count[cat] = max(instance_count[cat], inst_idx + 1)
        except Exception:
            continue

    return dict(instance_frames), dict(instance_count)


def _load_keyframes_for_instances(keyframes_dir, target_names):
    """加载keyframes/及可见性元数据

    参数:
        keyframes_dir: keyframes目录路径（含keyframes_metadata.json）
        target_names: 目标物体名称列表
    返回:
        keyframe_images: {vid_idx: (PIL.Image, source_str)}
        frame_visibility: {vid_idx: [matched_name, ...]}
    """
    from PIL import Image as PILImage

    metadata_path = os.path.join(keyframes_dir, "keyframes_metadata.json")
    if not os.path.exists(metadata_path):
        return {}, {}

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    raw_visibility = metadata.get("frame_visibility", {})
    keyframe_list = metadata.get("keyframes", [])

    keyframe_images = {}
    for kf in keyframe_list:
        vid_idx = kf["vid_idx"]
        path = kf["path"]
        full_path = os.path.join(keyframes_dir, path)
        if os.path.exists(full_path):
            try:
                img = PILImage.open(full_path).convert("RGB")
                keyframe_images[vid_idx] = (img, f"keyframes/{path}")
            except Exception:
                continue

    frame_visibility = {}
    for fidx_str, visible_names in raw_visibility.items():
        vid_idx = int(fidx_str)
        matched = []
        for raw_name in visible_names:
            json_name = _fuzzy_match_name(raw_name, target_names)
            matched.append(json_name if json_name else raw_name)
        frame_visibility[vid_idx] = matched

    return keyframe_images, frame_visibility


def _build_instance_frame_map(optimal_instance_frames, instance_count,
                               keyframe_images, keyframe_visibility,
                               target_categories):
    """合并 optimal_frames + keyframes 为每个实例的帧列表（去重）

    与 refine_other_objects_relations.py 的 build_object_to_frames 逻辑一致。

    参数:
        optimal_instance_frames: {inst_key: [(PIL.Image, src), ...]}
        instance_count: {category: int}
        keyframe_images: {vid_idx: (PIL.Image, src)}
        keyframe_visibility: {vid_idx: [name, ...]}
        target_categories: 目标类别列表
    返回:
        instance_to_frames: {inst_key: [(PIL.Image, src), ...]}
        instance_to_category: {inst_key: category}
    """
    instance_to_frames = defaultdict(list)
    instance_to_category = {}
    seen_sources = defaultdict(set)

    for cat in target_categories:
        n_inst = instance_count.get(cat, 1)
        if n_inst == 0:
            n_inst = 1

        for inst_idx in range(n_inst):
            inst_key = f"{cat}_{inst_idx}"
            instance_to_category[inst_key] = cat

            if inst_key in optimal_instance_frames:
                for img, src in optimal_instance_frames[inst_key]:
                    if src not in seen_sources[inst_key]:
                        instance_to_frames[inst_key].append((img, src))
                        seen_sources[inst_key].add(src)

    for vid_idx, visible_names in keyframe_visibility.items():
        if vid_idx not in keyframe_images:
            continue
        img, src = keyframe_images[vid_idx]
        for cat in target_categories:
            if cat in visible_names:
                n_inst = instance_count.get(cat, 1)
                if n_inst == 0:
                    n_inst = 1
                for inst_idx in range(n_inst):
                    inst_key = f"{cat}_{inst_idx}"
                    if src not in seen_sources[inst_key]:
                        instance_to_frames[inst_key].append((img, src))
                        seen_sources[inst_key].add(src)

    return dict(instance_to_frames), instance_to_category


def vlm_lookup_placement_strategy(supported_name, supporter_name, all_pairs,
                                  model, processor, instance_frames):
    """用VLM判断放置策略，按实例帧投票

    参数:
        supported_name: 被支撑物体名称（如 "bowl_0"）
        supporter_name: 支撑物名称（如 "table"）
        all_pairs: 所有支撑关系对
        model: VLM模型
        processor: VLM处理器
        instance_frames: [(PIL.Image, source_str), ...] 该实例的帧列表
    返回:
        策略字符串 (on_top/inside/against_side/hanging_below/leaning)
    """
    prompt = build_placement_prompt(supported_name, supporter_name, all_pairs)

    votes = Counter()
    for img, src in instance_frames:
        try:
            raw_text = _vlm_inference(img, model, processor, prompt, max_new_tokens=256)
            json_str = _extract_json_from_text(raw_text)
            if json_str:
                data = json.loads(json_str)
                strategy = data.get("strategy", "").strip().lower()
                if strategy in VALID_STRATEGIES:
                    votes[strategy] += 1
        except Exception:
            continue

    if votes:
        best_strategy = votes.most_common(1)[0][0]
        return best_strategy

    return "on_top"


def lookup_placement_strategy(supported_name, supporter_name):
    """无VLM时的规则回退，默认 on_top"""
    return "on_top"


def _get_transformed_mesh(instance_info, transform_matrix=None):
    """获取变换后的mesh"""
    mesh = instance_info["original_mesh"].copy()
    mesh.apply_transform(transform_matrix if transform_matrix is not None else instance_info["T"])
    return mesh


def _get_transformed_bounds(instance_info, transform_matrix=None):
    mesh = _get_transformed_mesh(instance_info, transform_matrix)
    return mesh.bounds


def _get_transformed_center(instance_info, transform_matrix=None):
    if transform_matrix is None:
        transform_matrix = instance_info["T"]
    center_local = np.mean(instance_info["original_mesh"].vertices, axis=0)
    center_world = trimesh.transformations.transform_points(
        np.array([center_local]), transform_matrix
    )[0]
    return center_world


def _align_upright(info):
    """旋转对齐: 让物体上方向对齐z轴 (与 refine_supported_by_floor_object 的旋转逻辑一致)"""
    transform_matrix = info["T"].copy()
    upper_real_vector = np.array([0, 0, 1])
    upper_transformed_vector = transform_matrix[:3, 1]
    upper_norm = np.linalg.norm(upper_transformed_vector)
    if upper_norm > 1e-8:
        upper_transformed_vector = upper_transformed_vector / upper_norm
        theta_gravity = np.arccos(np.clip(np.dot(upper_real_vector, upper_transformed_vector), -1.0, 1.0)) / np.pi * 180
        if theta_gravity < 10.0 or theta_gravity > 170.0:
            if theta_gravity < 90.0:
                upper_align_matrix = trimesh.geometry.align_vectors(upper_transformed_vector, upper_real_vector)
            else:
                upper_align_matrix = trimesh.geometry.align_vectors(upper_transformed_vector, -upper_real_vector)
            transform_matrix[:3, :3] = upper_align_matrix[:3, :3] @ transform_matrix[:3, :3]
    info["T"] = transform_matrix
    return info


def sp_refine_on_top(supported_info, supporter_info):
    """supported底面贴supporter顶面 — 等价于"把supporter顶面当作floor做refine_supported_by_floor_object"

    核心逻辑 (与 refine_supported_by_floor_object 完全一致，只是z=0换成supporter_top_z):
      Step 1: 旋转对齐 — 让物体上方向对齐z轴 (竖直)
      Step 2: z轴平移 — supported底面对齐到supporter顶面 (等同于floor的z=0对齐)

    物理约束:
      supported.bottom_z = supporter.top_z (底面贴顶面, 不穿模不悬空)
    """
    supported_info = _align_upright(supported_info)

    supporter_bounds = _get_transformed_bounds(supporter_info)
    supported_bounds = _get_transformed_bounds(supported_info)

    supporter_top_z = supporter_bounds[1, 2]
    supported_bottom_z = supported_bounds[0, 2]

    z_offset = supporter_top_z - supported_bottom_z

    if abs(z_offset) < 1e-6:
        return supported_info

    if z_offset < -0.5:
        return supported_info

    translation_vec = np.array([0.0, 0.0, z_offset])
    transform_matrix = supported_info["T"].copy()
    transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    supported_info["T"] = transform_matrix
    return supported_info


def sp_refine_inside(supported_info, supporter_info):
    """supported放入supporter内部，防止穿模

    物理约束:
      supported.bottom_z >= supporter.bottom_z (不穿出底部)
      supported.bottom_z <= supporter内部30%高度 (放在合理位置)
      supported.top_z <= supporter.top_z (不穿出顶部)

    实现:
      Step 1: 旋转对齐 — 让物体竖直
      Step 2: z轴平移 — supported底面对齐到supporter内部30%高度
      Step 3: 穿模检查 — 穿出顶部/底部则修正
    """
    supported_info = _align_upright(supported_info)

    transform_matrix = supported_info["T"].copy()

    supporter_bounds = _get_transformed_bounds(supporter_info)
    supported_bounds = _get_transformed_bounds(supported_info)

    supporter_bottom_z = supporter_bounds[0, 2]
    supporter_top_z = supporter_bounds[1, 2]
    supporter_height = supporter_top_z - supporter_bottom_z

    target_z = supporter_bottom_z + supporter_height * 0.3
    z_offset = target_z - supported_bounds[0, 2]

    translation_vec = np.array([0.0, 0.0, z_offset])
    transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    refined_bounds = _get_transformed_bounds(supported_info, transform_matrix)

    if refined_bounds[1, 2] > supporter_top_z:
        z_fix = supporter_top_z - refined_bounds[1, 2]
        transform_matrix = trimesh.transformations.translation_matrix(
            np.array([0.0, 0.0, z_fix])
        ) @ transform_matrix

    refined_bounds = _get_transformed_bounds(supported_info, transform_matrix)
    if refined_bounds[0, 2] < supporter_bottom_z:
        z_fix = supporter_bottom_z - refined_bounds[0, 2]
        transform_matrix = trimesh.transformations.translation_matrix(
            np.array([0.0, 0.0, z_fix])
        ) @ transform_matrix

    supported_info["T"] = transform_matrix
    return supported_info


def sp_refine_against_side(supported_info, supporter_info, walls_info=None):
    """supported靠在supporter侧面：底面贴地 + 侧面接触，防止穿模

    物理约束:
      1. 底面贴地: supported.bottom_z >= 0
      2. 侧面接触: supported与supporter在x/y轴上刚好接触，不穿入也不远离
      3. 不穿模: 侧面接触后，检查z轴是否穿入supporter内部，如果是则上移

    实现:
      Step 1: 旋转对齐 — 让物体竖直
      Step 2: z轴: 底面贴地
      Step 3: x/y轴: 找最近侧面方向，移动到刚好接触
      Step 4: 穿模检查
    """
    supported_info = _align_upright(supported_info)

    transform_matrix = supported_info["T"].copy()

    supported_bounds = _get_transformed_bounds(supported_info)
    if supported_bounds[0, 2] < 0.0:
        z_offset = -supported_bounds[0, 2]
        transform_matrix = trimesh.transformations.translation_matrix(
            np.array([0.0, 0.0, z_offset])
        ) @ transform_matrix

    supported_mesh = _get_transformed_mesh(supported_info, transform_matrix)
    supported_vertices = supported_mesh.vertices

    supporter_mesh = _get_transformed_mesh(supporter_info)
    supporter_vertices = supporter_mesh.vertices

    best_offset = 0.0
    best_dist = float("inf")
    best_axis = 0

    for axis_idx in [0, 1]:
        for sign in [1.0, -1.0]:
            if sign > 0:
                s_max = supported_vertices[:, axis_idx].max()
                r_min = supporter_vertices[:, axis_idx].min()
                offset = r_min - s_max
            else:
                s_min = supported_vertices[:, axis_idx].min()
                r_max = supporter_vertices[:, axis_idx].max()
                offset = r_max - s_min

            if abs(offset) < abs(best_dist):
                best_dist = offset
                best_offset = offset
                best_axis = axis_idx

    if abs(best_offset) < 1.0:
        translation_vec = np.array([0.0, 0.0, 0.0])
        translation_vec[best_axis] = best_offset
        transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    refined_mesh = _get_transformed_mesh(supported_info, transform_matrix)
    refined_bounds = refined_mesh.bounds
    supporter_bounds = supporter_mesh.bounds

    if (refined_bounds[0, 0] < supporter_bounds[1, 0] and
        refined_bounds[1, 0] > supporter_bounds[0, 0] and
        refined_bounds[0, 1] < supporter_bounds[1, 1] and
        refined_bounds[1, 1] > supporter_bounds[0, 1] and
        refined_bounds[0, 2] < supporter_bounds[1, 2]):

        z_fix = supporter_bounds[1, 2] - refined_bounds[0, 2]
        if z_fix > 0 and z_fix < supporter_bounds[1, 2] - supporter_bounds[0, 2]:
            transform_matrix = trimesh.transformations.translation_matrix(
                np.array([0.0, 0.0, z_fix])
            ) @ transform_matrix

    supported_info["T"] = transform_matrix
    return supported_info


def sp_refine_hanging_below(supported_info, supporter_info):
    """supported悬挂在supporter下方：顶面贴支撑物底面，防止穿模

    物理约束:
      supported.top_z <= supporter.bottom_z (不穿入支撑物)
      supported.top_z >= supporter.bottom_z - eps (不远离)

    实现:
      Step 1: 旋转对齐 — 让物体竖直
      Step 2: z轴平移 — supported顶面对齐到supporter底面
    """
    supported_info = _align_upright(supported_info)

    transform_matrix = supported_info["T"].copy()

    supporter_bounds = _get_transformed_bounds(supporter_info)
    supported_bounds = _get_transformed_bounds(supported_info)

    supporter_bottom_z = supporter_bounds[0, 2]
    supported_top_z = supported_bounds[1, 2]

    z_offset = supporter_bottom_z - supported_top_z

    if abs(z_offset) < 1e-6:
        return supported_info

    translation_vec = np.array([0.0, 0.0, z_offset])
    transform_matrix = trimesh.transformations.translation_matrix(translation_vec) @ transform_matrix

    supported_info["T"] = transform_matrix
    return supported_info


def sp_refine_leaning(supported_info, supporter_info):
    """斜靠在支撑物上：等同于 against_side"""
    return sp_refine_against_side(supported_info, supporter_info)


SP_REFINE_MAP = {
    "on_top": sp_refine_on_top,
    "inside": sp_refine_inside,
    "against_side": sp_refine_against_side,
    "hanging_below": sp_refine_hanging_below,
    "leaning": sp_refine_leaning,
}


def _find_supporter_instances(supporter_name, all_instances):
    """在all_instances中按名称查找实例列表，返回 (category, instances) 或 (None, None)

    支持实例级匹配:
      - "toy_2" → toy类别第2个实例(索引2)
      - "toy" → toy类别全部实例
      - "table" → table类别全部实例
    """
    supporter_lower = supporter_name.lower().strip().replace(" ", "_")

    instance_idx = None
    cat_name = supporter_lower
    if "_" in supporter_lower:
        parts = supporter_lower.rsplit("_", 1)
        if parts[1].isdigit():
            cat_name = parts[0]
            instance_idx = int(parts[1])

    for category, instances in all_instances.items():
        cat_lower = category.lower().strip().replace(" ", "_")
        if cat_lower == cat_name:
            if instance_idx is not None and instance_idx < len(instances):
                return category, [instances[instance_idx]]
            return category, instances

    for category, instances in all_instances.items():
        cat_base = category.lower().split("_")[0].split(" ")[0]
        supp_base = cat_name.split("_")[0].split(" ")[0]
        if cat_base == supp_base:
            if instance_idx is not None and instance_idx < len(instances):
                return category, [instances[instance_idx]]
            return category, instances

    for category, instances in all_instances.items():
        if cat_name in category.lower() or category.lower() in cat_name:
            if instance_idx is not None and instance_idx < len(instances):
                return category, [instances[instance_idx]]
            return category, instances

    return None, None


def _find_nearest_supporter_instance(supported_info, supporter_instances):
    """在多个支撑物实例中找到最近的那个"""
    if len(supporter_instances) == 1:
        return supporter_instances[0]

    supported_center = _get_transformed_center(supported_info)
    best_instance = supporter_instances[0]
    best_dist = float("inf")

    for inst in supporter_instances:
        inst_center = _get_transformed_center(inst)
        dist = np.linalg.norm(supported_center - inst_center)
        if dist < best_dist:
            best_dist = dist
            best_instance = inst

    return best_instance


def _find_relationship_for_category(category, categories_and_relations):
    """模糊匹配找到某个 category 对应的 relationship（支持 instance-level key）"""
    cat_lower = category.lower().strip()
    for key, rel in categories_and_relations.items():
        key_lower = key.lower().strip()
        if key_lower == cat_lower:
            return rel
        if key_lower.startswith(cat_lower) and (len(key_lower) == len(cat_lower) or key_lower[len(cat_lower)] == '_'):
            return rel
    return "supported by floor"


def _topological_sort_pairs(supported_pairs):
    """依赖排序: 支撑物先处理，被支撑物后处理

    例如: toy→box→table 的处理顺序应为 table, box, toy
    这样 box 放到 table 上之后，toy 再放到 box 上时位置才正确。
    """
    supporter_set = set()
    supported_set = set()
    for supported, supporter in supported_pairs:
        supporter_set.add(supporter)
        supported_set.add(supported)

    roots = supporter_set - supported_set

    order = []
    visited = set()
    queue = list(roots)

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for supported, supporter in supported_pairs:
            if supporter == current and supported not in visited:
                order.append((supported, supporter))
                queue.append(supported)

    for supported, supporter in supported_pairs:
        if (supported, supporter) not in order:
            order.append((supported, supporter))

    return order


def _get_aabb_overlap(mesh_a, mesh_b):
    """计算两个mesh的AABB重叠量

    返回: (overlaps, overlap_x, overlap_y, overlap_z)
        overlaps: 是否有重叠
        overlap_x/y/z: 各轴的重叠深度（负值表示不重叠）
    """
    bounds_a = mesh_a.bounds
    bounds_b = mesh_b.bounds

    overlap_x = min(bounds_a[1, 0], bounds_b[1, 0]) - max(bounds_a[0, 0], bounds_b[0, 0])
    overlap_y = min(bounds_a[1, 1], bounds_b[1, 1]) - max(bounds_a[0, 1], bounds_b[0, 1])
    overlap_z = min(bounds_a[1, 2], bounds_b[1, 2]) - max(bounds_a[0, 2], bounds_b[0, 2])

    overlaps = overlap_x > 0 and overlap_y > 0 and overlap_z > 0
    return overlaps, overlap_x, overlap_y, overlap_z


def _check_mesh_penetration(mesh_a, mesh_b, n_samples=500):
    """Check if mesh_a penetrates into mesh_b using vertex sampling.

    Returns:
        (penetrates, penetration_depth, sep_axis):
            penetrates: bool, whether penetration exists
            penetration_depth: float, estimated penetration depth
            sep_axis: int (0=x, 1=y, 2=z), best separation axis
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        overlaps, ox, oy, oz = _get_aabb_overlap(mesh_a, mesh_b)
        if not overlaps:
            return False, 0.0, 2
        min_overlap = min(ox, oy, oz)
        if oz == min_overlap:
            return True, oz, 2
        elif oy == min_overlap:
            return True, oy, 1
        return True, ox, 0

    if len(mesh_a.vertices) < 10 or len(mesh_b.vertices) < 10:
        overlaps, ox, oy, oz = _get_aabb_overlap(mesh_a, mesh_b)
        if not overlaps:
            return False, 0.0, 2
        min_overlap = min(ox, oy, oz)
        if oz == min_overlap:
            return True, oz, 2
        elif oy == min_overlap:
            return True, oy, 1
        return True, ox, 0

    overlaps, ox, oy, oz = _get_aabb_overlap(mesh_a, mesh_b)
    if not overlaps:
        return False, 0.0, 2

    try:
        pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    except Exception:
        pts_a = mesh_a.vertices
    if len(pts_a) == 0:
        pts_a = mesh_a.vertices

    tree_b = cKDTree(mesh_b.vertices)
    dists, _ = tree_b.query(pts_a)

    inside_count = (dists < 0.01).sum()
    if inside_count < 3:
        return False, 0.0, 2

    inside_pts = pts_a[dists < 0.01]
    center_a = mesh_a.bounds.mean(axis=0)
    center_b = mesh_b.bounds.mean(axis=0)
    diff = center_a - center_b

    abs_diff = np.abs(diff)
    sep_axis = int(np.argmax(abs_diff))

    penetration_depth = float(np.mean(dists[dists < 0.01])) + 0.005

    return True, penetration_depth, sep_axis


def resolve_penetrations(all_instances, refined_relations=None, verbose=True,
                         categories_and_relations=None):
    """全局穿模解决: 检测并修复所有物体对之间的穿模

    策略:
      1. 对每对物体先用AABB快速排除，再用顶点采样精确检测穿模
      2. 如果穿模，沿分离轴推开（优先推被支撑物，保留支撑物不动）
      3. 推开后确保不穿出地面(z=0)
      4. 迭代直到无穿模或达到最大迭代次数

    参数:
        all_instances: {category: [instance_info, ...]}
        refined_relations: 关系字典，用于判断谁是被支撑物（优先移动）
        verbose: 是否打印详细信息
    返回:
        更新后的 all_instances
    """
    if verbose:
        print("\n🛡️ 全局穿模检测与解决...", flush=True)

    supported_names = set()
    floor_names = set()
    wall_names = set()
    _all_rels = {}
    if categories_and_relations:
        _all_rels.update(categories_and_relations)
    if refined_relations:
        _all_rels.update(refined_relations)
    for name, rel in _all_rels.items():
        rel_lower = rel.lower()
        if rel.startswith("supported by ") and "floor" not in rel_lower and "other objects" not in rel_lower:
            base = name.rsplit('_', 1)[0] if '_' in name else name
            supported_names.add(base.lower().strip())
            supported_names.add(name.lower().strip())
        if "floor" in rel_lower:
            base = name.rsplit('_', 1)[0] if '_' in name else name
            floor_names.add(base.lower().strip())
            floor_names.add(name.lower().strip())
        if "wall" in rel_lower:
            base = name.rsplit('_', 1)[0] if '_' in name else name
            wall_names.add(base.lower().strip())
            wall_names.add(name.lower().strip())

    all_meshes = []
    for category, instances in all_instances.items():
        for idx, info in enumerate(instances):
            mesh = _get_transformed_mesh(info)
            all_meshes.append((category, idx, mesh, info))

    max_iterations = 1
    for iteration in range(max_iterations):
        any_resolved = False

        for i in range(len(all_meshes)):
            for j in range(i + 1, len(all_meshes)):
                cat_i, idx_i, mesh_i, info_i = all_meshes[i]
                cat_j, idx_j, mesh_j, info_j = all_meshes[j]

                overlaps, ox, oy, oz = _get_aabb_overlap(mesh_i, mesh_j)
                if not overlaps:
                    continue

                penetrates, pen_depth, sep_axis = _check_mesh_penetration(mesh_i, mesh_j)
                if not penetrates:
                    continue

                cat_i_is_supported = cat_i.lower().strip() in supported_names
                cat_j_is_supported = cat_j.lower().strip() in supported_names
                cat_i_is_floor = cat_i.lower().strip() in floor_names
                cat_j_is_floor = cat_j.lower().strip() in floor_names
                cat_i_is_wall = cat_i.lower().strip() in wall_names
                cat_j_is_wall = cat_j.lower().strip() in wall_names

                if (cat_i_is_floor or cat_i_is_wall) and (cat_j_is_floor or cat_j_is_wall):
                    continue

                if cat_i_is_supported and not cat_j_is_supported:
                    move_idx = i
                elif cat_j_is_supported and not cat_i_is_supported:
                    move_idx = j
                elif (cat_i_is_floor or cat_i_is_wall) and not (cat_j_is_floor or cat_j_is_wall):
                    move_idx = j
                elif (cat_j_is_floor or cat_j_is_wall) and not (cat_i_is_floor or cat_i_is_wall):
                    move_idx = i
                else:
                    center_i = mesh_i.bounds.mean(axis=0)
                    center_j = mesh_j.bounds.mean(axis=0)
                    if center_i[sep_axis] > center_j[sep_axis]:
                        move_idx = i
                    else:
                        move_idx = j

                move_cat, move_idx_orig, move_mesh, move_info = all_meshes[move_idx]
                other_mesh = mesh_j if move_idx == i else mesh_i
                move_is_fixed = move_cat.lower().strip() in floor_names or move_cat.lower().strip() in wall_names

                move_center = move_mesh.bounds.mean(axis=0)[sep_axis]
                other_center = other_mesh.bounds.mean(axis=0)[sep_axis]

                if move_center >= other_center:
                    direction = 1.0
                else:
                    direction = -1.0

                if move_is_fixed and sep_axis == 2:
                    continue

                sep_dist = pen_depth + 0.0005
                translation_vec = np.array([0.0, 0.0, 0.0])
                translation_vec[sep_axis] = direction * sep_dist

                new_T = trimesh.transformations.translation_matrix(translation_vec) @ move_info["T"]
                move_info["T"] = new_T

                new_mesh = _get_transformed_mesh(move_info)
                if new_mesh.bounds[0, 2] < 0.0:
                    z_fix = -new_mesh.bounds[0, 2]
                    move_info["T"] = trimesh.transformations.translation_matrix(
                        np.array([0.0, 0.0, z_fix])
                    ) @ move_info["T"]
                    new_mesh = _get_transformed_mesh(move_info)

                all_meshes[move_idx] = (move_cat, move_idx_orig, new_mesh, move_info)
                all_instances[move_cat][move_idx_orig] = move_info

                any_resolved = True
                if verbose:
                    pair_desc = f"{cat_i}_{idx_i} ↔ {cat_j}_{idx_j}"
                    axis_name = ['x', 'y', 'z'][sep_axis]
                    print(f"      🔧 穿模修复: {pair_desc} | {axis_name}轴分离 {sep_dist:.4f}m", flush=True)

        if not any_resolved:
            break

    if verbose:
        print(f"   ✅ 穿模检测完成 (迭代{iteration + 1}次)", flush=True)

    return all_instances


def refine_inter_object_relations(all_instances, refined_relations,
                                  walls_info=None, verbose=True,
                                  vlm_checkpoint=None, scene_dir=None,
                                  categories_and_relations=None):
    """
    主函数: 精修物体间支撑关系的空间位置

    策略判定优先级:
      1. VLM (如果提供 vlm_checkpoint + scene_dir) → 按实例帧投票
         帧来源: optimal_frames/ + keyframes/ (与 refine_other_objects_relations.py 一致)
      2. 规则回退 → 默认 on_top

    参数:
        all_instances: {category: [instance_info, ...]}
        refined_relations: {name: relationship}  细化后的关系
        walls_info: 墙面信息（供 against_side 策略使用）
        verbose: 是否打印详细信息
        vlm_checkpoint: VLM模型路径（可选）
        scene_dir: 场景输出目录（如 outputs/232，包含 optimal_frames/ 和 keyframes/）
    返回:
        更新后的 all_instances
    """
    if verbose:
        print("=" * 70, flush=True)
        print("🔧 物体间支撑关系空间位置精修", flush=True)
        print("=" * 70, flush=True)

    supported_pairs = []
    for name, rel in refined_relations.items():
        if rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel:
            supporter_name = rel.replace("supported by ", "").strip()
            supported_pairs.append((name, supporter_name))

    if verbose:
        print(f"\n📋 发现 {len(supported_pairs)} 对物体间支撑关系:", flush=True)
        for supported, supporter in supported_pairs:
            print(f"   {supported} ← {supporter}", flush=True)

    if not supported_pairs:
        if verbose:
            print(f"\n✅ 无需精修", flush=True)
        return all_instances

    # ── 策略判定 ──
    placement_strategies = {}

    use_vlm = vlm_checkpoint is not None and scene_dir is not None

    if use_vlm:
        resolved_ckpt = _resolve_vlm_checkpoint(vlm_checkpoint)
        if not resolved_ckpt:
            use_vlm = False
            if verbose:
                print(f"\n⚠️ VLM模型未找到，回退到规则匹配", flush=True)

    if use_vlm:
        if verbose:
            print(f"\n📐 放置策略（VLM判定，帧来源: optimal_frames/ + keyframes/）:", flush=True)

        # 收集所有涉及的物体类别（被支撑物 + 支撑物）
        all_categories = set()
        for supported_name, supporter_name in supported_pairs:
            all_categories.add(supported_name)
            all_categories.add(supporter_name)
        # 也加入 refined_relations 中的所有 key 以支持模糊匹配
        for name in refined_relations.keys():
            all_categories.add(name)

        # 加载帧: optimal_frames/ + keyframes/
        optimal_frames_dir = os.path.join(scene_dir, "optimal_frames")
        keyframes_dir = os.path.join(scene_dir, "keyframes")

        # 回退: 如果 {scene_dir}/keyframes/ 不存在，尝试 assets/key_frames/{scene_id}/
        if not os.path.isdir(keyframes_dir):
            scene_id = os.path.basename(os.path.normpath(scene_dir))
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            fallback_dir = os.path.join(project_root, "assets", "key_frames", scene_id)
            if os.path.isdir(fallback_dir):
                keyframes_dir = fallback_dir
                if verbose:
                    print(f"   ⚠️ {scene_dir}/keyframes/ 不存在，回退到 {fallback_dir}", flush=True)

        if verbose:
            print(f"\n   📷 [来源1] 加载optimal_frames: {optimal_frames_dir}", flush=True)
        optimal_instance_frames, instance_count = _load_optimal_frames_for_instances(
            optimal_frames_dir, list(all_categories)
        )
        if verbose:
            total_opt = sum(len(v) for v in optimal_instance_frames.values())
            print(f"      optimal_frames/: {total_opt} 帧, 覆盖 {len(optimal_instance_frames)} 个实例", flush=True)

        if verbose:
            print(f"   📷 [来源2] 加载keyframes: {keyframes_dir}", flush=True)
        keyframe_images, keyframe_visibility = _load_keyframes_for_instances(
            keyframes_dir, list(all_categories)
        )
        if verbose:
            print(f"      keyframes/: {len(keyframe_images)} 个关键帧, {len(keyframe_visibility)} 帧可见性数据", flush=True)

        # 合并帧来源
        instance_to_frames, instance_to_category = _build_instance_frame_map(
            optimal_instance_frames, instance_count,
            keyframe_images, keyframe_visibility,
            list(all_categories),
        )

        if verbose:
            print(f"\n   📋 实例→帧映射:", flush=True)
            for inst_key in sorted(instance_to_category.keys()):
                frames = instance_to_frames.get(inst_key, [])
                if frames:
                    sources = [src for _, src in frames]
                    print(f"      {inst_key}: {len(frames)} 帧 → {sources}", flush=True)
                else:
                    print(f"      {inst_key}: ❌ 无帧", flush=True)

        # 加载VLM
        vlm_model, vlm_processor = _load_vlm_model(resolved_ckpt)

        # 按实例投票判定策略
        for supported_name, supporter_name in supported_pairs:
            # 找到该物体的所有实例帧
            best_strategy = None
            best_votes = 0

            # 遍历所有实例 key，找到属于 supported_name 的
            for inst_key, frames in instance_to_frames.items():
                cat = instance_to_category.get(inst_key, "")
                # 模糊匹配: inst_key 的 category 部分匹配 supported_name
                if cat != supported_name and not _fuzzy_match_name(supported_name, [cat]):
                    continue
                if not frames:
                    continue

                strategy = vlm_lookup_placement_strategy(
                    inst_key, supporter_name, supported_pairs,
                    vlm_model, vlm_processor, frames,
                )
                # 多实例投票: 取出现最多的策略
                if strategy not in placement_strategies.values():
                    pass
                # 简化: 取第一个有帧实例的策略（多实例场景下可扩展为投票）
                if best_strategy is None:
                    best_strategy = strategy

            if best_strategy is None:
                best_strategy = "on_top"

            placement_strategies[supported_name] = best_strategy
            desc = PLACEMENT_STRATEGIES.get(best_strategy, "")
            if verbose:
                print(f"   {supported_name}: {best_strategy} — {desc} (VLM)", flush=True)

        import torch
        import gc
        del vlm_model, vlm_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if verbose:
            print(f"\n📐 放置策略（规则回退，默认 on_top）:", flush=True)
        for supported_name, supporter_name in supported_pairs:
            strategy = lookup_placement_strategy(supported_name, supporter_name)
            placement_strategies[supported_name] = strategy
            desc = PLACEMENT_STRATEGIES.get(strategy, "")
            if verbose:
                print(f"   {supported_name}: {strategy} — {desc}", flush=True)

    # ── SP精修 (按依赖排序: 支撑物先处理) ──
    if verbose:
        print(f"\n🔧 SP精修开始 (依赖排序: 支撑物优先)...", flush=True)

    sorted_pairs = _topological_sort_pairs(supported_pairs)
    if verbose and sorted_pairs != supported_pairs:
        print(f"   📋 处理顺序:", flush=True)
        for supported, supporter in sorted_pairs:
            print(f"      {supporter} → {supported}", flush=True)

    for supported_name, supporter_name in sorted_pairs:
        strategy = placement_strategies.get(supported_name, "on_top")

        supported_cat, supported_instances = _find_supporter_instances(supported_name, all_instances)
        if supported_cat is None:
            if verbose:
                print(f"   ⚠️ 跳过 '{supported_name}': 未找到对应实例", flush=True)
            continue

        supporter_cat, supporter_instances = _find_supporter_instances(supporter_name, all_instances)
        if supporter_cat is None:
            if verbose:
                print(f"   ⚠️ 跳过 '{supported_name}': 支撑物 '{supporter_name}' 未找到", flush=True)
            continue

        sp_refine_fn = SP_REFINE_MAP.get(strategy, sp_refine_on_top)

        for inst_idx, supported_info in enumerate(supported_instances):
            supporter_info = _find_nearest_supporter_instance(supported_info, supporter_instances)

            if verbose:
                print(f"   🔧 {supported_name}[{inst_idx}] → {strategy} → {supporter_name}", flush=True)

            try:
                if strategy in ("against_side", "leaning"):
                    supported_instances[inst_idx] = sp_refine_fn(
                        supported_info, supporter_info, walls_info
                    )
                else:
                    supported_instances[inst_idx] = sp_refine_fn(supported_info, supporter_info)
            except Exception as e:
                if verbose:
                    print(f"      ❌ 精修失败: {e}", flush=True)

    # ── 全局穿模解决 ──
    all_instances = resolve_penetrations(all_instances, refined_relations, verbose=verbose,
                                         categories_and_relations=categories_and_relations)

    if verbose:
        print(f"\n✅ 物体间关系精修完成!", flush=True)

    return all_instances


def refine_full_scene(all_instances, categories_and_relations, walls_info,
                      extrinsics=None, all_optimal_frame_ids=None, verbose=True,
                      vlm_checkpoint=None, scene_dir=None):
    """
    完整场景精修: 处理 floor + wall + inter-object 所有关系

    参数:
        all_instances: {category: [instance_info, ...]}
        categories_and_relations: {category: relationship}
        walls_info: 墙面信息
        extrinsics: 外参矩阵列表（用于 attached_to_wall 的 camera_pos）
        all_optimal_frame_ids: {category: [frame_id, ...]}
        verbose: 是否打印详细信息
        vlm_checkpoint: VLM模型路径（可选，用于策略判定）
        scene_dir: 场景输出目录（可选，包含 optimal_frames/ 和 keyframes/）
    返回:
        更新后的 all_instances
    """
    if verbose:
        print("=" * 70, flush=True)
        print("🔧 完整场景精修 (floor + wall + inter-object)", flush=True)
        print("=" * 70, flush=True)

    from src.sp_refinement import (
        refine_supported_by_floor_object,
        refine_embedded_in_wall_object,
        refine_attached_to_wall_object,
    )

    for category, category_instances in all_instances.items():
        relationship = _find_relationship_for_category(category, categories_and_relations)

        for instance_id, instance_info in enumerate(category_instances):
            if verbose:
                print(f"   {category}[{instance_id}]: {relationship}", flush=True)

            if relationship == "supported by floor" or relationship == "supported_by_floor":
                instance_info = refine_supported_by_floor_object(instance_info)

            elif relationship == "embedded in wall" or relationship == "embedded_in_wall":
                instance_info = refine_embedded_in_wall_object(instance_info, walls_info)

            elif relationship == "attached to wall" or relationship == "attached_to_wall":
                if extrinsics is not None and all_optimal_frame_ids is not None:
                    optimal_frame_id = all_optimal_frame_ids.get(category, [0])[
                        min(instance_id, len(all_optimal_frame_ids.get(category, [0])) - 1)
                    ]
                    extrinsic = extrinsics[optimal_frame_id]
                    camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
                else:
                    camera_pos = None
                instance_info = refine_attached_to_wall_object(instance_info, walls_info, camera_pos)

            category_instances[instance_id] = instance_info

    all_instances = refine_inter_object_relations(
        all_instances, categories_and_relations,
        walls_info=walls_info, verbose=verbose,
        vlm_checkpoint=vlm_checkpoint, scene_dir=scene_dir,
    )

    return all_instances


def _load_instances_from_glb(glb_path):
    """从 GLB 文件加载 all_instances (y-up → z-up)

    GLB文件是y-up格式，但SP精修函数需要z-up格式。
    加载时先做 y-up → z-up 反变换，精修完成后再转回y-up保存。
    """
    scene = trimesh.load(glb_path)

    y_up_to_z_up = np.array([
        [1, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    all_instances = defaultdict(list)
    g = scene.graph

    for node_name in g.nodes_geometry:
        if node_name == 'world':
            continue
        parts = node_name.rsplit('_', 1)
        if len(parts) != 2:
            continue
        category = parts[0]

        _, geom_name = g[node_name]
        if geom_name not in scene.geometry:
            continue

        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(y_up_to_z_up)
        all_instances[category].append({
            "original_mesh": mesh,
            "T": np.eye(4),
        })

    return dict(all_instances)


def _save_scene_to_glb(all_instances, output_glb):
    """将 all_instances 导出为 GLB 文件"""
    scene = trimesh.Scene()
    for category, category_instances in all_instances.items():
        for i, instance_info in enumerate(category_instances):
            mesh = instance_info['original_mesh'].copy()
            mesh.apply_transform(instance_info['T'])
            scene.add_geometry(mesh, node_name=f"{category}_{i}")
    scene.apply_transform(np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]))
    os.makedirs(os.path.dirname(output_glb) or '.', exist_ok=True)
    scene.export(output_glb)


def main():
    parser = argparse.ArgumentParser(description='物体间支撑关系空间位置精修')
    parser.add_argument('--input_glb', type=str, default=None,
                        help='输入 GLB 文件路径')
    parser.add_argument('--relations_json', type=str, required=True,
                        help='细化后的关系 JSON 路径')
    parser.add_argument('--output_glb', type=str, default=None,
                        help='输出 GLB 文件路径（默认为输入同目录下 *_refined.glb）')
    parser.add_argument('--instances_pkl', type=str, default=None,
                        help='[可选] all_instances pickle 路径')
    parser.add_argument('--vlm_checkpoint', type=str, default=None,
                        help=f'VLM模型路径 (默认: {DEFAULT_VLM_CHECKPOINT})')
    parser.add_argument('--scene_dir', type=str, default=None,
                        help='场景输出目录（含 optimal_frames/ 和 keyframes/，VLM用）')

    args = parser.parse_args()

    if not args.input_glb and not args.instances_pkl:
        parser.error("请提供 --input_glb 或 --instances_pkl 中的一个")

    print("=" * 70, flush=True)
    print("🚀 物体间支撑关系空间位置精修", flush=True)
    print("=" * 70, flush=True)

    with open(args.relations_json, 'r') as f:
        refined_relations = json.load(f)

    print(f"\n📋 关系列表:", flush=True)
    for name, rel in sorted(refined_relations.items()):
        print(f"   {name}: {rel}", flush=True)

    inter_object_pairs = [
        (name, rel.replace("supported by ", "").strip())
        for name, rel in refined_relations.items()
        if rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel
    ]

    if not inter_object_pairs:
        print(f"\n✅ 无物体间支撑关系，无需精修", flush=True)
        return

    import pickle

    if args.input_glb:
        print(f"\n📂 从 GLB 加载: {args.input_glb}", flush=True)
        all_instances = _load_instances_from_glb(args.input_glb)
    elif args.instances_pkl and os.path.exists(args.instances_pkl):
        print(f"\n📂 从 pickle 加载: {args.instances_pkl}", flush=True)
        with open(args.instances_pkl, 'rb') as f:
            all_instances = pickle.load(f)
    else:
        print(f"\n❌ 无有效输入文件", flush=True)
        return

    print(f"📦 加载 {sum(len(v) for v in all_instances.values())} 个物体, "
          f"{len(all_instances)} 个类别", flush=True)

    scene_dir = args.scene_dir
    if scene_dir is None and args.input_glb:
        scene_dir = os.path.dirname(args.input_glb)

    all_instances = refine_inter_object_relations(
        all_instances, refined_relations,
        vlm_checkpoint=args.vlm_checkpoint,
        scene_dir=scene_dir,
    )

    if args.input_glb:
        output_glb = args.output_glb or f"{os.path.splitext(args.input_glb)[0]}_refined.glb"
    else:
        output_glb = args.output_glb or os.path.join(
            os.path.dirname(args.relations_json), "refined_scene.glb"
        )

    _save_scene_to_glb(all_instances, output_glb)
    print(f"\n💾 保存精修后的场景: {output_glb}", flush=True)
    print(f"\n✅ 完成!", flush=True)


if __name__ == '__main__':
    main()
