"""
Stage 5.1 替代方案: 使用 SimRecon 场景图推断方式获取物体间关系
================================================================

与 refine_other_objects_relations.py 的区别:
  旧方式: 逐个物体查看图像，判断 "supported by other objects" 的具体支撑物
  新方式: 在一张带ID标签的场景图上，一次性推断所有物体的类别和关系

优势:
  1. 支持多级嵌套关系 (floor → table → cup)
  2. 单次VLM调用推断所有关系，上下文更完整
  3. 物理常识后处理更完善 (柜子不应挂在墙等)

精修方式仍使用 sp_refinement.py 的几何精修函数。

使用方式:
  # 从 mainv2.py 自动调用 (推荐)
  python mainv2.py --input_video ./232.mp4 --enable_stage5 --stage5_method scene_graph

  # 独立调用
  python tools/infer_relations_scene_graph.py \
      --scene_dir output_v2/232_vggt \
      --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B

  # 指定 ID 标注图
  python tools/infer_relations_scene_graph.py \
      --scene_dir output_v2/232_vggt \
      --id_scene_path output_v2/232_vggt/id_scene.png \
      --vlm_checkpoint /mnt/data/lza/models/Qwen3.5-9B
"""

import os
import sys
import json
import re
import argparse
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from PIL import Image


FILTER_CATEGORIES = {
    'floor', 'wall', 'ground', 'ceiling', 'floor_area', 'wall_section',
    'flooring', 'wall_surface', 'room', 'space'
}

MUST_BE_FLOOR_SUPPORTED = {
    'cabinet', 'wardrobe', 'closet', 'dresser', 'bookshelf', 'bookcase',
    'shelf', 'shelving', 'desk', 'table', 'chair', 'sofa', 'couch',
    'bed', 'mattress', 'nightstand', 'bench', 'stool', 'armchair',
    'refrigerator', 'fridge', 'washing_machine', 'dryer', 'oven', 'stove',
    'tv_stand', 'entertainment_center', 'sideboard', 'buffet',
    'filing_cabinet', 'storage_cabinet', 'kitchen_cabinet',
    'plant', 'potted_plant', 'trash_can', 'garbage_can', 'bin',
    'box', 'luggage', 'suitcase', 'bag', 'backpack',
    'carpet', 'rug', 'mat', 'ottoman', 'footstool'
}

MUST_BE_WALL_ATTACHED = {
    'picture', 'painting', 'photo', 'photo_frame', 'frame',
    'mirror', 'clock', 'wall_clock', 'poster', 'art',
    'whiteboard', 'blackboard', 'bulletin_board',
    'tv', 'television', 'monitor',
    'light_switch', 'outlet', 'socket',
    'curtain', 'drape', 'blind', 'shade',
    'window', 'door', 'vent', 'air_vent'
}


def create_id_labeled_image(
    frame_image: np.ndarray,
    instance_masks: Dict[str, List[Dict]],
    categories_and_relations: Dict[str, str],
    display_id_offset: int = 3,
    best_frame_id: int = None,
) -> np.ndarray:
    """
    在帧图像上为每个可见实例绘制ID标注 (绿色边框 + 红色编号)

    参数:
        frame_image: RGB图像 (H, W, 3)
        instance_masks: {category: [{frame_id, mask}, ...]} 每个实例的mask数据
        categories_and_relations: {category: relation} 物体关系
        display_id_offset: 显示ID偏移 (与SimRecon一致, floor=1, wall=2, 物体从3开始)
        best_frame_id: 当前标注的帧ID, 用于从每个实例的mask列表中选取对应帧的mask
    返回:
        标注后的RGB图像
    """
    annotated = frame_image.copy()
    display_id = display_id_offset

    category_to_display_ids = {}

    for category in categories_and_relations:
        if category.lower() in FILTER_CATEGORIES:
            continue

        cat_masks = instance_masks.get(category, [])
        if not cat_masks:
            continue

        category_to_display_ids[category] = []

        for inst_idx, inst_mask_list in enumerate(cat_masks):
            # 优先选 best_frame_id 对应的 mask; 找不到则回退到第一个
            mask_for_frame = None
            if best_frame_id is not None:
                for im in inst_mask_list:
                    if im.get('frame_id') == best_frame_id:
                        mask_for_frame = im['mask']
                        break
            if mask_for_frame is None and inst_mask_list:
                mask_for_frame = inst_mask_list[0]['mask']

            if mask_for_frame is None:
                category_to_display_ids[category].append(display_id)
                display_id += 1
                continue

            if mask_for_frame.shape[:2] != annotated.shape[:2]:
                category_to_display_ids[category].append(display_id)
                display_id += 1
                continue

            ys, xs = np.where(mask_for_frame > 0)
            if len(ys) == 0:
                category_to_display_ids[category].append(display_id)
                display_id += 1
                continue

            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())

            cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

            label = str(display_id)
            font_scale = max(0.8, min(2.0, (x_max - x_min) / 50.0))
            thickness = max(2, int(font_scale))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_x = x_min
            label_y = max(y_min - 5, th + 5)
            cv2.rectangle(annotated, (label_x, label_y - th - 5), (label_x + tw + 5, label_y + 5), (0, 0, 255), -1)
            cv2.putText(annotated, label, (label_x + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

            category_to_display_ids[category].append(display_id)
            display_id += 1

    return annotated, category_to_display_ids


def select_best_frame_for_labeling(
    instance_masks: Dict[str, List[Dict]],
    colors: List[np.ndarray],
) -> int:
    """
    选择显示最多物体的帧用于ID标注

    参数:
        instance_masks: {category: [{frame_id, mask}, ...]}
        colors: 帧图像列表
    返回:
        最佳帧ID
    """
    frame_object_count = defaultdict(int)

    for category, cat_masks in instance_masks.items():
        for inst_masks in cat_masks:
            visible_frames = set()
            for im in inst_masks:
                visible_frames.add(im['frame_id'])
            for fid in visible_frames:
                frame_object_count[fid] += 1

    if not frame_object_count:
        return 0

    best_frame = max(frame_object_count, key=frame_object_count.get)
    return best_frame


def infer_relations_from_scene_graph(
    model,
    processor,
    image_path: str,
    object_ids: List[int],
    category_names: List[str],
) -> Dict[str, Any]:
    """
    使用 SimRecon 风格的 VLM prompt 推断物体间关系

    参数:
        model: VLM模型
        processor: VLM处理器
        image_path: ID标注图路径
        object_ids: 显示ID列表 (从 display_id_offset 开始)
        category_names: 对应的类别名列表
    返回:
        场景图推断结果
    """
    display_ids_str = ", ".join(map(str, object_ids))
    num_objects = len(object_ids)

    id_to_name = {}
    for did, name in zip(object_ids, category_names):
        id_to_name[did] = name
    id_name_hints = "\n".join([f"- ID {did} = {name}" for did, name in zip(object_ids, category_names)])

    prompt = f'''You are analyzing a room scene image with numbered labels (green boxes with red numbers).

**Visible object IDs**: {display_ids_str} (Total: {num_objects} objects)

**Hidden root nodes** (not shown in image):
- ID 1 = "floor" (ground surface)
- ID 2 = "wall" (vertical wall surface)

**Object name hints** (use these as category names):
{id_name_hints}

## STEP-BY-STEP ANALYSIS (Follow this order strictly):

### Step 1: Identify ALL objects
For EACH of the {num_objects} IDs ({display_ids_str}), confirm what the object is.
- Use the provided name hints as the primary reference
- Two objects with the SAME category at different positions are DIFFERENT objects

### Step 2: Build FLOOR tree (support relations)
Think about what objects are SUPPORTED BY FLOOR (standing on ground):
- Furniture: desk, table, chair, sofa, bed, cabinet, wardrobe, shelf, bookcase, dresser, nightstand
- Appliances: refrigerator, washing machine, TV stand
- Other: carpet, rug, plant pot, trash can, box, luggage

Then think: what objects are ON TOP of these floor-supported objects?
- Example chain: floor → desk → lamp → cup

### Step 3: Build WALL tree (attached relations)
Think about what objects are ATTACHED TO WALL (hanging/fixed on vertical surface):
- ONLY these types: picture, painting, photo frame, mirror, clock, poster, whiteboard, TV (wall-mounted), window, door, curtain rod, light switch, outlet
- Curtains are attached to wall (via curtain rod)

### Step 4: Handle objects not on any surface
For objects that appear to be held by a person or floating in mid-air:
- Think about where the object would NATURALLY REST if not held
- If it's a small object that belongs on a table/desk → parent = that furniture's ID, relation = "support"
- If it's a large object that belongs on the floor → parent = 1, relation = "support"
- Do NOT use relation="held" — always assign a physical support relation based on where the object belongs

## PHYSICAL COMMON SENSE RULES:
1. **Cabinets, wardrobes, bookshelves** → ALWAYS supported by floor (parent=1), NOT attached to wall!
2. **Tables, desks, chairs, beds** → ALWAYS supported by floor (parent=1)
3. **Wall-attached** is RARE, only for: pictures, mirrors, clocks, posters, wall-mounted TVs, curtains, windows, doors
4. **Objects on furniture** → parent is that furniture's ID, relation is "support"
5. **Small items on desk** (lamp, monitor, keyboard, cup) → supported by desk
6. **Objects held by hand or floating** → assign their NATURAL support relation (where they belong when not held), NOT relation="held"
7. **If unsure**, default to: relation="support", parent=1 (floor)

## OUTPUT FORMAT:
```json
{{
  "objects": [
    {{"id": <display_id>, "category": "<name>", "relation": "support|attached", "parent": <parent_id>}}
  ]
}}
```
Where parent_id: 1=floor, 2=wall, or another object's display_id.

## CRITICAL REQUIREMENTS:
- You MUST output exactly {num_objects} objects (one for each ID: {display_ids_str})
- Do NOT skip any object!
- Do NOT use relation="held" — every object must have a physical support relation
- If unsure and the object appears to be resting on a surface, default to: relation="support", parent=1 (floor)

Now analyze the image and output the complete JSON with all {num_objects} objects.
'''

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
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
        generated_ids = model.generate(**inputs, max_new_tokens=1024)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def parse_scene_graph_output(output_text: str) -> Dict[str, Any]:
    """解析VLM输出为场景图JSON"""
    if '```' in output_text:
        text_after_thinking = output_text
        if '```json' in text_after_thinking:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', text_after_thinking)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        json_match = re.search(r'```\s*([\s\S]*?)\s*```', text_after_thinking)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    json_match = re.search(r'\{\s*"objects"\s*:\s*\[[\s\S]*?\]\s*\}', output_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return {"objects": [], "parse_error": True, "raw_output": output_text}


def post_process_scene_graph(objects: List[Dict]) -> List[Dict]:
    """
    物理常识后处理: 纠正VLM的常见错误

    规则:
      1. 柜子/桌子/椅子等必须 floor-supported
      2. 画/镜子/时钟等必须 wall-attached
      3. 过滤 floor/wall 本身
    """
    filtered = []
    for obj in objects:
        category = obj.get('category', '').lower().replace(' ', '_')
        relation = obj.get('relation', 'support')
        parent = obj.get('parent', 1)

        if category in FILTER_CATEGORIES:
            continue
        if 'duplicate' in category or 'same_as' in category:
            continue

        category_base = category.split('_')[0] if '_' in category else category

        if category in MUST_BE_FLOOR_SUPPORTED or category_base in MUST_BE_FLOOR_SUPPORTED:
            if relation == 'attached' and parent == 2:
                relation = 'support'
                parent = 1

        if category in MUST_BE_WALL_ATTACHED or category_base in MUST_BE_WALL_ATTACHED:
            if relation == 'support' and parent == 1:
                if category_base in {'picture', 'painting', 'photo', 'mirror', 'clock', 'poster'}:
                    relation = 'attached'
                    parent = 2

        filtered.append({
            'id': obj.get('id'),
            'category': obj.get('category', 'unknown'),
            'relation': relation,
            'parent': parent,
        })

    return filtered


def convert_scene_graph_to_relations(
    objects: List[Dict],
    category_to_display_ids: Dict[str, List[int]],
    original_relations: Dict[str, str],
    display_id_offset: int = 3,
) -> Dict[str, str]:
    """
    将场景图转换为 ReplicateAnyScene 的 categories_and_relations 格式

    核心规则:
      1. 只修改 "supported by other objects" 的物体关系
      2. 已确定的关系 (floor/wall/embedded/attached) 不改变
      3. 如果VLM判断不出具体关系, 保持原样 "supported by other objects"
      4. 生成 per-instance 关系键 (如 "toy_0", "toy_1")，同一类别不同实例可有不同关系
      5. 同时保留 category-level 键 (如 "toy")，值为该类别最常见的关系

    参数:
        objects: VLM推断的物体列表 [{id, category, relation, parent}, ...]
        category_to_display_ids: {category: [display_id, ...]} 类别到显示ID的映射
        original_relations: 原始的 categories_and_relations
        display_id_offset: 显示ID偏移
    返回:
        更新后的 categories_and_relations (包含 category-level 和 instance-level 键)
    """
    id_to_object = {}
    for obj in objects:
        id_to_object[obj['id']] = obj

    display_id_to_category_instance = {}
    for category, display_ids in category_to_display_ids.items():
        for inst_idx, did in enumerate(display_ids):
            display_id_to_category_instance[did] = (category, inst_idx)

    refined_relations = dict(original_relations)

    instance_relations = {}
    for obj in objects:
        obj_id = obj['id']
        relation = obj['relation']
        parent_id = obj['parent']

        if obj_id not in display_id_to_category_instance:
            continue

        category, inst_idx = display_id_to_category_instance[obj_id]

        if original_relations.get(category) != "supported by other objects":
            continue

        if parent_id == 0:
            continue
        elif parent_id == 1 and relation == 'support':
            rel_str = "supported by floor"
        elif parent_id == 2 and relation == 'attached':
            cat_lower = category.lower()
            if any(w in cat_lower for w in ['window', 'door']):
                rel_str = "embedded in wall"
            else:
                rel_str = "attached to wall"
        elif parent_id in display_id_to_category_instance:
            parent_category, parent_inst_idx = display_id_to_category_instance[parent_id]
            parent_n_instances = len(category_to_display_ids.get(parent_category, []))
            if relation == 'support':
                if parent_n_instances > 1:
                    rel_str = f"supported by {parent_category}_{parent_inst_idx}"
                else:
                    rel_str = f"supported by {parent_category}"
            elif relation == 'attached':
                if parent_n_instances > 1:
                    rel_str = f"attached to {parent_category}_{parent_inst_idx}"
                else:
                    rel_str = f"attached to {parent_category}"
            else:
                continue
        else:
            continue

        instance_relations[(category, inst_idx)] = rel_str

    for category in category_to_display_ids:
        if original_relations.get(category) != "supported by other objects":
            continue

        cat_instance_rels = {}
        for (cat, idx), rel in instance_relations.items():
            if cat == category:
                cat_instance_rels[idx] = rel

        if not cat_instance_rels:
            continue

        n_instances = len(category_to_display_ids[category])
        for idx, rel in cat_instance_rels.items():
            inst_key = f"{category}_{idx}"
            refined_relations[inst_key] = rel

        if n_instances == 1:
            only_rel = list(cat_instance_rels.values())[0]
            refined_relations[category] = only_rel
        else:
            from collections import Counter
            rel_counts = Counter(cat_instance_rels.values())
            most_common_rel = rel_counts.most_common(1)[0][0]
            refined_relations[category] = most_common_rel

    return refined_relations


def infer_relations_scene_graph(
    scene_dir: str,
    vlm_checkpoint: str,
    categories_and_relations: Dict[str, str],
    instance_masks: Dict[str, List[Dict]] = None,
    colors: List[np.ndarray] = None,
    id_scene_path: str = None,
    output_dir: str = None,
    preloaded_vlm: tuple = None,
) -> tuple:
    """
    使用 SimRecon 场景图推断方式获取物体间关系

    参数:
        scene_dir: 场景输出目录
        vlm_checkpoint: VLM模型路径
        categories_and_relations: 原始关系字典
        instance_masks: mask数据 (如果为None, 尝试从磁盘加载)
        colors: 帧图像列表 (如果为None, 尝试从磁盘加载)
        id_scene_path: 已有的ID标注图路径 (如果为None, 自动创建)
        output_dir: 输出目录 (默认为scene_dir)
        preloaded_vlm: 预加载的VLM (model, processor), 避免重复加载
    返回:
        (refined_relations, vlm_or_None): 更新后的关系字典 + VLM模型(供后续使用)
    """
    if output_dir is None:
        output_dir = scene_dir

    display_id_offset = 3

    # ── Step 1: 获取或创建 ID 标注图 ──
    if id_scene_path and os.path.exists(id_scene_path):
        print(f"   📷 使用已有ID标注图: {id_scene_path}", flush=True)
        category_to_display_ids = _load_category_display_ids(scene_dir, categories_and_relations, display_id_offset)
    else:
        print(f"   📷 创建ID标注图...", flush=True)

        if instance_masks is None:
            instance_masks = _load_instance_masks_from_disk(scene_dir)
        if colors is None:
            colors = _load_colors_from_disk(scene_dir)

        if instance_masks is None or colors is None:
            print("   ❌ 无法加载mask或图像数据，回退到逐物体推断", flush=True)
            return dict(categories_and_relations), None

        best_frame_id = select_best_frame_for_labeling(instance_masks, colors)
        frame_image = colors[best_frame_id]

        annotated, category_to_display_ids = create_id_labeled_image(
            frame_image, instance_masks, categories_and_relations, display_id_offset,
            best_frame_id=best_frame_id
        )

        id_scene_path = os.path.join(output_dir, "id_scene.png")
        cv2.imwrite(id_scene_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"   💾 ID标注图已保存: {id_scene_path}", flush=True)

        # 保存 category_to_display_ids 映射
        ids_map_path = os.path.join(output_dir, "id_scene_mapping.json")
        serializable = {cat: [int(did) for did in dids] for cat, dids in category_to_display_ids.items()}
        with open(ids_map_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    # ── Step 2: 构建 object_ids 和 category_names ──
    object_ids = []
    category_names = []
    for category, display_ids in category_to_display_ids.items():
        for did in display_ids:
            object_ids.append(did)
            category_names.append(category)

    if not object_ids:
        print("   ⚠️ 无可见物体，跳过场景图推断", flush=True)
        return dict(categories_and_relations), None

    print(f"   📋 推断 {len(object_ids)} 个物体的关系: {dict(zip(category_names, object_ids))}", flush=True)

    # ── Step 3: 加载VLM并推断 ──
    if preloaded_vlm is not None:
        model, processor = preloaded_vlm
        print(f"   🤖 使用预加载VLM", flush=True)
    else:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq, AutoProcessor
            except ImportError:
                print("   ❌ 无法导入transformers，跳过场景图推断", flush=True)
                return dict(categories_and_relations), None

        print(f"   🤖 加载VLM: {vlm_checkpoint}", flush=True)
        model = AutoModelForVision2Seq.from_pretrained(
            vlm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(vlm_checkpoint, trust_remote_code=True)

    raw_output = infer_relations_from_scene_graph(
        model, processor, id_scene_path, object_ids, category_names
    )

    print(f"\n   --- VLM Raw Output ---", flush=True)
    print(f"   {raw_output[:500]}{'...' if len(raw_output) > 500 else ''}", flush=True)
    print(f"   --- End of Raw Output ---\n", flush=True)

    # ── Step 4: 解析和后处理 ──
    parsed = parse_scene_graph_output(raw_output)
    objects = parsed.get("objects", [])

    if objects:
        print(f"   解析到 {len(objects)} 个物体，进行物理常识后处理...", flush=True)
        objects = post_process_scene_graph(objects)
        print(f"   后处理后剩余 {len(objects)} 个物体", flush=True)

    # ── Step 5: 转换为 ReplicateAnyScene 格式 ──
    refined_relations = convert_scene_graph_to_relations(
        objects, category_to_display_ids, categories_and_relations, display_id_offset
    )

    # 保存结果
    refined_json_path = os.path.join(output_dir, "relations_scene_graph.json")
    with open(refined_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'refined_relations': refined_relations,
            'scene_graph_objects': objects,
            'category_to_display_ids': {cat: [int(d) for d in dids] for cat, dids in category_to_display_ids.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"   💾 场景图关系已保存: {refined_json_path}", flush=True)

    # 不卸载VLM, 返回给调用者供后续使用 (5.2 SP精修需要VLM判断精修程度)
    return refined_relations, (model, processor)


def _load_category_display_ids(scene_dir, categories_and_relations, display_id_offset=3):
    """从磁盘加载 category_to_display_ids 映射"""
    ids_map_path = os.path.join(scene_dir, "id_scene_mapping.json")
    if os.path.exists(ids_map_path):
        with open(ids_map_path, 'r') as f:
            return json.load(f)

    category_to_display_ids = {}
    display_id = display_id_offset
    for category in categories_and_relations:
        if category.lower() in FILTER_CATEGORIES:
            continue
        instance_visibility_path = os.path.join(scene_dir, "optimal_frames", "instance_visibility.json")
        if os.path.exists(instance_visibility_path):
            with open(instance_visibility_path, 'r') as f:
                visibility = json.load(f)
            n_instances = len(visibility.get(category, {}))
        else:
            n_instances = 1
        category_to_display_ids[category] = list(range(display_id, display_id + n_instances))
        display_id += n_instances
    return category_to_display_ids


def _load_instance_masks_from_disk(scene_dir):
    """从磁盘加载mask数据 (回退方案: 使用optimal_frames推断)"""
    visibility_path = os.path.join(scene_dir, "optimal_frames", "instance_visibility.json")
    if not os.path.exists(visibility_path):
        return None

    with open(visibility_path, 'r') as f:
        visibility = json.load(f)

    instance_masks = {}
    for category, instances in visibility.items():
        cat_masks = []
        for inst_idx_str, frame_ids in instances.items():
            mask_list = []
            for fid in frame_ids:
                mask_list.append({'frame_id': fid, 'mask': np.zeros((1, 1), dtype=np.uint8)})
            cat_masks.append(mask_list)
        instance_masks[category] = cat_masks

    return instance_masks


def _load_colors_from_disk(scene_dir):
    """从磁盘加载帧图像"""
    color_dir = os.path.join(scene_dir, "color")
    if not os.path.isdir(color_dir):
        return None

    colors = []
    for cf in sorted(os.listdir(color_dir), key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 0):
        if cf.endswith(('.jpg', '.png', '.jpeg')):
            img = cv2.imread(os.path.join(color_dir, cf))
            if img is not None:
                colors.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return colors if colors else None


def main():
    parser = argparse.ArgumentParser(
        description="SimRecon风格的场景图关系推断",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scene_dir", type=str, required=True,
                        help="场景输出目录")
    parser.add_argument("--vlm_checkpoint", type=str, required=True,
                        help="VLM模型路径")
    parser.add_argument("--id_scene_path", type=str, default=None,
                        help="已有的ID标注图路径 (默认自动创建)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认为scene_dir)")

    args = parser.parse_args()

    # 加载原始关系
    stage1_json = None
    for f in os.listdir(args.scene_dir):
        if f.endswith('_stage1.json') and 'refined' not in f:
            stage1_json = os.path.join(args.scene_dir, f)
            break
    if stage1_json is None:
        for f in os.listdir(args.scene_dir):
            if f.endswith('.json') and 'stage1' in f:
                stage1_json = os.path.join(args.scene_dir, f)
                break

    if stage1_json is None:
        final_rel = os.path.join(args.scene_dir, "final_relations.json")
        if os.path.exists(final_rel):
            with open(final_rel, 'r') as f:
                categories_and_relations = json.load(f)
        else:
            print(f"❌ 未找到场景JSON", flush=True)
            sys.exit(1)
    else:
        with open(stage1_json, 'r') as f:
            categories_and_relations = json.load(f)

    print(f"📋 原始关系: {json.dumps(categories_and_relations, ensure_ascii=False)}", flush=True)

    refined, _ = infer_relations_scene_graph(
        scene_dir=args.scene_dir,
        vlm_checkpoint=args.vlm_checkpoint,
        categories_and_relations=categories_and_relations,
        id_scene_path=args.id_scene_path,
        output_dir=args.output_dir or args.scene_dir,
    )

    print(f"\n📋 推断后关系: {json.dumps(refined, ensure_ascii=False)}", flush=True)

    final_rel = os.path.join(args.output_dir or args.scene_dir, "final_relations.json")
    with open(final_rel, 'w') as f:
        json.dump(refined, f, indent=2, ensure_ascii=False)
    print(f"💾 最终关系已保存: {final_rel}", flush=True)


if __name__ == "__main__":
    main()
