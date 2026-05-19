"""
ReplicateAnyScene Stage 4 Refinement: 细化"supported by other objects"关系

核心思路:
  使用VLM进行视觉推理，结合3D空间信息作为辅助上下文，确定具体的支撑关系。
  
  在Stage3完成后，我们已经获得了每个物体的:
    - 3D资产(mesh)和位置(transform T)
    - 掩码(masks)和关键帧ID
    - 类别关系(来自stage1的JSON)
  
  本工具专门处理关系为"supported by other objects"的物体:
    1. 基于3D空间信息筛选候选支撑物（高度、投影重叠）
    2. VLM推理判断具体支撑关系（多帧投票提高鲁棒性）
    3. 物理常识规则后处理纠错
  
  输出: 更新后的场景JSON，包含具体的父子关系
"""

import argparse
import json
import os
import sys
import re
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ============================================================
# 常量定义
# ============================================================

# 物理常识规则：哪些类别通常放在哪些类别上
SUPPORT_RULES = {
    "cup": ["table", "desk", "shelf", "cabinet", "nightstand"],
    "plate": ["table", "desk", "shelf", "counter"],
    "bowl": ["table", "desk", "shelf", "counter"],
    "book": ["shelf", "table", "desk", "nightstand"],
    "laptop": ["table", "desk", "nightstand"],
    "keyboard": ["table", "desk"],
    "mouse": ["table", "desk"],
    "lamp": ["table", "desk", "nightstand", "shelf"],
    "pillow": ["bed", "sofa", "couch", "chair"],
    "blanket": ["bed", "sofa", "couch"],
    "toy": ["table", "shelf", "floor"],
    "plant": ["table", "shelf", "windowsill", "floor"],
    "vase": ["table", "shelf", "windowsill"],
    "clock": ["shelf", "table", "desk", "wall"],
    "picture": ["wall", "shelf", "table"],
    "mirror": ["wall", "dresser", "table"],
    "tv": ["tv_stand", "wall", "table"],
    "monitor": ["desk", "table"],
    "speaker": ["shelf", "table", "desk"],
    "remote": ["table", "coffee_table", "sofa"],
    "phone": ["table", "desk", "nightstand"],
    "bag": ["table", "chair", "floor"],
    "box": ["shelf", "table", "floor"],
}


# ============================================================
# VLM提示词构建
# ============================================================

def build_refinement_prompt(
    target_obj_name: str,
    target_category: str,
    candidate_supporters: List[Tuple[str, str]],
    spatial_context: str = ""
) -> str:
    """
    构建VLM推理提示词
    
    参数:
        target_obj_name: 目标物体名称
        target_category: 目标物体类别
        candidate_supporters: [(supporter_name, supporter_category), ...]
        spatial_context: 3D空间上下文信息字符串
    返回:
        完整的prompt文本
    """
    candidates_str = "\n".join([
        f"  - ID {i+1}: {name} (category: {cat})" 
        for i, (name, cat) in enumerate(candidate_supporters)
    ])
    
    num_candidates = len(candidate_supporters)
    
    prompt = f"""You are a precise spatial relationship analysis API.

**Task**: Determine which object is supporting "{target_obj_name}" (a {target_category}).

**Candidate supporting objects** ({num_candidates} objects):
{candidates_str}

{spatial_context}

**Analysis Guidelines**:
1. Look at the image carefully and identify "{target_obj_name}" (the {target_category})
2. Check which candidate object it is physically resting ON TOP OF
3. Consider physical common sense:
   - Small items (cup, plate, book, laptop, etc.) are usually on tables/desks/shelves
   - Pillows/blankets are usually on beds/sofas/chairs
   - Decorative items (vase, plant, clock) can be on tables/shelves/windowsills
   - Electronics (monitor, keyboard, speaker) are usually on desks/tables
4. If the object appears to be floating or unclear, choose the most likely support based on height and position
5. If NONE of the candidates seem correct, output "floor" as the supporter

**CRITICAL RULES**:
- Output MUST be valid JSON starting with {{ and ending with }}
- NO explanations, NO thinking process, NO text before/after JSON
- Choose EXACTLY ONE supporter from the candidate list, OR "floor" if none applies
- Do NOT invent new objects not in the candidate list

**Output format** (STRICT JSON):
{{
    "supporter_id": <integer 1-{num_candidates}, or null if floor>,
    "supporter_name": "<exact name from candidate list, or 'floor'>",
    "confidence": <float 0.0-1.0>,
    "reasoning": "<brief explanation in one sentence>"
}}

**Now analyze the image and OUTPUT JSON:**"""
    
    return prompt


# ============================================================
# 3D空间信息提取
# ============================================================

def extract_spatial_context(
    target_obj_info: dict,
    candidate_infos: List[dict]
) -> str:
    """从3D数据中提取空间关系上下文"""
    try:
        import trimesh
        
        target_mesh = target_obj_info['mesh'].copy()
        target_mesh.apply_transform(target_obj_info['T'])
        target_bbox_min = target_mesh.vertices.min(axis=0)
        target_bbox_max = target_mesh.vertices.max(axis=0)
        target_centroid = target_mesh.vertices.mean(axis=0)
        target_height = target_bbox_max[2] - target_bbox_min[2]
        target_bottom_z = target_bbox_min[2]
        
        context_lines = [
            f"\n**3D Spatial Context** (Z-axis is vertical, higher Z = higher position):",
            f"  Target object '{target_obj_info['name']}':",
            f"    - Centroid Z: {target_centroid[2]:.2f}m",
            f"    - Bottom Z: {target_bottom_z:.2f}m",
            f"    - Height: {target_height:.2f}m"
        ]
        
        for i, cand_info in enumerate(candidate_infos, 1):
            cand_mesh = cand_info['mesh'].copy()
            cand_mesh.apply_transform(cand_info['T'])
            cand_bbox_max = cand_mesh.vertices.max(axis=0)
            cand_top_z = cand_bbox_max[2]
            
            height_diff = target_bottom_z - cand_top_z
            
            context_lines.append(
                f"  Candidate {i} '{cand_info['name']}': "
                f"Top Z={cand_top_z:.2f}m, "
                f"Height diff to target bottom={height_diff:+.2f}m"
            )
        
        return "\n".join(context_lines)
    
    except Exception as e:
        print(f"      ⚠️  空间上下文提取失败: {e}", flush=True)
        return ""


def filter_candidates_by_geometry(
    target_obj_info: dict,
    all_instances: Dict[str, dict],
    height_threshold: float = 0.8,
    overlap_threshold: float = 0.1
) -> List[Tuple[str, dict]]:
    """基于几何关系筛选候选支撑物"""
    try:
        import trimesh
        
        target_mesh = target_obj_info['mesh'].copy()
        target_mesh.apply_transform(target_obj_info['T'])
        target_min = target_mesh.vertices.min(axis=0)
        target_max = target_mesh.vertices.max(axis=0)
        target_bottom_z = target_min[2]
        
        candidates = []
        
        for cand_name, cand_info in all_instances.items():
            if cand_name == target_obj_info['name']:
                continue
            
            if cand_info.get('relationship') == 'supported by other objects':
                continue
            
            cand_mesh = cand_info['mesh'].copy()
            cand_mesh.apply_transform(cand_info['T'])
            cand_min = cand_mesh.vertices.min(axis=0)
            cand_max = cand_mesh.vertices.max(axis=0)
            cand_top_z = cand_max[2]
            
            height_diff = target_bottom_z - cand_top_z
            if height_diff < -0.1 or height_diff > height_threshold:
                continue
            
            overlap_x_min = max(target_min[0], cand_min[0])
            overlap_x_max = min(target_max[0], cand_max[0])
            overlap_y_min = max(target_min[1], cand_min[1])
            overlap_y_max = min(target_max[1], cand_max[1])
            
            if overlap_x_max <= overlap_x_min or overlap_y_max <= overlap_y_min:
                continue
            
            overlap_area = (overlap_x_max - overlap_x_min) * (overlap_y_max - overlap_y_min)
            target_area = (target_max[0] - target_min[0]) * (target_max[1] - target_min[1])
            
            if target_area < 1e-6:
                continue
            
            overlap_ratio = overlap_area / target_area
            if overlap_ratio < overlap_threshold:
                continue
            
            candidates.append((cand_name, cand_info))
        
        candidates.sort(key=lambda x: abs(
            target_bottom_z - x[1]['mesh'].copy().apply_transform(x[1]['T']).vertices.max(axis=0)[2]
        ))
        
        return candidates[:5]
    
    except Exception as e:
        print(f"      ⚠️  几何筛选失败: {e}，返回所有非'other objects'物体", flush=True)
        return [
            (name, info) for name, info in all_instances.items()
            if name != target_obj_info['name'] and 
               info.get('relationship') != 'supported by other objects'
        ][:5]


# ============================================================
# VLM推理
# ============================================================

def vlm_inference_single_object(
    image: Image.Image,
    target_obj_name: str,
    target_category: str,
    candidate_supporters: List[Tuple[str, str]],
    model,
    processor,
    spatial_context: str = ""
) -> Optional[dict]:
    """对单个物体进行VLM推理"""
    import torch
    
    prompt = build_refinement_prompt(
        target_obj_name, target_category, candidate_supporters, spatial_context
    )
    
    try:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        json_match = re.search(r'\{[^{}]*\}', output_text)
        if json_match:
            result = json.loads(json_match.group())
            return result
        else:
            print(f"      ⚠️  无法从输出中提取JSON: {output_text[:200]}", flush=True)
            return None
    
    except Exception as e:
        print(f"      ❌ VLM推理失败: {e}", flush=True)
        return None


def multi_frame_voting(
    target_obj_info: dict,
    candidate_supporters: List[Tuple[str, dict]],
    vggt_results: dict,
    model,
    processor,
    max_frames: int = 3
) -> Optional[str]:
    """多帧VLM推理 + 投票机制"""
    optimal_frame_id = target_obj_info.get('optimal_frame_id', 0)
    colors = vggt_results['colors']
    
    frame_ids = [optimal_frame_id]
    for offset in [1, -1, 2, -2]:
        fid = optimal_frame_id + offset
        if 0 <= fid < len(colors):
            frame_ids.append(fid)
        if len(frame_ids) >= max_frames:
            break
    
    votes = Counter()
    total_votes = 0
    
    for frame_id in frame_ids:
        try:
            image_array = colors[frame_id]
            image = Image.fromarray(image_array)
            
            spatial_context = extract_spatial_context(
                target_obj_info,
                [info for _, info in candidate_supporters]
            )
            
            candidates_list = [(name, info['category']) for name, info in candidate_supporters]
            result = vlm_inference_single_object(
                image,
                target_obj_info['name'],
                target_obj_info['category'],
                candidates_list,
                model,
                processor,
                spatial_context
            )
            
            if result and 'supporter_name' in result:
                supporter = result['supporter_name']
                votes[supporter] += 1
                total_votes += 1
                confidence = result.get('confidence', 0.5)
                print(f"      帧#{frame_id}: 支持物='{supporter}', 置信度={confidence:.2f}", flush=True)
        
        except Exception as e:
            print(f"      ⚠️  帧#{frame_id}推理失败: {e}", flush=True)
            continue
    
    if total_votes == 0:
        return None
    
    best_supporter = votes.most_common(1)[0][0]
    vote_count = votes[best_supporter]
    print(f"      📊 投票结果: '{best_supporter}' ({vote_count}/{total_votes} 票)", flush=True)
    
    return best_supporter


# ============================================================
# 物理常识后处理
# ============================================================

def post_process_with_physics_rules(
    target_name: str,
    target_category: str,
    supporter_name: str,
    supporter_category: str
) -> str:
    """基于物理常识规则修正VLM的判断结果"""
    target_base = target_category.lower().replace(' ', '_').split('_')[0]
    supporter_base = supporter_category.lower().replace(' ', '_').split('_')[0]
    
    furniture_categories = {
        'table', 'desk', 'chair', 'sofa', 'bed', 'cabinet', 'shelf',
        'dresser', 'nightstand', 'bookshelf', 'wardrobe'
    }
    small_item_categories = {
        'cup', 'plate', 'bowl', 'book', 'laptop', 'keyboard', 'mouse',
        'phone', 'remote', 'toy', 'box'
    }
    
    if target_base in furniture_categories and supporter_base in small_item_categories:
        print(f"      [纠错] '{target_name}'({target_base}) 不应被 '{supporter_name}'({supporter_base}) 支撑 → 回退到floor", flush=True)
        return "floor"
    
    if target_base in SUPPORT_RULES:
        expected_supporters = SUPPORT_RULES[target_base]
        if supporter_base not in expected_supporters and supporter_name != "floor":
            print(f"      [警告] '{target_name}'({target_base}) 通常放在 {expected_supporters} 上，但VLM选择了 '{supporter_name}'({supporter_base})", flush=True)
    
    return supporter_name


# ============================================================
# 主流程
# ============================================================

def refine_other_objects_relations(
    stage3_data: dict,
    stage1_json_path: str,
    output_json_path: str,
    vlm_checkpoint: str,
    keyframe_images_dir: str = None,
    use_multi_frame_voting: bool = True,
    max_voting_frames: int = 3
):
    """主函数：使用VLM细化"supported by other objects"关系"""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor
    
    print("=" * 70, flush=True)
    print("🔧 Stage 4 Refinement: VLM驱动细化'supported by other objects'关系", flush=True)
    print("=" * 70, flush=True)
    
    # 1. 加载Stage1的关系信息
    print(f"\n📂 加载Stage1场景JSON: {stage1_json_path}", flush=True)
    with open(stage1_json_path, 'r') as f:
        stage1_relations = json.load(f)
    
    print(f"   总物体数: {len(stage1_relations)}", flush=True)
    other_objects = [
        name for name, rel in stage1_relations.items()
        if rel == "supported by other objects"
    ]
    print(f"   'other objects'关系物体: {len(other_objects)} 个", flush=True)
    if other_objects:
        print(f"   列表: {other_objects}", flush=True)
    
    if not other_objects:
        print(f"\n✅ 没有'supported by other objects'关系的物体，无需细化", flush=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(stage1_relations, f, indent=2, ensure_ascii=False)
        return stage1_relations
    
    # 2. 构建object_instances字典
    print(f"\n🏗️  构建物体实例数据结构...", flush=True)
    object_instances = {}
    
    all_instances = stage3_data['all_instances']
    all_optimal_frame_ids = stage3_data['all_optimal_frame_ids']
    vggt_results = stage3_data.get('vggt_prediction_results', {})
    
    for category, instances in all_instances.items():
        for i, instance_info in enumerate(instances):
            obj_name = f"{category}_{i}"
            
            matched_rel = None
            for stage1_name, stage1_rel in stage1_relations.items():
                if stage1_name.lower() in category.lower() or category.lower() in stage1_name.lower():
                    matched_rel = stage1_rel
                    break
            
            if matched_rel is None:
                matched_rel = "supported by floor"
            
            object_instances[obj_name] = {
                'mesh': instance_info['original_mesh'],
                'T': instance_info['T'],
                'category': category,
                'relationship': matched_rel,
                'optimal_frame_id': all_optimal_frame_ids.get(category, [0] * len(instances))[i],
            }
    
    print(f"   构建完成: {len(object_instances)} 个物体实例", flush=True)
    
    # 3. 加载VLM模型
    print(f"\n🤖 加载VLM模型: {vlm_checkpoint}", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        vlm_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(vlm_checkpoint, trust_remote_code=True)
    print(f"   ✅ VLM模型加载完成", flush=True)
    
    # 4. 逐个处理"supported by other objects"的物体
    print(f"\n🔍 开始VLM推理...", flush=True)
    final_relations = stage1_relations.copy()
    
    for idx, obj_name in enumerate(other_objects):
        if obj_name not in object_instances:
            print(f"   ⚠️  物体 '{obj_name}' 不在Stage3数据中，跳过", flush=True)
            continue
        
        obj_info = object_instances[obj_name]
        print(f"\n   [{idx + 1}/{len(other_objects)}] 分析 '{obj_name}' ({obj_info['category']})...", flush=True)
        
        # 4.1 几何筛选候选支撑物
        candidates = filter_candidates_by_geometry(obj_info, object_instances)
        
        if not candidates:
            print(f"      ⚠️  无候选支撑物，标记为 supported by floor", flush=True)
            final_relations[obj_name] = "supported by floor"
            continue
        
        print(f"      📋 候选支撑物 ({len(candidates)} 个): {[name for name, _ in candidates]}", flush=True)
        
        # 4.2 VLM推理
        if use_multi_frame_voting and vggt_results:
            best_supporter = multi_frame_voting(
                obj_info,
                candidates,
                vggt_results,
                model,
                processor,
                max_frames=max_voting_frames
            )
        else:
            optimal_frame_id = obj_info['optimal_frame_id']
            if keyframe_images_dir:
                image_path = os.path.join(keyframe_images_dir, f"frame_{optimal_frame_id:03d}.jpg")
                if os.path.exists(image_path):
                    image = Image.open(image_path).convert("RGB")
                    
                    spatial_context = extract_spatial_context(
                        obj_info,
                        [info for _, info in candidates]
                    )
                    
                    candidates_list = [(name, info['category']) for name, info in candidates]
                    result = vlm_inference_single_object(
                        image,
                        obj_name,
                        obj_info['category'],
                        candidates_list,
                        model,
                        processor,
                        spatial_context
                    )
                    
                    if result and 'supporter_name' in result:
                        best_supporter = result['supporter_name']
                    else:
                        best_supporter = None
                else:
                    print(f"      ⚠️  图像不存在: {image_path}", flush=True)
                    best_supporter = None
            else:
                print(f"      ⚠️  未提供关键帧图像目录", flush=True)
                best_supporter = None
        
        # 4.3 后处理
        if best_supporter and best_supporter != "floor":
            supporter_category = object_instances[best_supporter]['category'] if best_supporter in object_instances else "unknown"
            
            best_supporter = post_process_with_physics_rules(
                obj_name, obj_info['category'],
                best_supporter, supporter_category
            )
        
        # 4.4 更新关系
        if best_supporter and best_supporter != "floor":
            final_relations[obj_name] = f"supported by {best_supporter}"
            print(f"      ✅ 最终结果: supported by {best_supporter}", flush=True)
        else:
            final_relations[obj_name] = "supported by floor"
            print(f"      ✅ 最终结果: supported by floor", flush=True)
    
    # 清理VLM模型
    del model
    del processor
    torch.cuda.empty_cache()
    
    # 5. 保存结果
    print(f"\n💾 保存更新后的场景JSON: {output_json_path}", flush=True)
    os.makedirs(os.path.dirname(output_json_path) if os.path.dirname(output_json_path) else '.', exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_relations, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 细化完成!", flush=True)
    
    rel_counts = defaultdict(int)
    for rel in final_relations.values():
        rel_counts[rel] += 1
    
    print(f"\n📊 最终关系统计:", flush=True)
    for rel, count in sorted(rel_counts.items()):
        print(f"   {rel}: {count} 个物体", flush=True)
    
    return final_relations


# ============================================================
# 命令行接口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stage 4 Refinement: VLM驱动细化supported by other objects关系'
    )
    parser.add_argument('--stage3_data', type=str, required=True,
                       help='Stage3数据文件路径（pickle或json格式）')
    parser.add_argument('--stage1_json', type=str, required=True,
                       help='Stage1生成的场景JSON路径')
    parser.add_argument('--output_json', type=str, default=None,
                       help='输出JSON路径（默认: stage1_json路径_refined.json）')
    parser.add_argument('--vlm_checkpoint', type=str, required=True,
                       help='VLM模型路径（必需）')
    parser.add_argument('--keyframe_dir', type=str, default=None,
                       help='关键帧图像目录')
    parser.add_argument('--use_multi_frame', action='store_true', default=True,
                       help='启用多帧投票（默认开启）')
    parser.add_argument('--no_multi_frame', action='store_true', default=False,
                       help='禁用多帧投票，仅使用单帧')
    parser.add_argument('--max_frames', type=int, default=3,
                       help='多帧投票使用的最大帧数（默认3）')
    
    args = parser.parse_args()
    
    if args.no_multi_frame:
        args.use_multi_frame = False
    
    if args.output_json is None:
        base_path = Path(args.stage1_json)
        args.output_json = str(base_path.parent / f"{base_path.stem}_refined{base_path.suffix}")
    
    # 加载Stage3数据
    print(f"📂 加载Stage3数据: {args.stage3_data}", flush=True)
    if args.stage3_data.endswith('.pkl') or args.stage3_data.endswith('.pickle'):
        import pickle
        with open(args.stage3_data, 'rb') as f:
            stage3_data = pickle.load(f)
    elif args.stage3_data.endswith('.json'):
        with open(args.stage3_data, 'r') as f:
            stage3_data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {args.stage3_data}")
    
    # 执行细化
    refine_other_objects_relations(
        stage3_data=stage3_data,
        stage1_json_path=args.stage1_json,
        output_json_path=args.output_json,
        vlm_checkpoint=args.vlm_checkpoint,
        keyframe_images_dir=args.keyframe_dir,
        use_multi_frame_voting=args.use_multi_frame,
        max_voting_frames=args.max_frames
    )


if __name__ == '__main__':
    main()
