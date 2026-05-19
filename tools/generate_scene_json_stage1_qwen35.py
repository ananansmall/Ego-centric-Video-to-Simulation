"""
ReplicateAnyScene Stage 1: 渐进式物体发现 (Qwen3.5-9B 优化版)
- 使用 Qwen3.5-9B 模型（速度与质量的平衡）
- VGGT相机位姿引导的关键帧提取，最大化场景覆盖
- 动态类别注册表，逐帧查询新物体
- 同义词自动合并
"""
import argparse
import json
import os
import sys
import time
import subprocess
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from PIL import Image
try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    from transformers import AutoProcessor

from src.models import load_vggt_model, unload_model
from src.utils import load_video_frames
from src.vggt_predict import vggt_predict

SYNONYM_MAP = {
    'cup': ['cup', 'cups', 'mug', 'mugs', 'glass', 'glasses', 'tumbler'],
    'bowl': ['bowl', 'bowls', 'dish', 'dishes', 'plate', 'plates'],
    'bottle': ['bottle', 'bottles', 'jar', 'jars', 'container'],
    'box': ['box', 'boxes', 'carton', 'package', 'crate'],
    'bag': ['bag', 'bags', 'sack', 'pouch', 'backpack'],
    'chair': ['chair', 'chairs', 'seat', 'stool'],
    'table': ['table', 'tables', 'desk', 'countertop'],
    'toy': ['toy', 'toys', 'doll', 'action figure', 'teddy bear'],
    'ball': ['ball', 'balls', 'sphere', 'orb'],
    'book': ['book', 'books', 'notebook', 'magazine'],
    'phone': ['phone', 'smartphone', 'mobile', 'cellphone'],
    'laptop': ['laptop', 'computer', 'notebook computer'],
    'pillow': ['pillow', 'pillows', 'cushion'],
    'blanket': ['blanket', 'blankets', 'quilt', 'cover'],
    'shoe': ['shoe', 'shoes', 'sneaker', 'boot'],
    'cloth': ['cloth', 'clothes', 'shirt', 't-shirt', 'garment'],
}

VLM_PROMPT_INITIAL = """You are a specialized object detection API. Output ONLY valid JSON.

**CRITICAL**: 
- Output MUST be a single line JSON starting with { and ending with }
- NO explanations, NO analysis, NO thinking process
- NO text before or after the JSON
- Keep output SHORT and CONCISE

**Task**: Detect object categories in the image.

**Output format**:
{"category": "relationship", "category2": "relationship2"}

**Valid relationships**:
- "supported by floor"
- "supported by other objects"
- "attached to wall"
- "embedded in wall"

**Rules**:
- Use singular: "toy" not "toys"
- Simple names only
- Group similar items
- Ignore: hands, body parts, walls, floors

**Example**:
{"bed": "supported by floor", "toy": "supported by other objects", "plate": "supported by other objects"}

**OUTPUT JSON NOW (single line, no extra text):**"""

VLM_PROMPT_PROGRESSIVE = """You are a specialized object detection API. Output ONLY valid JSON.

**Already discovered**: {discovered_categories}

**Task**: Find NEW categories NOT in the list above.
If list is empty, detect ALL objects.

**CRITICAL**: 
- Output MUST be a single line JSON
- NO explanations or analysis
- Keep output SHORT

**Output format**:
{{"new_category": "relationship"}}

**Valid relationships**:
- "supported by floor"
- "supported by other objects"
- "attached to wall"
- "embedded in wall"

**Rules**:
- Singular form only
- Only NEW categories
- If no new: {{}}

**Examples**:
- New: {{"plate": "supported by other objects"}}
- None: {{}}

**OUTPUT JSON NOW (single line):**"""


def extract_keyframes_by_camera_motion(extrinsics, num_frames=20,
                                        min_displacement=0.1, min_rotation_deg=5.0):
    """
    基于VGGT相机位姿变化（位移+旋转）提取关键帧。
    （备选方案，已不推荐使用）
    """
    S = len(extrinsics)
    if S <= num_frames:
        return list(range(S))

    min_rotation_rad = np.deg2rad(min_rotation_deg)

    camera_positions = []
    camera_rotations = []
    for i in range(S):
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        pos = -R.T @ t
        camera_positions.append(pos)
        camera_rotations.append(R)
    camera_positions = np.array(camera_positions)

    selected = [0]
    accumulated_dist = 0.0
    accumulated_angle = 0.0

    for i in range(1, S):
        dist = np.linalg.norm(camera_positions[i] - camera_positions[i - 1])
        accumulated_dist += dist

        R_rel = camera_rotations[i] @ camera_rotations[i - 1].T
        trace = np.clip(np.trace(R_rel), -1.0, 3.0)
        angle = np.arccos((trace - 1.0) / 2.0)
        accumulated_angle += angle

        if accumulated_dist >= min_displacement or accumulated_angle >= min_rotation_rad:
            selected.append(i)
            accumulated_dist = 0.0
            accumulated_angle = 0.0

    if selected[-1] != S - 1:
        selected.append(S - 1)

    if len(selected) > num_frames:
        step = (len(selected) - 1) / max(num_frames - 1, 1)
        selected = [selected[int(i * step)] for i in range(num_frames)]

    selected = sorted(list(set(selected)))
    return selected


def compute_voxel_sets(world_points, world_points_conf_mask, voxel_size=0.05):
    """
    计算每一帧覆盖的体素集合（SimRecon方案）。
    
    Args:
        world_points: VGGT输出的世界坐标点云 (T, H, W, 3)
        world_points_conf_mask: 置信度掩码 (T, H, W)
        voxel_size: 体素大小（米），默认5cm
        
    Returns:
        List[Set]: 每帧的体素坐标集合
    """
    T = world_points.shape[0]
    voxel_sets = []
    
    # 计算场景边界
    valid_mask = world_points_conf_mask > 0
    if not valid_mask.any():
        print("   ⚠️  警告: 没有有效的点云数据", flush=True)
        return [set() for _ in range(T)]
    
    valid_points = world_points[valid_mask]
    x_min = valid_points[:, 0].min()
    y_min = valid_points[:, 1].min()
    z_min = valid_points[:, 2].min()
    
    offset = np.array([x_min, y_min, z_min])
    
    for t in range(T):
        mask = world_points_conf_mask[t].flatten()
        points = world_points[t].reshape(-1, 3)
        valid_pts = points[mask]
        
        if len(valid_pts) == 0:
            voxel_sets.append(set())
            continue
        
        # 计算体素坐标
        voxel_coords = np.floor((valid_pts - offset) / voxel_size).astype(int)
        
        # 去重
        unique_voxels = set(map(tuple, np.unique(voxel_coords, axis=0)))
        voxel_sets.append(unique_voxels)
    
    return voxel_sets


def maximum_coverage_sampling(voxel_sets, K):
    """
    贪心最大覆盖采样算法（SimRecon核心算法）。
    
    Args:
        voxel_sets: 每帧的体素集合列表
        K: 要选择的帧数
        
    Returns:
        List[int]: 选中的帧索引（已排序）
    """
    selected = []
    covered = set()
    remaining_frames = set(range(len(voxel_sets)))
    
    for iteration in range(K):
        if not remaining_frames:
            break
        
        max_gain = -1
        best_frame = None
        
        # 找到边际增益最大的帧
        for frame in remaining_frames:
            gain = len(voxel_sets[frame] - covered)
            if gain > max_gain:
                max_gain = gain
                best_frame = frame
        
        if best_frame is None or max_gain <= 0:
            break  # 没有新的覆盖
        
        selected.append(best_frame)
        covered.update(voxel_sets[best_frame])
        remaining_frames.remove(best_frame)
        
        if iteration < 3 or iteration >= K - 2:
            print(f"   🎯 第{iteration+1}轮: 选择帧#{best_frame}, 新增{max_gain}个体素, 累计覆盖{len(covered)}个", flush=True)
    
    # 计算覆盖率
    total_voxels = set()
    for vs in voxel_sets:
        total_voxels.update(vs)
    
    coverage_ratio = len(covered) / len(total_voxels) if total_voxels else 0
    print(f"   📊 最终覆盖率: {len(covered)}/{len(total_voxels)} 体素 ({coverage_ratio:.1%})", flush=True)
    
    return sorted(selected)


def _get_video_fps(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=10
        )
        num, den = result.stdout.strip().split('/')
        return float(num) / float(den)
    except Exception:
        return 30.0


def _get_video_total_frames(video_path):
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-count_frames', '-show_entries', 'stream=nb_read_frames',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=30
        )
        return int(result.stdout.strip())
    except Exception:
        return 0


def extract_specific_frames(video_path, frame_indices, output_dir):
    print(f"\n📹 正在提取 {len(frame_indices)} 个关键帧 (ffmpeg)...", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    fps = _get_video_fps(video_path)
    total_frames = _get_video_total_frames(video_path)
    if total_frames == 0:
        total_frames = int(fps * 60)

    sorted_indices = sorted(frame_indices)
    frame_paths_with_indices = []

    for i, frame_idx in enumerate(sorted_indices):
        if frame_idx >= total_frames:
            frame_idx = total_frames - 1

        timestamp = frame_idx / fps
        frame_path = os.path.join(output_dir, f"keyframe_{i:03d}_vid{frame_idx}.jpg")

        cmd = [
            'ffmpeg', '-y', '-ss', f'{timestamp:.6f}',
            '-i', video_path, '-frames:v', '1',
            '-q:v', '2', frame_path
        ]
        try:
            subprocess.run(cmd, capture_output=True, timeout=15)
            if os.path.exists(frame_path) and os.path.getsize(frame_path) > 100:
                frame_paths_with_indices.append((frame_idx, frame_path))
                if i < 3 or i >= len(sorted_indices) - 2:
                    print(f"   ✅ 关键帧 {i+1}/{len(sorted_indices)} (视频帧#{frame_idx}, ts={timestamp:.2f}s)", flush=True)
                elif i == 3:
                    print(f"   ... (中间帧省略) ...", flush=True)
            else:
                print(f"   ❌ 帧#{frame_idx} 提取失败（文件为空）", flush=True)
        except subprocess.TimeoutExpired:
            print(f"   ❌ 帧#{frame_idx} 提取超时", flush=True)

    print(f"✅ 成功提取 {len(frame_paths_with_indices)} 个关键帧\n", flush=True)
    return frame_paths_with_indices


def normalize_category_name(category_name):
    name = category_name.lower().strip()
    if name.endswith('ies'):
        name = name[:-3] + 'y'
    elif name.endswith('ves'):
        name = name[:-3] + 'f'
    elif name.endswith('es') and not name.endswith('ses'):
        name = name[:-2]
    elif name.endswith('s') and not name.endswith('ss'):
        name = name[:-1]
    return name


def merge_synonyms(category_name):
    name = normalize_category_name(category_name)
    for standard_name, synonyms in SYNONYM_MAP.items():
        if name in synonyms or name == standard_name:
            return standard_name
    return name


def _fix_json_string(json_str):
    import re
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_str)
    return json_str


def _extract_json_from_text(text):
    """从模型输出中提取 JSON，支持多种格式和被截断的情况"""
    import re
    
    # 尝试1: 查找 ```json ... ``` 代码块
    code_block_pattern = r'```(?:json)?\s*({.*?})\s*```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0]
    
    # 尝试2: 从最后一个 { 开始查找完整 JSON
    last_brace_start = text.rfind('{')
    if last_brace_start != -1:
        json_candidate = text[last_brace_start:]
        try:
            json.loads(json_candidate)
            return json_candidate
        except:
            # 如果解析失败，尝试找到匹配的 }
            brace_count = 0
            for i, char in enumerate(json_candidate):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        complete_json = json_candidate[:i+1]
                        try:
                            json.loads(complete_json)
                            return complete_json
                        except:
                            pass
            
            # 如果还是失败，可能是被截断了，尝试修复
            # 添加缺失的 } 和 "
            if json_candidate.count('{') > json_candidate.count('}'):
                # 补充缺少的 }
                missing_braces = json_candidate.count('{') - json_candidate.count('}')
                json_candidate = json_candidate.rstrip() + '}' * missing_braces
                try:
                    json.loads(json_candidate)
                    return json_candidate
                except:
                    pass
    
    # 尝试3: 查找第一个 { 到最后一个 }
    first_brace = text.find('{')
    last_brace = text.rfind('}') + 1
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace]
        try:
            json.loads(candidate)
            return candidate
        except:
            # 尝试修复
            candidate_fixed = _fix_json_string(candidate)
            try:
                json.loads(candidate_fixed)
                return candidate_fixed
            except:
                pass
    
    return None


def _vlm_inference(messages, model, processor, max_new_tokens=512):
    images = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                images.append(item["image"])

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


def progressive_object_discovery(frame_paths_with_indices, model, processor):
    print(f"\n{'='*70}", flush=True)
    print(f"🚀 Stage 1: 渐进式物体发现 (Qwen3.5-9B)", flush=True)
    print(f"📋 策略: 动态类别注册表 + 同义词自动合并", flush=True)
    print(f"{'='*70}\n", flush=True)

    category_registry = {}
    synonym_history = {}
    total_new = 0
    total_merged = 0

    system_message = {
        "role": "system",
        "content": "You are a specialized object detection API. Output ONLY valid JSON."
    }

    first_vid_idx, first_path = frame_paths_with_indices[0]
    print(f" [1/{len(frame_paths_with_indices)}] 初始帧分析 (视频帧#{first_vid_idx})...", flush=True)

    try:
        image = Image.open(first_path).convert("RGB")
        print(f"   📷 图像尺寸: {image.size}", flush=True)
        
        messages = [
            system_message,
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": VLM_PROMPT_INITIAL}
            ]}
        ]

        print(f"   🤖 开始VLM推理...", flush=True)
        start_time = time.time()
        output_text = _vlm_inference(messages, model, processor, max_new_tokens=1024)  # 大幅增加到1024
        elapsed = time.time() - start_time
        print(f"   ⏱️  推理耗时: {elapsed:.1f}s", flush=True)
        print(f"   📝 完整模型输出:\n{output_text}\n", flush=True)  # 显示完整输出

        json_str = _extract_json_from_text(output_text)
        
        if json_str is None:
            print(f"   ❌ 无法找到有效的JSON格式", flush=True)
            return category_registry
        
        print(f"   🔍 提取的JSON: {json_str[:150]}...", flush=True)
        
        try:
            new_categories = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"   ⚠️ JSON解析失败，尝试修复...", flush=True)
            json_str_fixed = _fix_json_string(json_str)
            try:
                new_categories = json.loads(json_str_fixed)
                print(f"   ✅ JSON修复成功", flush=True)
            except Exception as e2:
                print(f"   ❌ JSON修复也失败: {e2}", flush=True)
                return category_registry
        
        if not new_categories:
            print(f"   ⚠️ 模型返回空JSON", flush=True)
            return category_registry
        
        for orig_name, relationship in new_categories.items():
            standard_name = merge_synonyms(orig_name)
            if standard_name not in category_registry:
                category_registry[standard_name] = relationship
                synonym_history[orig_name] = standard_name
                total_new += 1
                print(f"   ✅ 新类别: '{orig_name}' → '{standard_name}'", flush=True)
            else:
                synonym_history[orig_name] = standard_name
                total_merged += 1
                print(f"   🔄 同义词合并: '{orig_name}' → '{standard_name}'", flush=True)
        
        print(f"   📊 初始帧共发现 {len(category_registry)} 个类别", flush=True)
        
    except Exception as e:
        import traceback
        print(f"   ❌ 初始帧分析异常: {e}", flush=True)
        print(f"   �� 详细堆栈:\n{traceback.format_exc()}", flush=True)
        return category_registry

    if not category_registry:
        print(f"\n❌ 初始帧未发现任何类别，终止后续分析", flush=True)
        return category_registry

    for i, (vid_idx, frame_path) in enumerate(frame_paths_with_indices[1:], start=2):
        print(f"\n [{i}/{len(frame_paths_with_indices)}] 渐进式分析 (视频帧#{vid_idx}, 已发现 {len(category_registry)} 个类别)...", flush=True)

        try:
            image = Image.open(frame_path).convert("RGB")
            discovered_list = ", ".join(sorted(category_registry.keys()))
            prompt = VLM_PROMPT_PROGRESSIVE.format(discovered_categories=discovered_list)

            messages = [
                system_message,
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]}
            ]

            start_time = time.time()
            output_text = _vlm_inference(messages, model, processor, max_new_tokens=1024)  # 大幅增加到1024
            elapsed = time.time() - start_time

            if i <= 3 or i >= len(frame_paths_with_indices) - 1:
                print(f"   📝 完整模型输出 ({elapsed:.1f}s):\n{output_text}\n", flush=True)  # 显示完整输出

            json_str = _extract_json_from_text(output_text)
            
            if json_str is None:
                print(f"   ⚠️ 无法解析输出 (耗时{elapsed:.1f}s)", flush=True)
                continue
            
            try:
                new_categories = json.loads(json_str)
            except json.JSONDecodeError:
                json_str_fixed = _fix_json_string(json_str)
                try:
                    new_categories = json.loads(json_str_fixed)
                except:
                    print(f"   ⚠️ JSON解析失败 (耗时{elapsed:.1f}s)", flush=True)
                    continue
            
            if new_categories:
                frame_new = 0
                frame_merged = 0
                for orig_name, relationship in new_categories.items():
                    standard_name = merge_synonyms(orig_name)
                    if standard_name not in category_registry:
                        category_registry[standard_name] = relationship
                        synonym_history[orig_name] = standard_name
                        frame_new += 1
                        total_new += 1
                    else:
                        synonym_history[orig_name] = standard_name
                        frame_merged += 1
                        total_merged += 1
                if frame_new > 0:
                    print(f"   ✅ 发现 {frame_new} 个新类别 ({elapsed:.1f}s)", flush=True)
                if frame_merged > 0:
                    print(f"   🔄 合并 {frame_merged} 个同义词", flush=True)
                if frame_new == 0 and frame_merged == 0:
                    print(f"   ⏭️ 无新类别 ({elapsed:.1f}s)", flush=True)
            else:
                print(f"   ⏭️ 无新类别 ({elapsed:.1f}s)", flush=True)
                
        except Exception as e:
            import traceback
            print(f"   ❌ 分析失败: {e}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"✅ Category Registry 完成", flush=True)
    print(f"   总类别数: {len(category_registry)}", flush=True)
    print(f"   新增类别: {total_new}", flush=True)
    print(f"   同义词合并: {total_merged}", flush=True)
    print(f"{'='*70}\n", flush=True)

    return category_registry


def generate_scene_json(category_registry, output_path):
    print(f"💾 保存 JSON: {output_path}", flush=True)
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    sorted_registry = dict(sorted(category_registry.items()))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_registry, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 已保存!", flush=True)
    print(f"\n�� 最终类别清单 ({len(sorted_registry)} 个):", flush=True)
    for i, (name, rel) in enumerate(sorted_registry.items(), 1):
        print(f"   {i:2d}. {name:20s} → {rel}", flush=True)


def load_vlm_model(checkpoint_path):
    print(f"📦 加载 VLM 模型: {checkpoint_path}", flush=True)
    print(f"   路径存在: {os.path.exists(checkpoint_path)}", flush=True)

    print("   [1/2] 加载 Processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )

    print("   [2/2] 加载 Model...", flush=True)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"✅ 模型加载完成! 设备: {model.device}", flush=True)
    return model, processor


def main():
    parser = argparse.ArgumentParser(description='ReplicateAnyScene Stage 1 (Qwen3.5-9B)')
    parser.add_argument('--input_video', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output_json', type=str, default=None, help='输出 JSON 路径')
    parser.add_argument('--vlm_checkpoint', type=str, default=None, help='VLM 模型路径')
    parser.add_argument('--num_frames', type=int, default=20, help='关键帧数（默认20）')
    parser.add_argument('--temp_dir', type=str, default='./temp_frames', help='临时帧目录')
    parser.add_argument('--no_vggt', action='store_true', help='禁用VGGT引导')
    parser.add_argument('--use_camera_motion', action='store_true', help='使用旧版相机运动方法（不推荐）')
    parser.add_argument('--min_displacement', type=float, default=0.1, help='VGGT最小位移阈值/米（仅旧方法）')
    parser.add_argument('--min_rotation', type=float, default=5.0, help='VGGT最小旋转角度/度（仅旧方法）')

    args = parser.parse_args()

    if args.output_json is None:
        video_stem = Path(args.input_video).stem
        args.output_json = f"./assets/json_configs/scene_{video_stem}.json"

    if args.vlm_checkpoint is None:
        default_model = "/mnt/data/lza/models/Qwen3.5-9B"
        if os.path.exists(default_model):
            args.vlm_checkpoint = default_model
            print(f"🔍 使用默认模型: Qwen3.5-9B\n", flush=True)
        else:
            fallback = "/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct"
            if os.path.exists(fallback):
                snapshots_dir = os.path.join(fallback, "snapshots")
                if os.path.exists(snapshots_dir):
                    snapshots = [d for d in os.listdir(snapshots_dir)
                                 if os.path.isdir(os.path.join(snapshots_dir, d))]
                    if snapshots:
                        args.vlm_checkpoint = os.path.join(snapshots_dir, snapshots[0])
                        print(f"🔍 回退到模型: {args.vlm_checkpoint}\n", flush=True)

    if args.vlm_checkpoint is None:
        print("❌ 错误: 未找到 VLM 模型", flush=True)
        return

    print("=" * 70, flush=True)
    print("🚀 ReplicateAnyScene Stage 1 (Qwen3.5-9B)", flush=True)
    print("=" * 70, flush=True)
    print(f"📥 输入视频: {args.input_video}")
    print(f"📤 输出 JSON: {args.output_json}")
    print(f"🤖 VLM模型: {args.vlm_checkpoint}")
    print(f"🖼️  目标帧数: {args.num_frames}")
    print(f"🔮 VGGT引导: {'禁用' if args.no_vggt else '启用'}")
    print("=" * 70 + "\n", flush=True)

    try:
        keyframe_indices = None

        if not args.no_vggt:
            print(f"{'='*70}", flush=True)
            method_name = "相机运动（旧方法）" if args.use_camera_motion else "3D空间覆盖（SimRecon）"
            print(f"🔮 Step 0: VGGT 3D场景重建 + {method_name}关键帧提取", flush=True)
            print(f"{'='*70}", flush=True)

            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                frames = load_video_frames(args.input_video, max_frames=160).to(device)
                print(f"   加载 {len(frames)} 帧用于VGGT重建", flush=True)

                vggt_model = load_vggt_model().to(device)
                vggt_results = vggt_predict(frames, vggt_model)

                if args.use_camera_motion:
                    # 使用旧版相机运动方法
                    print(f"   🎯 使用相机运动方法...", flush=True)
                    vggt_extrinsics = vggt_results['extrinsics']
                    
                    vggt_model = unload_model(vggt_model)
                    del frames, vggt_results
                    torch.cuda.empty_cache()
                    
                    keyframe_indices = extract_keyframes_by_camera_motion(
                        vggt_extrinsics,
                        num_frames=args.num_frames,
                        min_displacement=args.min_displacement,
                        min_rotation_deg=args.min_rotation
                    )
                    print(f"🎯 VGGT引导选择 {len(keyframe_indices)} 个关键帧 "
                          f"(位移阈值={args.min_displacement}m, 旋转阈值={args.min_rotation}°)", flush=True)
                    
                    del vggt_extrinsics
                else:
                    # 使用新版3D空间覆盖方法
                    world_points = vggt_results['world_points'][0]  # 已经是numpy数组
                    world_points_conf = vggt_results['world_points_conf'][0]  # 已经是numpy数组
                    
                    print(f"   📊 点云尺寸: {world_points.shape}", flush=True)
                    print(f"   🎯 计算体素集合...", flush=True)
                    
                    valid_mask = world_points_conf > 0.1
                    if valid_mask.any():
                        valid_pts = world_points[valid_mask]
                        scene_extent = np.max(valid_pts.max(axis=0) - valid_pts.min(axis=0))
                        voxel_size = scene_extent / 20.0
                        print(f"   📏 场景范围: {scene_extent:.2f}m, 体素大小: {voxel_size:.3f}m", flush=True)
                    else:
                        voxel_size = 0.05
                        print(f"   ⚠️  使用默认体素大小: {voxel_size}m", flush=True)
                    
                    voxel_sets = compute_voxel_sets(world_points, world_points_conf, voxel_size)
                    
                    print(f"   🎯 运行贪心最大覆盖算法 (目标{args.num_frames}帧)...", flush=True)
                    keyframe_indices = maximum_coverage_sampling(voxel_sets, args.num_frames)
                    
                    vggt_model = unload_model(vggt_model)
                    del frames, vggt_results, world_points, world_points_conf, voxel_sets
                    torch.cuda.empty_cache()
                
                print(f"✅ VGGT重建+关键帧提取完成\n", flush=True)

                frame_paths_with_indices = extract_specific_frames(
                    args.input_video, keyframe_indices, args.temp_dir
                )

            except Exception as e:
                print(f"⚠️  VGGT初始化失败: {e}", flush=True)
                import traceback
                traceback.print_exc()
                print(f"   回退到均匀采样模式...\n", flush=True)
                args.no_vggt = True
                keyframe_indices = None

        if args.no_vggt:
            print(f"\n⚠️  使用均匀采样模式", flush=True)
            from tools.generate_scene_json_stage1 import extract_frames_from_video
            frame_paths = extract_frames_from_video(
                args.input_video, args.temp_dir, args.num_frames
            )
            if not frame_paths:
                print("❌ 未能提取任何帧", flush=True)
                return
            frame_paths_with_indices = [(i, p) for i, p in enumerate(frame_paths)]

        if not frame_paths_with_indices:
            print("❌ 未能提取任何帧", flush=True)
            return

        model, processor = load_vlm_model(args.vlm_checkpoint)

        category_registry = progressive_object_discovery(
            frame_paths_with_indices, model, processor
        )

        if not category_registry:
            print("❌ 未能发现任何物体类别", flush=True)
            return

        generate_scene_json(category_registry, args.output_json)

    except KeyboardInterrupt:
        print("\n⚠️  用户中断", flush=True)
    except Exception as e:
        import traceback
        print(f"\n❌ 错误: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
    finally:
        if os.path.exists(args.temp_dir):
            import shutil
            shutil.rmtree(args.temp_dir)
            print(f"🧹 已清理临时目录: {args.temp_dir}", flush=True)


if __name__ == '__main__':
    main()
