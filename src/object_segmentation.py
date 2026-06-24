import numpy as np
from PIL import Image
import torch

def segment_wall_and_floor(images, sam3_image_model):
    """
    Use SAM3 to segment wall and floor from the input images.
    
    Args:
        images: numpy array of shape (S, H, W, 3)
        sam3_image_model: The loaded SAM3 image model object.
    Returns:
        wall_masks: A list of dictionaries containing 'frame_id' and 'mask' (binary mask of the wall).
        floor_masks: A list of dictionaries containing 'frame_id' and 'mask' (binary mask of the floor).
        [
        
            {
                'frame_id': int,
                'mask': numpy array of shape (H, W) with binary values (True or False)
            },
            ...
        ]
    """

    wall_masks = []
    floor_masks = []
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam3_image_model.set_image(image)
            sam3_image_model.reset_all_prompts(inference_state)
            inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt="single wall")
            masks = inference_state['masks'].cpu().numpy()
        for mask in masks:
            mask_arr = np.atleast_2d(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            if np.sum(mask_arr) > 500: # Filter out small masks.
                wall_masks.append({
                    'frame_id': i,
                    'mask': mask_arr
                })
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sam3_image_model.reset_all_prompts(inference_state)
            inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt="floor")
            masks = inference_state['masks'].cpu().numpy()
        for mask in masks:
            mask_arr = np.atleast_2d(mask)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[0]
            if np.sum(mask_arr) > 500: # Filter out small masks.
                floor_masks.append({
                    'frame_id': i,
                    'mask': mask_arr
                })
    return wall_masks, floor_masks


def _deduplicate_masks(masks, iou_threshold=0.5):
    """去重: 合并 IoU > threshold 的 mask"""
    if len(masks) <= 1:
        return masks
    unique = []
    for mask in masks:
        is_dup = False
        for existing in unique:
            intersection = np.sum(mask & existing)
            union = np.sum(mask | existing)
            if union > 0 and intersection / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(mask)
    return unique


def segment_large_flat_surfaces(images, sam3_image_model, min_mask_area=5000):
    """
    用宽泛文本提示分割大平面物体 (阶段3 fallback).

    当 "floor" 文本提示失败时, 用多个宽泛提示词尝试分割大平面.
    SAM3 不支持无文本提示的自动分割, 所以用 "flat surface"/"ground"/"horizontal surface" 替代.

    Args:
        images: numpy array of shape (S, H, W, 3)
        sam3_image_model: SAM3 image model
        min_mask_area: 最小 mask 面积 (像素), 过滤小 mask (默认5000, 比 segment_wall_and_floor 的500更大)
    Returns:
        large_plane_masks: list of dicts with 'frame_id' and 'mask'
    """
    large_plane_masks = []
    prompts = ["flat surface", "ground", "horizontal surface"]

    for i, image in enumerate(images):
        image = Image.fromarray(image)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam3_image_model.set_image(image)

        frame_masks = []
        for prompt in prompts:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sam3_image_model.reset_all_prompts(inference_state)
                inference_state = sam3_image_model.set_text_prompt(state=inference_state, prompt=prompt)
                masks = inference_state['masks'].cpu().numpy()
            for mask in masks:
                mask_arr = np.atleast_2d(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                if np.sum(mask_arr) > min_mask_area:
                    frame_masks.append(mask_arr)

        unique_masks = _deduplicate_masks(frame_masks)
        for mask_arr in unique_masks:
            large_plane_masks.append({
                'frame_id': i,
                'mask': mask_arr
            })

        sam3_image_model.reset_all_prompts(inference_state)

    return large_plane_masks


def detect_floor_reference_points_with_vlm(image, vlm_model, vlm_processor,
                                           num_points=4, max_new_tokens=256):
    """
    用VLM识别图像中地面的代表性参考点.

    SAM3的Sam3Processor没有直接的point prompt API, 但支持box prompt.
    因此先用VLM生成地面参考点, 再围绕每个点构造小box作为SAM3输入.

    Args:
        image: PIL.Image 或 numpy array (H, W, 3) RGB
        vlm_model: 已加载的VLM模型
        vlm_processor: 已加载的VLM processor
        num_points: 期望返回的参考点数量 (默认4)
        max_new_tokens: VLM最大输出token数

    Returns:
        points: list of (x, y) 归一化坐标 [0, 1], 或空列表
    """
    try:
        from PIL import Image as PILImage
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)

        prompt = (
            f"Identify {num_points} representative points on the floor/ground in this image. "
            "Return ONLY a JSON list of [x, y] normalized coordinates (0-1). "
            "Example: [[0.25, 0.75], [0.50, 0.80], [0.75, 0.75], [0.40, 0.85]]. "
            "Do not include any explanation."
        )

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]

        try:
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt", enable_thinking=False
            )
        except TypeError:
            inputs = vlm_processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
        inputs = inputs.to(vlm_model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": vlm_processor.tokenizer.eos_token_id,
            "do_sample": False,
        }

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, **gen_kwargs)

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 解析JSON
        import json
        import re
        # 尝试直接解析
        try:
            points = json.loads(output_text.strip())
            if isinstance(points, list) and len(points) > 0:
                normalized = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        x, y = float(p[0]), float(p[1])
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            normalized.append((x, y))
                return normalized[:num_points]
        except Exception:
            pass

        # 尝试从文本中提取JSON数组
        matches = re.findall(r'\[\s*\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]\s*(,\s*\[\s*\d+\.?\d*\s*,\s*\d+\.?\d*\s*\]\s*)*\]', output_text)
        for match in matches:
            try:
                points = json.loads(match)
                normalized = []
                for p in points:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        x, y = float(p[0]), float(p[1])
                        if 0 <= x <= 1 and 0 <= y <= 1:
                            normalized.append((x, y))
                if normalized:
                    return normalized[:num_points]
            except Exception:
                continue

        return []
    except Exception as e:
        print(f"   ⚠️ VLM地面参考点检测失败: {e}", flush=True)
        return []


def segment_floor_with_box_prompts(images, sam3_image_model, reference_points_per_frame,
                                   box_size=0.05, min_mask_area=500):
    """
    用VLM提供的地面参考点, 通过SAM3 box prompt生成floor mask.

    SAM3的Sam3Processor不支持point prompt, 但支持add_geometric_prompt(box).
    这里用围绕参考点的小box近似point prompt.

    Args:
        images: numpy array (S, H, W, 3) RGB
        sam3_image_model: SAM3 image processor
        reference_points_per_frame: dict {frame_id: list of (x, y) normalized coords}
        box_size: 小box的宽高 (归一化坐标, 默认0.05)
        min_mask_area: 最小mask面积 (像素)

    Returns:
        floor_masks: list of dicts with 'frame_id' and 'mask'
    """
    floor_masks = []

    for frame_id, points in reference_points_per_frame.items():
        if frame_id >= len(images):
            continue
        image = Image.fromarray(images[frame_id])

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            inference_state = sam3_image_model.set_image(image)

        frame_masks = []
        for (x, y) in points:
            # 围绕点构造小box [center_x, center_y, width, height], 归一化
            box = [x, y, box_size, box_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                sam3_image_model.reset_all_prompts(inference_state)
                inference_state = sam3_image_model.add_geometric_prompt(
                    box=box, label=True, state=inference_state
                )
            masks = inference_state['masks'].cpu().numpy()
            for mask in masks:
                mask_arr = np.atleast_2d(mask)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[0]
                if np.sum(mask_arr) > min_mask_area:
                    frame_masks.append(mask_arr)

        sam3_image_model.reset_all_prompts(inference_state)

        unique_masks = _deduplicate_masks(frame_masks)
        for mask_arr in unique_masks:
            floor_masks.append({
                'frame_id': frame_id,
                'mask': mask_arr
            })

    return floor_masks


def propagate_in_video(predictor, session_id):
    # we will just propagate from frame 0 to the end of the video
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]

    return outputs_per_frame

def segment_and_track(category, video_predictor, session_id):
    '''
    Segment raw instance masks using sam3 video tracking
    Args:
        category: A str means the category to segment
        video_predictor: the loaded sam3 video model  
        session_id: the session with loaded video frames corresponding to the video_predictor
    Returns:
        A list of list, each list represents a segmented instance and is composed
        of dicts with keys frame_id and mask (binary mask of the instance in that frame)
        [
            [
                {
                    'frame_id': int,
                    'mask': numpy array of shape (H, W) with binary values (True or False)
                },
                ...
            ],
            ...
        ]
    '''
    # Reset session and add text prompt for the category to segment
    _ = video_predictor.handle_request(request=dict(type="reset_session", session_id=session_id))
    video_predictor.handle_request(request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=category))
    outputs_per_frame = propagate_in_video(video_predictor, session_id)
    if not outputs_per_frame:
        return []

    # Collect all object IDs across frames, discontinuous segments will be split into different instances.
    all_obj_ids = set()
    for frame_idx in outputs_per_frame.keys():
        all_obj_ids.update(outputs_per_frame[frame_idx]['out_obj_ids'])
    if len(all_obj_ids) == 0:
        print(f'No object detected for {category}.')
        return []
    final_results = []
    sorted_obj_ids = sorted(list(all_obj_ids))
    for obj_id in sorted_obj_ids:
        raw_frame_ids = sorted([
            id for id in outputs_per_frame.keys() 
            if obj_id in outputs_per_frame[id]['out_obj_ids']
        ])
        
        if not raw_frame_ids:
            continue
        segments = []
        if len(raw_frame_ids) > 0:
            current_segment = [raw_frame_ids[0]]
            for i in range(1, len(raw_frame_ids)):
                if raw_frame_ids[i] == raw_frame_ids[i-1] + 1:
                    current_segment.append(raw_frame_ids[i])
                else:
                    segments.append(current_segment)
                    current_segment = [raw_frame_ids[i]]
            segments.append(current_segment)
        for frame_ids in segments:
            instance_track = []
            
            for frame_id in frame_ids:
                obj_indices = np.where(outputs_per_frame[frame_id]['out_obj_ids'] == obj_id)[0]
                
                if len(obj_indices) > 0:
                    idx = obj_indices[0]
                    raw_mask = outputs_per_frame[frame_id]['out_binary_masks'][idx].squeeze()
                    binary_mask = raw_mask > 0
                    
                    instance_track.append({
                        'frame_id': frame_id,
                        'mask': binary_mask
                    })
            if instance_track:
                final_results.append(instance_track)

    return final_results