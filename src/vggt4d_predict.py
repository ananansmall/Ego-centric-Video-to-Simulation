import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from einops import rearrange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'VGGT4D'))

from vggt4d.models.vggt4d import VGGTFor4D
from vggt4d.masks.dynamic_mask import (
    adaptive_multiotsu_variance,
    cluster_attention_maps,
    extract_dyn_map,
)
from vggt4d.utils.model_utils import organize_qk_dict
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

CHECKPOINT_PATH = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/VGGT4D/ckpts/model_tracker_fixed_e20.pt"


def load_vggt4d_model(checkpoint_path=CHECKPOINT_PATH):
    """
    加载 VGGT4D 模型权重。

    VGGT4D 继承自 VGGT, 额外包含 AggregatorFor4D (支持 dyn_masks 输入)。
    与原版 VGGT 一样有 PointHead + DepthHead + TrackHead。

    Args:
        checkpoint_path: .pt 权重文件路径

    Returns:
        VGGTFor4D 模型实例, eval 模式, 在 CPU 上 (需外部 .to("cuda"))
    """
    model = VGGTFor4D()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def load_vggt4d_frames(video_path, max_frames):
    import re
    import subprocess
    import tempfile

    if os.path.isdir(video_path):
        images = os.listdir(video_path)
        images = [img for img in images if img.endswith(('.jpg', '.png', '.jpeg'))]
        images = sorted(
            images,
            key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else -1,
        )
        total_frames = len(images)
        if total_frames == 0:
            raise ValueError(f"No image files found in directory: {video_path}")
        if total_frames > max_frames and max_frames > 0:
            indices = np.linspace(0, total_frames - 1, max_frames).astype(int)
            images = [os.path.join(video_path, images[i]) for i in indices]
        else:
            images = [os.path.join(video_path, img) for img in images]
        return load_and_preprocess_images(images)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    "ffmpeg", "-i", video_path,
                    "-vsync", "0",
                    os.path.join(temp_dir, "frame_%04d.png"),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return load_vggt4d_frames(temp_dir, max_frames)


def _temporal_filter_dyn_masks(dyn_masks, min_dynamic_ratio=0.5):
    """
    对 dyn_masks 做时间维度持续性过滤, 消除短暂遮挡导致的过度标记。

    原理:
      VGGT4D 的 dyn_map 公式检测的是"运动线索"而非"真正动态物体"。
      手快速扫过物体时, 被遮挡的背景区域也会被误标为动态。
      但真正动态物体(手)在连续多帧中持续出现, 而短暂遮挡只在少数帧中存在。

      对每个像素位置, 统计它在多少帧中被标记为动态:
        - 动态比例 >= min_dynamic_ratio → 真正动态物体, 保留
        - 动态比例 <  min_dynamic_ratio → 短暂遮挡, 恢复为静态

    示例 (min_dynamic_ratio=0.5, 共10帧):
      像素A: 帧2-8被标为动态 (7/10=0.7 > 0.5) → 保留 (手持续存在)
      像素B: 帧3-4被标为动态 (2/10=0.2 < 0.5) → 恢复 (手扫过, 短暂遮挡)

    Args:
        dyn_masks:          numpy (S, H, W) bool, 原始动态掩码
        min_dynamic_ratio:  float, 被视为真正动态所需的最小动态帧比例 (默认 0.5)
                            0.5 = 至少一半的帧中是动态才保留
                            值越大过滤越激进, 值越小保留越多

    Returns:
        filtered_masks: numpy (S, H, W) bool, 过滤后的动态掩码
    """
    dynamic_ratio = np.mean(dyn_masks.astype(np.float32), axis=0)
    truly_dynamic = dynamic_ratio >= min_dynamic_ratio
    filtered_masks = np.broadcast_to(truly_dynamic[None], dyn_masks.shape).copy()
    return filtered_masks


def _predictions_to_pcd(predictions, conf_thres=50.0):
    if "world_points_from_depth" in predictions:
        pred_world_points = predictions["world_points_from_depth"]
    else:
        pred_world_points = predictions["world_points"]
    pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    vertices_3d = pred_world_points.reshape(-1, 3)
    conf = pred_world_points_conf.reshape(-1)

    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])

    return trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)


def vggt4d_predict(images, model, enable_dyn_mask=True, filter_dynamic_points=True, min_dynamic_ratio=0.5, conf_thres=50.0):
    '''
    使用 VGGT4D 模型对帧序列进行推理, 输出格式与 vggt_predict() 一致, 额外返回 dyn_masks。

    推理流程 (两次前向):
      第1次推理: model(images) → VGGTFor4D 内部:
         - AggregatorFor4D: 与原版 Aggregator 相同, 但额外输出 qk_dict 和 enc_feat
         - CameraHead: 4次迭代精修 (与原版 VGGT 相同), 输出 9D pose_enc
         - DepthHead (DPTHead, output_dim=2, activation=exp):
           depth = exp(depth_logits),  depth_conf = 1 + exp(conf_logits)
         - PointHead (DPTHead, output_dim=4, activation=inv_log):
           world_points = sign(x) * (exp(|x|) - 1),  world_points_conf = 1 + exp(conf_logits)
           (PointHead 输出未被使用, 最终 world_points 来自深度反投影)

      动态掩码提取 (Stage 1):
         - organize_qk_dict(): 将 Q/K 按 (camera, register, token) 分类重组
         - extract_dyn_map(): 从5种注意力特征图组合计算动态区域得分
           dyn_map = (1-mean1) * (1-var1) * mean2 * (1-mean3) * var3
         - cluster_attention_maps(): KMeans(64类) 平滑动态得分
         - adaptive_multiotsu_variance(): 自适应多阈值 Otsu 分割 → dyn_masks

      时间持续性过滤 (Stage 2):
         - _temporal_filter_dyn_masks(): 对 dyn_masks 做时间维度持续性过滤
         - 原理: 真正动态物体在连续多帧中持续出现, 短暂遮挡只在少数帧中存在
         - 对每个像素统计动态帧比例, 低于 min_dynamic_ratio 的恢复为静态
         - 效果: 手扫过的背景区域不再被误标为动态

      第2次推理: model(images, dyn_masks=dyn_masks) →
         - AggregatorFor4D: dyn_masks 下采样到 patch 分辨率, 在帧间注意力中屏蔽动态 token
           (注意: 被注释掉的代码显示曾尝试将动态 token 置零, 但效果不好)
         - CameraHead: 基于静态场景约束精修位姿 → extrinsic2, intrinsic2
         - DepthHead/PointHead: 输出未使用 (只取 pose_enc2)

      后处理:
         - 用第1次的 depth + 第2次的 extrinsic 反投影得到 world_points
         - world_points_conf = depth_conf (第1次 DepthHead 输出)
         - 若 filter_dynamic_points=True: 将 dyn_masks 区域的 world_points_conf 设为 0
           → 下游所有置信度过滤 (predictions_to_pcd, self_category_deduplicate,
             verify_all_instances) 自动排除动态点, 无需修改 mainv2
         - dyn_masks: (S, H, W) bool, 经过时间持续性过滤后的动态区域掩码

    与 vggt_predict() 的核心差异:
      - 两次推理: 第1次获取深度+动态掩码, 第2次带掩码精修位姿
      - 有 PointHead 但未使用其 world_points 输出
      - world_points_conf 用 depth_conf (与 VGGT-Omega 相同)
      - 相机外参经过动态掩码精修, 对动态场景更准确
      - filter_dynamic_points: 利用 dyn_masks 过滤动态点, 这是 VGGT/VGGT-Omega 不具备的能力
      - min_dynamic_ratio: 时间持续性过滤, 解决手扫过背景被误标为动态的问题

    Args:
        images:                torch.Tensor (S, 3, 518, 518), 来自 load_vggt4d_frames()
        model:                 VGGTFor4D 模型实例 (eval 模式, 在 CUDA 上)
        enable_dyn_mask:       bool, 是否启用动态掩码提取+位姿精修 (默认 True)
        filter_dynamic_points: bool, 是否用 dyn_masks 将动态区域的置信度置零 (默认 True)
                               仅在 enable_dyn_mask=True 时生效
                               效果: 手/运动物体等动态点从点云中移除, 不会污染去重和mesh生成
        min_dynamic_ratio:     float, 时间持续性过滤阈值 (默认 0.5)
                               一个像素在 >= min_dynamic_ratio 比例的帧中被标为动态,
                               才被认为是真正动态物体。低于此比例的恢复为静态(短暂遮挡)。
                               0.5 = 至少一半帧中动态才保留; 0.0 = 不过滤; 1.0 = 极度严格
        conf_thres:            float, 置信度百分位阈值 (默认 50.0)

    Returns:
        dict (格式与 vggt_predict() 一致, 额外包含 dyn_masks):
          - point_cloud_data:   trimesh.PointCloud (动态点已过滤)
          - colors:             numpy (S, H, W, 3) uint8
          - depths:             numpy (S, H, W), 第1次推理的深度图 (动态区域深度保留, 但conf=0)
          - extrinsics:         numpy (S, 4, 4), 第2次推理精修后的外参
          - world_points:       numpy (S, H, W, 3), depth1 + extrinsic2 反投影
          - world_points_conf:  numpy (S, H, W), depth_conf, 动态区域已置零
          - intrinsic:          numpy (3, 3), 所有帧内参取平均
          - dyn_masks:          numpy (S, H, W) bool, 经过时间过滤后的动态区域掩码
    '''
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    n_img = images.shape[0]

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            predictions, qk_dict, enc_feat, agg_tokens_list = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions["images"] = images

    dyn_masks = None
    if enable_dyn_mask and n_img > 1:
        qk_dict = organize_qk_dict(qk_dict, n_img)
        dyn_maps = extract_dyn_map(qk_dict, images)
        dyn_maps = dyn_maps.cpu()

        h_tok = images.shape[-2] // 14
        w_tok = images.shape[-1] // 14
        feat_map = rearrange(
            enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok
        )

        norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)

        upsampled_map = F.interpolate(
            rearrange(norm_dyn_map.float(), "n h w -> n 1 h w"),
            size=(images.shape[-2], images.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        upsampled_map = rearrange(upsampled_map, "n 1 h w -> n h w")

        thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
        dyn_masks = upsampled_map > thres

        del enc_feat, feat_map, dyn_maps, qk_dict
        torch.cuda.empty_cache()

    if dyn_masks is not None:
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions2, _, _, _ = model(
                    images, dyn_masks=dyn_masks.to(images.device)
                )

        extrinsic2, intrinsic2 = pose_encoding_to_extri_intri(
            predictions2["pose_enc"], images.shape[-2:]
        )
        predictions["extrinsic"] = extrinsic2
        predictions["intrinsic"] = intrinsic2
        torch.cuda.empty_cache()

    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            arr = predictions[key].to(device="cpu", dtype=torch.float32).numpy()
            if arr.shape[0] == 1:
                arr = arr.squeeze(0)
            predictions[key] = arr
    predictions.pop("pose_enc_list", None)

    dyn_masks_np = dyn_masks.cpu().numpy() if dyn_masks is not None else None

    if dyn_masks_np is not None and min_dynamic_ratio > 0.0:
        dyn_masks_np = _temporal_filter_dyn_masks(dyn_masks_np, min_dynamic_ratio)

    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(
        depth_map, predictions["extrinsic"], predictions["intrinsic"]
    )
    predictions["world_points_from_depth"] = world_points

    if filter_dynamic_points and dyn_masks_np is not None:
        predictions["depth_conf"][dyn_masks_np] = 0.0

    point_cloud_data = _predictions_to_pcd(predictions, conf_thres=conf_thres)

    colors = (predictions["images"].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    depths = predictions["depth"].squeeze(-1)

    n_img_out = predictions["extrinsic"].shape[0]
    extrinsics = np.zeros((n_img_out, 4, 4), dtype=np.float32)
    extrinsics[:, :3, :4] = predictions["extrinsic"]
    extrinsics[:, 3, 3] = 1.0

    world_points_out = predictions["world_points_from_depth"].copy()
    world_points_conf = predictions.get("depth_conf", np.ones_like(depths))
    intrinsic_out = np.mean(predictions["intrinsic"], axis=0)

    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "point_cloud_data": point_cloud_data,
        "colors": colors,
        "depths": depths,
        "extrinsics": extrinsics,
        "world_points": world_points_out,
        "world_points_conf": world_points_conf,
        "intrinsic": intrinsic_out,
    }

    if dyn_masks_np is not None:
        result["dyn_masks"] = dyn_masks_np

    return result
