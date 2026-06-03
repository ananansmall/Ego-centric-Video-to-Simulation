"""
Step 1: VGGT TrackHead 联合点追踪
==================================

核心思路 (V-Dreamer):
  视频本身就是最好的运动先验。
  通过 VGGT 内置 TrackHead (CoTracker 架构变体) 联合追踪物体表面点，
  利用点间相关性和时序一致性，获得比逐帧 Procrustes 更鲁棒的轨迹。

数据流:
  VGGT4D Stage 1 (无 query_points)
    → depth, extrinsics, dynamic_mask, qk_dict
  采样 query_points (从 dynamic_mask - hand_mask)
  VGGT4D Stage 2 (带 dyn_masks + query_points)
    → refined extrinsics + tracked 2D trajectories
  深度反投影
    → 3D point trajectories (世界坐标系)

输出:
  tracks_3d:    (S, N, 3)  N个点在S帧中的3D世界坐标
  tracks_2d:    (S, N, 2)  N个点在S帧中的2D像素坐标
  visibility:   (S, N)     可见性概率
  confidence:   (S, N)     置信度概率
"""

import os
import sys

import cv2
import numpy as np
import torch

VGGT4D_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "VGGT4D"))
if VGGT4D_ROOT not in sys.path:
    sys.path.insert(0, VGGT4D_ROOT)

from vggt4d.utils.model_utils import inference, organize_qk_dict
from vggt4d.masks.dynamic_mask import (
    cluster_attention_maps,
    extract_dyn_map,
    adaptive_multiotsu_variance,
)
from vggt4d.masks.refine_dyn_mask import RefineDynMask
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def load_vggt4d_model(ckpt_path=None, device="cuda"):
    """加载 VGGT4D 模型

    Args:
        ckpt_path: 模型权重路径, 默认 VGGT4D/ckpts/model_tracker_fixed_e20.pt
        device: 推理设备

    Returns:
        model: VGGTFor4D 模型实例
    """
    from vggt4d.models.vggt4d import VGGTFor4D

    if ckpt_path is None:
        ckpt_path = os.path.join(VGGT4D_ROOT, "ckpts", "model_tracker_fixed_e20.pt")

    model = VGGTFor4D()
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()
    model = model.to(device)
    return model


def run_vggt4d_stage1(model, images, device="cuda"):
    """VGGT4D Stage 1: 预测深度图和动态图

    Returns:
        dict: {predictions, qk_dict, dyn_masks}
    """
    from einops import rearrange
    import torch.nn.functional as F

    predictions1, qk_dict, enc_feat, agg_tokens_list = inference(model, images)
    del agg_tokens_list

    qk_dict = organize_qk_dict(qk_dict, images.shape[0])
    dyn_maps = extract_dyn_map(qk_dict, images)

    n_img, _, h_img, w_img = images.shape
    h_tok, w_tok = h_img // 14, w_img // 14

    feat_map = rearrange(enc_feat, "n_img (h w) c -> n_img h w c", h=h_tok, w=w_tok)
    norm_dyn_map, _ = cluster_attention_maps(feat_map, dyn_maps)

    upsampled_map = F.interpolate(
        rearrange(norm_dyn_map, "n_img h w -> n_img 1 h w"),
        size=(h_img, w_img),
        mode="bilinear",
        align_corners=False,
    )
    upsampled_map = rearrange(upsampled_map, "n_img 1 h w -> n_img h w")

    thres = adaptive_multiotsu_variance(upsampled_map.cpu().numpy())
    dyn_masks = upsampled_map > thres

    if "enc_feat" in dir():
        del enc_feat
    if "feat_map" in dir():
        del feat_map
    torch.cuda.empty_cache()

    return {
        "predictions": predictions1,
        "qk_dict": qk_dict,
        "dyn_masks": dyn_masks,
    }


def run_vggt4d_stage2_with_tracking(model, images, dyn_masks, query_points, device="cuda"):
    """VGGT4D Stage 2: 用动态掩码精化外参 + TrackHead 联合追踪

    Args:
        model: VGGTFor4D 模型
        images: (S, 3, H, W) 图像张量
        dyn_masks: (S, H, W) bool 动态掩码
        query_points: (1, N, 2) 或 (N, 2) 查询点像素坐标
        device: 推理设备

    Returns:
        dict: {predictions, has_tracking}
    """
    if isinstance(query_points, np.ndarray):
        query_points = torch.from_numpy(query_points).float()
    if query_points.ndim == 2:
        query_points = query_points.unsqueeze(0)
    query_points = query_points.to(device)

    predictions2, _, _, _ = inference(
        model, images, dyn_masks=dyn_masks.to(device), query_points=query_points
    )

    has_tracking = "track" in predictions2
    return {
        "predictions": predictions2,
        "has_tracking": has_tracking,
    }


def run_vggt4d_stage3_refine_mask(model, images, predictions, dyn_masks, device="cuda"):
    """VGGT4D Stage 3: 精化动态掩码

    Returns:
        refined_mask: (S, H, W) bool 精化后的动态掩码
    """
    pred_intrinsic = predictions["intrinsic"]
    pred_cam2world = predictions["cam2world"]
    pred_depths = predictions["depth"]

    refiner = RefineDynMask(
        images,
        torch.tensor(pred_depths).to(device),
        dyn_masks.to(device),
        torch.tensor(pred_cam2world).float().to(device),
        torch.tensor(pred_intrinsic).to(device),
        device,
    )
    refined_mask = refiner.refine_masks()
    del refiner
    torch.cuda.empty_cache()
    return refined_mask


def sample_query_points_from_mask(mask, n_points=64, method="grid", margin=5):
    """从二值掩码中采样查询点

    Args:
        mask: (H, W) bool 二值掩码
        n_points: 采样点数
        method: 采样方法 ("grid"/"random"/"contour")
        margin: 距离边缘的最小像素数

    Returns:
        query_points: (N, 2) 像素坐标 (x, y 格式)
    """
    if mask.ndim == 3:
        mask = mask.any(axis=0)

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if margin > 0:
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask, iterations=margin)
        ys_inner, xs_inner = np.where(eroded)
        if len(xs_inner) > 10:
            xs, ys = xs_inner, ys_inner

    if method == "grid":
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        side = int(np.ceil(np.sqrt(n_points)))
        gx = np.linspace(x_min, x_max, side)
        gy = np.linspace(y_min, y_max, side)
        gx, gy = np.meshgrid(gx, gy)
        gx, gy = gx.flatten(), gy.flatten()
        valid = mask[gy.astype(int).clip(0, mask.shape[0] - 1),
                     gx.astype(int).clip(0, mask.shape[1] - 1)]
        points = np.stack([gx[valid], gy[valid]], axis=-1)
        if len(points) > n_points:
            idx = np.linspace(0, len(points) - 1, n_points, dtype=int)
            points = points[idx]
        return points.astype(np.float32)

    elif method == "random":
        idx = np.random.choice(len(xs), min(n_points, len(xs)), replace=False)
        return np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)

    elif method == "contour":
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_points = []
        for c in contours:
            for pt in c:
                edge_points.append(pt[0])
        edge_points = np.array(edge_points) if edge_points else np.zeros((0, 2))

        n_edge = min(len(edge_points), n_points // 2)
        n_inner = n_points - n_edge

        if n_edge > 0:
            idx = np.linspace(0, len(edge_points) - 1, n_edge, dtype=int)
            edge_pts = edge_points[idx].astype(np.float32)
        else:
            edge_pts = np.zeros((0, 2), dtype=np.float32)

        if n_inner > 0 and len(xs) > 0:
            idx = np.random.choice(len(xs), min(n_inner, len(xs)), replace=False)
            inner_pts = np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)
        else:
            inner_pts = np.zeros((0, 2), dtype=np.float32)

        points = np.concatenate([edge_pts, inner_pts], axis=0)
        return points

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def unproject_tracks_to_3d(tracks_2d, depths, extrinsics, intrinsic):
    """将 2D 追踪轨迹反投影到 3D 世界坐标

    Args:
        tracks_2d:   (S, N, 2) 像素坐标轨迹
        depths:      (S, H, W) 深度图 (米)
        extrinsics:  (S, 4, 4) 相机外参 (cam2world)
        intrinsic:   (3, 3) 相机内参

    Returns:
        tracks_3d: (S, N, 3) 世界坐标轨迹
        valid:     (S, N) bool 有效标记
    """
    S, N, _ = tracks_2d.shape
    H, W = depths.shape[1], depths.shape[2]
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    tracks_3d = np.full((S, N, 3), np.nan, dtype=np.float32)
    valid = np.zeros((S, N), dtype=bool)

    for t in range(S):
        c2w = extrinsics[t]
        R = c2w[:3, :3]
        T = c2w[:3, 3]

        for p in range(N):
            u, v = tracks_2d[t, p]
            ui, vi = int(round(u)), int(round(v))

            if ui < 0 or ui >= W or vi < 0 or vi >= H:
                continue

            d = depths[t, vi, ui]
            if d <= 0 or np.isnan(d):
                continue

            x_cam = (u - cx) / fx * d
            y_cam = (v - cy) / fy * d
            z_cam = d
            p_cam = np.array([x_cam, y_cam, z_cam])
            p_world = R @ p_cam + T

            tracks_3d[t, p] = p_world
            valid[t, p] = True

    return tracks_3d, valid


def subtract_hand_mask(dynamic_mask, hand_masks, scale_factor=None):
    """从 dynamic_mask 中减去手部区域, 只保留物体区域

    Args:
        dynamic_mask: (S, H_v, W_v) bool VGGT4D 动态掩码
        hand_masks: (S, H_orig, W_orig) bool HaWoR 手部掩码
        scale_factor: tuple (scale_y, scale_x) 缩放因子

    Returns:
        object_only_mask: (S, H_v, W_v) bool 物体区域掩码
    """
    if isinstance(dynamic_mask, torch.Tensor):
        dynamic_mask = dynamic_mask.cpu().numpy()
    if isinstance(hand_masks, torch.Tensor):
        hand_masks = hand_masks.cpu().numpy()

    S_dyn, H_v, W_v = dynamic_mask.shape
    S_hand, H_orig, W_orig = hand_masks.shape
    S = min(S_dyn, S_hand)

    if scale_factor is None:
        scale_y = H_v / H_orig
        scale_x = W_v / W_orig
    else:
        scale_y, scale_x = scale_factor

    object_only_mask = dynamic_mask.copy()

    for t in range(S):
        hand_m = hand_masks[t]
        hand_resized = cv2.resize(
            hand_m.astype(np.uint8), (W_v, H_v),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        object_only_mask[t] = dynamic_mask[t] & ~hand_resized

    return object_only_mask


def sample_query_points_from_dynamic_mask(dynamic_mask, hand_masks=None, n_points=64, method="grid", reference_frame=0):
    """从 dynamic_mask 采样物体查询点 (自动减去手部区域)

    Args:
        dynamic_mask: (S, H, W) bool VGGT4D 动态掩码
        hand_masks: (S, H_orig, W_orig) bool HaWoR 手部掩码 (可选)
        n_points: 采样点数
        method: 采样方法
        reference_frame: 参考帧

    Returns:
        dict: {obj_key: query_points (N, 2)}
    """
    if hand_masks is not None:
        obj_mask = subtract_hand_mask(dynamic_mask, hand_masks)
    else:
        obj_mask = dynamic_mask

    if isinstance(obj_mask, torch.Tensor):
        obj_mask = obj_mask.cpu().numpy()

    ref_mask = obj_mask[reference_frame]
    if ref_mask.ndim == 3:
        ref_mask = ref_mask.any(axis=0)

    if not ref_mask.any():
        for t in range(len(obj_mask)):
            m = obj_mask[t]
            if m.ndim == 3:
                m = m.any(axis=0)
            if m.any():
                ref_mask = m
                reference_frame = t
                print(f"[02_point_tracker] Using frame {t} as reference (first with object mask)")
                break

    if not ref_mask.any():
        return {}

    pts = sample_query_points_from_mask(ref_mask, n_points=n_points, method=method)
    if len(pts) == 0:
        return {}

    return {"object": pts}


def run_point_tracking(
    video_path_or_images,
    hand_masks=None,
    n_query_points=64,
    query_point_method="grid",
    reference_frame=0,
    ckpt_path=None,
    device="cuda",
    output_dir=None,
):
    """完整点追踪管线: VGGT4D 三阶段推理 + TrackHead 联合追踪

    Args:
        video_path_or_images: 视频路径 (str) 或预处理图像 (S,3,H,W) 张量
        hand_masks: (S, H_orig, W_orig) bool HaWoR 手部掩码 (可选)
        n_query_points: 每个物体的查询点数
        query_point_method: 采样方法 ("grid"/"random"/"contour")
        reference_frame: 参考帧 ID
        ckpt_path: VGGT4D 模型权重路径
        device: 推理设备
        output_dir: 输出目录 (可选)

    Returns:
        dict: {
            vggt_predictions: VGGT4D 预测结果,
            dynamic_mask: (S, H, W) bool 精化动态掩码,
            object_tracks: {obj_key: {tracks_2d, tracks_3d, visibility, confidence, valid_3d, query_points}},
        }
    """
    model = load_vggt4d_model(ckpt_path, device)

    if isinstance(video_path_or_images, str):
        from pathlib import Path

        image_dir = Path(video_path_or_images)
        if image_dir.is_file():
            import subprocess, tempfile

            tmp_dir = tempfile.mkdtemp()
            subprocess.run(
                ["ffmpeg", "-i", str(image_dir), "-q:v", "2", f"{tmp_dir}/%04d.jpg"],
                check=True,
                capture_output=True,
            )
            image_paths = sorted(Path(tmp_dir).glob("*.jpg"))
        else:
            image_paths = sorted(
                list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
            )
        images = load_and_preprocess_images([str(p) for p in image_paths]).to(device)
    else:
        images = video_path_or_images.to(device)

    S = images.shape[0]
    print(f"[02_point_tracker] Loaded {S} frames, running VGGT4D Stage 1...")

    stage1 = run_vggt4d_stage1(model, images, device)
    dyn_masks = stage1["dyn_masks"]
    print(f"[02_point_tracker] Stage 1 done. Dynamic mask: {dyn_masks.shape}")

    query_points_dict = sample_query_points_from_dynamic_mask(
        dyn_masks.cpu().numpy() if isinstance(dyn_masks, torch.Tensor) else dyn_masks,
        hand_masks=hand_masks,
        n_points=n_query_points,
        method=query_point_method,
        reference_frame=reference_frame,
    )

    if not query_points_dict:
        print("[02_point_tracker] No query points sampled, returning without tracking")
        return {
            "vggt_predictions": stage1["predictions"],
            "dynamic_mask": dyn_masks.cpu().numpy(),
            "object_tracks": {},
        }

    all_tracks = {}
    for obj_key, qp in query_points_dict.items():
        if len(qp) == 0:
            continue
        print(f"[02_point_tracker] Tracking {len(qp)} points for '{obj_key}'...")

        stage2 = run_vggt4d_stage2_with_tracking(
            model, images, dyn_masks, qp, device
        )

        if not stage2["has_tracking"]:
            print(f"[02_point_tracker] WARNING: No tracking result for '{obj_key}'")
            continue

        pred = stage2["predictions"]
        tracks_2d = pred["track"]
        visibility = pred["vis"]
        confidence = pred.get("conf", None)

        if isinstance(tracks_2d, torch.Tensor):
            tracks_2d = tracks_2d.cpu().numpy()
        if isinstance(visibility, torch.Tensor):
            visibility = visibility.cpu().numpy()
        if isinstance(confidence, torch.Tensor):
            confidence = confidence.cpu().numpy()

        if tracks_2d.ndim == 4:
            tracks_2d = tracks_2d[0]
        if visibility.ndim == 3:
            visibility = visibility[0]
        if confidence is not None and confidence.ndim == 3:
            confidence = confidence[0]

        all_tracks[obj_key] = {
            "tracks_2d": tracks_2d,
            "visibility": visibility,
            "confidence": confidence,
            "query_points": qp,
        }

    final_predictions = {}
    if all_tracks:
        last_stage2_pred = stage2["predictions"]
        final_predictions["extrinsic"] = last_stage2_pred["extrinsic"]
        final_predictions["intrinsic"] = last_stage2_pred["intrinsic"]
        final_predictions["cam2world"] = last_stage2_pred["cam2world"]
        final_predictions["depth"] = stage1["predictions"]["depth"]
        final_predictions["depth_conf"] = stage1["predictions"]["depth_conf"]
    else:
        final_predictions = stage1["predictions"]

    print("[02_point_tracker] Running Stage 3: refine dynamic mask...")
    refined_mask = run_vggt4d_stage3_refine_mask(
        model, images, final_predictions, dyn_masks, device
    )
    print("[02_point_tracker] Stage 3 done.")

    depths = final_predictions["depth"]
    if depths.ndim == 4 and depths.shape[-1] == 1:
        depths = depths[..., 0]
    extrinsics = final_predictions["cam2world"]
    intrinsic = final_predictions["intrinsic"]

    for obj_key, track_data in all_tracks.items():
        tracks_3d, valid_3d = unproject_tracks_to_3d(
            track_data["tracks_2d"], depths, extrinsics, intrinsic
        )
        track_data["tracks_3d"] = tracks_3d
        track_data["valid_3d"] = valid_3d
        n_valid = valid_3d.sum()
        n_total = valid_3d.size
        print(f"[02_point_tracker] '{obj_key}': {n_valid}/{n_total} valid 3D points")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "point_tracks.npz"),
            **{
                obj_key: {
                    "tracks_2d": t["tracks_2d"],
                    "tracks_3d": t["tracks_3d"],
                    "visibility": t["visibility"],
                    "confidence": t["confidence"],
                    "valid_3d": t["valid_3d"],
                    "query_points": t["query_points"],
                }
                for obj_key, t in all_tracks.items()
            },
        )
        print(f"[02_point_tracker] Results saved to {output_dir}/point_tracks.npz")

    return {
        "vggt_predictions": final_predictions,
        "dynamic_mask": refined_mask.cpu().numpy() if isinstance(refined_mask, torch.Tensor) else refined_mask,
        "object_tracks": all_tracks,
    }
