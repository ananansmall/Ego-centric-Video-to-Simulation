from html import parser
import os
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import argparse
import random
import numpy as np
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import cv2
from PIL import Image
from src.geometry_utils import predictions_to_pcd

def vggt_predict(images, model):
    '''
    使用原版 VGGT 模型对帧序列进行推理, 输出深度图、相机位姿和点云。

    推理流程:
      1. model(images) → VGGT 内部:
         - Aggregator: 24层交替帧内/帧间注意力, 输出 aggregated_tokens_list
         - CameraHead: 4次迭代精修 (DiT风格 AdaLN), 输出 9D pose_enc [T(3), quat(4), FoV(2)]
         - DepthHead (DPTHead, output_dim=2, activation=exp):
           depth = exp(depth_logits),  depth_conf = 1 + exp(conf_logits)
         - PointHead (DPTHead, output_dim=4, activation=inv_log):
           world_points = sign(x) * (exp(|x|) - 1),  world_points_conf = 1 + exp(conf_logits)
      2. pose_encoding_to_extri_intri(): 9D编码 → extrinsic (S,3,4) + intrinsic (S,3,3)
         - T 直接作为平移, quat → R (四元数转旋转矩阵), FoV → fx/fy (假设主点在中心)
      3. unproject_depth_map_to_point_map(): depth + extrinsic + intrinsic → world_points_from_depth
         - 逐帧: 像素→相机坐标 (针孔模型) → 世界坐标 (SE3逆变换)
      4. world_points_conf 来自 PointHead 直接预测 (3D点位置置信度, expp1激活)

    关键: VGGT 同时有 PointHead 和 DepthHead, 但最终返回的 world_points 来自深度反投影,
         world_points_conf 来自 PointHead 直接预测。PointHead 的 world_points 未被使用。

    Args:
        images: torch.Tensor (S, 3, 518, 518), 来自 load_video_frames(), 值域 [0,1]
        model:  VGGT 模型实例 (eval 模式, 在 CUDA 上)

    Returns:
        dict:
          - point_cloud_data:   trimesh.PointCloud, 置信度百分位50过滤后的合并点云
          - colors:             numpy (S, H, W, 3) uint8, RGB帧
          - depths:             numpy (S, H, W), 深度图 (exp激活, 始终正值)
          - extrinsics:         numpy (S, 4, 4), camera-from-world 外参
          - world_points:       numpy (S, H, W, 3), 深度反投影得到的世界坐标
          - world_points_conf:  numpy (S, H, W), PointHead直接预测的3D点置信度 (≥1.05)
          - intrinsic:          numpy (3, 3), 所有帧内参取平均
    '''
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16 
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions['images'] = images.cpu().numpy()

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    point_cloud_data = predictions_to_pcd(
            predictions,
            conf_thres=50.0,  # 默认值
            filter_by_frames="All",
            mask_black_bg=False,
            mask_white_bg=False,
            prediction_mode="Depthmap and Camera Branch",
        )
    
    colors = (predictions['images'].transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    depths = predictions['depth'].squeeze(-1)
    extrinsics = np.pad(predictions['extrinsic'], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    extrinsics[:, 3, 3] = 1
    world_points = predictions['world_points_from_depth'].copy()
    world_points_conf = predictions['world_points_conf'].copy()
    intrinsic = np.mean(predictions['intrinsic'], axis=0)

    return {
        "point_cloud_data": point_cloud_data,
        "colors": colors,
        "depths": depths,
        "extrinsics": extrinsics,
        "world_points": world_points,
        "world_points_conf": world_points_conf,
        "intrinsic": intrinsic,
    }