"""
轨迹精化模块 (trajectory_refiner.py)
=====================================

从 point_tracker 的联合追踪 3D 点轨迹, 通过 RANSAC + Procrustes
计算精确的物体 6DoF 轨迹, 并进行时序平滑。

相比 object_tracker.py 的逐帧 Procrustes:
  - RANSAC 剔除离群追踪点 → 更鲁棒
  - 置信度加权 → 低置信度点影响小
  - 时序平滑 → 消除帧间抖动
  - 释放后静止约束 → 物理一致性

输入:
  - point_tracker 输出的 tracks_3d (S, N, 3) + visibility + confidence
  - grasp_controller 输出的交互段 (抓取/释放帧)
输出:
  - object_6dof: (S, 4, 4) 精确 6DoF 轨迹
  - object_centroid: (S, 3) 质心轨迹
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def procrustes_align(source, target, weights=None):
    """加权 Procrustes 刚体对齐

    求解: target ≈ R @ source + t (最小化加权残差)

    Args:
        source: (N, 3) 源点云
        target: (N, 3) 目标点云
        weights: (N,) 点权重 (可选)

    Returns:
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
        scale: float 缩放因子 (固定为1, 刚体)
        residual: float 对齐残差
    """
    if weights is not None:
        w = weights / (weights.sum() + 1e-8)
        src_centered = source - np.sum(source * w[:, None], axis=0)
        tgt_centered = target - np.sum(target * w[:, None], axis=0)
        H = (tgt_centered * w[:, None]).T @ src_centered
    else:
        src_center = source.mean(axis=0)
        tgt_center = target.mean(axis=0)
        src_centered = source - src_center
        tgt_centered = target - tgt_center
        H = tgt_centered.T @ src_centered

    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt

    if weights is not None:
        t = np.sum(target * w[:, None], axis=0) - R @ np.sum(source * w[:, None], axis=0)
    else:
        t = tgt_center - R @ src_center

    aligned = R @ source.T + t[:, None]
    residual = np.linalg.norm(aligned.T - target, axis=1).mean()

    return R, t, 1.0, residual


def ransac_procrustes(source, target, confidence=None, n_iter=100, inlier_threshold=0.02, min_samples=5):
    """RANSAC + Procrustes 鲁棒对齐

    Args:
        source: (N, 3) 源点云
        target: (N, 3) 目标点云
        confidence: (N,) 点置信度 (用于采样权重)
        n_iter: RANSAC 迭代次数
        inlier_threshold: 内点阈值 (米)
        min_samples: 最小采样数

    Returns:
        R: (3, 3) 旋转矩阵
        t: (3,) 平移向量
        inlier_mask: (N,) bool 内点标记
        residual: float 内点残差
    """
    N = len(source)
    if N < min_samples:
        R, t, _, residual = procrustes_align(source, target)
        return R, t, np.ones(N, dtype=bool), residual

    best_R = np.eye(3)
    best_t = np.zeros(3)
    best_inliers = np.zeros(N, dtype=bool)
    best_residual = float("inf")

    sample_weights = confidence if confidence is not None else None

    for _ in range(n_iter):
        if sample_weights is not None and sample_weights.sum() > 0:
            probs = sample_weights / sample_weights.sum()
            idx = np.random.choice(N, min_samples, replace=False, p=probs)
        else:
            idx = np.random.choice(N, min_samples, replace=False)

        try:
            R_cand, t_cand, _, _ = procrustes_align(source[idx], target[idx])
        except np.linalg.LinAlgError:
            continue

        aligned = R_cand @ source.T + t_cand[:, None]
        errors = np.linalg.norm(aligned.T - target, axis=1)
        inlier_mask = errors < inlier_threshold

        if inlier_mask.sum() < min_samples:
            continue

        R_refined, t_refined, _, res = procrustes_align(
            source[inlier_mask], target[inlier_mask]
        )

        if inlier_mask.sum() > best_inliers.sum() or (
            inlier_mask.sum() == best_inliers.sum() and res < best_residual
        ):
            best_R = R_refined
            best_t = t_refined
            best_inliers = inlier_mask
            best_residual = res

    return best_R, best_t, best_inliers, best_residual


def estimate_6dof_from_tracked_points(
    tracks_3d, valid_mask, confidence=None, reference_frame=0, ransac=True
):
    """从追踪的 3D 点轨迹估计逐帧 6DoF 位姿

    以参考帧为基准, 对齐后续帧的点云, 得到相对变换。

    Args:
        tracks_3d: (S, N, 3) 3D 点轨迹
        valid_mask: (S, N) bool 有效标记
        confidence: (S, N) 置信度 (可选)
        reference_frame: 参考帧索引
        ransac: 是否使用 RANSAC

    Returns:
        poses: List[dict] 逐帧位姿, 每项 {R, t, valid, residual, inlier_ratio}
    """
    S, N, _ = tracks_3d.shape

    ref_valid = valid_mask[reference_frame]
    if ref_valid.sum() < 3:
        return [{"R": np.eye(3), "t": np.zeros(3), "valid": False, "residual": 0, "inlier_ratio": 0}] * S

    ref_points = tracks_3d[reference_frame, ref_valid]
    ref_conf = confidence[reference_frame, ref_valid] if confidence is not None else None

    poses = []
    for t in range(S):
        cur_valid = valid_mask[t] & ref_valid
        if cur_valid.sum() < 3:
            poses.append({
                "R": np.eye(3), "t": np.zeros(3),
                "valid": False, "residual": 0, "inlier_ratio": 0,
            })
            continue

        src = ref_points[cur_valid[ref_valid]]
        tgt = tracks_3d[t, cur_valid]
        conf = confidence[t, cur_valid] if confidence is not None else None

        if ransac and cur_valid.sum() >= 6:
            R, t_vec, inlier_mask, residual = ransac_procrustes(
                src, tgt, conf, n_iter=50, inlier_threshold=0.03
            )
            inlier_ratio = inlier_mask.mean()
        else:
            R, t_vec, _, residual = procrustes_align(src, tgt, conf)
            inlier_ratio = 1.0

        poses.append({
            "R": R, "t": t_vec,
            "valid": True, "residual": residual, "inlier_ratio": inlier_ratio,
        })

    return poses


def smooth_trajectory(poses, translation_window=11, rotation_window=11, poly_order=2):
    """时序平滑 6DoF 轨迹

    平移: Savitzky-Golay 滤波
    旋转: SLERP 插值 + 球面平均

    Args:
        poses: estimate_6dof_from_tracked_points() 输出
        translation_window: 平移平滑窗口
        rotation_window: 旋转平滑窗口
        poly_order: 多项式阶数

    Returns:
        smoothed_poses: List[dict] 平滑后的位姿
    """
    S = len(poses)
    valid_indices = [i for i, p in enumerate(poses) if p["valid"]]

    if len(valid_indices) < 3:
        return poses

    translations = np.array([poses[i]["t"] for i in valid_indices])
    rotations = Rotation.from_matrix([poses[i]["R"] for i in valid_indices])

    if len(valid_indices) >= translation_window and translation_window % 2 == 1:
        for axis in range(3):
            translations[:, axis] = savgol_filter(
                translations[:, axis], translation_window, poly_order
            )

    smoothed = [p.copy() for p in poses]

    for idx, vi in enumerate(valid_indices):
        smoothed[vi]["t"] = translations[idx]
        smoothed[vi]["R"] = rotations[idx].as_matrix()

    for i in range(S):
        if not smoothed[i]["valid"]:
            prev_valid = None
            next_valid = None
            for j in range(i - 1, -1, -1):
                if smoothed[j]["valid"]:
                    prev_valid = j
                    break
            for j in range(i + 1, S):
                if smoothed[j]["valid"]:
                    next_valid = j
                    break

            if prev_valid is not None and next_valid is not None:
                alpha = (i - prev_valid) / (next_valid - prev_valid)
                smoothed[i]["t"] = (1 - alpha) * smoothed[prev_valid]["t"] + alpha * smoothed[next_valid]["t"]

                R_prev = Rotation.from_matrix(smoothed[prev_valid]["R"])
                R_next = Rotation.from_matrix(smoothed[next_valid]["R"])
                slerp = Slerp([0, 1], Rotation.concatenate([R_prev, R_next]))
                smoothed[i]["R"] = slerp(alpha).as_matrix()
                smoothed[i]["valid"] = True
            elif prev_valid is not None:
                smoothed[i]["t"] = smoothed[prev_valid]["t"]
                smoothed[i]["R"] = smoothed[prev_valid]["R"]
                smoothed[i]["valid"] = True

    return smoothed


def apply_release_constraint(poses, release_frame, total_frames):
    """释放后物体保持静止

    Args:
        poses: 平滑后的位姿列表
        release_frame: 释放帧索引
        total_frames: 总帧数

    Returns:
        poses: 应用静止约束后的位姿列表
    """
    if release_frame is None or release_frame >= total_frames:
        return poses

    release_pose = poses[release_frame]
    if not release_pose["valid"]:
        for t in range(release_frame, -1, -1):
            if poses[t]["valid"]:
                release_pose = poses[t]
                break

    for t in range(release_frame + 1, total_frames):
        poses[t]["R"] = release_pose["R"]
        poses[t]["t"] = release_pose["t"]
        poses[t]["valid"] = True

    return poses


def poses_to_trajectory_array(poses, total_frames):
    """将位姿列表转换为 (S, 4, 4) 齐次矩阵数组

    Args:
        poses: 位姿列表
        total_frames: 总帧数

    Returns:
        trajectory: (S, 4, 4) 齐次变换矩阵
    """
    trajectory = np.eye(4)[None].repeat(total_frames, axis=0)
    for i, p in enumerate(poses):
        if i >= total_frames:
            break
        if p["valid"]:
            trajectory[i, :3, :3] = p["R"]
            trajectory[i, :3, 3] = p["t"]
    return trajectory


def compute_centroid_from_tracks(tracks_3d, valid_mask):
    """从追踪点计算质心轨迹

    Args:
        tracks_3d: (S, N, 3) 3D 点轨迹
        valid_mask: (S, N) bool 有效标记

    Returns:
        centroids: (S, 3) 质心轨迹
    """
    S = tracks_3d.shape[0]
    centroids = np.full((S, 3), np.nan)
    for t in range(S):
        v = valid_mask[t]
        if v.any():
            centroids[t] = np.nanmedian(tracks_3d[t, v], axis=0)

    for t in range(S):
        if np.isnan(centroids[t]).any() and t > 0 and not np.isnan(centroids[t - 1]).any():
            centroids[t] = centroids[t - 1]

    return centroids


def run_trajectory_refiner(
    object_tracks_3d,
    object_valid,
    object_confidence=None,
    object_visibility=None,
    interaction_segments=None,
    total_frames=None,
    reference_frame=0,
    smooth=True,
    translation_window=11,
    rotation_window=11,
):
    """完整轨迹精化管线

    Args:
        object_tracks_3d: (S, N, 3) 3D 点轨迹 (from point_tracker)
        object_valid: (S, N) bool 有效标记
        object_confidence: (S, N) 置信度 (可选)
        object_visibility: (S, N) 可见性 (可选, 用于过滤)
        interaction_segments: grasp_controller 输出的交互段
        total_frames: 总帧数
        reference_frame: 参考帧
        smooth: 是否时序平滑
        translation_window: 平移平滑窗口
        rotation_window: 旋转平滑窗口

    Returns:
        dict: {
            trajectory: (S, 4, 4) 6DoF 轨迹,
            centroids: (S, 3) 质心轨迹,
            poses: List[dict] 逐帧位姿详情,
            valid_frames: (S,) bool 有效帧标记,
            release_frame: int or None 释放帧,
        }
    """
    S, N, _ = object_tracks_3d.shape
    if total_frames is None:
        total_frames = S

    if object_visibility is not None:
        vis_threshold = 0.5
        combined_valid = object_valid & (object_visibility > vis_threshold)
    else:
        combined_valid = object_valid

    if object_confidence is not None:
        high_conf = object_confidence > 0.3
        combined_valid = combined_valid & high_conf

    n_valid_per_frame = combined_valid.sum(axis=1)
    if n_valid_per_frame.max() < 3:
        print("[trajectory_refiner] WARNING: Not enough valid points for 6DoF estimation")
        return {
            "trajectory": np.eye(4)[None].repeat(total_frames, axis=0),
            "centroids": compute_centroid_from_tracks(object_tracks_3d, object_valid),
            "poses": [],
            "valid_frames": np.zeros(total_frames, dtype=bool),
            "release_frame": None,
        }

    valid_frame_mask = n_valid_per_frame >= 3
    first_valid = np.where(valid_frame_mask)[0]
    if len(first_valid) > 0 and reference_frame < first_valid[0]:
        reference_frame = first_valid[0]
        print(f"[trajectory_refiner] Adjusted reference frame to {reference_frame}")

    poses = estimate_6dof_from_tracked_points(
        object_tracks_3d, combined_valid, object_confidence,
        reference_frame=reference_frame, ransac=True,
    )

    n_valid_poses = sum(1 for p in poses if p["valid"])
    print(f"[trajectory_refiner] Estimated 6DoF for {n_valid_poses}/{S} frames")

    if smooth and n_valid_poses >= 5:
        poses = smooth_trajectory(
            poses,
            translation_window=min(translation_window, n_valid_poses if n_valid_poses % 2 == 1 else n_valid_poses - 1),
            rotation_window=min(rotation_window, n_valid_poses if n_valid_poses % 2 == 1 else n_valid_poses - 1),
        )
        print("[trajectory_refiner] Applied temporal smoothing")

    release_frame = None
    if interaction_segments:
        all_segments = []
        for hand_label, segs in interaction_segments.items():
            all_segments.extend(segs)
        if all_segments:
            last_seg = max(all_segments, key=lambda s: s["release_frame"])
            release_frame = last_seg["release_frame"]
            poses = apply_release_constraint(poses, release_frame, total_frames)
            print(f"[trajectory_refiner] Applied release constraint at frame {release_frame}")

    trajectory = poses_to_trajectory_array(poses, total_frames)
    centroids = compute_centroid_from_tracks(object_tracks_3d, combined_valid)
    valid_frames = np.array([p["valid"] for p in poses])

    residuals = [p.get("residual", 0) for p in poses if p["valid"]]
    if residuals:
        print(f"[trajectory_refiner] Alignment residual: mean={np.mean(residuals):.4f}m, "
              f"max={np.max(residuals):.4f}m")

    return {
        "trajectory": trajectory,
        "centroids": centroids,
        "poses": poses,
        "valid_frames": valid_frames,
        "release_frame": release_frame,
    }
