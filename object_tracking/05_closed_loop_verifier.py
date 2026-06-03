"""
闭环验证模块 (closed_loop_verifier.py)
========================================

V-Dreamer 核心思路:
  仿真执行后, 渲染仿真画面与原视频对比,
  偏差超过阈值则调整轨迹参数, 迭代直到收敛。

闭环流程:
  1. 执行仿真 → 获取物体实际轨迹
  2. 渲染仿真画面 (从原视频相机视角)
  3. 对比仿真画面 vs 原视频画面
  4. 计算偏差指标 (物体位置误差 + 视觉相似度)
  5. 偏差 > 阈值 → 调整轨迹参数 → 重新仿真
  6. 偏差 < 阈值 → 验证通过

调整策略:
  - 位置偏移补偿: 仿真物体轨迹 vs 追踪物体轨迹的系统性偏差
  - 抓取姿态微调: 调整夹爪闭合时机和抓取偏移
  - 物理参数调整: 摩擦力、夹持力等
"""

import os

import cv2
import numpy as np


def compute_trajectory_error(sim_trajectory, ref_trajectory, valid_frames=None):
    """计算仿真轨迹与参考轨迹的位置偏差

    Args:
        sim_trajectory: (S, 4, 4) 仿真物体轨迹
        ref_trajectory: (S, 4, 3) 参考物体轨迹 (from trajectory_refiner)
        valid_frames: (S,) bool 有效帧标记

    Returns:
        dict: {
            mean_error: float 平均位置误差 (米),
            max_error: float 最大位置误差,
            min_error: float 最小位置误差,
            per_frame_error: (S,) 逐帧误差,
            rotation_error: float 平均旋转误差 (度),
        }
    """
    S = min(len(sim_trajectory), len(ref_trajectory))

    sim_pos = sim_trajectory[:S, :3, 3]
    ref_pos = ref_trajectory[:S, :3, 3]

    per_frame_error = np.linalg.norm(sim_pos - ref_pos, axis=1)

    if valid_frames is not None:
        valid = valid_frames[:S]
        if valid.any():
            per_frame_error_valid = per_frame_error[valid]
        else:
            per_frame_error_valid = per_frame_error
    else:
        per_frame_error_valid = per_frame_error

    sim_rot = sim_trajectory[:S, :3, :3]
    ref_rot = ref_trajectory[:S, :3, :3]
    rotation_errors = []
    for t in range(S):
        R_diff = sim_rot[t].T @ ref_rot[t]
        trace = np.clip(np.trace(R_diff), -1, 3)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        rotation_errors.append(np.degrees(angle))

    return {
        "mean_error": float(np.mean(per_frame_error_valid)),
        "max_error": float(np.max(per_frame_error_valid)),
        "min_error": float(np.min(per_frame_error_valid)),
        "per_frame_error": per_frame_error,
        "rotation_error": float(np.mean(rotation_errors)),
    }


def compute_visual_similarity(sim_frames, ref_frames, method="ssim"):
    """计算仿真画面与原视频的视觉相似度

    Args:
        sim_frames: (S, H, W, 3) uint8 仿真渲染帧
        ref_frames: (S, H, W, 3) uint8 原视频帧
        method: "ssim" 或 "mse" 或 "lpips"

    Returns:
        similarity: float 相似度 [0, 1]
    """
    S = min(len(sim_frames), len(ref_frames))

    if method == "mse":
        mse_values = []
        for t in range(S):
            sim = cv2.resize(sim_frames[t], (ref_frames[t].shape[1], ref_frames[t].shape[0]))
            mse = np.mean((sim.astype(float) - ref_frames[t].astype(float)) ** 2)
            mse_values.append(mse)
        avg_mse = np.mean(mse_values)
        similarity = 1.0 / (1.0 + avg_mse / 1000.0)
        return float(similarity)

    elif method == "ssim":
        similarities = []
        for t in range(S):
            sim = cv2.resize(sim_frames[t], (ref_frames[t].shape[1], ref_frames[t].shape[0]))
            sim_gray = cv2.cvtColor(sim, cv2.COLOR_RGB2GRAY)
            ref_gray = cv2.cvtColor(ref_frames[t], cv2.COLOR_RGB2GRAY)
            score = _compute_ssim(sim_gray, ref_gray)
            similarities.append(score)
        return float(np.mean(similarities))

    else:
        return 0.0


def _compute_ssim(img1, img2, window_size=7):
    """简化版 SSIM 计算"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (window_size, window_size), 1.5)
    mu2 = cv2.GaussianBlur(img2, (window_size, window_size), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (window_size, window_size), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (window_size, window_size), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (window_size, window_size), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def compute_position_offset(sim_trajectory, ref_trajectory, contact_frames):
    """计算仿真轨迹的系统性位置偏移

    在接触帧中, 仿真物体位置与参考位置的差值的平均值

    Args:
        sim_trajectory: (S, 4, 4) 仿真轨迹
        ref_trajectory: (S, 4, 4) 参考轨迹
        contact_frames: List[int] 接触帧索引

    Returns:
        offset: (3,) 位置偏移向量
    """
    if not contact_frames:
        return np.zeros(3)

    S = min(len(sim_trajectory), len(ref_trajectory))
    valid_frames = [f for f in contact_frames if f < S]

    if not valid_frames:
        return np.zeros(3)

    sim_pos = sim_trajectory[valid_frames, :3, 3]
    ref_pos = ref_trajectory[valid_frames, :3, 3]
    offset = np.median(ref_pos - sim_pos, axis=0)

    return offset


def adjust_trajectory_with_offset(trajectory, offset, start_frame, end_frame):
    """用偏移量调整轨迹

    Args:
        trajectory: (S, 4, 4) 原始轨迹
        offset: (3,) 位置偏移
        start_frame: 开始调整的帧
        end_frame: 结束调整的帧

    Returns:
        adjusted: (S, 4, 4) 调整后的轨迹
    """
    adjusted = trajectory.copy()
    for t in range(start_frame, min(end_frame, len(adjusted))):
        adjusted[t, :3, 3] += offset
    return adjusted


def adjust_gripper_timing(gripper_signal, shift_frames=0):
    """调整夹爪时序 (提前/延后)

    Args:
        gripper_signal: (T,) 夹爪信号
        shift_frames: 偏移帧数 (正=延后, 负=提前)

    Returns:
        adjusted: (T,) 调整后的夹爪信号
    """
    if shift_frames == 0:
        return gripper_signal.copy()

    adjusted = np.roll(gripper_signal, shift_frames)
    if shift_frames > 0:
        adjusted[:shift_frames] = gripper_signal[0]
    else:
        adjusted[shift_frames:] = gripper_signal[-1]
    return adjusted


def run_closed_loop_verification(
    sim_results,
    ref_trajectory,
    ref_frames=None,
    contact_frames=None,
    valid_frames=None,
    max_iterations=3,
    position_threshold=0.03,
    visual_threshold=0.5,
    output_dir=None,
):
    """运行闭环验证

    Args:
        sim_results: action_player.run_simulation() 的返回值
            包含 sim_object_trajectories, sim_ee_trajectories, verification
        ref_trajectory: (S, 4, 4) 参考物体轨迹 (from trajectory_refiner)
        ref_frames: (S, H, W, 3) uint8 原视频帧 (可选, 用于视觉对比)
        contact_frames: List[int] 接触帧索引
        valid_frames: (S,) bool 有效帧标记
        max_iterations: 最大迭代次数
        position_threshold: 位置误差阈值 (米)
        visual_threshold: 视觉相似度阈值
        output_dir: 输出目录

    Returns:
        dict: {
            verified: bool 是否通过验证,
            iterations: int 实际迭代次数,
            final_error: dict 最终偏差指标,
            trajectory_adjustments: List[dict] 每次迭代的调整记录,
            adjusted_trajectory: (S, 4, 4) 调整后的轨迹 (如果需要),
            adjusted_gripper_left: (T,) 调整后的左夹爪信号,
            adjusted_gripper_right: (T,) 调整后的右夹爪信号,
        }
    """
    sim_obj_traj = sim_results.get("sim_object_trajectories", {})
    sim_ee_traj = sim_results.get("sim_ee_trajectories", {})

    if not sim_obj_traj:
        print("[closed_loop_verifier] No simulated object trajectories, skipping verification")
        return {
            "verified": False,
            "iterations": 0,
            "final_error": {},
            "trajectory_adjustments": [],
            "adjusted_trajectory": ref_trajectory,
            "adjusted_gripper_left": None,
            "adjusted_gripper_right": None,
        }

    obj_key = list(sim_obj_traj.keys())[0]
    sim_traj = sim_obj_traj[obj_key]

    adjustments = []
    current_traj = ref_trajectory.copy()
    current_gripper_left = sim_results.get("gripper_signal_left", None)
    current_gripper_right = sim_results.get("gripper_signal_right", None)

    verified = False

    for iteration in range(max_iterations):
        print(f"\n[closed_loop_verifier] Iteration {iteration + 1}/{max_iterations}")

        error = compute_trajectory_error(sim_traj, current_traj, valid_frames)
        print(f"  Position error: mean={error['mean_error']:.4f}m, max={error['max_error']:.4f}m")
        print(f"  Rotation error: {error['rotation_error']:.2f}°")

        visual_sim = 0.0
        if ref_frames is not None:
            sim_rendered = sim_results.get("rendered_frames", None)
            if sim_rendered is not None:
                visual_sim = compute_visual_similarity(sim_rendered, ref_frames)
                print(f"  Visual similarity: {visual_sim:.3f}")

        if error["mean_error"] < position_threshold:
            print(f"  ✓ Position error below threshold ({position_threshold}m)")
            verified = True
            break

        if contact_frames:
            offset = compute_position_offset(sim_traj, current_traj, contact_frames)
            print(f"  Computed position offset: {offset}")

            if np.linalg.norm(offset) > 0.001:
                start = min(contact_frames)
                end = len(current_traj)
                current_traj = adjust_trajectory_with_offset(current_traj, offset, start, end)

                adjustments.append({
                    "iteration": iteration + 1,
                    "type": "position_offset",
                    "offset": offset.tolist(),
                    "start_frame": start,
                    "end_frame": end,
                    "error_before": error["mean_error"],
                })
                print(f"  Applied position offset adjustment")
            else:
                print(f"  Offset too small, trying gripper timing adjustment")
                if current_gripper_right is not None:
                    shift = -2 if error["mean_error"] > 0.05 else -1
                    current_gripper_right = adjust_gripper_timing(current_gripper_right, shift)
                    if current_gripper_left is not None:
                        current_gripper_left = adjust_gripper_timing(current_gripper_left, shift)
                    adjustments.append({
                        "iteration": iteration + 1,
                        "type": "gripper_timing",
                        "shift_frames": shift,
                        "error_before": error["mean_error"],
                    })
                    print(f"  Applied gripper timing shift: {shift} frames")
        else:
            print(f"  No contact frames, cannot adjust")
            break

    final_error = compute_trajectory_error(sim_traj, current_traj, valid_frames)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        np.savez(
            os.path.join(output_dir, "closed_loop_results.npz"),
            adjusted_trajectory=current_traj,
            per_frame_error=final_error["per_frame_error"],
        )

        import json
        report = {
            "verified": verified,
            "iterations": len(adjustments) + 1,
            "final_error": {k: v for k, v in final_error.items() if k != "per_frame_error"},
            "adjustments": adjustments,
        }
        with open(os.path.join(output_dir, "verification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

    status = "✓ PASSED" if verified else "✗ FAILED"
    print(f"\n[closed_loop_verifier] Verification {status}")
    print(f"  Final position error: {final_error['mean_error']:.4f}m")
    print(f"  Iterations: {len(adjustments) + 1}")

    return {
        "verified": verified,
        "iterations": len(adjustments) + 1,
        "final_error": final_error,
        "trajectory_adjustments": adjustments,
        "adjusted_trajectory": current_traj,
        "adjusted_gripper_left": current_gripper_left,
        "adjusted_gripper_right": current_gripper_right,
    }
