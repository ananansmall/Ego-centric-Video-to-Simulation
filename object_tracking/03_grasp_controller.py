"""
运动耦合检测 + 抓取时序生成 (grasp_controller.py)
==================================================

V-Dreamer 核心思路:
  不依赖深度对比检测接触, 而是通过物体点与手部点的运动耦合关系,
  精确判断抓取/释放时机。

原理:
  - 抓取: 物体点开始跟随手部点运动 → 运动方向一致, 速度耦合
  - 释放: 物体点停止跟随手部点 → 运动方向分离, 速度解耦
  - 搬运: 物体点与手部点刚性绑定 → 相对距离恒定

相比 contact_detector.py 的深度对比方法:
  - 更鲁棒: 不受深度噪声影响
  - 更精确: 基于运动学而非几何距离
  - 更及时: 能检测到"即将抓取"的前兆运动

输入:
  - point_tracker 输出的物体 3D 轨迹
  - HaWoR 手部顶点/指尖 3D 轨迹
输出:
  - gripper_timeline: 夹爪开闭时序
  - grasp_poses: 抓取姿态 (6DoF)
  - interaction_segments: 交互段 (抓取→搬运→释放)
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def compute_velocity(positions, dt=1.0, smooth_window=5):
    """计算位置序列的速度 (带平滑)

    Args:
        positions: (T, 3) 位置序列
        dt: 帧间隔 (秒)
        smooth_window: Savitzky-Golay 平滑窗口 (奇数)

    Returns:
        velocity: (T, 3) 速度序列
        speed: (T,) 速率标量
    """
    velocity = np.zeros_like(positions)
    if len(positions) > 2:
        velocity[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        velocity[0] = (positions[1] - positions[0]) / dt
        velocity[-1] = (positions[-1] - positions[-2]) / dt

    if smooth_window > 1 and len(velocity) >= smooth_window:
        for i in range(3):
            velocity[:, i] = savgol_filter(velocity[:, i], smooth_window, 2)

    speed = np.linalg.norm(velocity, axis=-1)
    return velocity, speed


def compute_motion_coupling(obj_positions, hand_positions, window=7):
    """计算物体-手部运动耦合度

    耦合度定义: 在滑动窗口内, 物体速度方向与手部速度方向的一致性

    Args:
        obj_positions: (T, 3) 物体质心位置
        hand_positions: (T, 3) 手部参考点位置 (如指尖中心)
        window: 滑动窗口大小

    Returns:
        coupling: (T,) 耦合度 [0, 1], 1=完全耦合, 0=完全解耦
        distance: (T,) 物体-手部距离
    """
    T = len(obj_positions)
    obj_vel, obj_speed = compute_velocity(obj_positions)
    hand_vel, hand_speed = compute_velocity(hand_positions)

    coupling = np.zeros(T)

    for t in range(T):
        half_w = window // 2
        t_start = max(0, t - half_w)
        t_end = min(T, t + half_w + 1)

        obj_v = obj_vel[t_start:t_end]
        hand_v = hand_vel[t_start:t_end]

        obj_s = obj_speed[t_start:t_end]
        hand_s = hand_speed[t_start:t_end]

        moving = (obj_s > 0.005) & (hand_s > 0.005)
        if moving.sum() < 2:
            coupling[t] = 0.0
            continue

        obj_v_norm = obj_v[moving] / (obj_s[moving, None] + 1e-8)
        hand_v_norm = hand_v[moving] / (hand_s[moving, None] + 1e-8)

        cos_sim = np.sum(obj_v_norm * hand_v_norm, axis=-1)
        cos_sim = np.clip(cos_sim, -1, 1)
        coupling[t] = np.mean(cos_sim)

    coupling = np.clip(coupling, 0, 1)

    distance = np.linalg.norm(obj_positions - hand_positions, axis=-1)

    return coupling, distance


def compute_relative_distance_stability(obj_positions, hand_positions, window=7):
    """计算物体-手部相对距离的稳定性

    抓取时: 相对距离恒定 (方差小)
    释放后: 相对距离变化 (方差大)

    Args:
        obj_positions: (T, 3) 物体质心位置
        hand_positions: (T, 3) 手部参考点位置
        window: 滑动窗口大小

    Returns:
        stability: (T,) 稳定性 [0, 1], 1=距离恒定 (抓取中), 0=距离变化
    """
    T = len(obj_positions)
    rel_dist = np.linalg.norm(obj_positions - hand_positions, axis=-1)

    stability = np.zeros(T)
    for t in range(T):
        half_w = window // 2
        t_start = max(0, t - half_w)
        t_end = min(T, t + half_w + 1)
        segment = rel_dist[t_start:t_end]
        if len(segment) < 2:
            stability[t] = 0.0
            continue
        mean_d = np.mean(segment)
        std_d = np.std(segment)
        stability[t] = 1.0 / (1.0 + std_d / (mean_d + 1e-6))

    return stability


def detect_grasp_release(
    coupling,
    stability,
    distance,
    obj_speed,
    hand_speed,
    coupling_threshold=0.5,
    stability_threshold=0.7,
    distance_threshold=0.15,
    min_grasp_frames=5,
    min_release_gap=10,
):
    """基于运动耦合和距离稳定性检测抓取/释放事件

    抓取条件 (全部满足):
      1. 运动耦合度 > coupling_threshold
      2. 距离稳定性 > stability_threshold
      3. 物体-手部距离 < distance_threshold
      4. 手部在运动 (hand_speed > 阈值)

    释放条件:
      1. 运动耦合度 < coupling_threshold
      2. 或距离稳定性 < stability_threshold
      3. 或物体-手部距离 > distance_threshold

    Args:
        coupling: (T,) 运动耦合度
        stability: (T,) 距离稳定性
        distance: (T,) 物体-手部距离
        obj_speed: (T,) 物体速率
        hand_speed: (T,) 手部速率
        coupling_threshold: 耦合度阈值
        stability_threshold: 稳定性阈值
        distance_threshold: 距离阈值 (米)
        min_grasp_frames: 最短抓取持续帧数
        min_release_gap: 最短释放间隔帧数

    Returns:
        segments: List[{grasp_frame, release_frame, type}] 交互段
    """
    T = len(coupling)

    grasp_score = np.zeros(T)
    for t in range(T):
        if hand_speed[t] < 0.005:
            continue
        score = 0.0
        if coupling[t] > coupling_threshold:
            score += 0.4
        if stability[t] > stability_threshold:
            score += 0.3
        if distance[t] < distance_threshold:
            score += 0.3
        grasp_score[t] = score

    is_grasping = grasp_score >= 0.7

    segments = []
    in_grasp = False
    grasp_start = 0

    for t in range(T):
        if is_grasping[t] and not in_grasp:
            grasp_start = t
            in_grasp = True
        elif not is_grasping[t] and in_grasp:
            duration = t - grasp_start
            if duration >= min_grasp_frames:
                segments.append({
                    "grasp_frame": grasp_start,
                    "release_frame": t - 1,
                    "type": "grasp_and_release",
                })
            in_grasp = False

    if in_grasp:
        duration = T - grasp_start
        if duration >= min_grasp_frames:
            segments.append({
                "grasp_frame": grasp_start,
                "release_frame": T - 1,
                "type": "held",
            })

    if len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg["grasp_frame"] - merged[-1]["release_frame"] <= min_release_gap:
                merged[-1]["release_frame"] = seg["release_frame"]
                merged[-1]["type"] = "grasp_and_release"
            else:
                merged.append(seg)
        segments = merged

    if len(segments) > 1:
        for seg in segments:
            seg["type"] = "repeated_manipulation"

    return segments


def compute_grasp_pose(hand_vertices, fingertips_3d, grasp_frame, hand_label="right"):
    """从手部顶点和指尖位置计算抓取姿态

    抓取姿态定义:
      - 位置: 接触指尖的中心
      - 朝向: 手掌法线方向 (拇指→小指方向为 x, 手掌→指尖方向为 z)

    Args:
        hand_vertices: (T, 778, 3) 手部顶点
        fingertips_3d: (T, 5, 3) 指尖3D位置
        grasp_frame: 抓取帧索引
        hand_label: "left" 或 "right"

    Returns:
        grasp_pose: (4, 4) 抓取位姿矩阵
    """
    ft = fingertips_3d[grasp_frame]
    palm_center = np.mean(hand_vertices[grasp_frame], axis=0)
    ft_center = np.mean(ft, axis=0)

    approach = ft_center - palm_center
    approach_norm = np.linalg.norm(approach)
    if approach_norm > 1e-6:
        approach = approach / approach_norm
    else:
        approach = np.array([0, 0, 1])

    thumb_to_pinky = ft[0] - ft[4]
    tp_norm = np.linalg.norm(thumb_to_pinky)
    if tp_norm > 1e-6:
        x_axis = thumb_to_pinky / tp_norm
    else:
        x_axis = np.array([1, 0, 0])

    z_axis = approach
    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm > 1e-6:
        y_axis = y_axis / y_norm
    x_axis = np.cross(y_axis, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm > 1e-6:
        x_axis = x_axis / x_norm

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 2] = -R[:, 2]

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = ft_center
    return pose


def generate_gripper_timeline(segments, total_frames, pre_grasp_frames=10):
    """从交互段生成夹爪开闭时序

    Args:
        segments: detect_grasp_release() 输出的交互段
        total_frames: 总帧数
        pre_grasp_frames: 抓取前提前闭合的帧数

    Returns:
        timeline: List[{frame, action, hand}] 夹爪事件列表
            action: "open" 或 "close"
            hand: "left" 或 "right"
    """
    timeline = []

    for seg in segments:
        close_frame = max(0, seg["grasp_frame"] - pre_grasp_frames)
        timeline.append({
            "frame": close_frame,
            "action": "close",
            "hand": "right",
        })
        timeline.append({
            "frame": seg["release_frame"] + 1,
            "action": "open",
            "hand": "right",
        })

    timeline.sort(key=lambda x: x["frame"])
    return timeline


def gripper_timeline_to_signal(timeline, total_frames, hand="right"):
    """将夹爪时序转换为逐帧信号

    Args:
        timeline: generate_gripper_timeline() 输出
        total_frames: 总帧数
        hand: "left" 或 "right"

    Returns:
        signal: (T,) 夹爪信号, 0.0=闭合, 0.05=张开
    """
    GRIPPER_OPEN = 0.05
    GRIPPER_CLOSED = 0.0

    signal = np.full(total_frames, GRIPPER_OPEN)

    for event in timeline:
        if event["hand"] != hand:
            continue
        f = event["frame"]
        if f < 0 or f >= total_frames:
            continue
        if event["action"] == "close":
            signal[f:] = GRIPPER_CLOSED
        elif event["action"] == "open":
            signal[f:] = GRIPPER_OPEN

    return signal


def run_grasp_controller(
    object_tracks_3d,
    object_valid,
    hand_vertices_left=None,
    hand_vertices_right=None,
    fingertips_3d_left=None,
    fingertips_3d_right=None,
    total_frames=None,
    distance_threshold=0.15,
    coupling_threshold=0.5,
):
    """运行完整的运动耦合检测 + 抓取时序生成管线

    Args:
        object_tracks_3d: (S, N, 3) 物体点3D轨迹
        object_valid: (S, N) bool 有效标记
        hand_vertices_left: (T, 778, 3) 左手顶点 (可选)
        hand_vertices_right: (T, 778, 3) 右手顶点 (可选)
        fingertips_3d_left: (T, 5, 3) 左手指尖 (可选)
        fingertips_3d_right: (T, 5, 3) 右手指尖 (可选)
        total_frames: 总帧数 (默认从轨迹推断)
        distance_threshold: 抓取距离阈值 (米)
        coupling_threshold: 运动耦合度阈值

    Returns:
        dict: {
            obj_centroid: (S, 3) 物体质心轨迹,
            obj_velocity: (S, 3) 物体速度,
            obj_speed: (S,) 物体速率,
            coupling: {hand_label: (S,)} 运动耦合度,
            stability: {hand_label: (S,)} 距离稳定性,
            distance: {hand_label: (S,)} 物体-手部距离,
            segments: {hand_label: List[segment]} 交互段,
            gripper_timeline: List[event] 夹爪时序,
            gripper_signal_left: (T,) 左夹爪信号,
            gripper_signal_right: (T,) 右夹爪信号,
            grasp_poses: {hand_label: (4,4)} 抓取姿态,
        }
    """
    S, N, _ = object_tracks_3d.shape
    if total_frames is None:
        total_frames = S

    valid_mask = object_valid
    obj_centroid = np.full((S, 3), np.nan)
    for t in range(S):
        v = valid_mask[t]
        if v.any():
            obj_centroid[t] = np.nanmedian(object_tracks_3d[t, v], axis=0)

    for t in range(S):
        if np.isnan(obj_centroid[t]).any():
            if t > 0 and not np.isnan(obj_centroid[t - 1]).any():
                obj_centroid[t] = obj_centroid[t - 1]

    obj_velocity, obj_speed = compute_velocity(obj_centroid)

    hand_data = {}
    if hand_vertices_right is not None:
        hand_data["right"] = {
            "vertices": hand_vertices_right,
            "fingertips": fingertips_3d_right,
        }
    if hand_vertices_left is not None:
        hand_data["left"] = {
            "vertices": hand_vertices_left,
            "fingertips": fingertips_3d_left,
        }

    coupling_dict = {}
    stability_dict = {}
    distance_dict = {}
    segments_dict = {}
    grasp_poses = {}

    for hand_label, hdata in hand_data.items():
        hand_verts = hdata["vertices"]
        T_hand = len(hand_verts)

        T_align = min(S, T_hand)
        hand_centroid = np.full((S, 3), np.nan)
        for t in range(T_align):
            hand_centroid[t] = np.mean(hand_verts[t], axis=0)
        for t in range(T_align, S):
            if T_align > 0:
                hand_centroid[t] = hand_centroid[T_align - 1]

        _, hand_speed = compute_velocity(hand_centroid)

        coupling, distance = compute_motion_coupling(obj_centroid, hand_centroid)
        stability = compute_relative_distance_stability(obj_centroid, hand_centroid)

        segments = detect_grasp_release(
            coupling,
            stability,
            distance,
            obj_speed,
            hand_speed,
            coupling_threshold=coupling_threshold,
            distance_threshold=distance_threshold,
        )

        coupling_dict[hand_label] = coupling
        stability_dict[hand_label] = stability
        distance_dict[hand_label] = distance
        segments_dict[hand_label] = segments

        if segments and hdata["fingertips"] is not None:
            grasp_poses[hand_label] = compute_grasp_pose(
                hand_verts, hdata["fingertips"], segments[0]["grasp_frame"], hand_label
            )

    active_hand = "right"
    if not segments_dict.get("right") and segments_dict.get("left"):
        active_hand = "left"

    gripper_timeline = generate_gripper_timeline(
        segments_dict.get(active_hand, []), total_frames
    )

    gripper_signal_right = gripper_timeline_to_signal(
        gripper_timeline, total_frames, hand="right"
    )
    gripper_signal_left = gripper_timeline_to_signal(
        gripper_timeline, total_frames, hand="left"
    )

    n_segments = sum(len(s) for s in segments_dict.values())
    print(f"[grasp_controller] Detected {n_segments} interaction segments")
    for hand_label, segs in segments_dict.items():
        for i, seg in enumerate(segs):
            print(f"  {hand_label} seg{i}: grasp@{seg['grasp_frame']}, "
                  f"release@{seg['release_frame']}, type={seg['type']}")

    return {
        "obj_centroid": obj_centroid,
        "obj_velocity": obj_velocity,
        "obj_speed": obj_speed,
        "coupling": coupling_dict,
        "stability": stability_dict,
        "distance": distance_dict,
        "segments": segments_dict,
        "gripper_timeline": gripper_timeline,
        "gripper_signal_left": gripper_signal_left,
        "gripper_signal_right": gripper_signal_right,
        "grasp_poses": grasp_poses,
    }
