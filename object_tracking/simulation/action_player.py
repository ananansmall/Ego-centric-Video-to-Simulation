"""
物理仿真动作回放模块

管线位置:
  scene_builder.py → 本文件 → run_simulation.py

输入:
  - robot: R1LiteRobot (已加载到场景中)
  - ee_trajectory: 机械臂末端执行器轨迹 (S,T,7) [位置3+四元数4]
  - gripper_timeline: 夹爪开闭时序 (来自 action_semantics)
  - object_actors: 物体 actor 字典 (用于验证)

输出:
  - 仿真结果: 每帧的物体位姿 + EE位姿
  - 验证报告: 仿真轨迹 vs VGGT追踪轨迹的偏差

核心逻辑:
  机械臂按手部轨迹运动 (主动方)
  物体由物理引擎计算被动响应 (被动方)
  夹爪按接触检测时序开闭
"""

import numpy as np
import sapien
from scipy.spatial.transform import Rotation

GRIPPER_OPEN = 0.05
GRIPPER_CLOSED = 0.0
ACTION_DIM = 16


def build_action(left_ee_pose, right_ee_pose, left_gripper, right_gripper):
    """构建 16 维 action 向量

    Args:
        left_ee_pose: (7,) [位置3, 四元数4] SAPIEN坐标系
        right_ee_pose: (7,) [位置3, 四元数4] SAPIEN坐标系
        left_gripper: float, 0.0=闭合, 0.05=张开
        right_gripper: float

    Returns:
        (16,) action 向量
    """
    action = np.zeros(ACTION_DIM)
    action[0:3] = left_ee_pose[:3]
    action[3:7] = left_ee_pose[3:7]
    action[7] = left_gripper
    action[8:11] = right_ee_pose[:3]
    action[11:15] = right_ee_pose[3:7]
    action[15] = right_gripper
    return action


def mano_trajectory_to_ee_trajectory(
    pred_trans, pred_rot, T_room_to_sapien=None
):
    """HaWoR 手部轨迹 → 机械臂 EE 轨迹

    Args:
        pred_trans: (2, T, 3) 左右手腕位置 (房间坐标系)
        pred_rot: (2, T, 3) 左右手腕旋转 (axis-angle, 房间坐标系)
        T_room_to_sapien: (4,4) 坐标系变换, 默认 z-up→y-up

    Returns:
        ee_trajectory: (2, T, 7) [位置3, 四元数4] SAPIEN坐标系
    """
    if T_room_to_sapien is None:
        T_room_to_sapien = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    num_hands, T, _ = pred_trans.shape
    ee_trajectory = np.zeros((num_hands, T, 7))

    for h in range(num_hands):
        for t in range(T):
            p_room = pred_trans[h, t]
            R_room = Rotation.from_rotvec(pred_rot[h, t]).as_matrix()

            T_homo = np.eye(4)
            T_homo[:3, :3] = R_room
            T_homo[:3, 3] = p_room

            T_sapien = T_room_to_sapien @ T_homo @ T_room_to_sapien.T
            p_sapien = T_sapien[:3, 3]
            R_sapien = T_sapien[:3, :3]
            q_sapien = Rotation.from_matrix(R_sapien).as_quat()

            ee_trajectory[h, t, :3] = p_sapien
            ee_trajectory[h, t, 3:7] = q_sapien

    return ee_trajectory


def gripper_timeline_to_signal(gripper_timeline, total_frames):
    """夹爪时序 → 逐帧夹爪信号

    Args:
        gripper_timeline: list of {"frame": int, "state": "open"|"closed", "object": str}
        total_frames: 总帧数

    Returns:
        left_gripper: (T,) float, 0.0=闭合, 0.05=张开
        right_gripper: (T,) float
    """
    left_gripper = np.full(total_frames, GRIPPER_OPEN)
    right_gripper = np.full(total_frames, GRIPPER_OPEN)

    for event in gripper_timeline:
        frame = event["frame"]
        state = event["state"]
        value = GRIPPER_CLOSED if state == "closed" else GRIPPER_OPEN

        if frame < total_frames:
            left_gripper[frame:] = value
            right_gripper[frame:] = value

    return left_gripper, right_gripper


def run_simulation(
    env,
    robot,
    ee_trajectory,
    left_gripper_signal,
    right_gripper_signal,
    object_actors=None,
    tracked_object_trajectories=None,
    max_steps=None,
):
    """执行物理仿真

    Args:
        env: GalaxeaManipSim 环境
        robot: R1LiteRobot
        ee_trajectory: (2, T, 7) EE轨迹 [位置3, 四元数4]
        left_gripper_signal: (T,) 左夹爪信号
        right_gripper_signal: (T,) 右夹爪信号
        object_actors: {name: sapien.Entity} 物体actor
        tracked_object_trajectories: {name: (T,4,4)} VGGT追踪的物体轨迹 (验证用)
        max_steps: 最大仿真步数, None=全部

    Returns:
        dict: {
            "sim_object_trajectories": {name: (T,4,4)},
            "sim_ee_trajectories": (2, T, 7),
            "verification": {name: {"mean_error": float, "max_error": float}},
        }
    """
    T = ee_trajectory.shape[1]
    if max_steps is not None:
        T = min(T, max_steps)

    sim_object_poses = {name: [] for name in (object_actors or {})}
    sim_ee_poses = [[], []]

    for t in range(T):
        left_ee = ee_trajectory[0, t]
        right_ee = ee_trajectory[1, t]
        left_grip = left_gripper_signal[t]
        right_grip = right_gripper_signal[t]

        action = build_action(left_ee, right_ee, left_grip, right_grip)

        obs, reward, terminated, truncated, info = env.step(action)

        for name, actor in (object_actors or {}).items():
            pose = actor.get_pose()
            sim_object_poses[name].append({
                "p": pose.p.copy(),
                "q": pose.q.copy(),
            })

        for h in range(2):
            ee_link = robot.left_ee_link if h == 0 else robot.right_ee_link
            ee_pose = ee_link.get_entity_pose()
            sim_ee_poses[h].append({
                "p": ee_pose.p.copy(),
                "q": ee_pose.q.copy(),
            })

    sim_object_trajectories = {}
    for name, poses in sim_object_poses.items():
        traj = np.zeros((len(poses), 4, 4))
        for i, p in enumerate(poses):
            traj[i, :3, :3] = Rotation.from_quat(p["q"]).as_matrix()
            traj[i, :3, 3] = p["p"]
            traj[i, 3, 3] = 1.0
        sim_object_trajectories[name] = traj

    sim_ee_trajectories = np.zeros((2, T, 7))
    for h in range(2):
        for i, p in enumerate(sim_ee_poses[h]):
            sim_ee_trajectories[h, i, :3] = p["p"]
            sim_ee_trajectories[h, i, 3:7] = p["q"]

    verification = {}
    if tracked_object_trajectories is not None:
        for name, tracked_traj in tracked_object_trajectories.items():
            if name in sim_object_trajectories:
                sim_traj = sim_object_trajectories[name]
                min_len = min(len(tracked_traj), len(sim_traj))
                pos_errors = np.linalg.norm(
                    sim_traj[:min_len, :3, 3] - tracked_traj[:min_len, :3, 3],
                    axis=1,
                )
                verification[name] = {
                    "mean_error": float(pos_errors.mean()),
                    "max_error": float(pos_errors.max()),
                    "min_error": float(pos_errors.min()),
                }

    return {
        "sim_object_trajectories": sim_object_trajectories,
        "sim_ee_trajectories": sim_ee_trajectories,
        "verification": verification,
    }
