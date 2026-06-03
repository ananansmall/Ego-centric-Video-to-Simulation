"""
物理仿真入口脚本

完整管线:
  视频输入
    → VGGT+VGGT4D (3D感知)
    → HaWoR (手部重建)
    → ReplicateAnyScene (场景重建)
    → object_tracking (运动耦合检测 + 物体追踪)
    → 本脚本 (物理仿真)

输入:
  --hawor_npz       HaWoR 重建结果 (pred_trans, pred_rot)
  --action_json     夹爪时序 (来自 grasp_controller)
  --scene_glb       静态场景 GLB
  --objects_json    物体信息 (mesh路径 + 初始位姿)
  --tracked_dir     VGGT追踪物体轨迹目录 (验证用)
  --output          输出目录

输出:
  {output}/
    sim_results.npz       仿真结果 (物体轨迹 + EE轨迹)
    verification.json     验证报告
    sim_video.mp4         仿真视频 (可选)
"""

import argparse
import json
import os
import sys
import numpy as np

GALAXEA_SIM_DIR = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/GalaxeaManipSim"
PROJECT_DIR = "/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene"


def parse_args():
    parser = argparse.ArgumentParser(description="物理仿真: 视频动作复刻")
    parser.add_argument("--hawor_npz", type=str, required=True,
                        help="HaWoR 重建结果 NPZ (pred_trans, pred_rot)")
    parser.add_argument("--action_json", type=str, required=True,
                        help="动作语义化 JSON (夹爪时序)")
    parser.add_argument("--scene_glb", type=str, default=None,
                        help="静态场景 GLB 文件")
    parser.add_argument("--objects_json", type=str, default=None,
                        help="物体信息 JSON (mesh路径 + 初始位姿)")
    parser.add_argument("--tracked_dir", type=str, default=None,
                        help="VGGT追踪物体轨迹目录 (验证用)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="最大仿真步数")
    parser.add_argument("--robot_base_position", type=float, nargs=3,
                        default=[0, 0, 0],
                        help="机器人基座位置 (房间坐标系, 米)")
    return parser.parse_args()


def load_hawor_data(npz_path):
    """加载 HaWoR 手部轨迹"""
    data = dict(np.load(npz_path, allow_pickle=True))
    pred_trans = data["pred_trans"]  # (2, T, 3)
    pred_rot = data["pred_rot"]      # (2, T, 3) axis-angle
    return pred_trans, pred_rot


def load_action_sequence(json_path):
    """加载动作语义化结果"""
    with open(json_path) as f:
        return json.load(f)


def load_objects_json(json_path):
    """加载物体信息"""
    with open(json_path) as f:
        return json.load(f)


def load_tracked_trajectories(tracked_dir):
    """加载 VGGT 追踪的物体轨迹"""
    trajectories = {}
    if tracked_dir is None or not os.path.isdir(tracked_dir):
        return trajectories

    for f in os.listdir(tracked_dir):
        if f.endswith(".npz"):
            name = f.replace(".npz", "")
            data = dict(np.load(os.path.join(tracked_dir, f), allow_pickle=True))
            if "trajectory" in data:
                trajectories[name] = data["trajectory"]
    return trajectories


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    sys.path.insert(0, GALAXEA_SIM_DIR)
    sys.path.insert(0, PROJECT_DIR)

    from object_tracking.simulation.scene_builder import build_scene
    from object_tracking.simulation.action_player import (
        mano_trajectory_to_ee_trajectory,
        gripper_timeline_to_signal,
        run_simulation,
    )

    # 1. 加载手部轨迹
    print("[1/6] 加载手部轨迹...")
    pred_trans, pred_rot = load_hawor_data(args.hawor_npz)
    T = pred_trans.shape[1]
    print(f"  手部轨迹: {pred_trans.shape} ({T}帧)")

    # 2. 加载夹爪时序
    print("[2/6] 加载夹爪时序...")
    action_seq = load_action_sequence(args.action_json)
    gripper_timeline = action_seq.get("gripper_timeline", [])
    left_gripper, right_gripper = gripper_timeline_to_signal(gripper_timeline, T)
    print(f"  夹爪事件: {len(gripper_timeline)}个")

    # 3. 加载物体信息
    print("[3/6] 加载物体信息...")
    objects = []
    if args.objects_json and os.path.exists(args.objects_json):
        objects_data = load_objects_json(args.objects_json)
        objects = objects_data if isinstance(objects_data, list) else []
    print(f"  物体数量: {len(objects)}")

    # 4. 构建仿真场景
    print("[4/6] 构建仿真场景...")
    sim_result = build_scene(
        scene_glb_path=args.scene_glb,
        objects=objects,
        robot_type="r1_lite",
        robot_base_pose_room={"position": args.robot_base_position},
    )
    scene = sim_result["scene"]
    robot = sim_result["robot"]
    object_actors = sim_result["object_actors"]
    print(f"  场景构建完成: {len(object_actors)}个动态物体")

    # 5. 转换轨迹
    print("[5/6] 转换手部轨迹 → EE轨迹...")
    ee_trajectory = mano_trajectory_to_ee_trajectory(pred_trans, pred_rot)
    print(f"  EE轨迹: {ee_trajectory.shape}")

    # 6. 执行仿真
    print("[6/6] 执行物理仿真...")
    tracked_trajectories = load_tracked_trajectories(args.tracked_dir)

    sim_output = run_simulation(
        env=None,
        robot=robot,
        ee_trajectory=ee_trajectory,
        left_gripper_signal=left_gripper,
        right_gripper_signal=right_gripper,
        object_actors=object_actors,
        tracked_object_trajectories=tracked_trajectories,
        max_steps=args.max_steps,
    )

    # 保存结果
    print("\n保存仿真结果...")
    np.savez(
        os.path.join(args.output, "sim_results.npz"),
        sim_ee_trajectories=sim_output["sim_ee_trajectories"],
        **{f"traj_{k}": v for k, v in sim_output["sim_object_trajectories"].items()},
    )

    if sim_output["verification"]:
        with open(os.path.join(args.output, "verification.json"), "w") as f:
            json.dump(sim_output["verification"], f, indent=2)
        print("验证报告:")
        for name, metrics in sim_output["verification"].items():
            print(f"  {name}: mean={metrics['mean_error']:.4f}m, max={metrics['max_error']:.4f}m")

    print(f"\n仿真完成! 结果保存在: {args.output}")


if __name__ == "__main__":
    main()
