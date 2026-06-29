"""
独立测试四阶段房间坐标系对齐模块 (带可视化输出).

从 assets/basic_pick_place 随机选 5 个视频, 对每个视频:
  1. VGGT-Omega 重建 (max_frames=16, 加速)
  2. SAM3 分割 floor/wall
  3. 依次尝试 4 个对齐阶段:
     - Stage 1: align_to_room_coordinate_system (严格: 阈值0.02, 要求正交wall)
     - Stage 2: align_via_objects (放宽: 阈值0.05, 无wall时PCA)
     - Stage 3: align_via_large_plane (大平面mask当floor)
     - Stage 4: align_via_geocalib (图像重力方向)
  4. 报告每阶段成功/失败、method、R正交性、z轴对齐度
  5. 输出可视化: 每个视频生成一张图, 6列(原始+4阶段) × 2行(侧视图+俯视图),
     点云按z高度着色, 直观看出z轴是否对齐到竖直

运行:
  cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
  /mnt/data/lza/conda_envs/ReplicateAnyScene/bin/python test_alignment_basic_pick_place.py
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import load_vggt_omega_model, load_sam3_image_model, unload_model
from src.vggt_omega_predict import load_vggt_omega_frames, vggt_omega_predict
from src.object_segmentation import segment_wall_and_floor
from src.geometry_utils import (
    align_to_room_coordinate_system,
    align_via_objects,
    align_via_large_plane,
    align_via_geocalib,
)


BASIC_PICK_PLACE = "./assets/basic_pick_place"
OUTPUT_DIR = "./output_v2/alignment_test_basic_pick_place"
MAX_FRAMES = 16  # 加速: 只用16帧 (原pipeline用160)
N_VIDEOS = 5
SEED = 42


def pick_random_videos(n=N_VIDEOS, seed=SEED):
    """从 basic_pick_place 随机选 n 个 .mp4 视频"""
    all_videos = sorted([
        f for f in os.listdir(BASIC_PICK_PLACE)
        if f.endswith(".mp4") and os.path.isfile(os.path.join(BASIC_PICK_PLACE, f))
    ])
    rng = random.Random(seed)
    chosen = rng.sample(all_videos, min(n, len(all_videos)))
    return chosen


def check_R_orthogonal(R):
    """检查 R 是否正交"""
    err = float(np.linalg.norm(R @ R.T - np.eye(3)))
    det = float(np.linalg.det(R))
    return err < 1e-4 and abs(det - 1) < 1e-4, err, det


def check_z_aligned(R):
    """检查 R 的第三行(z轴)是否接近 [0,0,1] (即 floor 法线对齐到世界z)"""
    new_z = R[2, :]  # R 的第三行 = floor_normal
    cos_to_world_z = abs(float(new_z[2]))
    return cos_to_world_z


def is_identity(R, t):
    """判断 R,t 是否为 identity (对齐失败)"""
    return np.allclose(R, np.eye(3), atol=1e-6) and np.allclose(t, 0, atol=1e-6)


def run_stage_1(world_points, wall_masks, floor_masks):
    """Stage 1: align_to_room_coordinate_system (严格)"""
    R, t = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
    return R, t, {"stage": 1, "name": "align_to_room_coordinate_system"}


def run_stage_2(world_points, wall_masks, floor_masks):
    """Stage 2: align_via_objects (放宽)"""
    R, t, info = align_via_objects(world_points, wall_masks, floor_masks)
    return R, t, {"stage": 2, "name": "align_via_objects", **info}


def run_stage_3(world_points, floor_masks):
    """Stage 3: align_via_large_plane (用floor_masks当大平面mask)"""
    R, t, info = align_via_large_plane(world_points, floor_masks)
    return R, t, {"stage": 3, "name": "align_via_large_plane", **info}


def run_stage_4(images, world_points, extrinsics):
    """Stage 4: align_via_geocalib (图像重力方向)"""
    R, t, info = align_via_geocalib(images, world_points, extrinsics, max_frames=8)
    return R, t, {"stage": 4, "name": "align_via_geocalib", **info}


def evaluate_alignment(R, t, world_points):
    """评估对齐质量"""
    is_id = is_identity(R, t)
    orth_ok, orth_err, det = check_R_orthogonal(R)
    z_cos = check_z_aligned(R)
    # 对齐后 floor 点的 z 分布 (越小越好, 理想=0)
    aligned_pts = world_points.reshape(-1, 3) @ R.T + t
    z_spread = float(np.std(aligned_pts[:, 2]))
    return {
        "is_identity": bool(is_id),
        "success": not is_id,
        "orthogonal": bool(orth_ok),
        "orth_err": orth_err,
        "det": det,
        "z_axis_cos_to_world_z": z_cos,
        "aligned_z_spread": z_spread,
        "R": R.tolist(),
        "t": t.tolist(),
    }


def _downsample_points(pts, max_n=8000):
    """下采样点云到 max_n 个点"""
    n = pts.shape[0]
    if n <= max_n:
        return pts, np.arange(n)
    idx = np.random.RandomState(42).choice(n, max_n, replace=False)
    return pts[idx], idx


def visualize_alignment(video_name, world_points, stage_results, output_dir):
    """生成可视化图: 6列(原始+4阶段) × 2行(侧视图XZ + 俯视图XY)

    点云按 z 高度着色 (蓝→红), 直观看出:
    - 侧视图: z轴是否竖直 (floor 应水平)
    - 俯视图: 场景结构是否合理
    """
    pts_orig = world_points.reshape(-1, 3)
    pts_ds, ds_idx = _downsample_points(pts_orig, max_n=8000)

    # z 范围 (用原始点云的 z 做颜色映射, 保持一致)
    z_vals_orig = pts_ds[:, 2]
    z_min, z_max = float(z_vals_orig.min()), float(z_vals_orig.max())
    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0

    col_labels = ["Original", "Stage1\n(strict)", "Stage2\n(relaxed)", "Stage3\n(large_plane)", "Stage4\n(geocalib)"]
    stage_keys = ["stage1", "stage2", "stage3", "stage4"]

    fig, axes = plt.subplots(2, 5, figsize=(28, 12))
    fig.suptitle(f"Four-Stage Room Coordinate Alignment — {video_name}", fontsize=16, fontweight="bold")

    for col, (label, skey) in enumerate(zip(col_labels, ["orig"] + stage_keys)):
        if skey == "orig":
            pts = pts_ds.copy()
            title = label
            z_for_color = pts[:, 2]
        else:
            sr = stage_results.get(skey, {})
            R = np.array(sr.get("R", np.eye(3).tolist()))
            t = np.array(sr.get("t", [0, 0, 0]))
            success = sr.get("success", False)
            z_cos = sr.get("z_axis_cos_to_world_z", 0)
            method = sr.get("method", sr.get("name", "?"))
            reason = sr.get("reason", "")
            pts = (pts_ds @ R.T + t)
            # 用对齐后的 z 着色
            z_for_color = pts[:, 2]
            status = "OK" if success else "FAIL"
            title = f"{label}\n[{status}] z_cos={z_cos:.2f}"
            if reason:
                title += f"\n{reason[:25]}"

        # Row 0: 侧视图 (X-Z plane, 看侧面的高度分布)
        ax = axes[0, col]
        sc = ax.scatter(pts[:, 0], pts[:, 2], c=z_for_color, cmap="jet",
                        s=0.3, alpha=0.6, vmin=z_min, vmax=z_max)
        ax.set_xlabel("X")
        ax.set_ylabel("Z (height)")
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        # 画水平参考线 (z=0)
        ax.axhline(y=0, color="green", linewidth=0.5, linestyle="--", alpha=0.5)

        # Row 1: 俯视图 (X-Y plane, 看平面分布)
        ax2 = axes[1, col]
        ax2.scatter(pts[:, 0], pts[:, 1], c=z_for_color, cmap="jet",
                    s=0.3, alpha=0.6, vmin=z_min, vmax=z_max)
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(f"Top view — {label}", fontsize=9)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"vis_{video_name.replace('.mp4', '')}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  📸 可视化已保存: {out_path}", flush=True)
    return out_path


def process_video(video_name, vggt_model, sam3_image_model, device):
    """处理单个视频: VGGT + SAM3 + 4阶段对齐"""
    video_path = os.path.join(BASIC_PICK_PLACE, video_name)
    result = {
        "video": video_name,
        "video_path": video_path,
        "max_frames": MAX_FRAMES,
        "stages": {},
        "error": None,
    }

    try:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"▶ 处理: {video_name}")
        print(f"{'='*60}")

        # 1. VGGT-Omega 重建
        print(f"  [1] VGGT-Omega 重建 (max_frames={MAX_FRAMES})...", flush=True)
        t1 = time.time()
        frames = load_vggt_omega_frames(video_path, MAX_FRAMES).to(device)
        predictions = vggt_omega_predict(frames, vggt_model)
        world_points = predictions['world_points']  # (T, H, W, 3)
        images = predictions['colors']  # (T, H, W, 3)
        print(f"      完成 ({time.time()-t1:.1f}s), world_points shape={world_points.shape}", flush=True)
        result["vggt_time"] = time.time() - t1
        result["world_points_shape"] = list(world_points.shape)

        # 2. SAM3 分割 floor/wall
        print(f"  [2] SAM3 分割 floor/wall...", flush=True)
        t2 = time.time()
        wall_masks, floor_masks = segment_wall_and_floor(images, sam3_image_model)
        print(f"      完成 ({time.time()-t2:.1f}s), walls={len(wall_masks)}, floors={len(floor_masks)}", flush=True)
        result["sam3_time"] = time.time() - t2
        result["n_wall_masks"] = len(wall_masks)
        result["n_floor_masks"] = len(floor_masks)

        # 3. 四阶段对齐
        stages_config = [
            ("stage1", lambda: run_stage_1(world_points, wall_masks, floor_masks)),
            ("stage2", lambda: run_stage_2(world_points, wall_masks, floor_masks)),
            ("stage3", lambda: run_stage_3(world_points, floor_masks)),
            ("stage4", lambda: run_stage_4(images, world_points, predictions['extrinsics'])),
        ]

        first_success_stage = None
        for stage_key, stage_fn in stages_config:
            stage_name = stage_key
            print(f"  [3] {stage_name}...", flush=True)
            t3 = time.time()
            try:
                R, t, info = stage_fn()
                elapsed = time.time() - t3
                eval_result = evaluate_alignment(R, t, world_points)
                stage_result = {
                    "time": elapsed,
                    **info,
                    **eval_result,
                }
                result["stages"][stage_key] = stage_result
                status = "✅成功" if eval_result["success"] else "❌失败(identity)"
                print(f"      {status} ({elapsed:.1f}s) method={info.get('name','?')} "
                      f"z_cos={eval_result['z_axis_cos_to_world_z']:.3f} "
                      f"z_spread={eval_result['aligned_z_spread']:.3f}", flush=True)
                if eval_result["success"] and first_success_stage is None:
                    first_success_stage = stage_key
            except Exception as e:
                elapsed = time.time() - t3
                result["stages"][stage_key] = {
                    "time": elapsed,
                    "error": str(e),
                    "success": False,
                }
                print(f"      ❌异常 ({elapsed:.1f}s): {e}", flush=True)

        result["first_success_stage"] = first_success_stage
        result["total_time"] = time.time() - t0
        print(f"  ✅ 视频完成 ({result['total_time']:.1f}s), 首个成功阶段: {first_success_stage}", flush=True)

        # 4. 可视化
        try:
            vis_path = visualize_alignment(video_name, world_points, result["stages"], OUTPUT_DIR)
            result["vis_path"] = vis_path
        except Exception as e:
            print(f"  ⚠️ 可视化失败: {e}", flush=True)
            result["vis_path"] = None

    except Exception as e:
        import traceback
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"  ❌ 视频处理失败: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"输出目录: {OUTPUT_DIR}")

    # 选 5 个视频
    videos = pick_random_videos(N_VIDEOS, SEED)
    print(f"\n随机选取 {len(videos)} 个视频 (seed={SEED}):")
    for v in videos:
        print(f"  - {v}")

    all_results = {
        "test_config": {
            "max_frames": MAX_FRAMES,
            "n_videos": N_VIDEOS,
            "seed": SEED,
            "device": device,
        },
        "videos": videos,
        "results": [],
    }

    # 加载模型 (VGGT omega + SAM3 image)
    print(f"\n加载 VGGT-Omega 模型...", flush=True)
    vggt_model = load_vggt_omega_model().to(device)
    print(f"加载 SAM3 image 模型...", flush=True)
    sam3_image_model = load_sam3_image_model()

    # 逐个处理视频
    for i, video_name in enumerate(videos, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(videos)}] {video_name}")
        print(f"{'#'*60}", flush=True)
        result = process_video(video_name, vggt_model, sam3_image_model, device)
        all_results["results"].append(result)

        # 增量保存 (防止中途崩溃丢结果)
        report_path = os.path.join(OUTPUT_DIR, "alignment_test_report.json")
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 卸载模型
    vggt_model = unload_model(vggt_model)
    sam3_image_model = unload_model(sam3_image_model)

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"汇总报告")
    print(f"{'='*60}")
    print(f"{'视频':<12} {'wall':>5} {'floor':>6} {'S1':>6} {'S2':>6} {'S3':>6} {'S4':>6} {'首个成功':>10}")
    print("-" * 70)
    for r in all_results["results"]:
        vid = r["video"]
        nw = r.get("n_wall_masks", "?")
        nf = r.get("n_floor_masks", "?")
        s1 = "✅" if r.get("stages", {}).get("stage1", {}).get("success") else "❌"
        s2 = "✅" if r.get("stages", {}).get("stage2", {}).get("success") else "❌"
        s3 = "✅" if r.get("stages", {}).get("stage3", {}).get("success") else "❌"
        s4 = "✅" if r.get("stages", {}).get("stage4", {}).get("success") else "❌"
        first = r.get("first_success_stage", "—")
        print(f"{vid:<12} {str(nw):>5} {str(nf):>6} {s1:>6} {s2:>6} {s3:>6} {s4:>6} {first:>10}")

    report_path = os.path.join(OUTPUT_DIR, "alignment_test_report.json")
    print(f"\n详细报告已保存: {report_path}")


if __name__ == "__main__":
    main()
