#!/usr/bin/env python3
"""
End-to-end Replica benchmark script for ReplicateAnyScene.
Usage:
    # 测试单个场景（完整pipeline + 评估）
    conda run -n ReplicateAnyScene python test_replica.py \
        --scene office0 --max_frames 25

    # 仅评估（跳过RAS pipeline，假设已有输出）
    conda run -n ReplicateAnyScene python test_replica.py \
        --scene office0 --skip_pipeline

    # 批量测试多个场景
    conda run -n ReplicateAnyScene python test_replica.py \
        --scenes office0 office1 room0 --max_frames 25

    # 仅评估多个场景
    conda run -n ReplicateAnyScene python test_replica.py \
        --scenes office0 office1 room0 --skip_pipeline --collect_results
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np

REPLICA_ROOT = "/mnt/data_8THDD/lza/dataset/Replica"
RAS_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CATEGORIES = {
    "chair": "supported_by_floor",
    "table": "supported_by_floor",
    "desk": "supported_by_floor",
    "cabinet": "supported_by_floor",
    "shelf": "supported_by_floor",
    "bed": "supported_by_floor",
    "sofa": "supported_by_floor",
    "couch": "supported_by_floor",
    "bookcase": "supported_by_floor",
    "counter": "supported_by_floor",
    "door": "embedded_in_wall",
    "window": "embedded_in_wall",
    "monitor": "supported_by_floor",
    "screen": "supported_by_floor",
    "keyboard": "supported_by_floor",
    "plant": "supported_by_floor",
    "pillow": "supported_by_floor",
    "lamp": "supported_by_floor",
    "light": "attached_to_wall",
    "picture": "attached_to_wall",
    "painting": "attached_to_wall",
    "frame": "attached_to_wall",
    "trash_can": "supported_by_floor",
    "bin": "supported_by_floor",
    "rug": "supported_by_floor",
    "carpet": "supported_by_floor",
}


def get_available_scenes():
    scenes = []
    for name in sorted(os.listdir(REPLICA_ROOT)):
        scene_dir = os.path.join(REPLICA_ROOT, name)
        mesh_path = os.path.join(REPLICA_ROOT, f"{name}_mesh.ply")
        traj_path = os.path.join(scene_dir, "traj.txt")
        results_dir = os.path.join(scene_dir, "results")
        if os.path.isdir(scene_dir) and os.path.exists(traj_path) and os.path.exists(results_dir) and os.path.exists(mesh_path):
            scenes.append(name)
    return scenes


def create_category_json(scene_dir, output_path):
    cat_json = {}
    results_dir = os.path.join(scene_dir, "results")
    jpg_files = [f for f in os.listdir(results_dir) if f.endswith(".jpg")]
    if not jpg_files:
        print(f"⚠️  {scene_dir} 中没有找到图片")
        return None

    print(f"   场景有 {len(jpg_files)} 帧图片")

    for cat, rel in DEFAULT_CATEGORIES.items():
        cat_json[cat] = rel

    json_path = os.path.join(output_path, "categories.json")
    with open(json_path, "w") as f:
        json.dump(cat_json, f, indent=2)
    print(f"   类别JSON已生成 ({len(cat_json)} 类): {json_path}")
    return json_path


def run_ras_pipeline(scene_name, output_path, max_frames, use_v2=True):
    scene_dir = os.path.join(REPLICA_ROOT, scene_name)
    results_dir = os.path.join(scene_dir, "results")
    os.makedirs(output_path, exist_ok=True)

    if use_v2:
        print(f"\n{'='*60}")
        print(f"🚀 运行 RAS pipeline (V2): {scene_name}")
        print(f"   输入目录: {results_dir}")
        print(f"   输出目录: {output_path}")
        print(f"   最大帧数: {max_frames}")
        print(f"   V2模式: Stage 1 自动发现物体 (VLM)")
        print(f"{'='*60}")

        cmd = [
            "python", os.path.join(RAS_ROOT, "mainv2.py"),
            "--input_images", results_dir,
            "--output_path", output_path,
            "--max_frames", str(max_frames),
        ]
    else:
        json_path = create_category_json(scene_dir, output_path)
        if json_path is None:
            return False

        print(f"\n{'='*60}")
        print(f"🚀 运行 RAS pipeline (V1): {scene_name}")
        print(f"   输入目录: {results_dir}")
        print(f"   输出目录: {output_path}")
        print(f"   最大帧数: {max_frames}")
        print(f"{'='*60}")

        cmd = [
            "python", os.path.join(RAS_ROOT, "main.py"),
            "--input_video", results_dir,
            "--output_path", output_path,
            "--category_path", json_path,
            "--max_frames", str(max_frames),
        ]

    print(f"   执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=RAS_ROOT)

    if result.returncode != 0:
        print(f"❌ Pipeline 失败 (exit code {result.returncode})")
        return False

    glb_path = os.path.join(output_path, "final_scene.glb")
    if not os.path.exists(glb_path):
        print(f"❌ GLB 文件未生成: {glb_path}")
        return False

    print(f"✅ Pipeline 完成: {glb_path}")
    return True


def run_assessment(output_path, scene_name, sample_count=None, skip_lpips=False):
    mesh_path = os.path.join(REPLICA_ROOT, f"{scene_name}_mesh.ply")
    glb_path = os.path.join(output_path, "final_scene.glb")

    if not os.path.exists(glb_path):
        print(f"❌ GLB 文件不存在: {glb_path}")
        return None

    if not os.path.exists(mesh_path):
        print(f"❌ GT mesh 不存在: {mesh_path}")
        return None

    print(f"\n{'='*60}")
    print(f"📊 评估: {scene_name}")
    print(f"   GLB: {glb_path}")
    print(f"   GT Mesh: {mesh_path}")
    print(f"{'='*60}")

    cmd = [
        "python", "-m", "assess.run_assessment",
        "--output_path", output_path,
        "--reference_mesh", mesh_path,
    ]

    if sample_count is not None:
        cmd.extend(["--sample_count", str(sample_count)])

    if skip_lpips:
        cmd.append("--skip_lpips")

    print(f"   执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=RAS_ROOT)

    if result.returncode != 0:
        print(f"⚠️  评估部分失败 (exit code {result.returncode})")

    results_path = os.path.join(output_path, "assessment_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        return results
    else:
        print(f"⚠️  评估结果文件未生成")
        return None


def print_scene_summary(scene_name, results):
    print(f"\n  ┌{'─'*50}┐")
    print(f"  │  {scene_name:^48} │")
    print(f"  └{'─'*50}┘")

    if not results:
        print("    (无结果)")
        return

    if "visual" in results:
        v = results["visual"]
        print(f"  视觉质量 (全图):")
        print(f"    PSNR: {v.get('PSNR', 'N/A'):.2f} dB")
        print(f"    SSIM: {v.get('SSIM', 'N/A'):.4f}")
        if "LPIPS" in v:
            print(f"    LPIPS: {v['LPIPS']:.4f}")
        if "render_coverage" in v:
            print(f"    coverage: {v['render_coverage']:.2%}")
        if "PSNR_masked" in v:
            print(f"  视觉质量 (masked):")
            print(f"    PSNR_m: {v['PSNR_masked']:.2f} dB")
            print(f"    SSIM_m: {v['SSIM_masked']:.4f}")
            if "LPIPS_masked" in v:
                print(f"    LPIPS_m: {v['LPIPS_masked']:.4f}")

    if "spatial" in results:
        s = results["spatial"]
        print(f"  空间精度:")
        print(f"    MaskIoU: {s.get('MaskIoU', 'N/A'):.4f}")
        print(f"    DepthRMSE: {s.get('DepthRMSE', 'N/A'):.4f} m")
        print(f"    DepthAbsRel: {s.get('DepthAbsRel', 'N/A'):.4f}")

    if "mesh_geometry" in results:
        g = results["mesh_geometry"]
        print(f"  几何精度 (Mesh vs GT, {'归一化' if g.get('normalized') else '原始'}):")
        print(f"    CD-L2: {g.get('ChamferDistance_L2', 'N/A'):.6f}")
        print(f"    CD-L1: {g.get('ChamferDistance_L1', 'N/A'):.6f}")
        print(f"    Accuracy: {g.get('Accuracy', 'N/A'):.6f}")
        print(f"    Completeness: {g.get('Completeness', 'N/A'):.6f}")
        print(f"    Overall: {g.get('Overall', 'N/A'):.6f}")
        for key in sorted(g.keys()):
            if key.startswith("F-Score_"):
                thresh = key.replace("_", ".").replace("F-Score.", "")
                print(f"    F-Score@{thresh}: {g[key]:.4f}")
        if "NormalConsistency" in g:
            print(f"    NC: {g['NormalConsistency']:.4f}")


def collect_all_results(scene_names, output_basedir):
    all_summary = {}
    for name in scene_names:
        output_path = os.path.join(output_basedir, f"replica_{name}")
        results_path = os.path.join(output_path, "assessment_results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                all_summary[name] = json.load(f)
        else:
            all_summary[name] = None

    print("\n" + "=" * 70)
    print("📊 汇总对比表")
    print("=" * 70)

    header = f"{'指标':<25}"
    for name in scene_names:
        header += f" {name:<18}"
    print(header)
    print("-" * 70)

    metrics = [
        ("PSNR (dB)", ["visual", "PSNR"], ".2f"),
        ("SSIM", ["visual", "SSIM"], ".4f"),
        ("LPIPS", ["visual", "LPIPS"], ".4f"),
        ("render_coverage", ["visual", "render_coverage"], ".2%"),
        ("PSNR_masked (dB)", ["visual", "PSNR_masked"], ".2f"),
        ("SSIM_masked", ["visual", "SSIM_masked"], ".4f"),
        ("MaskIoU", ["spatial", "MaskIoU"], ".4f"),
        ("DepthRMSE (m)", ["spatial", "DepthRMSE"], ".4f"),
        ("DepthAbsRel", ["spatial", "DepthAbsRel"], ".4f"),
        ("CD-L2 (norm)", ["mesh_geometry", "ChamferDistance_L2"], ".6f"),
        ("CD-L1 (norm)", ["mesh_geometry", "ChamferDistance_L1"], ".6f"),
        ("Accuracy (norm)", ["mesh_geometry", "Accuracy"], ".6f"),
        ("Completeness (norm)", ["mesh_geometry", "Completeness"], ".6f"),
        ("Overall (norm)", ["mesh_geometry", "Overall"], ".6f"),
        ("F-Score@0.05", ["mesh_geometry", "F-Score_0_05"], ".4f"),
        ("NC", ["mesh_geometry", "NormalConsistency"], ".4f"),
    ]

    for metric_name, keys, fmt in metrics:
        row = f"{metric_name:<25}"
        for name in scene_names:
            r = all_summary.get(name)
            if r is None:
                row += f" {'N/A':<18}"
                continue
            val = r
            for k in keys:
                if val is None:
                    break
                val = val.get(k)
            if val is not None:
                try:
                    row += f" {val:{fmt}}".ljust(19)
                except (ValueError, TypeError):
                    row += f" {str(val):<18}"
            else:
                row += f" {'-':<18}"
        print(row)

    print("\n总结:")
    for name in scene_names:
        r = all_summary.get(name)
        if r and "mesh_geometry" in r:
            g = r["mesh_geometry"]
            print(f"  {name}: CD-L2={g.get('ChamferDistance_L2', 'N/A'):.6f}, "
                  f"F@0.05={g.get('F-Score_0_05', 'N/A'):.4f}, "
                  f"NC={g.get('NormalConsistency', 'N/A'):.4f}")

    return all_summary


def main():
    parser = argparse.ArgumentParser(
        description="ReplicateAnyScene Replica Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看可用场景
  python test_replica.py --list

  # 测试单个场景 (完整 pipeline)
  python test_replica.py --scene office0 --max_frames 25

  # 仅评估（跳过 pipeline）
  python test_replica.py --scene office0 --skip_pipeline

  # 批量测试
  python test_replica.py --scenes office0 office1 room0 --max_frames 25

  # 仅收集评估结果
  python test_replica.py --scenes office0 office1 room0 --skip_pipeline --collect_results
        """
    )

    parser.add_argument("--scene", type=str, default=None, help="单个场景名称")
    parser.add_argument("--scenes", type=str, nargs="+", default=None, help="多个场景名称")
    parser.add_argument("--list", action="store_true", help="列出所有可用场景")
    parser.add_argument("--max_frames", type=int, default=25, help="RAS 最大帧数")
    parser.add_argument("--sample_count", type=int, default=None, help="评估采样帧数(默认:全部)")
    parser.add_argument("--skip_pipeline", action="store_true", help="跳过 RAS pipeline")
    parser.add_argument("--skip_lpips", action="store_true", help="跳过 LPIPS")
    parser.add_argument("--collect_results", action="store_true", help="仅汇总已有结果")
    parser.add_argument("--use_v1", action="store_true", help="使用 main.py (V1, 需要categories JSON)")
    parser.add_argument("--output_basedir", type=str,
                        default=os.path.join(RAS_ROOT, "outputs"),
                        help="输出基准目录")
    args = parser.parse_args()

    available = get_available_scenes()

    if args.list:
        print("可用场景:")
        for s in available:
            mesh_size = os.path.getsize(os.path.join(REPLICA_ROOT, f"{s}_mesh.ply")) / 1024 / 1024
            print(f"  {s:<12} (GT mesh: {mesh_size:.1f} MB)")
        return

    scene_names = []
    if args.scene:
        scene_names = [args.scene]
    elif args.scenes:
        scene_names = args.scenes
    else:
        print("请指定 --scene 或 --scenes")
        print(f"可用场景: {', '.join(available)}")
        return

    for name in scene_names:
        if name not in available:
            print(f"❌ 未知场景: {name}")
            print(f"   可用: {', '.join(available)}")
            return

    if args.collect_results:
        collect_all_results(scene_names, args.output_basedir)
        return

    print(f"测试场景: {scene_names}")
    print(f"最大帧数: {args.max_frames}")
    print(f"评估采样: {args.sample_count}")
    print(f"输出目录: {args.output_basedir}")

    for scene_name in scene_names:
        output_path = os.path.join(args.output_basedir, f"replica_{scene_name}")

        if not args.skip_pipeline:
            print(f"\n{'#'*60}")
            print(f"# 场景: {scene_name} - Stage 1: RAS Pipeline")
            print(f"{'#'*60}")
            success = run_ras_pipeline(scene_name, output_path, args.max_frames, use_v2=not args.use_v1)
            if not success:
                print(f"❌ {scene_name} pipeline 失败")
                continue

        print(f"\n{'#'*60}")
        print(f"# 场景: {scene_name} - Stage 2: 评估")
        print(f"{'#'*60}")
        results = run_assessment(output_path, scene_name,
                                 sample_count=args.sample_count,
                                 skip_lpips=args.skip_lpips)
        print_scene_summary(scene_name, results)

    if len(scene_names) > 1:
        collect_all_results(scene_names, args.output_basedir)


if __name__ == "__main__":
    main()