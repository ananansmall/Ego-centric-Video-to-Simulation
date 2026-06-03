"""
ReplicateAnyScene V2 — 完整自动化流水线
========================================

只需输入视频，自动完成所有阶段，输出最终 GLB。
与 main.py 相比，V2 新增了 Stage 1 自动物体发现、Stage 5.1/5.2 物体间关系精修。

流水线:
  Stage 1: VGGT引导的智能物体发现 (generate_scene_json_stage1.py)
           输入: 视频 → 输出: 场景JSON (物体类别+关系)
           原main.py需要手动提供JSON，V2自动生成

  Stage 2: 3D重建 + SAM3空间去重 (与main.py一致，可选3种模型)
           输入: 视频 → 输出: 去重后的实例masks + 3D重建中间结果
           模型选择: --vggt_model vggt(默认) / vggt_omega / vggt4d

  Stage 3: 最优视角资产生成 + 多票验证 (与main.py一致)
           输入: masks+点云 → 输出: 3D mesh实例

  基础精修: floor/wall/embedded 精修 (始终执行，与main.py一致)
           输出: final_scene_base.glb

  Stage 4: 迭代视觉-空间对齐 (stage4/, 默认关闭，--enable_stage4 启用)
           输入: GLB+3D数据 → 输出: 对齐后的GLB
           原main.py未实现此阶段

  Stage 5: 高级语义精修 (默认关闭，--enable_stage5 启用)
    5.1: 细化 "supported by other objects" 关系 (VLM判断具体支撑物)
    5.2: 物体间支撑关系空间位置精修 (纯几何计算)

GLB 文件说明:
  final_scene_base.glb    = 基础精修后固定起点 (供 run_post_pipeline.py 使用, 不再更改)
  final_scene.glb         = 最终场景 (始终为最新结果, 不启用stage4/5时等同base)
  all_instances.pkl       = 基础精修后的实例数据 (供 run_post_pipeline.py 独立调用stage4/5)

使用方式:
  ────────────────────────────────────────────────────────────
  参数总览:
  ────────────────────────────────────────────────────────────
  --input_video       输入视频路径 (与--input_images二选一)
  --input_images      输入图片目录路径 (与--input_video二选一)
  --output_path       输出目录 (默认: ./output_v2/{video_stem}_{模型名})
  --category_path     手动指定场景JSON (跳过Stage1自动发现)
  --vggt_model        3D重建模型: vggt(默认) | vggt_omega | vggt4d
  --max_frames        VGGT最大帧数 (默认160)
  --max_frames_stage1 Stage1采样关键帧数 (默认10)
  --vlm_checkpoint    VLM模型路径 (默认自动查找)
  --enable_stage4     启用Stage 4视觉-空间对齐
  --enable_stage5     启用Stage 5高级语义精修 (5.1+5.2)
  --stage4_iterations Stage4 ICP迭代次数 (默认8)
  --stage4_temporal_radius Stage4时序邻域半径 (默认5)
  --stage4_use_mast3r Stage4使用MASt3R匹配
  ────────────────────────────────────────────────────────────

  # 最简调用 (默认VGGT模型)
  python mainv2.py --input_video ./assets/example/hallway.mp4

  # 使用 VGGT-Ω 模型python mainv2.py --input_video ./232.mp4 --vggt_model vggt_omega
  

  # 使用 VGGT4D 模型 (带动态mask，适合动态场景)
  python mainv2.py --input_video ./232.mp4 --vggt_model vggt4d

  # VGGT4D + 全部高级阶段
  python mainv2.py --input_video ./232.mp4 --vggt_model vggt4d --enable_stage4 --enable_stage5

  # 手动指定场景JSON (跳过Stage1自动发现)
  python mainv2.py --input_video ./hallway.mp4 --category_path ./hallway.json

  # 指定输出目录
  python mainv2.py --input_video ./hallway.mp4 --output_path ./my_output

  # 启用 Stage 4 (视觉-空间对齐)
  python mainv2.py --input_video ./hallway.mp4 --enable_stage4

  # 启用 Stage 5 (高级语义精修)
  python mainv2.py --input_video ./hallway.mp4 --enable_stage5

  # 输入图片目录 (Replica/ScanNet等)
  python mainv2.py --input_images /path/to/images/ --output_path ./output_v2/scene_001

  ────────────────────────────────────────────────────────────
  后处理管线 (tools/run_post_pipeline.py):
  ────────────────────────────────────────────────────────────
  在 mainv2 完成 Stage 1~3 + 基础精修后，可单独运行后处理。
  参数:
    --scene_dir              场景输出目录 (必需)
    --stage4                 启用Stage 4
    --stage5                 启用Stage 5 (5.1+5.2)
    --only_refine_relations  只运行5.1 (细化关系)
    --only_sp_refinement     只运行5.2 (SP精修)
    --relations_json         手动指定关系JSON (配合--only_sp_refinement)
    --vlm_checkpoint         VLM模型路径
    --stage4_iterations      Stage4 ICP迭代次数 (默认8)
    --stage4_temporal_radius Stage4时序半径 (默认2)
    --stage4_use_mast3r      Stage4使用MASt3R匹配

  # 一键全功能: Stage4 + Stage5
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage4 --stage5

  # 只运行 Stage 5
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage5

  # 只运行 Stage 4
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --stage4

  # 只细化关系 (Stage 5.1)
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_refine_relations

  # 只做 SP 精修 (Stage 5.2), 手动指定关系JSON
  python tools/run_post_pipeline.py --scene_dir output_v2/hoi4d --only_sp_refinement \
      --relations_json output_v2/hoi4d/hoi4d_refined.json

  ────────────────────────────────────────────────────────────
  mainv2 与 run_post_pipeline 的关系:
  ────────────────────────────────────────────────────────────
  mainv2:          Stage1 → Stage2 → Stage3 → 基础精修 → [Stage4] → [Stage5] → final_scene.glb
  run_post_pipeline:                                      [Stage4] → [Stage5] → final_scene_stageX.glb

  mainv2 是完整流水线 (从视频到GLB)
  run_post_pipeline 只做后处理 (从已有输出目录开始)
  两者的 Stage4/5 逻辑完全一致，只是入口不同

输出目录结构:
  output_v2/{scene}_{模型名}/
  ├── scene_{scene}_stage1.json          # Stage1: 自动发现的场景JSON
  ├── scene_{scene}_stage1_refined.json  # Stage5.1: 细化后的关系JSON
  ├── final_relations.json               # 最终关系JSON
  ├── final_scene_base.glb               # 基础精修后固定起点 (不再更改)
  ├── final_scene.glb                    # 最终场景GLB (始终为最新结果)
  ├── all_instances.pkl                  # 实例数据 (供后处理管线使用)
  ├── point_cloud.ply                    # 3D重建点云
  ├── intrinsic.txt                      # 相机内参
  ├── color/                             # RGB帧
  ├── depth/                             # 深度图 (mm uint16)
  ├── extrinsics/                        # 相机外参
  ├── optimal_frames/                    # Stage3最优视角帧
  ├── keyframes/                         # Stage1关键帧+元数据
  └── instance_masks.mp4                 # 分割mask可视化
"""

import os
os.environ["LIDRA_SKIP_INIT"] = "true"
import argparse
import json
import logging
import subprocess
import sys
import time
import torch
import cv2
import numpy as np
import trimesh
from datetime import datetime

from src.models import load_vggt_model, load_vggt_omega_model, load_vggt4d_model, load_sam3_image_model, load_sam3_video_model, unload_model
from src.utils import load_video_frames, vis_instance_masks
from src.geometry_utils import (
    align_to_room_coordinate_system,
    align_vggt_predictions,
    get_optimal_view_frame_id,
    get_walls_info,
)
from src.vggt_predict import vggt_predict
from src.vggt_omega_predict import vggt_omega_predict
from src.vggt4d_predict import vggt4d_predict
from src.object_segmentation import segment_wall_and_floor, segment_and_track
from src.sg_deduplication import self_category_deduplicate, cross_category_deduplicate
from src.instance_generation import generate_3d_asset_in_subprocess
from src.sp_refinement import (
    refine_supported_by_floor_object,
    refine_attached_to_wall_object,
    refine_embedded_in_wall_object,
)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VLM_CHECKPOINT = "/mnt/data/lza/models/Qwen3.5-9B"


def _resolve_vlm_checkpoint(checkpoint_arg):
    """自动查找可用的VLM模型路径，优先级: 用户指定 > Qwen3.5-9B > Qwen2.5-VL-3B"""
    if checkpoint_arg and os.path.exists(checkpoint_arg):
        return checkpoint_arg
    if os.path.exists(DEFAULT_VLM_CHECKPOINT):
        return DEFAULT_VLM_CHECKPOINT
    fallback = "/mnt/data/lza/models/models--Qwen--Qwen2.5-VL-3B-Instruct"
    if os.path.exists(fallback):
        snapshots_dir = os.path.join(fallback, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = [d for d in os.listdir(snapshots_dir)
                         if os.path.isdir(os.path.join(snapshots_dir, d))]
            if snapshots:
                return os.path.join(snapshots_dir, snapshots[0])
    return None


def run_stage1(input_video, output_path, vlm_checkpoint, max_frames_stage1=10):
    """
    Stage 1: VGGT引导的智能物体发现 → 场景JSON

    调用 tools/generate_scene_json_stage1.py，流程:
      Step 0: VGGT 3D场景重建
      Step 1: SimRecon 3D空间覆盖采样 → 选择关键帧
      Step 2: 提取关键帧图像
      Step 3: 第一次VLM调用 → 逐帧物体检测（名称+位置）
      Step 4: 射线投射 → 将VLM像素位置映射到3D空间
      Step 5: 名称+3D位置联合去重
      Step 6: SAM分割floor和wall
      Step 7: 第二次VLM调用 → 关系判断（4种关系+物理常识后处理）
      Step 8: 输出场景JSON

    参数:
        input_video: 输入视频路径
        output_path: 输出目录
        vlm_checkpoint: VLM模型路径
        max_frames_stage1: 采样关键帧数
    返回:
        (stage1_json路径, categories_and_relations字典)
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 1: VGGT引导的智能物体发现", flush=True)
    print("=" * 70, flush=True)

    scene_id = os.path.splitext(os.path.basename(input_video))[0]
    stage1_json = os.path.join(output_path, f"scene_{scene_id}_stage1.json")
    keyframes_dir = os.path.join(output_path, "keyframes")

    # 通过subprocess调用，因为generate_scene_json_stage1内部有自己的VGGT/VLM生命周期管理
    cmd = [
        sys.executable, "-m", "tools.generate_scene_json_stage1",
        "--input_video", input_video,
        "--output_json", stage1_json,
        "--output_dir", keyframes_dir,
        "--vlm_checkpoint", vlm_checkpoint,
        "--max_frames", str(max_frames_stage1),
    ]

    print(f"   命令: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT)

    if result.returncode != 0:
        print(f"❌ Stage 1 失败 (返回码 {result.returncode})", flush=True)
        sys.exit(1)

    if not os.path.exists(stage1_json):
        print(f"❌ Stage 1 输出未找到: {stage1_json}", flush=True)
        sys.exit(1)

    # 读取Stage1输出的场景JSON
    with open(stage1_json, 'r') as f:
        categories_and_relations = json.load(f)

    # 同步 keyframes: generate_scene_json_stage1 将关键帧保存到
    # assets/key_frames/{scene_id}/，需要同步到 {output_path}/keyframes/
    # 以便后续 refine_other_objects_relations 和 refine_inter_object_placement 使用
    source_keyframes = os.path.join(REPO_ROOT, "assets", "key_frames", scene_id)
    target_keyframes = os.path.join(output_path, "keyframes")
    if os.path.isdir(source_keyframes) and not os.path.isdir(target_keyframes):
        import shutil
        shutil.copytree(source_keyframes, target_keyframes)
        print(f"   📋 同步 keyframes: {source_keyframes} → {target_keyframes}", flush=True)
    elif os.path.isdir(source_keyframes) and os.path.isdir(target_keyframes):
        # 目标已存在，检查是否缺少 metadata
        src_meta = os.path.join(source_keyframes, "keyframes_metadata.json")
        dst_meta = os.path.join(target_keyframes, "keyframes_metadata.json")
        if os.path.exists(src_meta) and not os.path.exists(dst_meta):
            import shutil
            shutil.copy2(src_meta, dst_meta)
            print(f"   📋 补充 keyframes_metadata.json → {target_keyframes}", flush=True)
        # 补充缺失的帧图片
        for fname in os.listdir(source_keyframes):
            src_f = os.path.join(source_keyframes, fname)
            dst_f = os.path.join(target_keyframes, fname)
            if os.path.isfile(src_f) and not os.path.exists(dst_f):
                import shutil
                shutil.copy2(src_f, dst_f)
                print(f"   📋 补充帧: {fname} → {target_keyframes}", flush=True)
    elif not os.path.isdir(source_keyframes) and not os.path.isdir(target_keyframes):
        print(f"   ⚠️ keyframes 目录不存在: {source_keyframes} 和 {target_keyframes}", flush=True)
        print(f"   ⚠️ 后续 VLM 帧投票可能无法使用 keyframes 来源", flush=True)

    print(f"✅ Stage 1 完成: {len(categories_and_relations)} 个物体", flush=True)
    for name, rel in sorted(categories_and_relations.items()):
        print(f"   {name}: {rel}", flush=True)

    return stage1_json, categories_and_relations


def run_stage2(input_video, output_path, categories_and_relations, max_frames=160, vggt_model_type="vggt"):
    """
    Stage 2: VGGT 3D重建 + SAM3空间去重 (与main.py一致)

    流程:
      1. VGGT预测3D属性（点云、深度、外参）
      2. SAM3分割floor/wall → 对齐到房间坐标系
      3. 保存VGGT中间结果（color/depth/extrinsics/intrinsic/point_cloud）
      4. SAM3 video模型逐类别分割+跟踪
      5. 类内去重（self_category_deduplicate）
      6. 跨类去重（cross_category_deduplicate）
      7. JSON白名单过滤（只保留Stage1发现的类别）
      8. 可视化mask结果

    参数:
        input_video: 输入视频路径
        output_path: 输出目录
        categories_and_relations: Stage1输出的类别-关系字典
        max_frames: VGGT最大帧数
    返回:
        (vggt_prediction_results, deduplicated_all_masks, wall_masks, floor_masks)
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 2: VGGT 3D重建 + SAM3空间去重", flush=True)
    print("=" * 70, flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    detected_categories = list(categories_and_relations.keys())
    print(f"   检测类别: {detected_categories}", flush=True)

    # 3D重建预测
    # 注意: VGGT-omega 使用 patch_size=16, image_resolution=512
    #       VGGT/VGGT4D 使用 patch_size=14, image_resolution=518
    #       每个模型必须用自己的 load 函数，不能用 ReplicateAnyScene 的 load_video_frames

    if vggt_model_type == "vggt":
        frames = load_video_frames(input_video, max_frames).to(device)
        vggt_model = load_vggt_model().to(device)
        vggt_prediction_results = vggt_predict(frames, vggt_model)
    elif vggt_model_type == "vggt_omega":
        from src.vggt_omega_predict import load_vggt_omega_frames
        frames = load_vggt_omega_frames(input_video, max_frames).to(device)
        vggt_model = load_vggt_omega_model().to(device)
        vggt_prediction_results = vggt_omega_predict(frames, vggt_model)
    elif vggt_model_type == "vggt4d":
        from src.vggt4d_predict import load_vggt4d_frames
        frames = load_vggt4d_frames(input_video, max_frames).to(device)
        vggt_model = load_vggt4d_model().to(device)
        vggt_prediction_results = vggt4d_predict(frames, vggt_model)
    else:
        raise ValueError(f"Unknown vggt_model_type: {vggt_model_type}. Choose from: vggt, vggt_omega, vggt4d")
    print(f"   加载 {len(frames)} 帧", flush=True)
    print(f"   使用 3D重建模型: {vggt_model_type} (分辨率={frames.shape[-1]}x{frames.shape[-2]})", flush=True)
    vggt_model = unload_model(vggt_model)

    # SAM3分割floor/wall → 对齐到房间坐标系
    sam3_image_model = load_sam3_image_model()
    wall_masks, floor_masks = segment_wall_and_floor(
        vggt_prediction_results['colors'], sam3_image_model
    )
    R, t = align_to_room_coordinate_system(
        vggt_prediction_results['world_points'], wall_masks, floor_masks
    )
    vggt_prediction_results = align_vggt_predictions(vggt_prediction_results, R, t)

    # 保存VGGT中间结果（供Stage4和调试使用）
    os.makedirs(os.path.join(output_path, 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'extrinsics'), exist_ok=True)
    np.savetxt(os.path.join(output_path, 'intrinsic.txt'), vggt_prediction_results['intrinsic'])
    vggt_prediction_results['point_cloud_data'].export(
        os.path.join(output_path, 'point_cloud.ply')
    )
    for i, image in enumerate(vggt_prediction_results['colors']):
        cv2.imwrite(
            os.path.join(output_path, 'color', f"{i}.jpg"),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
    for i, depth in enumerate(vggt_prediction_results['depths']):
        cv2.imwrite(
            os.path.join(output_path, 'depth', f"{i}.png"),
            (depth * 1000).astype(np.uint16),  # 深度单位: mm
        )
    for i, extrinsic in enumerate(vggt_prediction_results['extrinsics']):
        np.savetxt(os.path.join(output_path, 'extrinsics', f"{i}.txt"), extrinsic)

    sam3_image_model = unload_model(sam3_image_model)

    # SAM3 video模型: 逐类别分割+跟踪
    sam3_video_model = load_sam3_video_model()
    video_path = os.path.join(output_path, 'color')
    response = sam3_video_model.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    # 逐类别分割跟踪 + 类内去重
    all_masks = {}
    for category in detected_categories:
        print(f"   分割跟踪: {category}", flush=True)
        category_masks = segment_and_track(category, sam3_video_model, session_id)
        deduplicated_category_masks = self_category_deduplicate(
            category_masks,
            vggt_prediction_results['world_points'],
            vggt_prediction_results['world_points_conf'],
        )
        all_masks[category] = deduplicated_category_masks

    # 跨类去重
    json_categories_set = set(categories_and_relations.keys())
    deduplicated_all_masks = cross_category_deduplicate(
        all_masks,
        vggt_prediction_results['world_points'],
        vggt_prediction_results['world_points_conf'],
        protected_categories=json_categories_set,
    )

    # JSON白名单过滤: 只保留Stage1发现的类别
    filtered_masks = {}
    for category, category_masks in deduplicated_all_masks.items():
        if category in json_categories_set:
            filtered_masks[category] = category_masks
        else:
            print(f"   🚫 白名单过滤: '{category}' 不在JSON中", flush=True)
    deduplicated_all_masks = filtered_masks

    # 可视化分割mask结果
    vis_instance_masks(
        vggt_prediction_results['colors'], deduplicated_all_masks,
        os.path.join(output_path, 'instance_masks.mp4'),
    )
    sam3_video_model = unload_model(sam3_video_model)

    print(f"✅ Stage 2 完成", flush=True)
    return vggt_prediction_results, deduplicated_all_masks, wall_masks, floor_masks


def run_stage3(output_path, vggt_prediction_results, deduplicated_all_masks):
    """
    Stage 3: 最优视角资产生成 + 多票验证 (与main.py一致)

    流程:
      1. 计算每个实例的最优视角帧ID（最大3D表面积）
      2. 保存最优视角帧图像（供Stage5.1 VLM使用）
      3. 在SAM3D子进程中生成3D资产（避免CUDA冲突）
      4. 多票验证: 用多帧验证生成的3D资产质量

    参数:
        output_path: 输出目录
        vggt_prediction_results: VGGT预测结果
        deduplicated_all_masks: 去重后的masks
    返回:
        (all_instances, all_optimal_frame_ids)
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 3: 最优视角资产生成", flush=True)
    print("=" * 70, flush=True)

    # 计算每个实例的最优视角帧ID
    all_optimal_frame_ids = {}
    dynamic_count = 0
    static_count = 0
    for category, category_masks in deduplicated_all_masks.items():
        all_optimal_frame_ids[category] = []
        for inst_idx, instance_masks in enumerate(category_masks):
            optimal_frame_id, is_dynamic, motion_info = get_optimal_view_frame_id(
                vggt_prediction_results['world_points'], instance_masks
            )
            all_optimal_frame_ids[category].append(optimal_frame_id)
            tag = "DYNAMIC" if is_dynamic else "STATIC"
            if is_dynamic:
                dynamic_count += 1
            else:
                static_count += 1
            print(f"   {category}_{inst_idx}: [{tag}] median_disp={motion_info['median_disp']}m, "
                  f"max_disp={motion_info['max_disp']}m, "
                  f"global_disp={motion_info['global_disp']}m, "
                  f"valid_frames={motion_info['num_valid_frames']} → frame {optimal_frame_id}", flush=True)

    print(f"\n   📊 动态/静态统计: {dynamic_count} 动态, {static_count} 静态", flush=True)

    # 保存最优视角帧图像（文件名格式: {category}_inst{idx}_frame{fid}.jpg）
    optimal_frames_dir = os.path.join(output_path, 'optimal_frames')
    os.makedirs(optimal_frames_dir, exist_ok=True)
    for category, frame_ids in all_optimal_frame_ids.items():
        for inst_idx, frame_id in enumerate(frame_ids):
            image_rgb = vggt_prediction_results['colors'][frame_id]
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            save_name = f"{category}_inst{inst_idx}_frame{frame_id}.jpg"
            cv2.imwrite(os.path.join(optimal_frames_dir, save_name), image_bgr)

    instance_visibility = {}
    for category, category_masks in deduplicated_all_masks.items():
        instance_visibility[category] = {}
        for inst_idx, instance_masks in enumerate(category_masks):
            frame_ids = sorted([im["frame_id"] for im in instance_masks])
            instance_visibility[category][str(inst_idx)] = frame_ids
    visibility_path = os.path.join(optimal_frames_dir, "instance_visibility.json")
    with open(visibility_path, 'w', encoding='utf-8') as f:
        json.dump(instance_visibility, f, indent=2, ensure_ascii=False)
    print(f"   💾 Saved instance_visibility.json ({sum(len(v) for v in instance_visibility.values())} instances)", flush=True)

    # 在子进程中生成3D资产（避免CUDA内存冲突）
    all_instances = generate_3d_asset_in_subprocess(
        deduplicated_all_masks,
        all_optimal_frame_ids,
        vggt_prediction_results['colors'],
        vggt_prediction_results['world_points'],
        vggt_prediction_results['extrinsics'],
    )

    # Stage 3.5: 多票验证生成的3D资产
    from tools.asset_verifier import verify_all_instances
    all_instances = verify_all_instances(
        all_instances,
        all_optimal_frame_ids,
        deduplicated_all_masks,
        vggt_prediction_results['world_points'],
        vggt_prediction_results['world_points_conf'],
        min_votes=2,
    )

    print(f"✅ Stage 3 完成", flush=True)
    return all_instances, all_optimal_frame_ids


def run_stage4(output_path, vggt_prediction_results, all_instances, args,
               categories_and_relations=None, walls_info=None):
    """
    Stage 4: 迭代视觉-空间对齐 (可选，默认关闭)

    流程:
      1. 从VGGT数据重建世界点云
      2. 基于深度图创建实例mask
      3. 对每个实例执行2D-3D对应关系对齐:
         - Phase A: MASt3R/深度匹配 → Umeyama变换
         - Phase B: ICP精调 → 渐进阈值+RANSAC

    参数:
        output_path: 输出目录
        vggt_prediction_results: VGGT预测结果
        all_instances: 3D资产实例
        args: 命令行参数（stage4_iterations, stage4_temporal_radius等）
    返回:
        对齐后的 all_instances
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 4: 迭代视觉-空间对齐", flush=True)
    print("=" * 70, flush=True)

    from stage4.run_alignment import (
        reconstruct_world_points,
        create_depth_based_masks,
        compute_optimal_frame_ids,
    )
    from stage4.combined_alignment import refine_single_instance_combined

    # 从VGGT深度+外参重建世界点云
    vggt = {
        'depths': vggt_prediction_results['depths'],
        'extrinsics': vggt_prediction_results['extrinsics'],
        'intrinsic': vggt_prediction_results['intrinsic'],
        'colors': vggt_prediction_results['colors'],
    }
    world_points = reconstruct_world_points(
        vggt['depths'], vggt['extrinsics'], vggt['intrinsic']
    )
    world_points_conf = np.ones_like(vggt['depths'], dtype=np.float32)
    vggt['world_points'] = world_points
    vggt['world_points_conf'] = world_points_conf

    # 基于深度图创建实例mask
    all_masks = create_depth_based_masks(
        all_instances, vggt['depths'], vggt['extrinsics'],
        vggt['intrinsic'], world_points,
    )
    all_optimal_frame_ids = compute_optimal_frame_ids(all_masks, world_points)

    total_instances = sum(len(insts) for insts in all_instances.values())
    current_instance = 0

    # 逐实例执行2D-3D对齐
    for category, cat_insts in all_instances.items():
        cat_masks = all_masks.get(category, [])
        cat_fids = all_optimal_frame_ids.get(category, [])

        if len(cat_insts) != len(cat_masks):
            print(f"   [Warning] {category}: 实例/mask数量不匹配，跳过", flush=True)
            current_instance += len(cat_insts)
            continue

        for iid, (inst, masks) in enumerate(zip(cat_insts, cat_masks)):
            opt_fid = cat_fids[iid] if iid < len(cat_fids) else 0
            print(f"   [{current_instance+1}/{total_instances}] {category} #{iid}", flush=True)

            relationship = categories_and_relations.get(category) if categories_and_relations else None
            camera_pos = None
            if relationship and walls_info:
                rel_lower = relationship.lower().replace("_", " ")
                if "wall" in rel_lower:
                    extrinsic = vggt_prediction_results['extrinsics'][opt_fid]
                    camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]

            inst = refine_single_instance_combined(
                instance_info=inst,
                instance_masks=masks,
                optimal_frame_id=opt_fid,
                world_points=world_points,
                world_points_conf=world_points_conf,
                depths=vggt['depths'],
                extrinsics=vggt['extrinsics'],
                intrinsic=vggt['intrinsic'],
                colors=vggt['colors'],
                num_icp_iterations=args.stage4_iterations,
                temporal_radius=args.stage4_temporal_radius,
                instance_index=current_instance,
                total_instances=total_instances,
                instance_name=f"{category}_{iid}",
                use_mast3r=args.stage4_use_mast3r,
                mast3r_device='cuda',
                relationship=relationship,
                walls_info=walls_info,
                camera_pos=camera_pos,
            )
            cat_insts[iid] = inst
            current_instance += 1

    print(f"✅ Stage 4 完成", flush=True)
    return all_instances


def run_stage5(output_path, categories_and_relations, all_instances,
               vggt_prediction_results, all_optimal_frame_ids,
               wall_masks, vlm_checkpoint, stage1_json):
    """
    Stage 5: 高级语义精修 (5.1 + 5.2, 需 --enable_stage5 启用)

    注意: 5.0 基础精修 (floor/wall/embedded) 已移至 main() 中始终执行，
    与 main.py 行为一致。此函数只处理 5.1 和 5.2。
    """
    print("\n" + "=" * 70, flush=True)
    print("🚀 Stage 5: 高级语义精修 (5.1 + 5.2)", flush=True)
    print("=" * 70, flush=True)

    walls_info = get_walls_info(vggt_prediction_results['world_points'], wall_masks)
    refined_relations = dict(categories_and_relations)

    has_other_objects = any(
        rel == "supported by other objects"
        for rel in categories_and_relations.values()
    )

    if has_other_objects:
        print("\n   📍 5.1: 细化 'supported by other objects' 关系", flush=True)
        scene_id = os.path.splitext(os.path.basename(stage1_json))[0]
        refined_json = os.path.join(output_path, f"{scene_id}_refined.json")

        from tools.refine_other_objects_relations import refine_other_objects_relations
        refined_relations = refine_other_objects_relations(
            stage1_json_path=stage1_json,
            output_json_path=refined_json,
            scene_dir=output_path,
            vlm_checkpoint=vlm_checkpoint,
            optimal_frames_dir=os.path.join(output_path, 'optimal_frames'),
            keyframes_dir=os.path.join(output_path, 'keyframes'),
        )
    else:
        print("\n   📍 5.1: 无 'supported by other objects' 关系，跳过细化", flush=True)

    has_inter_object = any(
        rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel
        for rel in refined_relations.values()
    )

    if has_inter_object:
        print("\n   📍 5.2: 物体间支撑关系空间位置精修", flush=True)
        from tools.refine_inter_object_placement import refine_inter_object_relations
        all_instances = refine_inter_object_relations(
            all_instances, refined_relations,
            walls_info=walls_info, verbose=True,
            vlm_checkpoint=vlm_checkpoint,
            scene_dir=output_path,
        )
    else:
        print("\n   📍 5.2: 无物体间支撑关系，跳过精修", flush=True)

    print(f"✅ Stage 5 完成", flush=True)
    return all_instances, refined_relations


def save_final_glb(all_instances, output_path, filename="final_scene.glb"):
    """
    保存最终GLB文件

    流程:
      1. 将每个实例的mesh应用变换矩阵T
      2. 添加到trimesh.Scene（node_name格式: {category}_{id}）
      3. z-up → y-up 变换（GLB标准坐标系）
      4. 导出GLB

    参数:
        all_instances: 精修后的3D资产实例
        output_path: 输出目录
        filename: 输出文件名
    返回:
        GLB文件路径
    """
    scene = trimesh.Scene()
    for category, category_instances in all_instances.items():
        for i, instance_info in enumerate(category_instances):
            mesh = instance_info['original_mesh']
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(instance_info['T'])
            scene.add_geometry(transformed_mesh, node_name=f"{category}_{i}")
    # z-up → y-up (GLB标准)
    scene.apply_transform(np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ]))
    glb_path = os.path.join(output_path, filename)
    scene.export(glb_path)
    print(f"💾 GLB 已保存: {glb_path}", flush=True)
    return glb_path


def main(args):
    """主流水线入口"""
    total_start = time.time()

    if not os.path.exists(args.input_video):
        raise FileNotFoundError(f"输入视频未找到: {args.input_video}")

    os.makedirs(args.output_path, exist_ok=True)
    
    # 配置日志输出
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(args.output_path, f"mainv2_{timestamp}.log")
    
    # 配置日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 清除已有handler

    # 格式化器: [时间戳] [级别] 消息
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    # 文件handler
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 定义一个方便的print替代函数，同时输出到日志和控制台
    def log_print(msg, flush=False):
        logger.info(msg)
    
    # 替换print为log_print的功能
    import builtins
    original_print = builtins.print
    def new_print(*args, sep=' ', end='\n', file=sys.stdout, flush=False):
        msg = sep.join(str(arg) for arg in args) + end
        if not msg.endswith('\n'):
            msg += '\n'
        logger.info(msg.rstrip('\n'))
        original_print(*args, sep=sep, end=end, file=file, flush=flush)
    builtins.print = new_print
    
    log_print(f"📋 开始运行 ReplicateAnyScene V2")
    log_print(f"=" * 70)
    log_print(f"输入视频: {args.input_video}")
    log_print(f"输出目录: {args.output_path}")
    log_print(f"3D重建模型: {args.vggt_model}")
    log_print(f"日志文件: {log_filename}")
    log_print(f"=" * 70)

    # 自动查找VLM模型
    vlm_checkpoint = _resolve_vlm_checkpoint(args.vlm_checkpoint)
    if vlm_checkpoint is None:
        print("❌ 未找到VLM模型，请通过 --vlm_checkpoint 指定", flush=True)
        sys.exit(1)
    print(f"🤖 VLM模型: {vlm_checkpoint}", flush=True)

    # ── Stage 1: VGGT引导的智能物体发现 ──
    stage1_json, categories_and_relations = run_stage1(
        args.input_video, args.output_path, vlm_checkpoint,
        max_frames_stage1=args.max_frames_stage1,
    )

    # ── Stage 2: VGGT 3D重建 + SAM3空间去重 ──
    vggt_prediction_results, deduplicated_all_masks, wall_masks, floor_masks = run_stage2(
        args.input_video, args.output_path, categories_and_relations,
        max_frames=args.max_frames, vggt_model_type=args.vggt_model,
    )

    # ── Stage 3: 最优视角资产生成 + 多票验证 ──
    all_instances, all_optimal_frame_ids = run_stage3(
        args.output_path, vggt_prediction_results, deduplicated_all_masks,
    )

    # ── 基础精修: floor/wall/embedded (始终执行，与main.py一致，在Stage4之前) ──
    walls_info = get_walls_info(vggt_prediction_results['world_points'], wall_masks)

    for category, category_instances in all_instances.items():
        relationship = categories_and_relations[category]
        for instance_id, (optimal_frame_id, instance_info) in enumerate(zip(all_optimal_frame_ids[category], category_instances)):
            if relationship == "supported_by_floor" or relationship == "supported by floor":
                print(f"  基础精修: {category}_{instance_id} → supported by floor", flush=True)
                instance_info = refine_supported_by_floor_object(instance_info)
            elif relationship == "embedded_in_wall" or relationship == "embedded in wall":
                print(f"  基础精修: {category}_{instance_id} → embedded in wall", flush=True)
                instance_info = refine_embedded_in_wall_object(instance_info, walls_info)
            elif relationship == "attached_to_wall" or relationship == "attached to wall":
                print(f"  基础精修: {category}_{instance_id} → attached to wall", flush=True)
                extrinsic = vggt_prediction_results['extrinsics'][optimal_frame_id]
                camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
                instance_info = refine_attached_to_wall_object(instance_info, walls_info, camera_pos)
            else:
                continue
            category_instances[instance_id] = instance_info

    # 保存基础精修后的结果 (供 run_post_pipeline.py 作为固定起点)
    save_final_glb(all_instances, args.output_path, "final_scene_base.glb")

    import pickle
    pkl_path = os.path.join(args.output_path, "all_instances.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'all_instances': all_instances,
            'all_optimal_frame_ids': all_optimal_frame_ids,
            'categories_and_relations': categories_and_relations,
            'walls_info': walls_info,
        }, f)
    print(f"💾 all_instances.pkl 已保存: {pkl_path}", flush=True)

    # ── Stage 4: 迭代视觉-空间对齐 (可选，默认关闭) ──
    if args.enable_stage4:
        all_instances = run_stage4(
            args.output_path, vggt_prediction_results, all_instances, args,
            categories_and_relations=categories_and_relations,
            walls_info=walls_info,
        )
        from tools.refine_inter_object_placement import resolve_penetrations
        all_instances = resolve_penetrations(all_instances, categories_and_relations, verbose=True)
    else:
        print("\n⏭️  Stage 4 已跳过 (使用 --enable_stage4 启用)", flush=True)

    # ── Stage 5: 高级语义精修 (可选，默认关闭) ──
    refined_relations = dict(categories_and_relations)
    if args.enable_stage5:
        # 5.1: 细化 "supported by other objects" 关系
        has_other_objects = any(
            rel == "supported by other objects"
            for rel in categories_and_relations.values()
        )
        if has_other_objects:
            print("\n   📍 5.1: 细化 'supported by other objects' 关系", flush=True)
            scene_id = os.path.splitext(os.path.basename(stage1_json))[0]
            refined_json = os.path.join(args.output_path, f"{scene_id}_refined.json")

            from tools.refine_other_objects_relations import refine_other_objects_relations
            refined_relations = refine_other_objects_relations(
                stage1_json_path=stage1_json,
                output_json_path=refined_json,
                scene_dir=args.output_path,
                vlm_checkpoint=vlm_checkpoint,
                optimal_frames_dir=os.path.join(args.output_path, 'optimal_frames'),
                keyframes_dir=os.path.join(args.output_path, 'keyframes'),
            )
        else:
            print("\n   📍 5.1: 无 'supported by other objects' 关系，跳过细化", flush=True)

        # 5.2: 物体间支撑关系空间位置精修
        has_inter_object = any(
            rel.startswith("supported by ") and "floor" not in rel and "other objects" not in rel
            for rel in refined_relations.values()
        )
        if has_inter_object:
            print("\n   📍 5.2: 物体间支撑关系空间位置精修", flush=True)
            from tools.refine_inter_object_placement import refine_inter_object_relations
            all_instances = refine_inter_object_relations(
                all_instances, refined_relations,
                walls_info=walls_info, verbose=True,
                vlm_checkpoint=vlm_checkpoint,
                scene_dir=args.output_path,
            )
        else:
            print("\n   📍 5.2: 无物体间支撑关系，跳过精修", flush=True)
    else:
        print("\n⏭️  Stage 5 高级精修已跳过 (使用 --enable_stage5 启用)", flush=True)

    # ── 保存最终结果 ──
    glb_path = save_final_glb(all_instances, args.output_path)

    # 保存最终关系JSON
    with open(os.path.join(args.output_path, "final_relations.json"), 'w') as f:
        json.dump(refined_relations, f, indent=2, ensure_ascii=False)

    total_time = time.time() - total_start
    print("\n" + "=" * 70, flush=True)
    print(f"🎉 全部完成! 耗时 {total_time:.1f}s", flush=True)
    print(f"📂 输出目录: {args.output_path}", flush=True)
    print(f"📦 最终GLB: {glb_path}", flush=True)
    print("=" * 70, flush=True)
    print(f"📝 运行日志已保存到: {log_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReplicateAnyScene V2 — 完整自动化流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_video", type=str, default=None,
                        help="输入视频路径")
    parser.add_argument("--input_images", type=str, default=None,
                        help="输入图片目录路径 (与--input_video二选一)")
    parser.add_argument("--output_path", type=str, default=None,
                        help="输出目录 (默认: ./output_v2/{video_stem})")
    parser.add_argument("--vlm_checkpoint", type=str, default=None,
                        help="VLM模型路径 (默认: /mnt/data/lza/models/Qwen3.5-9B)")
    parser.add_argument("--max_frames", type=int, default=160,
                        help="VGGT最大帧数 (Stage 2, 默认160)")
    parser.add_argument("--vggt_model", type=str, default="vggt",
                        choices=["vggt", "vggt_omega", "vggt4d"],
                        help="3D重建模型选择: vggt(默认), vggt_omega, vggt4d")
    parser.add_argument("--max_frames_stage1", type=int, default=10,
                        help="Stage1采样关键帧数 (默认10)")
    parser.add_argument("--enable_stage4", action="store_true",
                        help="启用Stage 4视觉-空间对齐 (默认关闭)")
    parser.add_argument("--enable_stage5", action="store_true",
                        help="启用Stage 5语义感知场景精修 (默认关闭)")
    parser.add_argument("--stage4_iterations", type=int, default=8,
                        help="Stage4 ICP迭代次数 (默认8)")
    parser.add_argument("--stage4_temporal_radius", type=int, default=5,
                        help="Stage4 时序邻域半径 (默认5)")
    parser.add_argument("--stage4_use_mast3r", action="store_true",
                        help="Stage4 使用MASt3R匹配 (需要GPU)")

    args = parser.parse_args()

    if args.input_images and args.input_video:
        parser.error("--input_video 和 --input_images 不能同时使用")
    if not args.input_images and not args.input_video:
        parser.error("必须指定 --input_video 或 --input_images")

    if args.input_images:
        if not os.path.isdir(args.input_images):
            parser.error(f"图片目录不存在: {args.input_images}")
        args.input_video = args.input_images

    if args.output_path is None:
        video_stem = os.path.splitext(os.path.basename(args.input_video))[0]
        args.output_path = os.path.join(".", "output_v2", f"{video_stem}_{args.vggt_model}")
    
    try:
        main(args)
    except Exception as e:
        # 确保异常也被记录到日志中
        import traceback
        
        # 尝试先初始化日志记录器（如果还没初始化）
        if not logging.getLogger().handlers:
            try:
                os.makedirs(args.output_path, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_filename = os.path.join(args.output_path, f"mainv2_error_{timestamp}.log")
                logging.basicConfig(
                    level=logging.INFO,
                    format='[%(asctime)s] [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler(log_filename, encoding='utf-8'),
                        logging.StreamHandler()
                    ]
                )
            except:
                pass
        
        logging.error(f"运行过程中出现异常: {e}")
        logging.error(traceback.format_exc())
        print(f"❌ 错误详情已记录到日志文件中")
        raise
