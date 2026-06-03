import os
os.environ["LIDRA_SKIP_INIT"] = "true"
import argparse
import torch
import cv2
import numpy as np
import json
import re
import shutil
import trimesh


def prepare_scannet_images(scannet_scene_dir, work_dir, max_frames=None):
    color_src = os.path.join(scannet_scene_dir)
    color_dst = os.path.join(work_dir, 'input_images')
    os.makedirs(color_dst, exist_ok=True)

    images = sorted([f for f in os.listdir(color_src) if f.endswith('.jpg')])
    if not images:
        raise ValueError(f"No .jpg files found in {color_src}")

    if max_frames and max_frames < len(images):
        indices = np.linspace(0, len(images) - 1, max_frames).astype(int)
        images = [images[i] for i in indices]

    for i, img_name in enumerate(images):
        src = os.path.join(color_src, img_name)
        dst = os.path.join(color_dst, f"{i:05d}.jpg")
        img = cv2.imread(src)
        h, w = img.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        cv2.imwrite(dst, img)

    print(f"✅ 准备了 {len(images)} 张图片到 {color_dst}")
    return color_dst


def generate_category_json(categories, output_path):
    default_relations = {
        "chair": "supported by floor",
        "table": "supported by floor",
        "desk": "supported by floor",
        "bed": "supported by floor",
        "sofa": "supported by floor",
        "cabinet": "supported by floor",
        "shelf": "supported by floor",
        "bookshelf": "supported by floor",
        "door": "embedded in wall",
        "window": "embedded in wall",
        "picture": "attached to wall",
        "mirror": "attached to wall",
        "lamp": "supported by other objects",
        "pillow": "supported by other objects",
        "curtain": "attached to wall",
        "plant": "supported by floor",
        "toilet": "supported by floor",
        "sink": "supported by other objects",
        "bathtub": "supported by floor",
        "counter": "supported by floor",
        "refrigerator": "supported by floor",
        "oven": "supported by floor",
        "microwave": "supported by other objects",
        "dishwasher": "supported by other objects",
        "trash can": "supported by floor",
        "clothes": "supported by other objects",
        "towel": "attached to wall",
        "rug": "supported by floor",
        "floor": "supported by floor",
        "wall": "embedded in wall",
        "ceiling": "embedded in wall",
        "box": "supported by other objects",
        "bag": "supported by other objects",
        "bottle": "supported by other objects",
        "bowl": "supported by other objects",
        "cup": "supported by other objects",
        "mug": "supported by other objects",
        "plate": "supported by other objects",
        "keyboard": "supported by other objects",
        "monitor": "supported by other objects",
        "tv": "supported by other objects",
        "book": "supported by other objects",
        "paper": "supported by other objects",
        "backpack": "supported by other objects",
    }

    cat_dict = {}
    for cat in categories:
        cat_lower = cat.lower().strip()
        cat_dict[cat_lower] = default_relations.get(cat_lower, "supported by other objects")

    with open(output_path, 'w') as f:
        json.dump(cat_dict, f, indent=2)

    print(f"✅ 生成类别JSON: {output_path}")
    print(f"   类别: {list(cat_dict.keys())}")
    return output_path


def main(args):
    from src.models import load_vggt_model, load_sam3_image_model, load_sam3_video_model, unload_model
    from src.utils import load_video_frames, vis_instance_masks
    from src.geometry_utils import align_to_room_coordinate_system, align_vggt_predictions, get_optimal_view_frame_id, get_walls_info
    from src.vggt_predict import vggt_predict
    from src.object_segmentation import segment_wall_and_floor, segment_and_track
    from src.sg_deduplication import self_category_deduplicate, cross_category_deduplicate
    from src.instance_generation import generate_3d_asset_in_subprocess
    from src.sp_refinement import refine_supported_by_floor_object, refine_attached_to_wall_object, refine_embedded_in_wall_object

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_path, exist_ok=True)

    # Step 1: Prepare input images from ScanNet directory
    print("\n" + "=" * 50)
    print("Step 1: 准备输入图片")
    print("=" * 50)
    input_dir = prepare_scannet_images(args.scannet_scene, args.output_path, args.max_frames)

    # Step 2: Category JSON
    print("\n" + "=" * 50)
    print("Step 2: 准备类别JSON")
    print("=" * 50)
    if args.category_path and os.path.exists(args.category_path):
        cat_path = args.category_path
        print(f"使用指定JSON: {cat_path}")
    elif args.categories:
        cat_path = os.path.join(args.output_path, 'categories.json')
        generate_category_json(args.categories, cat_path)
    else:
        cat_path = os.path.join(args.output_path, 'categories.json')
        default_cats = ["chair", "table", "door", "window", "cabinet", "bed", "sofa", "shelf"]
        generate_category_json(default_cats, cat_path)

    with open(cat_path, 'r') as f:
        categories_and_relations = json.load(f)
    detected_categories = list(categories_and_relations.keys())
    print(f"Detected categories: {detected_categories}")

    # Step 3: VGGT prediction
    print("\n" + "=" * 50)
    print("Step 3: VGGT 3D预测")
    print("=" * 50)
    frames = load_video_frames(input_dir, args.max_frames).to(device)
    print(f"Loaded {len(frames)} frames for processing.")
    vggt_model = load_vggt_model().to(device)
    vggt_prediction_results = vggt_predict(frames, vggt_model)
    vggt_model = unload_model(vggt_model)

    # Step 4: Align to room coordinate system
    print("\n" + "=" * 50)
    print("Step 4: 房间坐标系对齐")
    print("=" * 50)
    sam3_image_model = load_sam3_image_model()
    wall_masks, floor_masks = segment_wall_and_floor(vggt_prediction_results['colors'], sam3_image_model)
    R, t = align_to_room_coordinate_system(vggt_prediction_results['world_points'], wall_masks, floor_masks)
    vggt_prediction_results = align_vggt_predictions(vggt_prediction_results, R, t)

    # Save intermediate results
    os.makedirs(os.path.join(args.output_path, 'color'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(args.output_path, 'extrinsics'), exist_ok=True)
    np.savetxt(os.path.join(args.output_path, 'intrinsic.txt'), vggt_prediction_results['intrinsic'])
    vggt_prediction_results['point_cloud_data'].export(os.path.join(args.output_path, 'point_cloud.ply'))
    for i, image in enumerate(vggt_prediction_results['colors']):
        cv2.imwrite(os.path.join(args.output_path, 'color', f"{i}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    for i, depth in enumerate(vggt_prediction_results['depths']):
        cv2.imwrite(os.path.join(args.output_path, 'depth', f"{i}.png"), (depth * 1000).astype(np.uint16))
    for i, extrinsic in enumerate(vggt_prediction_results['extrinsics']):
        np.savetxt(os.path.join(args.output_path, 'extrinsics', f"{i}.txt"), extrinsic)
    sam3_image_model = unload_model(sam3_image_model)

    # Step 5: SAM3 segmentation and tracking
    print("\n" + "=" * 50)
    print("Step 5: SAM3 分割与跟踪")
    print("=" * 50)
    sam3_video_model = load_sam3_video_model()
    video_path = os.path.join(args.output_path, 'color')
    response = sam3_video_model.handle_request(
        request=dict(type="start_session", resource_path=video_path)
    )
    session_id = response["session_id"]

    all_masks = {}
    for category in detected_categories:
        print(f"Segmenting and tracking category: {category}")
        category_masks = segment_and_track(category, sam3_video_model, session_id)
        deduplicated_category_masks = self_category_deduplicate(
            category_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf']
        )
        all_masks[category] = deduplicated_category_masks

    deduplicated_all_masks = cross_category_deduplicate(
        all_masks, vggt_prediction_results['world_points'], vggt_prediction_results['world_points_conf']
    )

    json_categories = set(categories_and_relations.keys())
    filtered_masks = {}
    for category, category_masks in deduplicated_all_masks.items():
        if category in json_categories:
            filtered_masks[category] = category_masks
        else:
            print(f"   🚫 白名单过滤: '{category}' 不在JSON配置中，已删除")
    deduplicated_all_masks = filtered_masks

    vis_instance_masks(vggt_prediction_results['colors'], deduplicated_all_masks,
                       os.path.join(args.output_path, 'instance_masks.mp4'))
    sam3_video_model = unload_model(sam3_video_model)

    # Step 6: 3D asset generation
    print("\n" + "=" * 50)
    print("Step 6: 3D资产生成")
    print("=" * 50)
    all_optimal_frame_ids = {}
    for category, category_masks in deduplicated_all_masks.items():
        all_optimal_frame_ids[category] = []
        for instance_masks in category_masks:
            optimal_frame_id = get_optimal_view_frame_id(vggt_prediction_results['world_points'], instance_masks)
            all_optimal_frame_ids[category].append(optimal_frame_id)

    all_instances = generate_3d_asset_in_subprocess(
        deduplicated_all_masks,
        all_optimal_frame_ids,
        vggt_prediction_results['colors'],
        vggt_prediction_results['world_points'],
        vggt_prediction_results['extrinsics'],
    )

    # Step 6.5: Asset verification
    print("\n" + "=" * 50)
    print("Step 6.5: 资产验证")
    print("=" * 50)
    from tools.asset_verifier import verify_all_instances
    all_instances = verify_all_instances(
        all_instances,
        all_optimal_frame_ids,
        deduplicated_all_masks,
        vggt_prediction_results['world_points'],
        vggt_prediction_results['world_points_conf'],
        min_votes=2,
    )

    # Step 7: Spatial refinement
    print("\n" + "=" * 50)
    print("Step 7: 空间关系优化")
    print("=" * 50)
    walls_info = get_walls_info(vggt_prediction_results['world_points'], wall_masks)

    for category, category_instances in all_instances.items():
        relationship = categories_and_relations[category]
        for instance_id, (optimal_frame_id, instance_info) in enumerate(
                zip(all_optimal_frame_ids[category], category_instances)):
            print(f"Refining {category}: {instance_id} with relationship: {relationship}")
            if relationship == "supported_by_floor":
                instance_info = refine_supported_by_floor_object(instance_info)
            elif relationship == "embedded_in_wall":
                instance_info = refine_embedded_in_wall_object(instance_info, walls_info)
            elif relationship == "attached_to_wall":
                extrinsic = vggt_prediction_results['extrinsics'][optimal_frame_id]
                camera_pos = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
                instance_info = refine_attached_to_wall_object(instance_info, walls_info, camera_pos)
            else:
                continue

    # Step 8: Save final scene
    print("\n" + "=" * 50)
    print("Step 8: 保存最终场景")
    print("=" * 50)
    scene = trimesh.Scene()
    for category, category_instances in all_instances.items():
        for i, instance_info in enumerate(category_instances):
            mesh = instance_info['original_mesh']
            transformed_mesh = mesh.copy()
            transformed_mesh.apply_transform(instance_info['T'])
            scene.add_geometry(transformed_mesh, node_name=f"{category}_{i}")
    scene.apply_transform(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]))
    scene.export(os.path.join(args.output_path, "final_scene.glb"))

    print(f"\n✅ 完成！输出目录: {args.output_path}")
    print(f"   final_scene.glb: {os.path.join(args.output_path, 'final_scene.glb')}")
    print(f"\n📊 运行评估:")
    print(f"   python -m assess.run_assessment --output_path {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReplicateAnyScene - ScanNet Test")
    parser.add_argument("--scannet_scene", type=str, required=True,
                        help="ScanNet scene directory (e.g. /path/to/scannet/posed_images/scene0161_00)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output directory (e.g. ./outputs/scannet_scene0161_00)")
    parser.add_argument("--category_path", type=str, default=None,
                        help="Category JSON path (optional, auto-generate if not provided)")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                        help="Category list (e.g. --categories chair table door)")
    parser.add_argument("--max_frames", type=int, default=25,
                        help="Max frames to process (default: 25)")
    args = parser.parse_args()

    if not os.path.exists(args.scannet_scene):
        raise FileNotFoundError(f"ScanNet scene directory not found: {args.scannet_scene}")

    main(args)
