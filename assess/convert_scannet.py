import os
import argparse
import cv2
import numpy as np


def convert_scannet_to_ras(scannet_scene_dir, output_dir, max_frames=None, depth_scale=1000.0):
    os.makedirs(os.path.join(output_dir, 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'extrinsics'), exist_ok=True)

    intrinsic_path = os.path.join(scannet_scene_dir, 'intrinsic.txt')
    if not os.path.exists(intrinsic_path):
        print(f"⚠️  未找到 intrinsic.txt: {intrinsic_path}")
        return

    intrinsic_orig = np.loadtxt(intrinsic_path)
    intrinsic = intrinsic_orig.copy()

    color_files = sorted([f for f in os.listdir(scannet_scene_dir) if f.endswith('.jpg')])
    if max_frames and max_frames < len(color_files):
        indices = np.linspace(0, len(color_files) - 1, max_frames, dtype=int)
        color_files = [color_files[i] for i in indices]

    sample_img = cv2.imread(os.path.join(scannet_scene_dir, color_files[0]))
    if sample_img is None:
        print("⚠️  无法读取图像")
        return

    h_orig, w_orig = sample_img.shape[:2]
    scale = 1.0
    if max(h_orig, w_orig) > 640:
        scale = 640.0 / max(h_orig, w_orig)
        intrinsic[0, 0] *= scale
        intrinsic[0, 2] *= scale
        intrinsic[1, 1] *= scale
        intrinsic[1, 2] *= scale

    np.savetxt(os.path.join(output_dir, 'intrinsic.txt'), intrinsic)

    print(f"转换 {len(color_files)} 帧 (scale={scale:.4f})...")
    print(f"原始内参: fx={intrinsic_orig[0,0]:.2f}, fy={intrinsic_orig[1,1]:.2f}, cx={intrinsic_orig[0,2]:.2f}, cy={intrinsic_orig[1,2]:.2f}")
    print(f"缩放内参: fx={intrinsic[0,0]:.2f}, fy={intrinsic[1,1]:.2f}, cx={intrinsic[0,2]:.2f}, cy={intrinsic[1,2]:.2f}")

    for i, color_file in enumerate(color_files):
        frame_id = os.path.splitext(color_file)[0]

        color_src = os.path.join(scannet_scene_dir, color_file)
        depth_src = os.path.join(scannet_scene_dir, frame_id + '.png')
        pose_src = os.path.join(scannet_scene_dir, frame_id + '.txt')

        color_img = cv2.imread(color_src)
        if color_img is None:
            continue

        if scale != 1.0:
            h, w = color_img.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            color_img = cv2.resize(color_img, (new_w, new_h))

        cv2.imwrite(os.path.join(output_dir, 'color', f'{i}.jpg'), color_img)

        if os.path.exists(depth_src):
            depth_img = cv2.imread(depth_src, cv2.IMREAD_UNCHANGED)
            if depth_img is not None:
                depth_m = depth_img.astype(np.float32) / depth_scale
                depth_out = (depth_m * 1000.0).astype(np.uint16)
                if depth_img.shape[:2] != color_img.shape[:2]:
                    depth_out = cv2.resize(depth_out, (color_img.shape[1], color_img.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(output_dir, 'depth', f'{i}.png'), depth_out)

        if os.path.exists(pose_src):
            pose = np.loadtxt(pose_src)
            np.savetxt(os.path.join(output_dir, 'extrinsics', f'{i}.txt'), pose)

    print(f"✅ 转换完成: {output_dir}")
    print(f"   帧数: {len(color_files)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert ScanNet scene to RAS assessment format")
    parser.add_argument("--scannet_scene", type=str, required=True, help="ScanNet scene directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Max frames to convert")
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale factor (ScanNet default: 1000)")
    args = parser.parse_args()
    convert_scannet_to_ras(args.scannet_scene, args.output_dir, args.max_frames, args.depth_scale)
