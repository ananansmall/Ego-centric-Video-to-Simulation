import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_fn = None


def compute_psnr(img1, img2):
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
    return peak_signal_noise_ratio(img1, img2)


def compute_ssim(img1, img2):
    if img1.dtype != np.uint8:
        img1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)
    if img1.ndim == 3:
        return structural_similarity(img1, img2, channel_axis=-1)
    return structural_similarity(img1, img2)


def compute_psnr_masked(gt_img, rend_img, mask):
    gt_m = gt_img.copy()
    rend_m = rend_img.copy()
    gt_m[~mask] = 0
    rend_m[~mask] = 0
    return peak_signal_noise_ratio(gt_m, rend_m)


def compute_ssim_masked(gt_img, rend_img, mask):
    gt_m = gt_img.copy()
    rend_m = rend_img.copy()
    gt_m[~mask] = 0
    rend_m[~mask] = 0
    if gt_m.ndim == 3:
        return structural_similarity(gt_m, rend_m, channel_axis=-1)
    return structural_similarity(gt_m, rend_m)


def _get_lpips_fn(net='alex'):
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        _lpips_fn = lpips.LPIPS(net=net)
        if __import__('torch').cuda.is_available():
            _lpips_fn = _lpips_fn.cuda()
    return _lpips_fn


def compute_lpips(img1, img2, net='alex'):
    import torch
    fn = _get_lpips_fn(net)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    if torch.cuda.is_available():
        t1 = t1.cuda()
        t2 = t2.cuda()
    return fn(t1, t2).item()


def compute_lpips_masked(gt_img, rend_img, mask, net='alex'):
    import torch
    fn = _get_lpips_fn(net)
    gt_m = gt_img.copy()
    rend_m = rend_img.copy()
    gt_m[~mask] = 0
    rend_m[~mask] = 0
    t1 = torch.from_numpy(gt_m).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    t2 = torch.from_numpy(rend_m).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    if torch.cuda.is_available():
        t1 = t1.cuda()
        t2 = t2.cuda()
    return fn(t1, t2).item()


def evaluate_visual_quality(rendered_dir, gt_dir, sample_count=None, compute_lpips_flag=True):
    import os
    import glob

    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.jpg")) + glob.glob(os.path.join(gt_dir, "*.png")))
    rendered_files = sorted(glob.glob(os.path.join(rendered_dir, "*.jpg")) + glob.glob(os.path.join(rendered_dir, "*.png")))

    if not gt_files:
        print("⚠️  未找到GT图像文件")
        return {}

    if sample_count and sample_count < len(gt_files):
        indices = np.linspace(0, len(gt_files) - 1, sample_count, dtype=int)
        gt_files = [gt_files[i] for i in indices]
        rendered_files = [rendered_files[i] for i in indices if i < len(rendered_files)]

    if compute_lpips_flag:
        _get_lpips_fn()

    psnr_list, ssim_list, lpips_list = [], [], []
    psnr_masked_list, ssim_masked_list, lpips_masked_list = [], [], []
    coverage_list = []

    for gt_path in gt_files:
        fname = os.path.basename(gt_path)
        rend_path = os.path.join(rendered_dir, fname)
        if not os.path.exists(rend_path):
            name_no_ext = os.path.splitext(fname)[0]
            for ext in ['.jpg', '.png']:
                candidate = os.path.join(rendered_dir, name_no_ext + ext)
                if os.path.exists(candidate):
                    rend_path = candidate
                    break
            else:
                continue

        gt_img = cv2.imread(gt_path)
        rend_img = cv2.imread(rend_path)
        if gt_img is None or rend_img is None:
            continue

        h, w = gt_img.shape[:2]
        rend_img = cv2.resize(rend_img, (w, h))

        psnr_list.append(compute_psnr(gt_img, rend_img))
        ssim_list.append(compute_ssim(gt_img, rend_img))

        rend_mask = rend_img.sum(axis=2) > 0
        gt_mask = gt_img.sum(axis=2) > 0
        overlap_mask = rend_mask & gt_mask
        coverage_list.append(float(rend_mask.mean()))

        if overlap_mask.sum() > 100:
            psnr_masked_list.append(compute_psnr_masked(gt_img, rend_img, overlap_mask))
            ssim_masked_list.append(compute_ssim_masked(gt_img, rend_img, overlap_mask))

        if compute_lpips_flag:
            try:
                lpips_list.append(compute_lpips(gt_img, rend_img))
                if overlap_mask.sum() > 100:
                    lpips_masked_list.append(compute_lpips_masked(gt_img, rend_img, overlap_mask))
            except Exception as e:
                print(f"   LPIPS计算失败: {e}")

    results = {}
    if psnr_list:
        results['PSNR'] = float(np.mean(psnr_list))
        results['SSIM'] = float(np.mean(ssim_list))
        results['PSNR_per_frame'] = psnr_list
        results['SSIM_per_frame'] = ssim_list
    if lpips_list:
        results['LPIPS'] = float(np.mean(lpips_list))
        results['LPIPS_per_frame'] = lpips_list
    if psnr_masked_list:
        results['PSNR_masked'] = float(np.mean(psnr_masked_list))
        results['SSIM_masked'] = float(np.mean(ssim_masked_list))
        results['PSNR_masked_per_frame'] = psnr_masked_list
        results['SSIM_masked_per_frame'] = ssim_masked_list
    if lpips_masked_list:
        results['LPIPS_masked'] = float(np.mean(lpips_masked_list))
        results['LPIPS_masked_per_frame'] = lpips_masked_list
    if coverage_list:
        results['render_coverage'] = float(np.mean(coverage_list))
        results['render_coverage_per_frame'] = coverage_list

    return results
