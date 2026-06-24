"""
快速测试三阶段Z轴对齐逻辑 (不需要 GPU/SAM3)

用合成的 world_points 和 mask 数据测试:
  阶段1: align_to_room_coordinate_system (SAM3 "floor"/"wall" 文本提示)
  阶段2: align_via_objects (放宽阈值 + 只用floor + PCA)
  阶段3: align_via_large_plane (大平面 mask 拟合)

运行:
  cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
  python test_stage2_alignment.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.geometry_utils import (
    align_to_room_coordinate_system,
    align_via_objects,
    align_via_large_plane,
    _orient_floor_normal,
    _build_R_t_from_floor,
    get_plane_info,
    align_vggt_predictions,
)


def make_synthetic_scene(T=4, H=100, W=100, seed=42):
    """
    生成合成场景: 一个房间, floor 在 z=0, wall 在 x=2.

    VGGT 相机坐标系: z 轴朝前 (不是朝上), 所以 floor 法线在 y 方向.
    但为了测试简单, 我们直接用 z-up 坐标系, floor 法线 = [0,0,1].

    Returns:
        world_points: (T, H, W, 3)
        floor_masks: list of dicts
        wall_masks: list of dicts
    """
    rng = np.random.RandomState(seed)
    world_points = np.zeros((T, H, W, 3), dtype=np.float32)

    for t in range(T):
        # floor: z=0 平面, y in [0, 4], x in [-2, 2]
        for i in range(H):
            for j in range(W):
                x = (j / W - 0.5) * 4.0  # x in [-2, 2]
                y = (i / H) * 4.0         # y in [0, 4]
                z = 0.0
                # 加噪声
                z += rng.normal(0, 0.005)
                world_points[t, i, j] = [x, y, z]

    # floor mask: 下半部分 (y < 2)
    floor_mask = np.zeros((H, W), dtype=bool)
    floor_mask[H//2:, :] = True
    floor_masks = [{'frame_id': 0, 'mask': floor_mask}]

    # wall mask: 右边缘 (x ≈ 2), z in [0, 3]
    wall_mask = np.zeros((H, W), dtype=bool)
    wall_mask[:, W//2:] = True
    # 修改 wall 区域的点云: wall 在 x=2 平面
    for t in range(T):
        for i in range(H):
            for j in range(W//2, W):
                x = 2.0 + rng.normal(0, 0.005)
                y = (i / H) * 4.0
                z = (j - W//2) / (W//2) * 3.0
                world_points[t, i, j] = [x, y, z]
    wall_masks = [{'frame_id': 0, 'mask': wall_mask}]

    return world_points, floor_masks, wall_masks


def make_tilted_scene(T=4, H=100, W=100, seed=42):
    """
    生成倾斜场景: floor 法线不朝 z 轴, 模拟 VGGT 相机坐标系.
    floor 平面绕 x 轴旋转 30°, 法线 = [0, -sin30, cos30] = [0, -0.5, 0.866]
    """
    rng = np.random.RandomState(seed)
    world_points = np.zeros((T, H, W, 3), dtype=np.float32)

    angle = np.radians(30)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    for t in range(T):
        for i in range(H):
            for j in range(W):
                # 原始 floor 坐标 (z=0 平面)
                x0 = (j / W - 0.5) * 4.0
                y0 = (i / H) * 4.0
                z0 = 0.0
                # 绕 x 轴旋转
                y1 = y0 * cos_a - z0 * sin_a
                z1 = y0 * sin_a + z0 * cos_a
                z1 += rng.normal(0, 0.005)
                world_points[t, i, j] = [x0, y1, z1]

    floor_mask = np.zeros((H, W), dtype=bool)
    floor_mask[H//2:, :] = True
    floor_masks = [{'frame_id': 0, 'mask': floor_mask}]

    # wall: 绕 x 轴旋转后的 x=2 平面
    wall_mask = np.zeros((H, W), dtype=bool)
    wall_mask[:, W//2:] = True
    for t in range(T):
        for i in range(H):
            for j in range(W//2, W):
                x0 = 2.0 + rng.normal(0, 0.005)
                y0 = (i / H) * 4.0
                z0 = (j - W//2) / (W//2) * 3.0
                y1 = y0 * cos_a - z0 * sin_a
                z1 = y0 * sin_a + z0 * cos_a
                world_points[t, i, j] = [x0, y1, z1]
    wall_masks = [{'frame_id': 0, 'mask': wall_mask}]

    return world_points, floor_masks, wall_masks


def check_R_orthogonal(R, name="R"):
    """检查 R 是否正交"""
    err = np.linalg.norm(R @ R.T - np.eye(3))
    det = np.linalg.det(R)
    print(f"   {name}: det={det:.4f}, orth_err={err:.6f}", end="")
    if err < 1e-4 and abs(det - 1) < 1e-4:
        print(" ✅")
        return True
    else:
        print(" ❌")
        return False


def check_z_aligned(R, expected_cos=1.0, tol=0.1, name=""):
    """检查 R 的 z 轴是否对齐到世界 z 轴"""
    new_z = R @ np.array([0, 0, 1])
    cos_to_world_z = abs(new_z[2])
    print(f"   {name}z轴对齐: cos(z, world_z)={cos_to_world_z:.3f} (期望≈{expected_cos})", end="")
    if abs(cos_to_world_z - expected_cos) < tol:
        print(" ✅")
        return True
    else:
        print(" ❌")
        return False


def test_orient_floor_normal():
    """测试 _orient_floor_normal"""
    print("\n=== 测试 _orient_floor_normal ===")
    all_points = np.array([[0, 0, 1], [0, 0, 2], [1, 0, 1.5]])  # 场景在 z>0
    floor_centroid = np.array([0, 0, 0])  # floor 在 z=0

    # 法线朝上 [0,0,1] → 应保持
    n = np.array([0, 0, 1])
    result = _orient_floor_normal(n, floor_centroid, all_points)
    assert np.allclose(result, [0, 0, 1]), f"朝上应保持, got {result}"
    print("   朝上 [0,0,1] → 保持 ✅")

    # 法线朝下 [0,0,-1] → 应翻转
    n = np.array([0, 0, -1])
    result = _orient_floor_normal(n, floor_centroid, all_points)
    assert np.allclose(result, [0, 0, 1]), f"朝下应翻转, got {result}"
    print("   朝下 [0,0,-1] → 翻转 ✅")


def test_build_R_t_from_floor():
    """测试 _build_R_t_from_floor"""
    print("\n=== 测试 _build_R_t_from_floor ===")
    # 简单场景: floor 在 z=0, 法线 [0,0,1]
    world_points = np.zeros((2, 10, 10, 3), dtype=np.float32)
    for t in range(2):
        for i in range(10):
            for j in range(10):
                world_points[t, i, j] = [j - 5, i - 5, 0]

    floor_normal = np.array([0, 0, 1], dtype=np.float32)
    floor_centroid = np.array([0, 0, 0], dtype=np.float32)

    R, t = _build_R_t_from_floor(world_points, floor_normal, floor_centroid, wall_normal_1=None)
    print(f"   R={R.round(3).tolist()}, t={t.round(3).tolist()}")
    check_R_orthogonal(R, "R")
    check_z_aligned(R, name="")

    # floor 应该在 z=0
    assert abs(t[2]) < 0.01, f"floor z 应为 0, got t[2]={t[2]}"
    print("   floor z=0 ✅")


def test_stage1_align_to_room():
    """测试阶段1: align_to_room_coordinate_system"""
    print("\n=== 测试阶段1: align_to_room_coordinate_system ===")
    world_points, floor_masks, wall_masks = make_synthetic_scene()

    R, t = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
    R_is_identity = np.allclose(R, np.eye(3), atol=1e-6)
    t_is_zero = np.allclose(t, 0, atol=1e-6)

    print(f"   R={R.round(3).tolist()}")
    print(f"   t={t.round(3).tolist()}")
    print(f"   is_identity={R_is_identity}, is_zero={t_is_zero}")

    if R_is_identity and t_is_zero:
        print("   ❌ 阶段1失败 (合成数据应该成功)")
        return False
    else:
        print("   ✅ 阶段1成功")
        check_R_orthogonal(R, "R")
        check_z_aligned(R, name="")
        return True


def test_stage2_align_via_objects():
    """测试阶段2: align_via_objects (用倾斜场景)"""
    print("\n=== 测试阶段2: align_via_objects ===")
    world_points, floor_masks, wall_masks = make_tilted_scene()

    # 先测试阶段1是否失败 (倾斜场景可能阶段1成功, 也可能失败)
    R1, t1 = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
    R1_is_identity = np.allclose(R1, np.eye(3), atol=1e-6)
    print(f"   阶段1: is_identity={R1_is_identity}")

    # 测试阶段2
    R, t, info = align_via_objects(world_points, wall_masks, floor_masks)
    R_is_identity = np.allclose(R, np.eye(3), atol=1e-6)
    t_is_zero = np.allclose(t, 0, atol=1e-6)

    print(f"   阶段2: method={info.get('method')}, is_identity={R_is_identity}")
    print(f"   R={R.round(3).tolist()}")
    print(f"   t={t.round(3).tolist()}")
    print(f"   info={info}")

    if R_is_identity and t_is_zero:
        print(f"   ❌ 阶段2失败: {info.get('reason')}")
        return False
    else:
        print("   ✅ 阶段2成功")
        check_R_orthogonal(R, "R")
        # 倾斜30°的场景, 对齐后 z 轴应该接近世界 z
        check_z_aligned(R, name="")
        return True


def test_stage3_align_via_large_plane():
    """测试阶段3: align_via_large_plane"""
    print("\n=== 测试阶段3: align_via_large_plane ===")
    world_points, floor_masks, wall_masks = make_tilted_scene()

    # 用 floor_masks 作为 large_plane_masks 测试
    R, t, info = align_via_large_plane(world_points, floor_masks)
    R_is_identity = np.allclose(R, np.eye(3), atol=1e-6)
    t_is_zero = np.allclose(t, 0, atol=1e-6)

    print(f"   阶段3: method={info.get('method')}, is_identity={R_is_identity}")
    print(f"   R={R.round(3).tolist()}")
    print(f"   t={t.round(3).tolist()}")
    print(f"   info={info}")

    if R_is_identity and t_is_zero:
        print(f"   ❌ 阶段3失败: {info.get('reason')}")
        return False
    else:
        print("   ✅ 阶段3成功")
        check_R_orthogonal(R, "R")
        check_z_aligned(R, name="")
        return True


def test_empty_masks():
    """测试空 mask 的情况"""
    print("\n=== 测试空 mask (应返回 identity) ===")
    world_points = np.random.randn(2, 10, 10, 3).astype(np.float32)

    R, t, info = align_via_objects(world_points, [], [])
    assert np.allclose(R, np.eye(3)) and np.allclose(t, 0), "空 mask 应返回 identity"
    print(f"   align_via_objects 空 mask: reason={info.get('reason')} ✅")

    R, t, info = align_via_large_plane(world_points, [])
    assert np.allclose(R, np.eye(3)) and np.allclose(t, 0), "空 mask 应返回 identity"
    print(f"   align_via_large_plane 空 mask: reason={info.get('reason')} ✅")


def test_align_vggt_predictions():
    """测试 align_vggt_predictions 是否正确应用 R, t"""
    print("\n=== 测试 align_vggt_predictions ===")
    world_points, floor_masks, wall_masks = make_synthetic_scene()

    # 构造 predictions 字典
    predictions = {
        'extrinsics': np.tile(np.eye(4), (2, 1, 1)).astype(np.float32),
        'world_points': world_points.copy(),
        'point_cloud_data': trimesh.PointCloud(world_points.reshape(-1, 3)),
    }

    R, t = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
    if np.allclose(R, np.eye(3)):
        print("   跳过 (阶段1失败)")
        return

    predictions = align_vggt_predictions(predictions, R, t)

    # 检查 world_points 是否正确变换
    new_wp = predictions['world_points']
    old_wp = world_points
    expected = old_wp @ R.T + t
    err = np.linalg.norm(new_wp - expected) / max(np.linalg.norm(expected), 1e-6)
    print(f"   world_points 变换误差: {err:.6f}", end="")
    if err < 1e-4:
        print(" ✅")
    else:
        print(" ❌")


def test_high_noise_scene():
    """测试高噪声场景 (mean_distance > 0.02 但 < 0.05)"""
    print("\n=== 测试高噪声场景 (阶段1失败, 阶段2应成功) ===")
    rng = np.random.RandomState(123)
    T, H, W = 2, 80, 80
    world_points = np.zeros((T, H, W, 3), dtype=np.float32)

    # floor 有较大噪声 (0.03m)
    for t in range(T):
        for i in range(H):
            for j in range(W):
                x = (j / W - 0.5) * 4.0
                y = (i / H) * 4.0
                z = rng.normal(0, 0.03)  # 噪声 0.03m > 0.02 阈值
                world_points[t, i, j] = [x, y, z]

    floor_mask = np.zeros((H, W), dtype=bool)
    floor_mask[H//2:, :] = True
    floor_masks = [{'frame_id': 0, 'mask': floor_mask}]
    wall_masks = []

    # 阶段1: 应该失败 (mean_distance ≈ 0.03 > 0.02)
    R1, t1 = align_to_room_coordinate_system(world_points, wall_masks, floor_masks)
    r1_id = np.allclose(R1, np.eye(3), atol=1e-6)
    print(f"   阶段1: is_identity={r1_id} (期望 True, 因为噪声>0.02)")

    # 阶段2: 应该成功 (放宽到 0.05)
    R2, t2, info2 = align_via_objects(world_points, wall_masks, floor_masks,
                                       floor_mean_distance_thres=0.05)
    r2_id = np.allclose(R2, np.eye(3), atol=1e-6)
    print(f"   阶段2: is_identity={r2_id}, method={info2.get('method')}")
    if not r2_id:
        print("   ✅ 阶段2成功 (放宽阈值后通过)")
        check_z_aligned(R2, name="")
    else:
        print(f"   ❌ 阶段2也失败: {info2.get('reason')}")


if __name__ == "__main__":
    import trimesh

    print("=" * 60)
    print("三阶段Z轴对齐快速测试")
    print("=" * 60)

    test_orient_floor_normal()
    test_build_R_t_from_floor()
    test_stage1_align_to_room()
    test_stage2_align_via_objects()
    test_stage3_align_via_large_plane()
    test_empty_masks()
    test_align_vggt_predictions()
    test_high_noise_scene()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
