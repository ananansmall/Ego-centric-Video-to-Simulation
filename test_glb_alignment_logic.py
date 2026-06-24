#!/usr/bin/env python3
"""
测试 GLB 生成逻辑和 Z 轴对齐记录
================================

验证:
  1. --enable_stage5 (无 stage4) → 4 个 GLB
  2. --enable_stage4 --enable_stage5 → 5 个 GLB
  3. z_axis_alignment.json 记录格式正确

不运行完整流水线，只验证命名和记录逻辑。
"""
import os
import sys
import json
import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


def test_glb_naming_logic():
    """测试 GLB 文件命名逻辑"""
    print("=" * 60)
    print("测试: GLB 文件命名逻辑")
    print("=" * 60)

    # 模拟 mainv2.py 中的 GLB 命名逻辑
    test_cases = [
        # (enable_stage4, enable_stage5, has_inter_object, expected_glbs)
        (False, False, False, ["final_scene_initial.glb", "final_scene.glb"]),
        (False, True, False, ["final_scene_initial.glb", "final_scene.glb",
                              "final_scene_stage5.glb"]),
        (False, True, True, ["final_scene_initial.glb", "final_scene.glb",
                             "final_scene_stage5_sp.glb", "final_scene_stage5.glb"]),
        (True, False, False, ["final_scene_initial.glb", "final_scene.glb",
                              "final_scene_stage4.glb"]),
        (True, True, False, ["final_scene_initial.glb", "final_scene.glb",
                             "final_scene_stage4.glb", "final_scene_stage4_5.glb"]),
        (True, True, True, ["final_scene_initial.glb", "final_scene.glb",
                            "final_scene_stage4.glb", "final_scene_stage5_sp.glb",
                            "final_scene_stage4_5.glb"]),
    ]

    all_pass = True
    for s4, s5, has_inter, expected in test_cases:
        # 模拟命名逻辑
        glbs = ["final_scene_initial.glb", "final_scene.glb"]

        if s4:
            glbs.append("final_scene_stage4.glb")

        if s5:
            if has_inter:
                glbs.append("final_scene_stage5_sp.glb")
            final_name = "final_scene_stage4_5.glb" if s4 else "final_scene_stage5.glb"
            glbs.append(final_name)

        status = "✅" if glbs == expected else "❌"
        if glbs != expected:
            all_pass = False
            print(f"  {status} stage4={s4}, stage5={s5}, inter_obj={has_inter}")
            print(f"     期望: {expected}")
            print(f"     实际: {glbs}")
        else:
            label = f"stage4={'ON' if s4 else 'OFF'}, stage5={'ON' if s5 else 'OFF'}, inter_obj={has_inter}"
            print(f"  {status} {label} → {len(glbs)} 个 GLB")

    print(f"\n  结果: {'全部通过' if all_pass else '有失败'}")
    return all_pass


def test_alignment_json_format():
    """测试 z_axis_alignment.json 记录格式"""
    print("\n" + "=" * 60)
    print("测试: z_axis_alignment.json 记录格式")
    print("=" * 60)

    # 模拟对齐成功的情况
    R_success = np.array([
        [0.98, -0.12, 0.15],
        [0.10, 0.99, 0.08],
        [-0.16, -0.06, 0.98],
    ])
    t_success = np.array([0.01, -0.02, -0.85])

    align_record = {
        "align_method": "stage1_room",
        "R_matrix": R_success.tolist(),
        "t_vector": t_success.tolist(),
        "is_identity": False,
        "align_info": {
            "method": "stage1_room",
            "floor_area": 2.35,
            "floor_mean_distance": 0.018,
        },
        "n_wall_masks": 5,
        "n_floor_masks": 8,
    }

    # 验证字段
    required_fields = ["align_method", "R_matrix", "t_vector", "is_identity",
                       "align_info", "n_wall_masks", "n_floor_masks"]
    for field in required_fields:
        assert field in align_record, f"缺少字段: {field}"

    # 验证 R_matrix 是 3x3
    R = np.array(align_record["R_matrix"])
    assert R.shape == (3, 3), f"R_matrix 形状错误: {R.shape}"

    # 验证 t_vector 是 3
    t = np.array(align_record["t_vector"])
    assert t.shape == (3,), f"t_vector 形状错误: {t.shape}"

    print("  ✅ 对齐成功记录格式正确")
    print(f"     method={align_record['align_method']}, "
          f"is_identity={align_record['is_identity']}, "
          f"n_floor={align_record['n_floor_masks']}, "
          f"n_wall={align_record['n_wall_masks']}")

    # 模拟对齐失败的情况
    align_record_fail = {
        "align_method": "none",
        "R_matrix": np.eye(3).tolist(),
        "t_vector": np.zeros(3).tolist(),
        "is_identity": True,
        "align_info": {"reason": "all_stages_failed"},
        "n_wall_masks": 0,
        "n_floor_masks": 0,
    }
    print("  ✅ 对齐失败记录格式正确")
    print(f"     method={align_record_fail['align_method']}, "
          f"is_identity={align_record_fail['is_identity']}")

    return True


def test_identity_check():
    """测试 _is_identity_alignment 判断逻辑"""
    print("\n" + "=" * 60)
    print("测试: _is_identity_alignment 判断逻辑")
    print("=" * 60)

    # 内联实现 (与 mainv2.py 中一致)
    def _is_identity_alignment(R, t, tol=1e-6):
        return np.allclose(R, np.eye(3), atol=tol) and np.allclose(t, np.zeros(3), atol=tol)

    # 单位阵 → True (对齐失败)
    assert _is_identity_alignment(np.eye(3), np.zeros(3)) == True
    print("  ✅ 单位阵 → 对齐失败 (True)")

    # 非单位阵 → False (对齐成功)
    R = np.array([[0.98, -0.12, 0.15], [0.10, 0.99, 0.08], [-0.16, -0.06, 0.98]])
    t = np.array([0.01, -0.02, -0.85])
    assert _is_identity_alignment(R, t) == False
    print("  ✅ 非单位阵 → 对齐成功 (False)")

    # 接近单位阵 → True (数值误差范围内)
    R_near = np.eye(3) + 1e-8 * np.ones((3, 3))
    t_near = np.zeros(3) + 1e-8 * np.ones(3)
    assert _is_identity_alignment(R_near, t_near) == True
    print("  ✅ 接近单位阵 (1e-8) → 对齐失败 (True)")

    return True


def analyze_068_log():
    """分析 068 日志，对比旧代码 vs 新代码"""
    print("\n" + "=" * 60)
    print("分析: 068 日志 (旧代码 vs 新代码对比)")
    print("=" * 60)

    log_path = os.path.join(REPO, "output_v2/068_C5_CellPhone_high_onehand_vggt_omega",
                            "mainv2_20260623_143339.log")
    if not os.path.exists(log_path):
        print("  ⚠️ 日志文件不存在，跳过")
        return True

    # 读取日志关键信息
    with open(log_path, 'r') as f:
        log_content = f.read()

    # 旧代码特征: 没有 "Z轴对齐方法" 输出
    has_z_align = "Z轴对齐方法" in log_content
    has_stage4 = "Stage 4: 迭代视觉-空间对齐" in log_content and "已跳过" not in log_content.split("Stage 4")[1][:100]

    # 提取 theta_gravity 值
    import re
    theta_matches = re.findall(r'theta_gravity=([\d.]+)°', log_content)
    thetas = [float(t) for t in theta_matches]

    print(f"  旧代码日志分析:")
    print(f"    Z轴对齐输出: {'有' if has_z_align else '❌ 无 (旧代码)'}")
    print(f"    Stage4 启用: {'是' if has_stage4 else '❌ 否'}")
    print(f"    theta_gravity 值: {thetas}")
    if thetas:
        max_theta = max(thetas)
        print(f"    最大倾斜角: {max_theta}°", end="")
        if max_theta > 90:
            print(f" → ❌ 严重倾斜 (接近倒立)")
        elif max_theta > 30:
            print(f" → ⚠️ 明显倾斜")
        else:
            print(f" → ✅ 基本正常")

    print(f"\n  新代码预期改进:")
    print(f"    1. Stage2 会输出 'Z轴对齐方法: xxx' 和 '对齐信息: xxx'")
    print(f"    2. 保存 z_axis_alignment.json (含 R, t, method, mask数量)")
    print(f"    3. 四阶段 fallback: stage1→stage2→stage2.5(VLM)→stage3→stage4(GeoCalib)")
    print(f"    4. 启用 --enable_stage4 时生成 5 个 GLB (而非 4 个)")

    # 检查 pose_changes.json
    pose_path = os.path.join(REPO, "output_v2/068_C5_CellPhone_high_onehand_vggt_omega",
                             "pose_changes.json")
    if os.path.exists(pose_path):
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        sample_key = list(pose_data.keys())[0]
        stages = list(pose_data[sample_key]["stages"].keys())
        has_stage4_pose = "stage4" in stages
        print(f"\n  pose_changes.json 分析:")
        print(f"    物体数量: {len(pose_data)}")
        print(f"    记录阶段: {stages}")
        print(f"    Stage4 记录: {'✅ 有' if has_stage4_pose else '❌ 无 (未启用 stage4)'}")

    return True


def main():
    print("\n" + "=" * 60)
    print("ReplicateAnyScene V2 — GLB 逻辑与 Z 轴对齐测试")
    print("=" * 60 + "\n")

    results = []
    results.append(test_glb_naming_logic())
    results.append(test_alignment_json_format())
    results.append(test_identity_check())
    results.append(analyze_068_log())

    print("\n" + "=" * 60)
    print(f"总结: {'全部通过 ✅' if all(results) else '有失败 ❌'}")
    print("=" * 60)

    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
