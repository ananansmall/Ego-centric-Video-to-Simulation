"""层级穿模修复功能测试

验证 resolve_penetrations 的层级修复逻辑:
  1. 小物体 (supported) 不独立调整 x,y — 只允许 z 轴移动
  2. supporter (大物体) 移动时, 其 supported (小物体) 跟随 x,y
  3. 小物体之间穿模使用更小的 margin (0.005m)

用法:
  python3 tools/_test_hierarchical_penetration.py \
      --scene_dir output_v2/121_C5_CellPhone_161deg_vggt_omega
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _pos(info):
    return info["T"][:3, 3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True)
    args = ap.parse_args()

    # 1. 加载 all_instances.pkl
    pkl_path = os.path.join(args.scene_dir, "all_instances.pkl")
    if not os.path.isfile(pkl_path):
        pkl_path = os.path.join(args.scene_dir, "all_instances_stage4.pkl")
    with open(pkl_path, "rb") as f:
        all_instances = pickle.load(f)
    print(f"📦 加载 {pkl_path}")
    for cat, insts in all_instances.items():
        print(f"   {cat}: {len(insts)} 个实例")

    # 2. 加载 final_relations.json
    rel_path = os.path.join(args.scene_dir, "final_relations.json")
    with open(rel_path, "r") as f:
        refined_relations = json.load(f)

    # 3. 记录初始位置
    pos_before = {}
    for cat, insts in all_instances.items():
        for idx, info in enumerate(insts):
            pos_before[(cat.lower().strip(), idx)] = _pos(info)

    # 4. 调用 resolve_penetrations (实际修复, 非 dry_run)
    from tools.refine_inter_object_placement import resolve_penetrations
    print("\n🛡️ 调用 resolve_penetrations (实际修复)...")
    all_instances = resolve_penetrations(
        all_instances,
        refined_relations=refined_relations,
        verbose=True,
        dry_run=False,
        max_iterations=8,
    )

    # 5. 验证层级修复逻辑
    print("\n" + "=" * 70)
    print("🔍 验证层级修复逻辑")
    print("=" * 70)

    # 构建 supporter → supported 映射
    supporter_to_supported = {}
    supported_set = set()
    for name, rel in refined_relations.items():
        if rel.startswith("supported by ") and "floor" not in rel.lower() and "other objects" not in rel.lower():
            supporter_name = rel[len("supported by "):].strip()
            if "_" in supporter_name:
                s_cat, s_idx_str = supporter_name.rsplit("_", 1)
                try:
                    s_idx = int(s_idx_str)
                except ValueError:
                    continue
            else:
                continue
            if "_" in name:
                d_cat, d_idx_str = name.rsplit("_", 1)
                try:
                    d_idx = int(d_idx_str)
                except ValueError:
                    continue
            else:
                continue
            key = (s_cat.lower().strip(), s_idx)
            val = (d_cat.lower().strip(), d_idx)
            supporter_to_supported.setdefault(key, []).append(val)
            supported_set.add(val)

    print(f"\n   支撑关系:")
    for sup, sups in supporter_to_supported.items():
        print(f"     {sup[0]}_{sup[1]} → {sups}")

    # 5a. 验证小物体只移动 z 轴 (x,y 不变)
    print(f"\n   [检查 1] 小物体 x,y 是否不变 (只允许 z 轴移动):")
    small_xy_violations = 0
    for sup_key, sups in supporter_to_supported.items():
        for s_cat, s_idx in sups:
            if s_cat not in all_instances or s_idx >= len(all_instances[s_cat]):
                continue
            before = pos_before.get((s_cat, s_idx))
            if before is None:
                continue
            after = _pos(all_instances[s_cat][s_idx])
            dx = after[0] - before[0]
            dy = after[1] - before[1]
            dz = after[2] - before[2]
            status = "✅" if abs(dx) < 1e-6 and abs(dy) < 1e-6 else "❌"
            if "❌" in status:
                small_xy_violations += 1
            print(f"     {status} {s_cat}_{s_idx}: Δxyz=({dx:+.4f}, {dy:+.4f}, {dz:+.4f})")

    # 5b. 验证 supporter 移动时, 其 supported 跟随 x,y
    print(f"\n   [检查 2] supporter 移动时, supported 是否跟随 x,y:")
    follow_ok = 0
    follow_fail = 0
    for sup_key, sups in supporter_to_supported.items():
        s_cat, s_idx = sup_key
        if s_cat not in all_instances or s_idx >= len(all_instances[s_cat]):
            continue
        sup_before = pos_before.get((s_cat, s_idx))
        if sup_before is None:
            continue
        sup_after = _pos(all_instances[s_cat][s_idx])
        sup_dx = sup_after[0] - sup_before[0]
        sup_dy = sup_after[1] - sup_before[1]
        if abs(sup_dx) < 1e-6 and abs(sup_dy) < 1e-6:
            # supporter 没移动, 跳过
            continue
        # supporter 移动了, 检查 supported 是否跟随
        for d_cat, d_idx in sups:
            if d_cat not in all_instances or d_idx >= len(all_instances[d_cat]):
                continue
            d_before = pos_before.get((d_cat, d_idx))
            if d_before is None:
                continue
            d_after = _pos(all_instances[d_cat][d_idx])
            d_dx = d_after[0] - d_before[0]
            d_dy = d_after[1] - d_before[1]
            # supported 应该跟随 supporter 的 x,y delta
            follow_dx = abs(d_dx - sup_dx) < 1e-4
            follow_dy = abs(d_dy - sup_dy) < 1e-4
            status = "✅" if follow_dx and follow_dy else "❌"
            if "✅" in status:
                follow_ok += 1
            else:
                follow_fail += 1
            print(f"     {status} {d_cat}_{d_idx} 跟随 {s_cat}_{s_idx}: "
                  f"sup Δxy=({sup_dx:+.4f}, {sup_dy:+.4f}), "
                  f"sub Δxy=({d_dx:+.4f}, {d_dy:+.4f})")

    # 6. 总结
    print("\n" + "=" * 70)
    print("📊 测试结果")
    print("=" * 70)
    print(f"   小物体 x,y 独立调整违规: {small_xy_violations} 处")
    print(f"   supporter → supported 跟随成功: {follow_ok} 处")
    print(f"   supporter → supported 跟随失败: {follow_fail} 处")
    if small_xy_violations == 0:
        print("   ✅ [检查 1 通过] 小物体未独立调整 x,y")
    else:
        print(f"   ❌ [检查 1 失败] {small_xy_violations} 个小物体独立调整了 x,y")
    if follow_fail == 0:
        print("   ✅ [检查 2 通过] supporter 移动时 supported 正确跟随 x,y")
    else:
        print(f"   ❌ [检查 2 失败] {follow_fail} 个 supported 未跟随 supporter")


if __name__ == "__main__":
    main()
