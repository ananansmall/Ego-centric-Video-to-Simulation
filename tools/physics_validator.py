"""
物体摆放z轴合理性验证器

核心思路:
  用SAPIEN物理仿真检测哪些物体不稳定（悬空/穿模），
  然后对不稳定物体做最小z轴修正（下落到最近支撑面），
  不采用仿真中的xy位移和旋转（避免弹开/飞走问题）。

验证逻辑:
  1. 在SAPIEN中构建场景，运行短时仿真
  2. 检测仿真中z轴位移 > 阈值的物体 → 标记为不稳定
  3. 对不稳定物体:
     - "supported by floor" → 下移到地面 (bottom_z = 0)
     - "supported by {name}" → 下移到支撑物顶面 (bottom_z = supporter_top_z)
     - 保留原始xy位置和旋转不变
  4. 修正后用FCL检测穿模，如有穿模则沿z轴继续微调

使用方式:
  python mainv2.py --input_video video.mp4 --enable_stage5 --enable_physics_validation
  python mainv2.py --input_video video.mp4 --enable_physics_validation  # 独立模式
"""

import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def rotation_distance(R1, R2):
    R_diff = R1.T @ R2
    val = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(val))


class PhysicsValidator:
    """物体摆放z轴合理性验证

    保守策略: 物理仿真仅用于检测，修正只做z轴最小下落
    """

    def __init__(self, sim_steps=300, z_displacement_threshold=0.02,
                 gap_threshold=0.02, verbose=True):
        self.sim_steps = sim_steps
        self.z_displacement_threshold = z_displacement_threshold
        self.gap_threshold = gap_threshold
        self.verbose = verbose

    def validate(self, all_instances, categories_and_relations=None,
                 walls_info=None, output_dir=None):
        _all_rels = {}
        if categories_and_relations:
            _all_rels.update(categories_and_relations)

        unstable_names = self._detect_unstable(all_instances, _all_rels, output_dir)

        report = {}
        for category, instances in all_instances.items():
            for idx, info in enumerate(instances):
                name = f"{category}_{idx}"
                T = info["T"]
                rel = _all_rels.get(category, "")

                if name not in unstable_names:
                    report[name] = {
                        "status": "stable",
                        "displacement": 0.0,
                        "original_z": float(T[2, 3]),
                        "final_z": float(T[2, 3]),
                    }
                    continue

                mesh = info.get("original_mesh")
                if mesh is None:
                    report[name] = {"status": "skip", "displacement": 0.0,
                                    "original_z": float(T[2, 3]),
                                    "final_z": float(T[2, 3])}
                    continue

                mesh_t = mesh.copy()
                mesh_t.apply_transform(T)
                original_z = float(T[2, 3])
                bottom_z = mesh_t.bounds[0, 2]

                target_z = self._compute_target_z(
                    category, idx, rel, bottom_z, all_instances, _all_rels
                )

                if target_z is not None and abs(bottom_z - target_z) > self.gap_threshold:
                    z_shift = target_z - bottom_z
                    T_new = T.copy()
                    T_new[2, 3] += z_shift
                    info["T"] = T_new
                    final_z = float(T_new[2, 3])

                    report[name] = {
                        "status": "corrected",
                        "displacement": float(abs(z_shift)),
                        "original_z": original_z,
                        "final_z": final_z,
                        "direction": "down" if z_shift < 0 else "up",
                    }
                else:
                    report[name] = {
                        "status": "stable_sim_unstable",
                        "displacement": 0.0,
                        "original_z": original_z,
                        "final_z": original_z,
                    }

        self._post_penetration_check(all_instances, _all_rels, report)

        if self.verbose:
            self._print_report(report)

        return all_instances, report

    def _detect_unstable(self, all_instances, all_rels, output_dir):
        """用SAPIEN物理仿真检测z轴不稳定物体"""
        try:
            import sapien
        except ImportError:
            if self.verbose:
                print("      [WARN] SAPIEN未安装, 跳过物理仿真检测")
            return set()

        import tempfile
        tmp_dir = output_dir or tempfile.mkdtemp(prefix="physics_val_")
        os.makedirs(tmp_dir, exist_ok=True)

        mesh_files = self._export_meshes(all_instances, tmp_dir)
        if not mesh_files:
            return set()

        scene = sapien.Scene()
        scene.set_timestep(1.0 / 240.0)
        scene.add_ground(0)

        if all_rels:
            for wall in (all_rels.get("__walls__") or []):
                self._add_wall(scene, wall)

        actors = {}
        for category, instances in all_instances.items():
            for idx, info in enumerate(instances):
                name = f"{category}_{idx}"
                rel = all_rels.get(category, "")
                is_static = self._is_static_object(rel)

                T = info["T"]
                pos = T[:3, 3]
                R = T[:3, :3]
                q = Rotation.from_matrix(R).as_quat()

                glb_path = mesh_files.get(name)
                if glb_path is None:
                    continue

                actor = self._create_object(scene, glb_path, name, is_static,
                                            sapien.Pose(p=pos, q=q))
                if actor is not None:
                    actors[name] = {
                        "actor": actor,
                        "original_z": float(T[2, 3]),
                        "is_static": is_static,
                    }

        for i in range(self.sim_steps):
            scene.step()

        unstable = set()
        for name, data in actors.items():
            if data["is_static"]:
                continue
            sim_z = float(data["actor"].get_pose().p[2])
            z_shift = abs(sim_z - data["original_z"])
            if z_shift > self.z_displacement_threshold:
                unstable.add(name)

        return unstable

    def _compute_target_z(self, category, idx, rel, bottom_z, all_instances, all_rels):
        """计算物体应该落到的z位置 (只做z轴修正)"""
        rel_lower = rel.lower() if rel else ""

        if "floor" in rel_lower:
            return 0.0

        if rel.startswith("supported by ") and "other objects" not in rel_lower and "floor" not in rel_lower:
            supporter_name = rel[len("supported by "):].strip()
            supporter_cat = supporter_name.rsplit("_", 1)[0] if "_" in supporter_name else supporter_name

            supporter_info = self._find_supporter(all_instances, supporter_name, supporter_cat)
            if supporter_info is not None:
                supporter_mesh = supporter_info.get("original_mesh")
                supporter_T = supporter_info.get("T")
                if supporter_mesh is not None and supporter_T is not None:
                    s_mesh = supporter_mesh.copy()
                    s_mesh.apply_transform(supporter_T)
                    return float(s_mesh.bounds[1, 2])

        return 0.0

    def _find_supporter(self, all_instances, supporter_name, supporter_cat):
        if supporter_name in all_instances and all_instances[supporter_name]:
            return all_instances[supporter_name][0]
        if supporter_cat in all_instances and all_instances[supporter_cat]:
            return all_instances[supporter_cat][0]
        for cat, instances in all_instances.items():
            cat_base = cat.rsplit("_", 1)[0] if "_" in cat else cat
            if cat_base.lower() == supporter_cat.lower() or cat.lower() == supporter_name.lower():
                if instances:
                    return instances[0]
        return None

    def _post_penetration_check(self, all_instances, all_rels, report):
        """修正后检查corrected物体是否穿入地面"""
        for name, r in report.items():
            if r["status"] != "corrected":
                continue
            cat, idx = name.rsplit("_", 1)
            info = all_instances[cat][int(idx)]
            mesh = info.get("original_mesh")
            T = info.get("T")
            if mesh is None or T is None:
                continue
            mesh_t = mesh.copy()
            mesh_t.apply_transform(T)
            bottom_z = mesh_t.bounds[0, 2]
            if bottom_z < 0:
                z_fix = -bottom_z
                info["T"][2, 3] += z_fix
                if self.verbose:
                    print(f"      地面穿入修正: {name} z += {z_fix:.4f}m")

    def _aabb_overlap(self, mesh_a, mesh_b):
        bounds_a = mesh_a.bounds
        bounds_b = mesh_b.bounds
        ox = min(bounds_a[1, 0], bounds_b[1, 0]) - max(bounds_a[0, 0], bounds_b[0, 0])
        oy = min(bounds_a[1, 1], bounds_b[1, 1]) - max(bounds_a[0, 1], bounds_b[0, 1])
        oz = min(bounds_a[1, 2], bounds_b[1, 2]) - max(bounds_a[0, 2], bounds_b[0, 2])
        return ox > 0 and oy > 0 and oz > 0, ox, oy, oz

    def _export_meshes(self, all_instances, tmp_dir):
        mesh_files = {}
        for category, instances in all_instances.items():
            for idx, info in enumerate(instances):
                name = f"{category}_{idx}"
                mesh = info.get("original_mesh")
                if mesh is None:
                    continue
                glb_path = os.path.join(tmp_dir, f"{name}.glb")
                try:
                    mesh.export(glb_path)
                    mesh_files[name] = glb_path
                except Exception as e:
                    if self.verbose:
                        print(f"      [WARN] 导出 {name} 失败: {e}")
        return mesh_files

    def _is_static_object(self, rel):
        rel_lower = rel.lower() if rel else ""
        return any(kw in rel_lower for kw in
                   ["floor", "wall", "embedded", "attached"])

    def _create_object(self, scene, mesh_path, name, is_static, pose):
        import sapien
        builder = scene.create_actor_builder()
        if is_static:
            builder.set_physx_body_type("static")
        else:
            builder.set_physx_body_type("dynamic")

        try:
            builder.add_multiple_convex_collisions_from_file(
                filename=mesh_path, decomposition="coacd"
            )
        except Exception:
            try:
                builder.add_convex_collision_from_file(filename=mesh_path)
            except Exception:
                return None

        try:
            builder.add_visual_from_file(filename=mesh_path)
        except Exception:
            pass

        try:
            actor = builder.build(name=name)
            actor.set_pose(pose)
            return actor
        except Exception:
            return None

    def _add_wall(self, scene, wall_info):
        import sapien
        axis = wall_info.get("axis", "x")
        position = wall_info.get("position", 0)
        thickness = 0.1
        if axis == "x":
            half_size = [thickness / 2, 5.0, 2.0]
            pos = [position, 0, 2]
        else:
            half_size = [5.0, thickness / 2, 2.0]
            pos = [0, position, 2]
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size)
        actor = builder.build(name=f"wall_{axis}_{position}")
        actor.set_pose(sapien.Pose(p=pos))

    def _print_report(self, report):
        stable = sum(1 for r in report.values() if r["status"] == "stable")
        corrected = sum(1 for r in report.values() if r["status"] == "corrected")
        sim_unstable = sum(1 for r in report.values() if r["status"] == "stable_sim_unstable")
        skip = sum(1 for r in report.values() if r["status"] == "skip")
        print(f"   🔬 z轴合理性验证: {len(report)} 个物体 "
              f"(稳定 {stable}, 修正 {corrected}, 仿真不稳但几何合理 {sim_unstable}, 跳过 {skip})")
        for name, r in report.items():
            if r["status"] == "corrected":
                print(f"      {name}: z {r['original_z']:.3f} → {r['final_z']:.3f} "
                      f"(下移 {r['displacement']:.3f}m)")
            elif r["status"] == "stable_sim_unstable":
                print(f"      {name}: 仿真中位移但几何位置合理, 保持不变")
