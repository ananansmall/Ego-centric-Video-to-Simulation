# Stage5 物理引擎集成方案

> 目标: 在 Stage5 SP精修中引入物理引擎，解决当前纯几何方法的穿模漏检、支撑不稳、迭代不足等问题
> 状态: 方案设计阶段，不涉及代码修改

---

## 1. 当前问题分析

### 1.1 碰撞检测缺陷

| 问题 | 当前实现 | 后果 |
|------|---------|------|
| AABB精度不足 | SP精修全部基于AABB bounds | 非轴对齐物体（旋转的椅子）AABB远大于实际形状，被推到过远位置 |
| 顶点距离≠穿模检测 | `cKDTree(mesh_b.vertices)` 查询采样点到顶点距离 | mesh_b顶点稀疏时漏检，采样点实际在mesh内部但离顶点远 |
| 硬编码阈值 | `dist < 0.01m`、`inside_count < 3` | 大物体阈值太严，小物体太松 |
| 分离轴粗糙 | `sep_axis = argmax(\|center_a - center_b\|)` | L形穿入等复杂情况单轴分离不够 |
| 只迭代1次 | `resolve_penetrations` max_iterations=1 | 推开A可能让A穿入C，不会再次检测 |
| 无旋转修正 | 穿模解决只做平移 | 倾斜穿入无法解决 |

### 1.2 SP精修缺陷

| 问题 | 当前实现 | 后果 |
|------|---------|------|
| `inside` 30%高度硬编码 | `target_z = supporter_bottom_z + height * 0.3` | 浅容器/高容器都不合理 |
| `leaning` 等同 `against_side` | 直接调用 `against_side`，无倾斜 | 斜靠物体没有倾斜角度 |
| `on_top` 无悬空阈值 | 任何z_offset都会对齐 | 误判关系时物体被错误吸附 |
| `_align_upright` 过度竖直化 | 强制Y轴对齐Z轴 | 躺着的瓶子被强制竖直 |

### 1.3 架构缺陷

- **无物理约束**: 不考虑重力、摩擦力、支撑面积
- **无稳定性验证**: 物体可能"放"在支撑面边缘，实际会掉落
- **无物理仿真**: 无法预测物体在重力下的最终稳定位置

---

## 2. 候选方案

### 方案A: trimesh CollisionManager (轻量级，推荐)

**核心思路**: 用 trimesh 自带的 `CollisionManager`（底层 FCL 库）替换手写的 AABB+顶点采样碰撞检测，保持当前SP精修的几何逻辑不变，仅升级碰撞检测精度。

**技术栈**:
- `trimesh.collision.CollisionManager` — 基于FCL (Flexible Collision Library) 的精确mesh-mesh碰撞检测
- `trimesh.proximity.closest_point` — 精确的mesh表面距离计算
- 无需安装额外依赖（trimesh已安装，FCL为trimesh的可选依赖）

**改动范围**:
- 替换 `_check_mesh_penetration()` → 用 `CollisionManager.in_collision_single()` + `CollisionManager.min_distance_single()`
- 替换 `_get_aabb_overlap()` → 用 FCL 的精确碰撞检测
- `resolve_penetrations()` 增加迭代次数，用 FCL 的穿透深度和方向指导分离

**优点**:
- 最小改动，不改变现有SP精修逻辑
- trimesh已安装，FCL为pip可装（`pip install python-fcl`）
- mesh-mesh碰撞精度远高于AABB+顶点采样
- 可获取精确的穿透深度和分离方向
- 不引入GPU依赖

**缺点**:
- 仍然是纯几何方法，不考虑物理约束（重力、摩擦力）
- 无法验证物体稳定性（是否会在重力下滑落）
- 对复杂穿模（多物体连锁）仍需多轮迭代

**适用场景**: 快速提升碰撞检测精度，解决当前漏检和误推问题

---

### 方案B: PyBullet 物理仿真 (中等重量)

**核心思路**: 在 PyBullet 物理引擎中重建场景，让物体在重力下自然稳定，利用物理仿真验证和修正SP精修结果。

**技术栈**:
- `pybullet` — 开源物理引擎，支持刚体动力学、碰撞检测、约束求解
- 需要新增依赖: `pip install pybullet`

**改动范围**:
- 新增 `PhysicsValidator` 类，封装PyBullet场景构建和仿真
- Stage5 SP精修后，调用 `PhysicsValidator` 验证结果
- 不替换现有SP精修逻辑，作为后置验证+修正层

**流程**:
```
Stage5 SP精修 (现有逻辑)
  ↓
PhysicsValidator.validate(all_instances, relations)
  ├─ 1. 构建PyBullet场景
  │     - 地面: 静态平面 (z=0)
  │     - 墙壁: 静态box (从walls_info)
  │     - 每个物体: 动态刚体 (mesh → convex hull / VHACD分解)
  │     - 初始位姿: SP精修后的T矩阵
  │
  ├─ 2. 添加物理约束
  │     - floor/wall物体: 设为STATIC (kinematic)
  │     - 支撑关系: 添加接触约束 (或直接让物理求解)
  │
  ├─ 3. 运行仿真 (500-1000步)
  │     - 物体在重力下自然稳定
  │     - 碰撞检测自动处理穿模
  │     - 记录每个物体的最终位姿
  │
  ├─ 4. 结果验证
  │     - 对比仿真前后位姿变化
  │     - 位移 > 阈值 → SP精修结果不可靠，用仿真位姿替换
  │     - 位移 < 阈值 → SP精修结果有效，保持不变
  │
  └─ 5. 更新T矩阵
        - 用仿真最终位姿更新 all_instances
```

**mesh处理**:
PyBullet 要求刚体为凸包或三角形网格（带VHACD分解）:
```python
# 凸包 (简单，适合简单形状)
convex_hull = mesh.convex_hull

# VHACD分解 (精确，适合凹形状)
# pip install v-hacd
import vhacd
hulls = vhacd.decompose(mesh)  # 分解为多个凸包
```

**优点**:
- 真正的物理约束: 重力、摩擦力、支撑面积
- 自动处理穿模: 物理引擎的碰撞响应自然推开穿模物体
- 稳定性验证: 物体如果放不稳会自然滑落到稳定位置
- 多物体连锁穿模: 物理仿真自然处理，不需要手动迭代

**缺点**:
- 需要安装PyBullet（~50MB）
- mesh需要转凸包或VHACD分解，增加预处理步骤
- 仿真参数（摩擦系数、恢复系数）需要调优
- 仿真时间: 500步约0.5-2秒/场景
- 物理仿真可能改变VLM/SP精修的意图（如VLM说"靠墙"但物理仿真让物体滑到地面）

**适用场景**: 需要物理正确性验证，确保物体不会穿模、不会悬浮、不会滑落

---

### 方案C: PyBullet 物理仿真 + 约束求解 (完整方案)

**核心思路**: 在方案B基础上，利用PyBullet的约束系统显式编码支撑关系，而非仅靠重力自然求解。

**额外技术**:
- `pybullet.createConstraint()` — 创建固定/铰接/滑动约束
- 根据VLM判定的关系类型选择约束:
  - `on_top` → 接触约束 (point-to-plane)
  - `inside` → 容器约束 (限制在AABB内)
  - `against_side` → 接触约束 (point-to-plane, 侧面)
  - `hanging_below` → 固定约束 (fixed)
  - `leaning` → 铰接约束 (revolute, 允许倾斜)

**流程**:
```
Stage5 SP精修 (现有逻辑)
  ↓
PhysicsValidator.validate_with_constraints(all_instances, relations)
  ├─ 1. 构建PyBullet场景 (同方案B)
  │
  ├─ 2. 添加物理约束 (新增)
  │     for each relation:
  │       if "supported by floor":
  │         → 无需约束，重力自然处理
  │       if "supported by {name}":
  │         → createConstraint(type=CONTACT, on supporter top surface)
  │       if "inside {name}":
  │         → createConstraint(type=GENERIC, limit position to container AABB)
  │       if "against_side of {name}":
  │         → createConstraint(type=CONTACT, on supporter side surface)
  │       if "hanging_below {name}":
  │         → createConstraint(type=FIXED, attach to supporter bottom)
  │       if "leaning on {name}":
  │         → createConstraint(type=REVOLUTE, pivot at contact point)
  │
  ├─ 3. 运行仿真 (同方案B)
  │     - 约束确保物体不会离开指定的支撑关系
  │     - 物理引擎在约束范围内求解最优位姿
  │
  ├─ 4. 约束验证 (新增)
  │     - 检查约束是否满足 (物体是否仍在支撑面上)
  │     - 约束力过大 → 支撑关系可能有误，标记为需人工确认
  │
  └─ 5. 更新T矩阵 (同方案B)
```

**优点**:
- 方案B的所有优点
- 约束保证物体不会离开VLM指定的支撑关系
- 约束力可作为支撑关系正确性的量化指标
- 更精确地实现5种放置策略的物理语义

**缺点**:
- 方案B的所有缺点
- 约束参数需要仔细调优
- 约束与物理仿真的交互可能产生不自然的结果
- 实现复杂度最高

**适用场景**: 需要精确控制支撑关系，同时保证物理正确性

---

## 3. 方案对比

| 维度 | 方案A: trimesh FCL | 方案B: PyBullet仿真 | 方案C: PyBullet+约束 |
|------|-------------------|--------------------|--------------------|
| 碰撞检测精度 | ★★★★★ mesh-mesh | ★★★★★ mesh-mesh | ★★★★★ mesh-mesh |
| 穿模解决 | ★★★ 多轮迭代推开 | ★★★★★ 物理仿真自然解决 | ★★★★★ 物理仿真+约束 |
| 稳定性验证 | ★ 无 | ★★★★ 重力下自然稳定 | ★★★★★ 约束保证稳定 |
| 支撑关系保持 | ★★★ 依赖SP精修 | ★★★ 重力可能改变意图 | ★★★★★ 约束显式编码 |
| 实现复杂度 | ★★ 低 | ★★★ 中 | ★★★★★ 高 |
| 新增依赖 | python-fcl (~5MB) | pybullet (~50MB) | pybullet (~50MB) |
| GPU需求 | 无 | 无 | 无 |
| 运行时间 | ~0.1s/场景 | ~1-2s/场景 | ~2-3s/场景 |
| 对现有代码影响 | 替换碰撞检测函数 | 新增验证层 | 新增验证层+约束层 |

---

## 4. 推荐方案: 渐进式实施

**推荐: 先A后B，C作为远期目标**

### Phase 1: trimesh FCL 碰撞检测升级 (方案A)

**目标**: 解决当前最紧急的碰撞检测精度问题

**改动**:
1. 安装 `python-fcl`
2. 替换 `_check_mesh_penetration()` → FCL精确碰撞
3. 替换 `_get_aabb_overlap()` → FCL碰撞检测
4. `resolve_penetrations()` 增加迭代次数 (1→5)，用FCL穿透方向指导分离
5. 增加稳定性检查: 计算支撑面积占比，低于阈值则标记不稳定

**验证标准**:
- 穿模漏检率降低 (对比当前AABB+顶点采样方法)
- 穿模解决成功率提升 (多轮迭代)
- 不引入新的误推问题

### Phase 2: PyBullet 物理验证 (方案B)

**目标**: 引入物理约束，确保物体放置稳定

**改动**:
1. 安装 `pybullet`
2. 新增 `PhysicsValidator` 类
3. Stage5 SP精修后调用 `PhysicsValidator.validate()`
4. 仿真结果与SP精修结果对比，选择更合理的位姿
5. 新增 `--enable_physics_validation` 命令行参数

**验证标准**:
- 悬浮物体被正确下放到支撑面
- 不稳定物体滑落到稳定位置
- 不改变已正确放置的物体位姿 (位移 < 1cm)

### Phase 3: PyBullet 约束求解 (方案C, 可选)

**目标**: 精确控制支撑关系，实现5种放置策略的物理语义

**改动**:
1. 扩展 `PhysicsValidator`，添加约束系统
2. 根据5种策略创建对应约束
3. 约束力作为支撑关系正确性指标
4. 新增 `--enable_physics_constraints` 命令行参数

---

## 5. Phase 1 详细设计: trimesh FCL 碰撞检测升级

### 5.1 核心API

```python
from trimesh.collision import CollisionManager

# 构建碰撞管理器
manager = CollisionManager()
for name, mesh in all_meshes:
    manager.add_object(name, mesh)

# 检测两物体碰撞
collides = manager.in_collision_single(mesh_b)

# 获取碰撞详情 (穿透深度+方向)
contact_data = manager.in_collision_single(mesh_b, return_data=True)
# contact_data 包含: contact_normals, penetrations, contacts

# 计算最小距离
min_dist = manager.min_distance_single(mesh_b)
```

### 5.2 替换 _check_mesh_penetration

```python
# 当前: AABB + 顶点采样 (不精确)
def _check_mesh_penetration(mesh_a, mesh_b, n_samples=500):
    overlaps, ox, oy, oz = _get_aabb_overlap(mesh_a, mesh_b)
    if not overlaps:
        return False, 0.0, 2
    pts_a, _ = trimesh.sample.sample_surface(mesh_a, n_samples)
    tree_b = cKDTree(mesh_b.vertices)
    dists, _ = tree_b.query(pts_a)
    inside_count = (dists < 0.01).sum()
    ...

# 替换为: FCL精确碰撞
def _check_mesh_penetration(mesh_a, mesh_b):
    manager = CollisionManager()
    manager.add_object("b", mesh_b)
    collides, contacts = manager.in_collision_single(mesh_a, return_data=True)
    if not collides:
        return False, 0.0, None
    # 从contacts提取最大穿透深度和方向
    max_pen = max(c.penetration for c in contacts)
    avg_normal = np.mean([c.contact_normal for c in contacts], axis=0)
    sep_axis = int(np.argmax(np.abs(avg_normal)))
    return True, max_pen, sep_axis
```

### 5.3 resolve_penetrations 多轮迭代

```python
# 当前: max_iterations=1
# 改为: max_iterations=5, 每轮用FCL检测剩余穿模
for iteration in range(max_iterations):
    any_penetration = False
    for i, j in object_pairs:
        collides, depth, axis = _check_mesh_penetration(mesh_i, mesh_j)
        if collides:
            any_penetration = True
            # 用FCL的穿透方向指导分离 (比中心差更准确)
            _separate_objects(i, j, depth, axis)
    if not any_penetration:
        break  # 无穿模，提前退出
```

### 5.4 稳定性检查 (新增)

```python
def _check_stability(supported_mesh, supporter_mesh, contact_threshold=0.3):
    """检查支撑面积是否足够"""
    # 1. 计算接触区域
    supported_bottom = supported_mesh.bounds[0, 2]  # z_min
    supporter_top = supporter_mesh.bounds[1, 2]      # z_max

    # 2. 投影到xy平面，计算重叠面积
    supported_proj = _project_to_xy(supported_mesh, z=supported_bottom)
    supporter_proj = _project_to_xy(supporter_mesh, z=supporter_top)
    overlap_area = supported_proj.intersection(supporter_proj).area

    # 3. 支撑面积占比
    supported_base_area = supported_proj.area
    support_ratio = overlap_area / supported_base_area if supported_base_area > 0 else 0

    # 4. 判定稳定性
    if support_ratio < contact_threshold:
        return False, support_ratio  # 不稳定
    return True, support_ratio
```

---

## 6. Phase 2 详细设计: PyBullet 物理验证

### 6.1 PhysicsValidator 类设计

```python
class PhysicsValidator:
    """用PyBullet物理仿真验证Stage5 SP精修结果"""

    def __init__(self, gravity=-9.81, friction=0.5, restitution=0.1, sim_steps=500):
        self.gravity = gravity
        self.friction = friction
        self.restitution = restitution
        self.sim_steps = sim_steps

    def validate(self, all_instances, categories_and_relations, walls_info):
        """
        验证SP精修结果的物理正确性

        Returns:
            validated_instances: 更新后的all_instances
            validation_report: 每个物体的验证结果
        """
        # 1. 初始化PyBullet场景
        # 2. 添加地面、墙壁 (静态)
        # 3. 添加物体 (动态, 初始位姿=SP精修结果)
        # 4. 运行仿真
        # 5. 对比仿真前后位姿
        # 6. 更新不可靠物体的T矩阵
        pass

    def _mesh_to_bullet_body(self, mesh, T, is_static=False):
        """trimesh.Trimesh → PyBullet刚体"""
        # 1. 凸包或VHACD分解
        # 2. 创建碰撞形状
        # 3. 创建刚体
        # 4. 设置位姿 (T矩阵 → PyBullet位置+四元数)
        pass

    def _bullet_pose_to_T(self, body_id):
        """PyBullet位姿 → 4x4变换矩阵"""
        pass

    def _add_walls(self, walls_info):
        """从walls_info创建静态墙壁"""
        pass

    def _add_floor(self):
        """创建地面平面 (z=0)"""
        pass
```

### 6.2 mesh处理策略

| mesh类型 | 处理方式 | 适用场景 |
|----------|---------|---------|
| 简单凸形状 | `mesh.convex_hull` | 球、盒子、圆柱 |
| 复杂凹形状 | VHACD分解为多个凸包 | 椅子、桌子、容器 |
| 退化mesh (无体积) | 跳过物理验证，保持SP精修结果 | 平面、薄片 |

### 6.3 仿真结果判定

```python
def _judge_result(self, original_T, simulated_T, category):
    """判定SP精修结果是否可靠"""
    # 计算位移
    displacement = np.linalg.norm(simulated_T[:3, 3] - original_T[:3, 3])
    # 计算旋转变化
    rotation_diff = _rotation_distance(original_T[:3, :3], simulated_T[:3, :3])

    if displacement < 0.01 and rotation_diff < 5:  # 1cm + 5度
        return "stable", original_T     # SP精修结果可靠
    elif displacement < 0.05 and rotation_diff < 15:  # 5cm + 15度
        return "adjusted", simulated_T  # SP精修接近，用仿真微调
    else:
        return "unstable", simulated_T  # SP精修不可靠，用仿真结果
```

### 6.4 与Stage5的集成方式

```python
# mainv2.py Stage5调用
if args.enable_stage5:
    run_stage5(...)

    if args.enable_physics_validation:
        from tools.physics_validator import PhysicsValidator
        validator = PhysicsValidator(sim_steps=500)
        all_instances, report = validator.validate(
            all_instances, categories_and_relations, walls_info
        )
        for category, result in report.items():
            print(f"  {category}: {result['status']} (位移={result['displacement']:.3f}m)")
```

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| FCL安装失败 (编译依赖) | Phase 1无法实施 | 回退到当前AABB+顶点采样，增加迭代次数 |
| PyBullet仿真时间过长 | Stage5运行时间增加 | 可配置sim_steps，默认500步约1-2秒 |
| VHACD分解质量差 | 凹物体碰撞检测不准 | 对分解失败的物体回退到凸包 |
| 物理仿真改变VLM意图 | "靠墙"变"落地" | 方案C的约束系统解决；Phase 2用位移阈值保护 |
| 凸包丢失凹特征 | 容器内部空间被填充 | 对"inside"关系使用VHACD而非凸包 |
| 仿真参数不通用 | 不同场景需要不同摩擦系数 | 提供参数配置接口，默认值覆盖常见场景 |

---

## 8. 实施优先级

| 优先级 | 任务 | 方案 | 预计工作量 |
|--------|------|------|-----------|
| P0 | 替换碰撞检测为FCL | A | 替换2个函数 + 增加迭代 |
| P0 | resolve_penetrations多轮迭代 | A | 修改1个参数 + 增加收敛检测 |
| P1 | 稳定性检查 (支撑面积) | A | 新增1个函数 |
| P1 | PhysicsValidator基础框架 | B | 新增1个类 |
| P2 | mesh凸包/VHACD预处理 | B | 新增预处理模块 |
| P2 | 仿真结果判定与T矩阵更新 | B | 新增判定逻辑 |
| P3 | 约束系统 | C | 扩展PhysicsValidator |
| P3 | 约束力分析 | C | 新增分析模块 |
