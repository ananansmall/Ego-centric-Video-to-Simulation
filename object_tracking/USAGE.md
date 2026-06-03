# 物体追踪管线使用指南

## 一句话概述

从第一人称操作视频中，检测手-物接触事件，追踪被操作物体的 6DoF 轨迹，输出可用于仿真环境的物体位姿序列。

## 管线总览

```
输入 MP4 视频 (如 232.mp4)
    │
    ├─→ Step 0a: VGGT-Omega predict.py  ──→ 232/color/ depth/ extrinsics/ intrinsic.txt
    ├─→ Step 0b: HaWoR demov2.py         ──→ 232/hawor_results.npz
    └─→ Step 0c: ReplicateAnyScene main.py ──→ 232/object_masks.npz
           │
           ▼
    ┌──────────────────────────────────────────┐
    │  run_tracking.py --video 232.mp4         │
    │                                          │
    │  Step 1: 加载 VGGT-Omega (深度+相机)     │
    │  Step 2: 加载 HaWoR (手部 MANO 参数)     │
    │  Step 3: 加载 SAM3 (物体分割 mask)        │
    │  Step 4: 接触检测 (指尖深度 vs 场景深度)  │
    │  Step 5: 6DoF 轨迹追踪 (Procrustes 对齐) │
    │  Step 6: 保存结果                         │
    └──────────────────────────────────────────┘
           │
           ▼
    232_tracking/
    ├── trajectories/donut_inst0.npz    ← 物体 6DoF 轨迹 (S,4,4)
    ├── tracking_summary.json           ← 接触信息汇总
    ├── vis_contact/                    ← 接触可视化 (红色=接触, 绿点=指尖)
    ├── vis_trajectory/                 ← 轨迹投影可视化
    └── trajectories_3d.glb             ← 3D 轨迹 (MeshLab 打开)
```

## 前置步骤：生成中间文件

### Step 0a: VGGT-Omega 3D 感知

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/vggt-omega
conda activate ReplicateAnyScene

python predict.py \
  --input_video /mnt/data_8THDD/lza/workspace/robot_world_ws/src/egodex_small/test/basic_pick_place/232.mp4 \
  --output_dir /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs \
  --output_name 232 \
  --sample_fps 2.0 \
  --max_frames 160
```

**输出**: `ReplicateAnyScene/outputs/232/`
- `color/0.jpg, 1.jpg, ...` — RGB 帧
- `depth/0.png, 1.png, ...` — 深度图 (16位, 单位mm)
- `extrinsics/0.txt, 1.txt, ...` — 4×4 外参矩阵
- `intrinsic.txt` — 3×3 内参矩阵

### Step 0b: HaWoR 手部重建

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/HaWoR
conda activate hawor

python demov2.py \
  --video /mnt/data_8THDD/lza/workspace/robot_world_ws/src/egodex_small/test/basic_pick_place/232.mp4
```

**输出**: `HaWoR/example/232/reconstruction/hawor_results_0_XXX.npz`
- `pred_trans` (2, T, 3) — 双手平移
- `pred_rot` (2, T, 3) — 双手旋转
- `pred_hand_pose` (2, T, 45) — 双手关节
- `pred_betas` (2, T, 10) — 双手 shape

### Step 0c: SAM3 物体分割

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
conda activate ReplicateAnyScene

python main.py \
  --video_path /mnt/data_8THDD/lza/workspace/robot_world_ws/src/egodex_small/test/basic_pick_place/232.mp4 \
  --category_path /path/to/categories.json \
  --output_path outputs/232
```

**输出**: `ReplicateAnyScene/outputs/232/`
- `final_scene.glb` — 完整 3D 场景
- `instance_masks.mp4` — 实例分割可视化
- `optimal_frames/` — 最优视角帧

## 运行物体追踪

### 最简用法 (自动查找中间文件)

```bash
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/object_tracking
conda activate ReplicateAnyScene

python run_tracking.py \
  --video /mnt/data_8THDD/lza/workspace/robot_world_ws/src/egodex_small/test/basic_pick_place/232.mp4
```

脚本会自动从 `ReplicateAnyScene/outputs/232/` 查找 VGGT-Omega 输出，从 `HaWoR/example/232/` 查找 HaWoR 输出。

### 手动指定中间文件

```bash
python run_tracking.py \
  --video /path/to/232.mp4 \
  --vggt_dir /path/to/vggt_output/232 \
  --hawor_npz /path/to/hawor_results.npz \
  --masks_file /path/to/object_masks.npz \
  --output /path/to/output/232_tracking
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--video` | (必填) | 输入 MP4 文件路径 |
| `--base_dir` | `ReplicateAnyScene/outputs/` | 中间文件基础目录 |
| `--output` | `{base_dir}/{video_name}_tracking/` | 输出目录 |
| `--vggt_dir` | 自动查找 | 手动指定 VGGT-Omega 输出目录 |
| `--hawor_npz` | 自动查找 | 手动指定 HaWoR npz 文件 |
| `--masks_file` | 自动查找 | 手动指定物体 mask 文件 |
| `--depth_margin` | 0.05 | 接触检测深度容差 (米) |
| `--dilate_radius` | 10 | 接触区域膨胀半径 (像素) |
| `--conf_thres` | 30.0 | 点云置信度百分位阈值 |
| `--skip_contact_vis` | False | 跳过接触可视化 |
| `--skip_trajectory_vis` | False | 跳过轨迹可视化 |

## 输出文件说明

### `232_tracking/` 目录结构

```
232_tracking/
├── trajectories/                    ← 逐物体 6DoF 轨迹
│   ├── donut_inst0.npz              每个被操作的物体一个文件
│   │   ├── trajectory (S, 4, 4)     齐次变换矩阵: 物体局部→世界坐标
│   │   ├── centroids (S, 3)         物体质心轨迹
│   │   ├── valid_frames (S,)        有效帧标记
│   │   └── contact_frames (N,)      接触帧索引
│   └── ...
├── tracking_summary.json            ← 接触信息汇总
│   {
│     "donut_inst0": {
│       "category": "donut",
│       "interaction_type": "grasp_and_release",
│       "contact_frames": [14, 15, 16, ...],
│       "segments": [[14, 45]],
│       "release_frame": 45
│     }
│   }
├── vis_contact/                     ← 接触可视化帧
│   ├── frame_0000.jpg               红色overlay=接触区域
│   └── ...                          绿点=接触指尖, 蓝点=未接触指尖
├── vis_trajectory/                  ← 轨迹投影可视化帧
│   ├── frame_0000.jpg               黄色圆点+标签=物体质心投影
│   └── ...
└── trajectories_3d.glb              ← 3D 轨迹 (MeshLab/Blender 打开)
```

### `trajectory (S, 4, 4)` 的含义

每个 4×4 矩阵是一个齐次变换，将物体局部坐标映射到世界坐标：

```python
p_world = trajectory[t][:3, :3] @ p_local + trajectory[t][:3, 3]
```

- `trajectory[t][:3, :3]` — 3×3 旋转矩阵
- `trajectory[t][:3, 3]` — 3×1 平移向量 (世界坐标系中的位置)

## 文件查找逻辑

脚本按以下顺序自动查找中间文件：

| 数据 | 查找路径 |
|------|---------|
| VGGT-Omega | `{base_dir}/{video_name}/` → `vggt-omega/output/{video_name}/` → `ReplicateAnyScene/outputs/{video_name}/` |
| HaWoR | `{base_dir}/{video_name}/hawor_results.npz` → `HaWoR/example/{video_name}/reconstruction/hawor_results*.npz` |
| 物体 mask | `{base_dir}/{video_name}/object_masks.npz` → `object_masks.json` |

## 与仿真环境对接

物体追踪的输出可以直接用于 GalaxeaManipSim 仿真：

```python
import numpy as np
from galaxea_sim.utils.robotwin_utils import create_glb

# 加载物体轨迹
traj = np.load("232_tracking/trajectories/donut_inst0.npz")
trajectory = traj["trajectory"]  # (S, 4, 4)
contact_frames = traj["contact_frames"]

# 加载物体 mesh (来自 ReplicateAnyScene)
# 将物体导入仿真为动态刚体
actor = create_glb(scene, "outputs/232/final_scene.glb", is_static=False)

# 逐帧设置物体位姿
for t in range(len(trajectory)):
    R = trajectory[t, :3, :3]
    t_vec = trajectory[t, :3, 3]
    actor.set_pose(sapien.Pose(t_vec, R))
```

## 当前 232.mp4 的运行状态

| 数据 | 状态 | 路径 |
|------|------|------|
| VGGT-Omega | ✅ 已有 | `ReplicateAnyScene/outputs/232/` (120帧) |
| HaWoR | ❌ 未生成 | 需要运行 `HaWoR/demov2.py` |
| 物体 mask | ❌ 未生成 | 需要运行 `ReplicateAnyScene/main.py` 或 `extract_masks.py` |
| 追踪结果 | ⚠️ 部分完成 | `ReplicateAnyScene/outputs/232_tracking/` (无接触/轨迹, 因为缺 HaWoR 和 mask) |
