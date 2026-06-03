# 232 场景分析工具

这个目录包含了对 ReplicateAnyScene 在 232 动态场景下的详细分析脚本，用于诊断点云质量、坐标系对齐、物体摆放等问题。

## 分析脚本说明

| 文件名 | 主要分析内容 |
| ------ | ----------- |
| `analyze_232_pointcloud.py` | 点云质量与统计分布分析，包括平面结构检测、相机轨迹分析、深度图一致性检查 |
| `analyze_232_core.py` | 核心问题深度分析：桌子悬浮为什么没被 refine 修复？mesh 位置 vs 点云位置？ |
| `analyze_232_deep.py` | 坐标系对齐验证、地板/墙壁平面拟合、物体摆放位置合理性分析 |
| `analyze_232_mask_pointcloud.py` | 掩码与点云关系分析：相同掩码区域在不同帧的 3D 坐标一致性 |

## 使用方式

直接运行 Python 脚本，需要确保在 ReplicateAnyScene 根目录下或正确配置了 `sys.path`：

```bash
# 从 ReplicateAnyScene 根目录运行
cd /mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene
python tools/analysis/analyze_232_pointcloud.py
```

## 输入数据要求

脚本默认分析 `/mnt/data_8THDD/lza/workspace/robot_world_ws/src/ReplicateAnyScene/outputs/232` 目录下的数据，需要该目录包含：

- `point_cloud.ply` - 点云文件
- `final_scene.glb` / `final_scene_refined.glb` - 最终场景
- `color/` - RGB 帧目录
- `depth/` - 深度图目录
- `extrinsics/` - 相机外参目录
- `intrinsic.txt` - 相机内参
- `optimal_frames/` - 最优帧目录
