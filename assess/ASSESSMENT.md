# ReplicateAnyScene 评估系统

## 1. 评估维度总览

| 维度 | 文件 | GT 来源 | 独立性 | 可靠性 |
|------|------|---------|--------|--------|
| 视觉质量 | `visual_metrics.py` | 原始视频帧 | ✅ 独立 | ⭐⭐⭐ 高 |
| 空间精度 | `geometry_metrics.py` | VGGT 深度图 | ⚠️ 半独立 | ⭐⭐ 中 |
| 几何精度 (Mesh) | `geometry_metrics.py` | 外部 GT Mesh | ✅ 独立 | ⭐⭐⭐ 高 |
| 几何精度 (点云) | `geometry_metrics.py` | 外部 GT 点云 | ✅ 独立 | ⭐⭐⭐ 高 |
| 文本完整性 | `textual_metrics.py` | 人工标注 JSON | ✅ 独立 | ⭐⭐⭐ 高 |

---

## 2. 详细指标说明

### 2.1 视觉质量 (visual_metrics.py)

**原理**：用 pyrender 从 VGGT 相机位姿渲染 `final_scene.glb`，和原始视频帧对比。

```
final_scene.glb + VGGT相机位姿 → pyrender渲染 → 渲染图
                                                    ↓ 对比
                                              原始视频帧 (color/)
```

**全图指标**：

| 指标 | 含义 | 方向 | 论文参考值 |
|------|------|------|-----------|
| PSNR | 像素级重建误差 (dB) | ↑ 越高越好 | ~19.45 dB |
| SSIM | 结构相似性 | ↑ 越高越好 | ~0.854 |
| LPIPS | 感知相似度 | ↓ 越低越好 | 默认开启 |
| render_coverage | 渲染覆盖率（非黑色像素比例） | ↑ | - |

**仅物体区域指标 (masked)**：

| 指标 | 含义 | 说明 |
|------|------|------|
| PSNR_masked | 仅在渲染图和GT都有内容的区域计算 | 排除空白区域的影响 |
| SSIM_masked | 同上 | 排除空白区域的影响 |
| LPIPS_masked | 同上 | 排除空白区域的影响 |

**为什么独立**：渲染图来自 SAM3D 生成的 3D mesh，原始帧来自相机拍摄。两者完全独立。

**注意**：RAS 的 GLB 只包含物体 mesh，不包含墙壁/地板/天花板。全图指标会被大面积空白拉低，**masked 指标才是物体重建质量的真实反映**。

### 2.2 空间精度 (geometry_metrics.py)

**原理**：渲染场景深度图，和 VGGT 输出的深度图对比。

```
final_scene.glb + VGGT相机位姿 → pyrender渲染 → 渲染深度图
                                                    ↓ 对比
                                              VGGT深度图 (depth/)
```

| 指标 | 含义 | 方向 | 说明 |
|------|------|------|------|
| MaskIoU | 轮廓重合度 | ↑ | 渲染 mask vs VGGT depth mask |
| DepthRMSE | 深度绝对误差 (m) | ↓ | 只在两者都有深度的像素计算 |
| DepthAbsRel | 深度相对误差 | ↓ | RMSE / GT深度，消除尺度影响 |

**为什么半独立**：
- VGGT 深度图 = 多视角 3D 重建（输入：原始视频帧）
- SAM3D mesh = 单帧分割 + 3D 生成（输入：SAM3 mask + VGGT 点云）
- 迭代对齐 = 把 mesh 摆到正确位置（输入：VGGT 深度 + 相机位姿）
- **测的是"摆得对不对"，不是"3D模型好不好"**
- 但 VGGT 深度本身有误差，不是真实 GT

### 2.3 几何精度 - Mesh vs GT Mesh (geometry_metrics.py)

**原理**：从生成 mesh 和 GT mesh 表面采样点云，计算标准 3D 重建指标。

```
final_scene.glb → 采样点云 → pred_points, pred_normals
GT mesh (GLB/OBJ/PLY) → 采样点云 → gt_points, gt_normals
                                    ↓ 对比
                              CD / F-Score / NC / Accuracy / Completeness
```

**指标**（遵循 ObjectSDF++ / MonoSDF 评估协议）：

| 指标 | 含义 | 方向 | 来源 |
|------|------|------|------|
| Chamfer Distance L2 | 双向平均最近邻距离² | ↓ | ObjectSDF++, MonoSDF |
| Chamfer Distance L1 | 双向平均最近邻距离 | ↓ | MonoSDF |
| Accuracy | pred→GT 平均最近邻距离 | ↓ | MonoSDF |
| Completeness | GT→pred 平均最近邻距离 | ↓ | MonoSDF |
| Overall | (Accuracy + Completeness) / 2 | ↓ | MonoSDF |
| F-Score@τ | 距离 < τ 的点比例 | ↑ | ObjectSDF++ |
| Normal Consistency | 法线点积均值 | ↑ | ObjectSDF++, MonoSDF |

**需要**：`--reference_mesh` 参数提供 GT mesh 文件路径。

### 2.4 几何精度 - 点云 vs GT 点云 (geometry_metrics.py)

**原理**：直接对比点云，无需 mesh 采样。指标与 Mesh 评估相同。

| 指标 | 说明 |
|------|------|
| CD-L2/L1, Accuracy, Completeness, Overall, F-Score | 同 Mesh 评估 |
| Normal Consistency | 需要点云包含法线信息 |

**需要**：`--reference_ply` 参数提供 GT 点云文件路径。

### 2.5 文本完整性 (textual_metrics.py)

**原理**：对比检测到的类别和最终 GLB 中生成的类别。

```
Stage1 JSON: {"cup", "table", "bottle"}     ← 检测到的
final_scene.glb geometry names: {"cup_0", "table_0", "bottle_0"}  ← 生成的
                                                    ↓ 对比
                                              人工标注 JSON (可选)
```

| 指标 | 含义 | 方向 | 需要 GT |
|------|------|------|---------|
| Recall | 检测到了多少真实类别 | ↑ | 是 |
| Precision | 检测结果中有多少是对的 | ↑ | 是 |
| F1 | 精确率和召回率的调和平均 | ↑ | 是 |
| SRR | 语义冗余率（重复检测比例） | ↓ | 否 |

---

## 3. 依赖库

| 库 | 安装方式 | 提供的指标 | 是否必需 |
|------|---------|-----------|---------|
| scikit-image | `pip install scikit-image` | PSNR, SSIM | ✅ 必需 |
| lpips | `pip install lpips` | LPIPS | 可选（默认开启） |
| scipy | `pip install scipy` | CD, F-Score, NC, Accuracy, Completeness | ✅ 必需 |
| pyrender | `pip install pyrender` | 渲染 | ✅ 必需 |
| open3d | `pip install open3d` | 点云操作 | 点云评估需要 |
| trimesh | `pip install trimesh` | Mesh 操作 | ✅ 必需 |

**注意**：未使用 PyTorch3D。所有核心功能（CD、点云操作）已用 `scipy.spatial.KDTree` + `trimesh` + `open3d` 实现，避免 PyTorch3D 复杂的编译安装。

---

## 4. 运行方式

```bash
# 基础评估（视觉 + 空间精度，不需要 GT mesh）
python -m assess.run_assessment --output_path ./outputs/232

# 采样 10 帧加速
python -m assess.run_assessment --output_path ./outputs/232 --sample_count 10

# 跳过 LPIPS 加速
python -m assess.run_assessment --output_path ./outputs/232 --skip_lpips

# 有 GT mesh 时，计算完整几何指标（CD/F-Score/NC/Accuracy/Completeness）
python -m assess.run_assessment --output_path ./outputs/232 \
    --reference_mesh /path/to/ground_truth_mesh.glb

# 有 GT 点云时
python -m assess.run_assessment --output_path ./outputs/232 \
    --reference_ply /path/to/ground_truth.ply

# 自定义 F-Score 阈值
python -m assess.run_assessment --output_path ./outputs/232 \
    --reference_mesh ./gt.glb --f_thresholds 0.01 0.05 0.1

# 自定义采样点数
python -m assess.run_assessment --output_path ./outputs/232 \
    --reference_mesh ./gt.glb --num_sample_points 200000

# 文本评估（需要 GT JSON）
python -m assess.run_assessment --output_path ./outputs/232 \
    --category_path ./categories.json \
    --ground_truth_json ./gt_categories.json

# 跳过某些评估
python -m assess.run_assessment --output_path ./outputs/232 \
    --skip_visual --skip_geometry --skip_textual
```

**输出**：`{output_path}/assessment_results.json`

---

## 5. CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_path` | (必需) | main.py 的输出目录 |
| `--glb_path` | 自动检测 | 自定义 GLB 文件路径 |
| `--category_path` | 自动检测 | 类别 JSON 路径 |
| `--ground_truth_json` | None | GT 类别 JSON |
| `--reference_mesh` | None | GT mesh 路径（GLB/OBJ/PLY） |
| `--reference_ply` | None | GT 点云路径（PLY） |
| `--skip_visual` | False | 跳过视觉评估 |
| `--skip_geometry` | False | 跳过空间精度评估 |
| `--skip_textual` | False | 跳过文本评估 |
| `--skip_lpips` | False | 跳过 LPIPS（加速） |
| `--sample_count` | None | 采样帧数 |
| `--num_sample_points` | 100000 | Mesh 评估采样点数 |
| `--f_thresholds` | [0.01, 0.02, 0.05] | F-Score 阈值列表 |

---

## 6. 当前测试结果 (232 场景)

### 6.1 全图 vs 仅物体区域

| 指标 | 全图 | 仅物体区域 (masked) | 说明 |
|------|------|---------------------|------|
| PSNR | 12.27 dB | **16.07 dB** | 全图被空白拉低 |
| SSIM | 0.3748 | **0.7304** | 同上 |
| LPIPS | 0.6821 | **0.4527** | 同上 |
| 渲染覆盖率 | 62.5% | - | 37%画面为黑色 |

### 6.2 空间精度

| 指标 | 值 | 说明 |
|------|-----|------|
| MaskIoU | 0.6205 | 轮廓重合约 62% |
| DepthRMSE | 0.4123 m | 深度误差约 41cm |
| DepthAbsRel | 0.9875 | 相对深度误差约 99% |

### 6.3 几何精度

需要 GT mesh 才能计算，待 Replica 数据集下载完成后评估。

---

## 7. 论文评估方式对比

| 维度 | 论文方法 | 我们的实现 | 状态 |
|------|----------|-----------|------|
| 视觉 | Blender 渲染 GT vs 生成 | pyrender 渲染 vs 原始帧 | ✅ 已实现 |
| 几何 (深度) | GT 深度 vs 渲染深度 | VGGT 深度 vs 渲染深度 | ✅ 已实现 |
| 几何 (Mesh) | GT mesh vs 生成 mesh → CD/F-Score/NC | ✅ 同 | ✅ 已实现 |
| 几何 (点云) | GT 点云 vs 生成点云 → CD/F-Score/NC | ✅ 同 | ✅ 已实现 |
| 文本 | 人工标注 vs 检测 → Rec/F1 | ✅ 同 | ✅ 已实现 |

**与论文的差距**：
- 视觉评估：论文用 Blender 渲染 GT 场景，我们用原始帧（更直接但光照可能不同）
- 几何评估：论文用专业建模师手动建模作为 GT，我们用 Replica/ScanNet 的 GT mesh
- **核心指标（CD/F-Score/NC/Accuracy/Completeness）已与论文对齐**

---

## 8. 数据集支持

### 8.1 Replica 数据集（推荐用于正式评估）

- 18 个室内场景，完美 GT mesh + HDR 纹理 + 语义标注
- 下载：`https://github.com/facebookresearch/Replica-Dataset`
- 大小：~32GB（17个 part 文件）
- 评估时使用 `--reference_mesh` 指向 `mesh.ply`

### 8.2 ScanNet 数据集

- 1,513 个真实室内场景，RGB-D + GT mesh + 语义标注
- 申请：`http://www.scan-net.org/`
- 测试脚本：`test_scannet.py`
- 评估时使用 `--reference_mesh` 指向 `_vh_clean.ply`

### 8.3 NeRF Synthetic 数据集

- 8 个合成物体场景，多视角图片 + 相机位姿
- **无 GT mesh**，只能做视觉质量评估
- 下载：`wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip`
- 大小：~700MB

---

## 9. 文件结构

```
assess/
├── ASSESSMENT.md              ← 本文档
├── __init__.py
├── visual_metrics.py          ← PSNR / SSIM / LPIPS (全图 + masked) / render_coverage
├── geometry_metrics.py        ← MaskIoU / DepthRMSE / DepthAbsRel / CD / F-Score / NC / Accuracy / Completeness
├── textual_metrics.py         ← Recall / Precision / F1 / SRR
└── run_assessment.py          ← 主脚本：渲染 + 评估 + 保存结果

test_scannet.py                ← ScanNet 测试脚本（图片输入 + 自动生成JSON）
```

---

## 10. 修复记录

### 10.1 渲染坐标系修复 (已完成)

**问题**：main.py 保存 GLB 时将场景从 z-up 转换为 y-up，VGGT 的 extrinsic 是 z-up 坐标系。

**解决方案**：
```python
c2w_opencv = np.linalg.inv(extrinsic)
opencv_to_opengl = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
c2w_opengl = c2w_opencv @ opencv_to_opengl
zup_to_yup = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]])
cam_pose = zup_to_yup @ c2w_opengl
```

### 10.2 深度图单位

当前假设 depth/*.png 是 uint16 毫米：
```python
gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.0
```

### 10.3 LPIPS 模型缓存

LPIPS 模型只加载一次（全局变量 `_lpips_fn`），避免重复加载。

---

## 11. 变更历史

| 日期 | 变更 |
|------|------|
| 2026-05-19 | 初始版本：视觉+空间+文本三维评估 |
| 2026-05-19 | 修复 pyrender EGL 渲染（服务器无显示器） |
| 2026-05-19 | 几何评估从 VGGT 点云 CD 改为渲染深度 vs VGGT 深度（避免循环论证） |
| 2026-05-22 | 新增 Mesh 几何评估：CD-L2/L1, Accuracy, Completeness, Overall, F-Score, Normal Consistency |
| 2026-05-22 | 新增点云几何评估：同上指标 |
| 2026-05-22 | LPIPS 默认开启，新增 `--skip_lpips` 参数 |
| 2026-05-22 | 新增 masked 视觉指标：PSNR_masked, SSIM_masked, LPIPS_masked |
| 2026-05-22 | 新增 render_coverage 指标 |
| 2026-05-22 | 新增 `--reference_mesh`, `--reference_ply`, `--num_sample_points`, `--f_thresholds` 参数 |
| 2026-05-22 | 创建 test_scannet.py ScanNet 测试脚本 |
| 2026-05-23 | 更新 ASSESSMENT.md，对齐实际代码 |
