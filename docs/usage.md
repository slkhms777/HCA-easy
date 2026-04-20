# 使用流程

## 1. 物体视频分割

```bash
python tools/01_video_segmentation.py --sequence_folder <path_to_sequence_folder>
```

提示：

- 请确保显存和内存充足。
- 需提前使用半自动标注 notebook 来提供初始值：`sam2.ipynb`。
- 详细标注步骤见 [SAM2_USAGE.md](./SAM2_USAGE.md)。
- 对每个视角，至少需要标注一帧，通常可从第 `0` 帧开始。
- 每个物体的 mask 值必须严格对应 `meta.yaml` 中 `object_ids` 的顺序。
- 例如：第 1 个物体的 mask value 应为 `1`，第 4 个物体的 mask value 应为 `4`。
- 请确保每个物体至少在某一帧中被标注到；若第 `0` 帧中某个物体被遮挡，需要额外选择包含该物体的帧进行标注。

## 2. 物体位姿估计与多视角融合

对每个物体分别运行 FoundationPose：

```bash
python tools/04-1_fd_pose_solver.py --sequence_folder <path_to_sequence_folder> --object_idx <object_idx>
```

如果场景中有 4 个物体，则需要分别运行 4 次，例如：

```bash
python tools/04-1_fd_pose_solver.py --sequence_folder <path_to_sequence_folder> --object_idx 1
python tools/04-1_fd_pose_solver.py --sequence_folder <path_to_sequence_folder> --object_idx 2
python tools/04-1_fd_pose_solver.py --sequence_folder <path_to_sequence_folder> --object_idx 3
python tools/04-1_fd_pose_solver.py --sequence_folder <path_to_sequence_folder> --object_idx 4
```

随后合并所有物体结果，并进行插值、平滑和可视化：

```bash
python tools/04-2_fd_pose_merger.py --sequence_folder <path_to_sequence_folder>
```

调试建议：

- 如果某个视角中物体不可见，或观察角度较差，可以直接在 `meta.yaml` 中注释掉对应视角的 `serial id`。并在运行结束后恢复注释，避免不经意间影响后续代码。
- 如果渲染报错，通常与多进程渲染有关，可加上 `--single_process` 参数改为单进程渲染。

## 3. 2D 手部关键点检测

```bash
python tools/02_mp_hand_detection.py --sequence_folder <path_to_sequence_folder>
```

## 4. 3D 手部关键点生成

```bash
python tools/03_mp_3d_joints_generation.py --sequence_folder <path_to_sequence_folder>
```

## 5. MANO 手部拟合

```bash
python tools/05_mano_pose_solver_wosdf.py --sequence_folder <path_to_sequence_folder>
```

调试建议：

- 如果某个视角的 2D 关键点检测质量较差，可以在 `meta.yaml` 中注释掉该视角的 `serial id`。
- 一旦某个视角被注释移除，后续 3D 关键点拟合与 MANO 拟合阶段应保持一致。
- 如果渲染报错，通常与多进程渲染有关，可加上 `--single_process` 参数改为单进程渲染。
