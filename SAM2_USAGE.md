# SAM2 半自动标注使用说明

本文档说明如何使用仓库根目录下的 [`sam2.ipynb`](/mnt/16T/gjx/HO-Cap-Annotation/sam2.ipynb) 为后续视频分割提供初始化 mask。

## 1. 适用场景

- 用于给 `tools/01_video_segmentation.py` 提供初始标注。
- 标注结果会保存到序列目录下的 `processed/segmentation/init/<serial>/mask_XXXXXX.png`。

## 2. 标注前要求

- 对每个视角，至少需要标注一帧，通常可从第 `0` 帧开始。
- 每个物体的 mask 值必须严格对应 `meta.yaml` 中 `object_ids` 的顺序。
- 例如：第 1 个物体的 mask value 应为 `1`，第 4 个物体的 mask value 应为 `4`。
- 请确保每个物体至少在某一帧中被标注到；若第 `0` 帧中某个物体被遮挡，需要额外选择包含该物体的帧进行标注。

## 3. 启动方式

在项目根目录启动 Jupyter，然后打开 [`sam2.ipynb`](/mnt/16T/gjx/HO-Cap-Annotation/sam2.ipynb)。

notebook 中默认会读取：

- `SEQUENCE_DIR`：当前要标注的序列目录
- `OUTPUT_SUBDIR`：默认输出目录 `processed/segmentation/init`

如需切换序列，优先修改 notebook 里的 `SEQUENCE_DIR`，或在界面中的 `Sequence` 输入框中直接填写。

## 4. 基本工作流

### 4.1 选择序列与视角

- 点击 `Refresh`，读取当前序列信息。
- 在 `Serial` 下拉框中选择一个视角。
- 在 `Frame` 或 `Frame idx` 中选择待标注帧，通常先从第 `0` 帧开始。
- 点击 `Load Frame` 加载图像。

### 4.2 添加提示点


- `Click mode` 选择前景点 `Foreground (+)` 或背景点 `Background (-)`。
- 如果 notebook 支持 live canvas，可以直接在左侧图像上点击加点。
- 如果不支持 live canvas，可以手动填写 `X`、`Y` 后点击 `Add Point`。
- 点错时可用 `Remove Last Point` 删除最后一个点。
*一般而言，每个物体选1-2个点即可，如果物体和背景颜色相近，可以在背景加一个`Background (-)`点*

### 4.3 预览与提交 mask

- 点击 `Preview Mask` 生成当前物体的候选 mask。
- 在 `Pixel value` 中填写该物体对应的标签值。
- 标签值必须与 `meta.yaml` 里的 `object_ids` 顺序一一对应，不能跳号，也不能随意重排。
- 点击 `Commit Mask` 将当前物体写入 composite mask。

## 5. 标签值规则

假设 `meta.yaml` 中：

```yaml
object_ids:
  - G01_1
  - G05_1
  - G12_1
  - G15_4
```

则标注时必须使用：

- 第 1 个物体 `G01_1` -> `Pixel value = 1`
- 第 2 个物体 `G05_1` -> `Pixel value = 2`
- 第 3 个物体 `G12_1` -> `Pixel value = 3`
- 第 4 个物体 `G15_4` -> `Pixel value = 4`

不要把 mask value 写成任意实例编号或 CAD 名称编号，后续程序按顺序解析。

## 6. 多物体与遮挡处理

- 同一帧中可连续标注多个物体，每个物体提交一次 `Commit Mask`。
- 如果某个物体在第 `0` 帧不可见或被严重遮挡，请切换到能看见它的其他帧补标。
- 最终目标是：每个视角至少有一帧初始化标注，且每个物体至少在某一帧出现过一次有效标注。

## 7. 保存与检查

- 点击 `Save Mask` 将当前帧的 composite mask 保存到磁盘。
- `Load Saved Mask` 可加载当前帧已保存结果继续修改。
- 建议每标完一帧就保存一次，并切换回查看确认标签值是否正确。

保存路径示例：

```text
datasets/subject_5/20231027_112303/processed/segmentation/init/241222074933/mask_000000.png
```

## 8. 标注完成后

完成初始化标注后，运行：

```bash
python tools/01_video_segmentation.py --sequence_folder <path_to_sequence_folder>
```

该脚本会读取 `processed/segmentation/init` 下的初始 mask，并生成完整视频分割结果。

## 9. 常见错误

- 只标了部分视角：会导致某些相机没有初始化 mask。
- `Pixel value` 与 `object_ids` 顺序不一致：会导致物体 ID 对错。
- 第 `0` 帧没看到某个物体也没有补标其他帧：后续传播时该物体可能完全缺失。
- 预览后没有 `Commit Mask` 就直接保存：最终文件里不会包含该物体。
