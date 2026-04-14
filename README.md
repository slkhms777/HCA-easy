# HO-Cap Annotation Pipeline (cleaned version)

## Installation

推荐环境：

- Python `3.10`
- CUDA `12.4`

### 1. 创建 Conda 环境

```bash
conda create -n hocap-annotation python=3.10
conda activate hocap-annotation
```

### 2. 安装 PyTorch `cu124`

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装 HO-Cap Annotation

```bash
bash ./scripts/install_hocap-annotation.sh
```

### 4. 下载 MANO 模型

从 [MANO 官网](https://mano.is.tue.mpg.de/) 下载 `mano_v1_2.zip`，解压后将 `.pkl` 文件放到 `config/mano_models` 目录下：

```text
config/mano_models
├── MANO_LEFT.pkl
└── MANO_RIGHT.pkl
```

### 5. 安装第三方工具

#### 5.1 安装 FoundationPose

初始化并编译：

```bash
bash ./scripts/install_foundationpose.sh
```

下载模型权重：

```bash
bash ./scripts/download_models.sh --foundationpose
```

#### 5.2 安装 SAM2

初始化并编译：

```bash
bash ./scripts/install_sam2.sh
```

下载模型权重：

```bash
bash ./scripts/download_models.sh --sam2
```

## 数据组织

### 1. 相机标定数据

#### 外参

路径：`datasets/calibration/extrinsics`

每个相机的外参文件位于：

```text
datasets/calibration/extrinsics/subject_{i}/{serial_idx}.yaml
```

示例格式：

```yaml
serial: '037522251142'
cam2world:
- - -0.8553213331977578
  - 0.1661480065171818
  - 0.4907348764212444
  - -0.6471051327967781
- - -0.5017561880194157
  - -0.5016557184433581
  - -0.7046856757906143
  - 0.4788695981184158
- - 0.12909754379323998
  - -0.8489616320203276
  - 0.512442288848331
  - 0.06658482352375994
- - 0.0
  - 0.0
  - 0.0
  - 1.0
world2cam:
- - -0.8553210379840743
  - -0.5017565411020417
  - 0.129097886857523
  - -0.32180264069144
- - 0.1661479640392903
  - -0.5016556836240906
  - -0.8489622821860211
  - 0.4042708596821313
- - 0.4907344638383433
  - -0.7046859689407771
  - 0.5124428272296073
  - 0.6208885619193231
- - 0.0
  - 0.0
  - 0.0
  - 1.0

```

#### 内参

路径：`datasets/calibration/intrinsics`

每个相机的内参文件位于：

```text
datasets/calibration/intrinsics/{serial_idx}.yaml
```

示例格式：

```yaml
serial: '037522251142'
color:
  width: 640
  height: 480
  fx: 378.4029541015625
  fy: 377.9852600097656
  ppx: 322.846435546875
  ppy: 243.7872772216797
  coeffs:
    - -0.05652647837996483
    - 0.06795103847980499
    - 0.0005720899207517505
    - -0.00012941579916514456
    - -0.021466698497533798
depth:
  width: 640
  height: 480
  fx: 387.5823974609375
  fy: 387.5823974609375
  ppx: 316.75390625
  ppy: 241.99844360351562
  coeffs:
    - 0.0
    - 0.0
    - 0.0
    - 0.0
    - 0.0
depth2color:
  - 0.999993622303009
  - -0.0006601770292036235
  - -0.003510251408442855
  - -0.0590653121471405
  - 0.0006539513124153018
  - 0.9999982118606567
  - -0.001774423522874713
  - 6.566795036633266e-06
  - 0.0035114167258143425
  - 0.0017721166368573904
  - 0.9999922513961792
  - 0.0005424162955023348
```

### 2. MANO 参数

路径：`datasets/calibration/mano`

每个被试的 `betas` 文件位于：

```text
datasets/calibration/mano/subject_{i}
```

示例格式：

```yaml
betas:
  - -0.5082021951675415
  - -0.24375076591968536
  - -0.31839972734451294
  - -0.17650160193443298
  - 0.4463734030723572
  - 0.05645660683512688
  - -0.6338333487510681
  - 0.15932975709438324
  - -0.18557298183441162
  - -0.08094348013401031
```

如果该文件不存在，则默认使用平均手型参数。

### 3. 物体 CAD 模型

路径：`datasets/models`

目录结构示例：

```text
datasets/models/G01_1/
├── textured_mesh.mtl
├── textured_mesh.obj
└── *.png 或 *.jpg
```

### 4. 序列数据

目录结构示例：

```text
datasets/subject_{i}/{date_日期}/
├── serial_1/
│   ├── color_{frame_idx:06d}.jpg
│   └── depth_{frame_idx:06d}.png
├── serial_2/
├── ...
├── serial_8/
├── processed/      # 处理中间结果
└── initial_poses/  # 最终导出的初始手-物姿态
```

### 5. 元数据

路径：`datasets/subject_{i}/meta.yaml`

示例格式：

```yaml
realsense:
  serials:
    - '037522251142'
    - '043422252387'
    - '046122250168'
    - '105322251225'
    - '105322251564'
    - '108222250342'
    - '115422250549'
    - '117222250549'
  width: 640
  height: 480
extrinsics: datasets/calibration/extrinsics/subject_5
subject_id: subject_5
object_ids:
  - G15_1
  - G15_2
  - G15_3
  - G15_4
mano_sides:
  - right
# 双手示例：
# mano_sides:
#   - right
#   - left
num_frames: 702
task_id: 1
```

说明：

- `object_ids` 表示场景中参与交互的物体 CAD 模型 ID。
- `mano_sides` 用于指定当前序列中需要拟合的手，支持单手或双手。
- `task_id` 任务类型标记，可随便填

## 使用流程

### 1. 物体视频分割

```bash
python tools/01_video_segmentation.py --sequence_folder <path_to_sequence_folder>
```

提示：

- 请确保显存和内存充足。
- 需提前使用半自动标注 notebook 来提供初始值：`sam2.ipynb`。
- 详细标注步骤见 [`SAM2_USAGE.md`](/mnt/16T/gjx/HO-Cap-Annotation/SAM2_USAGE.md)。
- 对每个视角，至少需要标注一帧，通常可从第 `0` 帧开始。
- 每个物体的 mask 值必须严格对应 `meta.yaml` 中 `object_ids` 的顺序。
- 例如：第 1 个物体的 mask value 应为 `1`，第 4 个物体的 mask value 应为 `4`。
- 请确保每个物体至少在某一帧中被标注到；若第 `0` 帧中某个物体被遮挡，需要额外选择包含该物体的帧进行标注。

### 2. 物体位姿估计与多视角融合

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

### 3. 2D 手部关键点检测

```bash
python tools/02_mp_hand_detection.py --sequence_folder <path_to_sequence_folder>
```

![2d_hand_detection](./docs/resources/02_2d_hand_detection.png)

### 4. 3D 手部关键点生成

```bash
python tools/03_mp_3d_joints_generation.py --sequence_folder <path_to_sequence_folder>
```

![3d_hand_joints_estimation](./docs/resources/03_3d_hand_joints.png)

### 5. MANO 手部拟合

```bash
python tools/05_mano_pose_solver_wosdf.py --sequence_folder <path_to_sequence_folder>
```

调试建议：

- 如果某个视角的 2D 关键点检测质量较差，可以在 `meta.yaml` 中注释掉该视角的 `serial id`。
- 一旦某个视角被注释移除，后续 3D 关键点拟合与 MANO 拟合阶段应保持一致。
- 如果渲染报错，通常与多进程渲染有关，可加上 `--single_process` 参数改为单进程渲染。
