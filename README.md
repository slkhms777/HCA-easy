# HO-Cap Annotation Pipeline (cleaned version)

## Clone Repository

只包含代码和普通小文件，不自动下载 Git LFS 管理的大文件，请使用：

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone <repo_url>
cd HO-Cap-Annotation
```

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

## 文档

- [数据组织](/mnt/16T/gjx/HO-Cap-Annotation/docs/data_organization.md)
- [测试数据](/mnt/16T/gjx/HO-Cap-Annotation/docs/test_data.md)
- [使用流程](/mnt/16T/gjx/HO-Cap-Annotation/docs/usage.md)
- [SAM2 半自动标注说明](/mnt/16T/gjx/HO-Cap-Annotation/docs/SAM2_USAGE.md)
