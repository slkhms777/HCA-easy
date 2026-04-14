#!/bin/bash
set -e

# 1. conda activate hocap-annotation
# 2. 需要提前标注好sam2的初始mask

# 物体视频分割 + 预测物体pose
CUDA_VISIBLE_DEVICES=1 python tools/01_video_segmentation.py --sequence_folder datasets/subject_example1/20231027_112303
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example1/20231027_112303 --object_idx 1
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example1/20231027_112303 --object_idx 2
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example1/20231027_112303 --object_idx 3

# 实测发现object_idx 4的物体在部分视角中被遮挡较严重，在meta.yaml注释掉 037522251142，046122250168，115422250549 再运行下面这一行即可
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example1/20231027_112303 --object_idx 4

CUDA_VISIBLE_DEVICES=1 python tools/04-2_fd_pose_merger.py --sequence_folder datasets/subject_example1/20231027_112303 --single_process

# 2d关键点检测 + 3d关键点拟合 + MANO pose 拟合
CUDA_VISIBLE_DEVICES=1 python tools/02_mp_hand_detection.py --sequence_folder datasets/subject_example1/20231027_112303
CUDA_VISIBLE_DEVICES=1 python tools/03_mp_3d_joints_generation.py --sequence_folder datasets/subject_example1/20231027_112303
CUDA_VISIBLE_DEVICES=1 python tools/05_mano_pose_solver_wosdf.py --sequence_folder datasets/subject_example1/20231027_112303 --single_process
