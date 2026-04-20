#!/bin/bash
set -e

# 1. conda activate hocap-annotation
# 2. 需要提前标注好sam2的初始mask

# 物体视频分割 + 预测物体pose
python tools/01_video_segmentation.py --sequence_folder datasets/subject_example2/card_box3
python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example2/card_box3 --object_idx 1
python tools/04-2_fd_pose_merger.py --sequence_folder datasets/subject_example2/card_box3 --single_process

# 2d关键点检测 + 3d关键点拟合 + MANO pose 拟合
python tools/02_mp_hand_detection.py --sequence_folder datasets/subject_example2/card_box3
python tools/03_mp_3d_joints_generation.py --sequence_folder datasets/subject_example2/card_box3
python tools/05_mano_pose_solver_wosdf.py --sequence_folder datasets/subject_example2/card_box3 --single_process
python tools/06_merge_hand_object.py --sequence_folder datasets/subject_example2/card_box3
