# conda activate hocap-annotation

# 物体视频分割 + 预测物体pose
# CUDA_VISIBLE_DEVICES=1 python tools/01_video_segmentation.py --sequence_folder datasets/subject_example/20231027_112303
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example/20231027_112303 --object_idx 1
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example/20231027_112303 --object_idx 2
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example/20231027_112303 --object_idx 3
CUDA_VISIBLE_DEVICES=1 python tools/04-1_fd_pose_solver.py --sequence_folder datasets/subject_example/20231027_112303 --object_idx 4
CUDA_VISIBLE_DEVICES=1 python tools/04-2_fd_pose_merger.py --sequence_folder datasets/subject_example/20231027_112303 --single_process

# 2d关键点检测 + 3d关键点拟合 + MANO pose 拟合
CUDA_VISIBLE_DEVICES=1 python tools/02_mp_hand_detection.py --sequence_folder datasets/subject_example/20231027_112303
CUDA_VISIBLE_DEVICES=1 python tools/03_mp_3d_joints_generation.py --sequence_folder datasets/subject_example/20231027_112303
CUDA_VISIBLE_DEVICES=1 python tools/05_mano_pose_solver_wosdf.py --sequence_folder datasets/subject_example/20231027_112303 --single_process