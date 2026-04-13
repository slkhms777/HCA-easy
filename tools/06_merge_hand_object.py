import argparse
from pathlib import Path
import numpy as np
import os
from hocap_annotation.utils import *
from hocap_annotation.loaders import HOCapLoader

def main(sequence_folder, object_ids : list):
    sequence_folder = Path(sequence_folder)
    data_loader = HOCapLoader(sequence_folder)

    object_pose_path = sequence_folder / "processed" / "fd_pose_solver" / "fd_poses_merged_fixed.npy"
    hand_pose_path = sequence_folder / "processed" / "hand_pose_solver" / "poses_m.npy"
    initial_pose_dir = sequence_folder / "initial_poses"
    os.makedirs(initial_pose_dir, exist_ok=True)

    obj_pose = np.load(object_pose_path)
    hand_pose = np.load(hand_pose_path)
    os.system(f"cp {object_pose_path} {initial_pose_dir / 'object_initial_pose.npy'}")
    os.system(f"cp {hand_pose_path} {initial_pose_dir / 'hand_initial_pose.npy'}")

    total_objs = len(data_loader.object_textured_files)
    print(f"Total objects in the scene: {total_objs}")
    obj_ids = data_loader.object_ids
    print(f"Object IDs in the scene: {obj_ids}")
    print(obj_pose.shape)  # (num_objs, num_frames, 7)   
    # num_objs： 场景中物体数目
    # num_frames： 帧数
    # 7： 物体位姿， 格式为[qx, qy, qz, qw, x, y, z]
    print(hand_pose.shape) # (2, num_frames, 51)
    # 2： 2只手，0为右手，1为左手
    # num_frames： 帧数
    # 51： MANO参数，48 维 pose + 3 维 translation


    pose_format_path = initial_pose_dir / "pose_format.txt"
    pose_format_lines = [
        f"object_initial_pose 的 shape : {tuple(obj_pose.shape)}",
        "shape 格式 : (num_objs, num_frames, 7)\n",
        "- num_objs   :  场景中的物体数",
        "- num_frames :  帧数",
        "- 7          :  pose格式为[qx, qy, qz, qw, x, y, z]",
        f"- 场景中的物体 :  {obj_ids}",
        "\n\n",
        f"hand_initial_pose 的 shape : {tuple(hand_pose.shape)}",
        "shape 格式 : (2, num_frames, 51)\n",
        "- 2          :  2只手，0表示右手，1表示左手",
        "- num_frames :  帧数",
        "- 51         :  48 维 pose + 3 维 translation（如果某只手无效或不在视野中则全为-1）",
    ]
    pose_format_path.write_text("\n".join(pose_format_lines) + "\n", encoding="utf-8")


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Pose Solver")
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--object_ids", type=int, nargs="+", default=[1,2,3,4], help="Object IDs to merge with hand."
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder!")
    main(args.sequence_folder, args.object_ids)
"""
python tools/06_merge_hand_object.py --sequence_folder datasets/subject_5/20231027_112303
"""
