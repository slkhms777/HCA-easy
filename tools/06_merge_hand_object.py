import argparse
from pathlib import Path
import numpy as np
import shutil
from ruamel.yaml import YAML


def _check_file_exists(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Required file does not exist: {file_path}")


def _read_meta(sequence_folder):
    meta_path = sequence_folder / "meta.yaml"
    _check_file_exists(meta_path)
    yaml = YAML(typ="safe")
    with meta_path.open("r", encoding="utf-8") as f:
        return yaml.load(f)


def main(sequence_folder, object_ids: list):
    sequence_folder = Path(sequence_folder)
    meta = _read_meta(sequence_folder)
    rs_serials = meta["realsense"]["serials"]
    meta_object_ids = meta["object_ids"]

    object_pose_path = sequence_folder / "processed" / "fd_pose_solver" / "fd_poses_merged_fixed.npy"
    hand_pose_path = sequence_folder / "processed" / "hand_pose_solver" / "poses_m.npy"
    kpt_2d_path = sequence_folder / "processed" / "hand_detection" / "mp_joints_3d_projection.npz"
    kpt_3d_path = sequence_folder / "processed" / "hand_detection" / "mp_joints_3d_interpolated.npy"
    for file_path in [object_pose_path, hand_pose_path, kpt_2d_path, kpt_3d_path]:
        _check_file_exists(file_path)

    initial_pose_dir = sequence_folder / "initial_poses"
    initial_pose_dir.mkdir(parents=True, exist_ok=True)

    obj_pose = np.load(object_pose_path)
    hand_pose = np.load(hand_pose_path)
    kpt_3d = np.load(kpt_3d_path)
    with np.load(kpt_2d_path) as kpt_2d:
        kpt_2d_shapes = {key: tuple(kpt_2d[key].shape) for key in kpt_2d.files}

    saved_files = {
        "object": initial_pose_dir / "object_initial_pose.npy",
        "hand": initial_pose_dir / "hand_initial_pose.npy",
        "kpt_2d": initial_pose_dir / "hand_kpt2d.npz",
        "kpt_3d": initial_pose_dir / "hand_kpt3d.npy",
    }
    shutil.copy2(object_pose_path, saved_files["object"])
    shutil.copy2(hand_pose_path, saved_files["hand"])
    shutil.copy2(kpt_2d_path, saved_files["kpt_2d"])
    shutil.copy2(kpt_3d_path, saved_files["kpt_3d"])

    total_objs = len(meta_object_ids)
    print(f"Total objects in the scene: {total_objs}")
    obj_ids = meta_object_ids
    print(f"Object IDs in the scene: {obj_ids}")
    print(obj_pose.shape)  # (num_objs, num_frames, 7)
    # num_objs： 场景中物体数目
    # num_frames： 帧数
    # 7： 物体位姿， 格式为[qx, qy, qz, qw, x, y, z]
    print(hand_pose.shape)  # (2, num_frames, 51)
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

    kpt_info_path = initial_pose_dir / "kpt_info.txt"
    kpt_2d_lines = [
        f"- {serial}: {kpt_2d_shapes[serial]}"
        for serial in rs_serials
        if serial in kpt_2d_shapes
    ]
    extra_kpt_2d_lines = [
        f"- {serial}: {shape}"
        for serial, shape in kpt_2d_shapes.items()
        if serial not in rs_serials
    ]
    kpt_info_lines = [
        f"hand_kpt2d.npz",
        "文件格式: npz，每个 key 是一个 RealSense/kinect 相机序列号。",
        "每个相机数组 shape 格式: (2, num_frames, 21, 2)",
        "- 2          : 2只手，0表示右手，1表示左手",
        "- num_frames : 帧数",
        "- 21         : MediaPipe hand joints",
        "- 2          : 图像坐标 [u, v]",
        "- 无效或缺失的关键点为 -1",
        "相机序列号与对应 shape:",
        *kpt_2d_lines,
        *extra_kpt_2d_lines,
        "",
        f"hand_kpt3d.npy",
        f"shape: {tuple(kpt_3d.shape)}",
        "shape 格式: (2, num_frames, 21, 3)",
        "- 2          : 2只手，0表示右手，1表示左手",
        "- num_frames : 帧数",
        "- 21         : MediaPipe hand joints",
        "- 3          : 世界坐标 [x, y, z]",
        "- 无效或缺失的关键点 -1",
    ]
    kpt_info_path.write_text("\n".join(kpt_info_lines) + "\n", encoding="utf-8")

    print(f"Saved merged initial files to: {initial_pose_dir}")
    print(f"Saved pose format info to: {pose_format_path}")
    print(f"Saved kpt info to: {kpt_info_path}")


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
