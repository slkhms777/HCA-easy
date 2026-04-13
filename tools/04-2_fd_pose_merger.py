import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d, CubicSpline
from hocap_annotation.utils import *
from hocap_annotation.loaders import HOCapLoader
from hocap_annotation.rendering import HOCapRenderer


def complete_positions_by_interp1d(positions, valid_indices):
    if len(valid_indices) < 2:
        raise ValueError("No valid poses to interpolate from.")

    if len(valid_indices) == len(positions):
        return positions

    if isinstance(positions, list):
        pts = np.stack(positions, axis=0)
    else:
        pts = positions.copy()

    # Handle the beginning and end of the sequence
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]
    if start_idx > 0:
        pts[:start_idx] = pts[start_idx]
    if end_idx < len(pts) - 1:
        pts[end_idx + 1 :] = pts[end_idx]

    # Interpolate the missing positions
    f_x = interp1d(
        valid_indices, pts[valid_indices, 0], kind="linear", fill_value="extrapolate"
    )
    f_y = interp1d(
        valid_indices, pts[valid_indices, 1], kind="linear", fill_value="extrapolate"
    )
    f_z = interp1d(
        valid_indices, pts[valid_indices, 2], kind="linear", fill_value="extrapolate"
    )

    for idx in range(start_idx, end_idx + 1):
        if idx not in valid_indices:
            pts[idx] = [f_x(idx), f_y(idx), f_z(idx)]

    return pts.astype(np.float32)


def complete_positions_by_cubic_spline(positions, valid_indices):
    if len(valid_indices) < 2:
        raise ValueError("No valid poses to interpolate from.")

    if len(valid_indices) == len(positions):
        return positions

    if isinstance(positions, list):
        pts = np.stack(positions, axis=0)
    else:
        pts = positions.copy()

    # Handle the beginning and end of the sequence
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]
    if start_idx > 0:
        pts[:start_idx] = pts[start_idx]
    if end_idx < len(pts) - 1:
        pts[end_idx + 1 :] = pts[end_idx]

    # Interpolate the missing positions
    f_x = CubicSpline(valid_indices, pts[valid_indices, 0])
    f_y = CubicSpline(valid_indices, pts[valid_indices, 1])
    f_z = CubicSpline(valid_indices, pts[valid_indices, 2])

    for idx in range(start_idx, end_idx + 1):
        if idx not in valid_indices:
            pts[idx] = [f_x(idx), f_y(idx), f_z(idx)]

    return pts.astype(np.float32)


def complete_rotations_by_slerp(quat_rots, valid_indices):
    if len(valid_indices) < 2:
        raise ValueError("No valid poses to interpolate from.")

    if len(valid_indices) == len(quat_rots):
        return quat_rots

    if isinstance(quat_rots, list):
        quats = np.stack(quat_rots, axis=0)
    else:
        quats = quat_rots.copy()

    # Handle the beginning and end of the sequence
    start_idx = valid_indices[0]
    end_idx = valid_indices[-1]
    if start_idx > 0:
        quats[:start_idx] = quats[start_idx]
    if end_idx < len(quats) - 1:
        quats[end_idx + 1 :] = quats[end_idx]

    # Interpolate the missing rotations
    key_rots = R.from_quat(quats[valid_indices])
    slerp = Slerp(valid_indices, key_rots)
    slerp_rots = slerp(range(start_idx, end_idx + 1))
    quats[start_idx : end_idx + 1] = slerp_rots.as_quat()

    return quats.astype(np.float32)


def pose_jitter_smooth(
    poses: np.ndarray, window_size: int = 1, rot_thresh=0.5, trans_thresh=0.002
) -> np.ndarray:
    """Smooth poses using a simple moving average over a specified window size.

    Args:
        poses (np.ndarray): Array of poses with shape (N, 7).
        window_size (int): Window size for smoothing.

    Returns:
        np.ndarray: Smoothed poses.
    """
    N = poses.shape[0]
    smoothed_poses = poses.copy()
    mean_poses = np.zeros_like(poses)

    for i in range(N):
        start = max(0, i - window_size)
        end = min(N, i + window_size + 1)
        window_quats = poses[start:end, :4]
        window_trans = poses[start:end, 4:]
        norm_quats = normalize_quats(window_quats)
        mean_quat = average_quats(norm_quats)
        mean_quat = normalize_quats(mean_quat)
        mean_trans = np.mean(window_trans, axis=0)
        mean_poses[i] = np.concatenate([mean_quat, mean_trans], axis=0)

    quat_diffs = quat_distance(poses[:, :4], mean_poses[:, :4], in_degree=True)
    smoothed_quats = complete_rotations_by_slerp(
        mean_poses[:, :4], np.where(quat_diffs < rot_thresh)[0]
    )

    trans_diffs = np.linalg.norm(poses[:, 4:] - mean_poses[:, 4:], axis=1)
    smoothed_trans = complete_positions_by_interp1d(
        mean_poses[:, 4:], np.where(trans_diffs < trans_thresh)[0]
    )

    smoothed_poses = np.concatenate([smoothed_quats, smoothed_trans], axis=1)
    return smoothed_poses.astype(np.float32)


class FoundationPoseMerger:
    def __init__(self, sequence_folder, single_process=False):
        self._data_folder = Path(sequence_folder)
        self._single_process = single_process
        self._fd_pose_folder = self._data_folder / "processed" / "fd_pose_solver"
        self._log_file = self._fd_pose_folder / "fd_pose_merger.log"

        if not self._fd_pose_folder.exists():
            raise ValueError("Foundation pose folder does not exist.")

        self._logger = get_logger(
            log_name=__class__.__name__,
            log_file=self._log_file,
        )

        self._loader = HOCapLoader(sequence_folder)
        self._num_frames = self._loader.num_frames
        self._rs_serials = self._loader.rs_serials
        self._num_cameras = len(self._rs_serials)
        self._rs_width = self._loader.rs_width
        self._rs_height = self._loader.rs_height
        self._cam_Ks = self._loader.rs_Ks
        self._cam_RTs = self._loader.extr2world
        self._object_ids = self._loader.object_ids
        self._num_objects = len(self._object_ids)
        self._object_mesh_files = self._loader.object_textured_files

    def _get_color_images_by_frame_id(self, frame_id):
        color_images = [None] * self._num_cameras
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._loader.get_color, s, frame_id): cam_idx
                for cam_idx, s in enumerate(self._rs_serials)
            }
            for future in concurrent.futures.as_completed(futures):
                cam_idx = futures[future]
                color_images[cam_idx] = future.result()
            futures.clear()
        return color_images

    def _complete_fd_poses(self, poses):
        valid_indices = np.where(poses[:, -1] > 0)[0]
        rots = complete_rotations_by_slerp(poses[:, :4], valid_indices)
        trans = complete_positions_by_cubic_spline(poses[:, 4:7], valid_indices)
        return np.concatenate([rots, trans], axis=1).astype(np.float32)

    def _render_world_frame(self, color_images, obj_poses, cam_indices, save_path):
        rgb_images = [color_images[idx] for idx in cam_indices]
        render_images = self._renderer.get_render_colors(
            width=self._rs_width,
            height=self._rs_height,
            cam_names=[self._rs_serials[idx] for idx in cam_indices],
            cam_poses=[self._cam_RTs[idx] for idx in cam_indices],
            mesh_names=self._object_ids,
            mesh_poses=obj_poses,
        )
        vis_image = draw_image_grid(
            images=[
                draw_image_overlay(rgb_image, render_image, 0.7)
                for rgb_image, render_image in zip(rgb_images, render_images)
            ],
            names=[self._rs_serials[idx] for idx in cam_indices],
            max_cols=2,
            facecolor="black",
            titlecolor="white",
        )
        write_rgb_image(save_path, vis_image)

    def render_fd_poses(self, pose_file, save_vis=True):
        self._logger.info(f">>>>>>>>>> Start rendering for {pose_file.name} <<<<<<<<<<")
        save_vis_folder = self._fd_pose_folder / f"vis"
        save_video_file = self._fd_pose_folder / f"vis_{pose_file.stem}.mp4"
        poses_o, verts_m, faces_m, colors_m, joints_m = None, None, None, None, None
        poses_o = np.load(pose_file)
        poses_o = np.stack([quat_to_mat(p) for p in poses_o], axis=1)
        self._logger.debug(f"Loaded fd_poses: {poses_o.shape}")
        renderer = HOCapRenderer(self._data_folder, log_file=self._log_file)
        renderer.update_render_dict(poses_o, verts_m, faces_m, colors_m, joints_m)
        renderer.render_pose_images(
            save_vis_folder,
            save_video_file,
            vis_only=True,
            save_vis=save_vis,
            single_process=self._single_process,
        )

    def merge_fd_poses(self):
        self._logger.info(f">>>>>>>>>> Merging foundation poses <<<<<<<<<<")

        raw_file = self._fd_pose_folder / f"fd_poses_merged_raw.npy"
        interp_file = self._fd_pose_folder / f"fd_poses_merged_interp.npy"
        fixed_file = self._fd_pose_folder / f"fd_poses_merged_fixed.npy"

        # Generate raw poses
        self._logger.info("Generating raw poses...")
        fd_poses_raw = [[None] * self._num_frames for _ in range(self._num_objects)]
        tqbar = tqdm(total=self._num_frames * self._num_objects, ncols=100)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    read_pose_from_txt,
                    self._fd_pose_folder
                    / obj_id
                    / "ob_in_world"
                    / f"{frame_id:06d}.txt",
                ): (obj_idx, frame_id)
                for frame_id in range(self._num_frames)
                for obj_idx, obj_id in enumerate(self._object_ids)
            }
            for future in concurrent.futures.as_completed(futures):
                obj_idx, frame_id = futures[future]
                fd_poses_raw[obj_idx][frame_id] = future.result()
                tqbar.update()
            futures.clear()
        tqbar.close()

        fd_poses_raw = np.stack(fd_poses_raw, axis=0).astype(np.float32)
        self._logger.info(f"fd_poses_raw: {fd_poses_raw.shape}")

        np.save(raw_file, fd_poses_raw[:, :, :7])
        self._logger.info(f"Saved raw poses to {raw_file.name}")

        # Interpolate the missing poses
        self._logger.info("Interpolating the missing poses...")
        fd_poses_interp = [
            self._complete_fd_poses(fd_poses_raw[obj_idx])
            for obj_idx in range(self._num_objects)
        ]
        fd_poses_interp = np.stack(fd_poses_interp, axis=0).astype(np.float32)
        self._logger.info(f"fd_poses_interp: {fd_poses_interp.shape}")

        np.save(interp_file, fd_poses_interp)
        self._logger.info(f"Saved interpolated poses to {interp_file.name}")

        # Smooth the interpolated poses
        self._logger.info("Smoothing the interpolated poses...")
        fd_poses_fixed = []
        # Smooth the poses
        for idx in range(self._num_objects):
            poses_fixed = fd_poses_interp[idx]

            poses_fixed = evaluate_and_fix_poses(
                poses_fixed,
                window_size=15,
                rot_thresh=5.0,
                trans_thresh=0.01,
                seperate_rot_trans=True,
                use_mean_pose=True,
            )

            poses_fixed = pose_jitter_smooth(
                poses_fixed,
                window_size=1,
                rot_thresh=0.5,
                trans_thresh=0.003,
            )

            poses_fixed = evaluate_and_fix_poses(
                poses_fixed,
                window_size=10,
                rot_thresh=1.0,
                trans_thresh=0.005,
                seperate_rot_trans=True,
                use_mean_pose=True,
            )

            poses_fixed = evaluate_and_fix_poses(
                poses_fixed,
                window_size=15,
                rot_thresh=2.0,
                trans_thresh=0.01,
                seperate_rot_trans=False,
                use_mean_pose=True,
            )

            poses_fixed = evaluate_and_fix_poses(
                poses_fixed,
                window_size=5,
                rot_thresh=1.0,
                trans_thresh=0.003,
                seperate_rot_trans=False,
                use_mean_pose=True,
            )

            poses_fixed = pose_jitter_smooth(
                poses_fixed,
                window_size=1,
                rot_thresh=0.5,
                trans_thresh=0.001,
            )

            poses_fixed = evaluate_and_fix_poses(
                poses_fixed,
                window_size=45,
                rot_thresh=2.0,
                trans_thresh=0.003,
                seperate_rot_trans=True,
                use_mean_pose=False,
            )

            fd_poses_fixed.append(poses_fixed)

        fd_poses_fixed = np.stack(fd_poses_fixed, axis=0).astype(np.float32)
        self._logger.info(f"fd_poses_fixed: {fd_poses_fixed.shape}")

        np.save(fixed_file, fd_poses_fixed)
        self._logger.info(f"Saved fixed poses to {fixed_file.name}")

    def run(self):
        self._logger.info(f">>>>>>>>>> Start Foundation Pose Merger <<<<<<<<<<")
        t_s = time.time()

        # Merge the foundation poses
        self.merge_fd_poses()

        # Render the foundation poses
        # self.render_fd_poses(
        #     self._fd_pose_folder / f"fd_poses_merged_raw.npy", save_vis=False
        # )
        # self.render_fd_poses(
        #     self._fd_pose_folder / f"fd_poses_merged_interp.npy", save_vis=False
        # )
        self.render_fd_poses(
            self._fd_pose_folder / f"fd_poses_merged_fixed.npy", save_vis=True
        )

        self._logger.info(
            f">>>>>>>>>> Done!!! ({time.time() - t_s:.2f} seconds) <<<<<<<<<<"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--single_process",
        action="store_true",
        help="Use single-process rendering to avoid multiprocessing pickle errors.",
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder path.")

    pose_merger = FoundationPoseMerger(
        args.sequence_folder, single_process=args.single_process
    )
    pose_merger.run()
