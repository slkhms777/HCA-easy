import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from itertools import combinations
from hocap_annotation.utils import *
from hocap_annotation.loaders import HOCapLoader
from hocap_annotation.wrappers.foundationpose import (
    FoundationPose,
    ScorePredictor,
    PoseRefinePredictor,
    set_logging_format,
    set_seed,
    dr,
)


def slerp(q1, q2, t):
    """
    Spherical Linear Interpolation (SLERP) between two quaternions.

    Args:
        q1: Starting quaternion as [qx, qy, qz, qw].
        q2: Ending quaternion as [qx, qy, qz, qw].
        t: Interpolation factor, 0 <= t <= 1.

    Returns:
        Interpolated quaternion as [qx, qy, qz, qw].
    """
    # Normalize input quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute the dot product
    dot_product = np.dot(q1, q2)

    # Ensure the shortest path is used
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product

    # Clamp the dot product to avoid numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angle between quaternions
    theta_0 = np.arccos(dot_product)  # Angle between q1 and q2
    sin_theta_0 = np.sin(theta_0)

    if sin_theta_0 < 1e-6:
        # Quaternions are almost identical
        return (1 - t) * q1 + t * q2

    # Perform SLERP
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    s1 = np.sin(theta_0 - theta_t) / sin_theta_0
    s2 = sin_theta_t / sin_theta_0

    return s1 * q1 + s2 * q2


def predict_current_rotation(prev_quats, prev_flags):
    """
    Predict the current frame rotation based on previous quaternions and flags, weighted by temporal proximity.

    Args:
        prev_quats: List of previous frame quaternions [[qx, qy, qz, qw], ...].
        prev_flags: List of flags corresponding to each quaternion (1 = valid, 0 = invalid).

    Returns:
        Predicted quaternion [qx, qy, qz, qw] for the current frame.
    """
    # Step 1: Filter valid quaternions and assign temporal weights
    valid_quats = []
    weights = []
    for i, (q, flag) in enumerate(zip(prev_quats, prev_flags)):
        if flag == 1:
            valid_quats.append(q)
            weights.append(1 / (len(prev_quats) - i))

    if len(valid_quats) == 0:
        return np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

    if len(valid_quats) == 1:
        return valid_quats[0]

    # Step 4: Compute the weighted mean quaternion
    weights = np.array(weights) / np.sum(weights)  # Normalize weights
    weighted_quat = np.zeros(4)
    for q, w in zip(valid_quats, weights):
        weighted_quat += w * np.array(q)
    weighted_quat /= np.linalg.norm(weighted_quat)  # Normalize the mean quaternion

    # Step 5: Interpolate between the weighted mean quaternion and the most recent valid quaternion
    most_recent_valid_quat = valid_quats[-1]
    predicted_quat = slerp(weighted_quat, most_recent_valid_quat, t=0.5)

    return predicted_quat


def predict_current_position(prev_positions, prev_flags):
    """
    Predict the current frame position using Cubic Spline Interpolation.

    Args:
        prev_positions: List of previous frame positions [[x, y, z], ...].
        prev_flags: List of flags corresponding to each position (1 = valid, 0 = invalid).

    Returns:
        Predicted position [x, y, z] for the current frame.
    """
    # Step 1: Filter valid positions and their corresponding time indices
    valid_positions = []
    valid_times = []
    for i, (pos, flag) in enumerate(zip(prev_positions, prev_flags)):
        if flag == 1:
            valid_positions.append(pos)
            valid_times.append(i)  # Use the index as the "time" axis

    if len(valid_positions) == 0:
        return np.array([-1.0, -1.0, -1.0], dtype=np.float32)

    if len(valid_positions) == 1:
        return np.array(valid_positions[0])

    # Step 4: Fit cubic splines for x, y, z components
    valid_positions = np.array(valid_positions)
    x_coords = valid_positions[:, 0]
    y_coords = valid_positions[:, 1]
    z_coords = valid_positions[:, 2]

    spline_x = CubicSpline(valid_times, x_coords)
    spline_y = CubicSpline(valid_times, y_coords)
    spline_z = CubicSpline(valid_times, z_coords)

    # Step 5: Predict position for the current frame (next time index)
    t_current = len(prev_positions)  # Current frame index
    x_pred = spline_x(t_current)
    y_pred = spline_y(t_current)
    z_pred = spline_z(t_current)

    return np.array([x_pred, y_pred, z_pred], dtype=np.float32)


# Helper Function 1: Calculate Pairwise Distances
def calculate_pairwise_distances(poses):
    """
    Calculate pairwise distances for rotations and translations.

    Args:
        poses: List of poses, where each pose is [qx, qy, qz, qw, x, y, z].

    Returns:
        rot_dists: Pairwise rotation distances.
        trans_dists: Pairwise translation distances.
        pairwise_indices: List of tuples (i, j) indicating which poses were used to calculate each distance.
    """
    num_poses = len(poses)
    rot_dists = []
    trans_dists = []
    pairwise_indices = []  # To track which poses are involved in each pair

    for i in range(num_poses):
        for j in range(i + 1, num_poses):
            # Extract rotations and translations
            q1, q2 = poses[i][:4], poses[j][:4]
            t1, t2 = poses[i][4:], poses[j][4:]

            # Normalize quaternions
            q1 = q1 / np.linalg.norm(q1)
            q2 = q2 / np.linalg.norm(q2)

            # Calculate rotation geodesic distance
            dot_product = np.dot(q1, q2)
            theta = 2 * np.arccos(np.clip(abs(dot_product), -1, 1))
            rot_dists.append(theta)

            # Calculate translation Euclidean distance
            trans_dists.append(np.linalg.norm(np.array(t1) - np.array(t2)))

            # Save pairwise indices
            pairwise_indices.append((i, j))
    rot_dists = np.array(rot_dists, dtype=np.float32)
    trans_dists = np.array(trans_dists, dtype=np.float32)
    return rot_dists, trans_dists, pairwise_indices


# Helper Function 2: Analyze Distances
def analyze_distances(distances, threshold_factor=2.0, outlier_ratio=0.2):
    """
    Analyze distance distribution to identify noise and inliers.

    Args:
        distances: Array of pairwise distances.
        threshold_factor: Factor to determine outlier threshold (default 2.0 for 95% confidence).
        outlier_ratio: Maximum acceptable ratio of outliers (default 0.2).

    Returns:
        is_noisy: Boolean, True if distances are noisy.
        inlier_distance_indices: Indices of distances considered inliers.
    """
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + threshold_factor * std_dist

    # Identify inliers
    inlier_indices = np.where(distances <= threshold)[0]
    outlier_fraction = 1 - (len(inlier_indices) / len(distances))

    is_noisy = outlier_fraction > outlier_ratio
    return is_noisy, inlier_indices


def detect_pose_outliers(poses, threshold_factor=2.0, outlier_ratio=0.2):
    """
    Detect outlier rotations and translations, and return inliers.

    Args:
        poses: List of poses, where each pose is [qx, qy, qz, qw, x, y, z].
        threshold_factor: Factor to determine outlier threshold (default 2.0 for 95% confidence).
        outlier_ratio: Maximum acceptable ratio of outliers (default 0.2).

    Returns:
        inlier_rots: List of inlier rotations as [qx, qy, qz, qw].
        inlier_trans: List of inlier translations as [x, y, z].
        is_rot_noisy: Boolean, True if rotations are noisy.
        is_trans_noisy: Boolean, True if translations are noisy.
    """
    # Step 1: Calculate pairwise distances
    rot_dists, trans_dists, pairwise_indices = calculate_pairwise_distances(poses)

    # Step 2: Analyze rotation distances
    is_rot_noisy, rot_inlier_dist_indices = analyze_distances(
        rot_dists, threshold_factor, outlier_ratio
    )

    # Step 3: Analyze translation distances
    is_trans_noisy, trans_inlier_dist_indices = analyze_distances(
        trans_dists, threshold_factor, outlier_ratio
    )

    # Step 4: Find pose inliers
    rot_inlier_indices = set(
        idx
        for pair_idx in rot_inlier_dist_indices
        for idx in pairwise_indices[pair_idx]
    )
    trans_inlier_indices = set(
        idx
        for pair_idx in trans_inlier_dist_indices
        for idx in pairwise_indices[pair_idx]
    )

    # Convert sets to sorted lists for consistent output
    rot_inlier_indices = sorted(rot_inlier_indices)
    trans_inlier_indices = sorted(trans_inlier_indices)

    # Extract inlier rotations and translations
    inlier_rots = [poses[i][:4] for i in rot_inlier_indices]
    inlier_trans = [poses[i][4:] for i in trans_inlier_indices]

    return inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy


def is_valid_pose(pose_w):
    """
    Check if pose in world space is valid.

    Args:
        pose_w: Pose in world space as [qx, qy, qz, qw, x, y, z].

    Returns:
        Boolean, True if pose is valid.
    """
    x, y, z = pose_w[-3:]
    return -0.6 < x < 0.6 and -0.4 < y < 0.4 and -0.6 < z < 0.6


def transform_poses_to_world(mat_poses_c, cam_RTs):
    poses_w = []
    for mat_pose, cam_RT in zip(mat_poses_c, cam_RTs):
        if np.all(mat_pose == -1):  # invalid pose
            continue
        mat_pose_w = cam_RT @ mat_pose
        quat_pose_w = mat_to_quat(mat_pose_w)
        if is_valid_pose(quat_pose_w):
            poses_w.append(quat_pose_w)
    return poses_w


def ransac_consistent_rotation(inlier_rots, threshold):
    """
    Estimate the consistent rotation using RANSAC on inlier rotations.

    Args:
        inlier_rots: List of inlier rotations as [qx, qy, qz, qw].
        threshold: Geodesic distance threshold for inlier classification (in radians).

    Returns:
        Consistent rotation quaternion [qx, qy, qz, qw].
    """
    if len(inlier_rots) == 1:
        return inlier_rots[0]  # Return directly if only one inlier exists

    best_rotation = None
    max_inliers = 0

    # Step 1: Generate candidate rotations by averaging all possible combinations
    for r in range(1, len(inlier_rots) + 1):  # From 1 to all inliers
        for comb in combinations(inlier_rots, r):
            candidate = np.mean(comb, axis=0)
            candidate /= np.linalg.norm(candidate)  # Normalize quaternion

            # Step 2: Evaluate candidate using RANSAC
            inlier_count = 0
            for rot in inlier_rots:
                dot_product = np.dot(candidate, rot)
                loss = 2 * np.arccos(np.clip(abs(dot_product), -1, 1))
                if loss <= threshold:
                    inlier_count += 1

            # Step 3: Update best candidate
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_rotation = candidate

    return best_rotation


def ransac_consistent_translation(inlier_trans, threshold):
    """
    Estimate the consistent translation using RANSAC on inlier translations.

    Args:
        inlier_trans: List of inlier translations as [x, y, z].
        threshold: Euclidean distance threshold for inlier classification.

    Returns:
        Consistent translation [x, y, z].
    """
    if len(inlier_trans) == 1:
        return inlier_trans[0]  # Return directly if only one inlier exists

    best_translation = None
    max_inliers = 0

    # Step 1: Generate candidate translations by averaging all possible combinations
    for r in range(1, len(inlier_trans) + 1):  # From 1 to all inliers
        for comb in combinations(inlier_trans, r):
            candidate = np.mean(comb, axis=0)

            # Step 2: Evaluate candidate using RANSAC
            inlier_count = 0
            for trans in inlier_trans:
                loss = np.linalg.norm(candidate - trans)
                if loss <= threshold:
                    inlier_count += 1

            # Step 3: Update best candidate
            if inlier_count > max_inliers:
                max_inliers = inlier_count
                best_translation = candidate

    return best_translation


def get_consistent_pose_w(
    mat_poses_c,
    cam_RTs,
    prev_poses_w,
    rot_thresh=5.0,
    trans_thresh=0.01,
    thresh_factor=2.0,
    outlier_ratio=0.2,
):
    """
    Get consistent pose in world space using RANSAC on inlier rotations and translations.

    Args:
        mat_poses_c: List of poses in camera space as 4x4 matrix.
        cam_RTs: List of camera extrinsics as 4x4 matrix.
        prev_poses_w: List of previous poses in world space as [qx, qy, qz, qw, x, y, z, flag].
        rot_thresh: Rotational threshold in degrees.
        trans_thresh: Translation threshold in meters.
        thresh_factor: Factor to determine outlier threshold (default 2.0 for 95% confidence).
        outlier_ratio: Maximum acceptable ratio of outliers (default 0.2).

    Returns:
        Consistent pose in world space as [qx, qy, qz, qw, x, y, z, flag].
    """
    rot_thresh = np.deg2rad(rot_thresh)
    curr_rot = None
    curr_trans = None
    flag = 1

    # Step 1: transform all poses to world space
    poses_w = transform_poses_to_world(mat_poses_c, cam_RTs)

    # if len(poses_w) == 0:
    if len(poses_w) < 3:
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
        return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)
    # elif len(poses_w) == 1:
    #     curr_rot = poses_w[0][:4]
    #     curr_trans = poses_w[0][4:]
    #     return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)

    # Stack poses for processing
    poses_w = np.stack(poses_w, axis=0)

    # Step 2: detect check if poses are noisy
    inlier_rots, inlier_trans, is_rot_noisy, is_trans_noisy = detect_pose_outliers(
        poses_w, thresh_factor, outlier_ratio
    )

    # Step 3: Handle noisy scenarios
    if is_rot_noisy and is_trans_noisy:
        # Predict both rotation and translation
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    elif is_rot_noisy:
        # Predict rotation, estimate translation via RANSAC
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        curr_trans = ransac_consistent_translation(inlier_trans, trans_thresh)
        flag = 0
    elif is_trans_noisy:
        # Predict translation, estimate rotation via RANSAC
        curr_rot = ransac_consistent_rotation(inlier_rots, rot_thresh)
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    else:
        # Use RANSAC for both rotation and translation
        curr_rot = ransac_consistent_rotation(inlier_rots, rot_thresh)
        curr_trans = ransac_consistent_translation(inlier_trans, trans_thresh)

    # Ensure both rotation and translation are defined
    if curr_rot is None:
        curr_rot = predict_current_rotation(
            [pose[:4] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0
    if curr_trans is None:
        curr_trans = predict_current_position(
            [pose[4:7] for pose in prev_poses_w], [pose[-1] for pose in prev_poses_w]
        )
        flag = 0

    return np.concatenate([curr_rot, curr_trans, [flag]], axis=0)


def is_valid_ob_pose(ob_in_cam, cam_RT=None):
    if np.all(ob_in_cam == -1):
        return False
    elif cam_RT is None:
        x, y, z = ob_in_cam[:3, 3]
    else:
        ob_in_world = cam_RT @ ob_in_cam
        x, y, z = ob_in_world[:3, 3]
    return -0.6 < x < 0.6 and -0.4 < y < 0.4 and -0.6 < z < 0.6


def initialize_fd_pose_estimator(textured_mesh_path, cleaned_mesh_path, debug_dir):
    textured_mesh = trimesh.load(textured_mesh_path, process=False, force='mesh')
    cleaned_mesh = trimesh.load(cleaned_mesh_path, process=False, force='mesh')
    return FoundationPose(
        model_pts=cleaned_mesh.vertices.astype(np.float32),
        model_normals=cleaned_mesh.vertex_normals.astype(np.float32),
        mesh=textured_mesh,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=0,
        debug_dir=debug_dir,
        rotation_grid_min_n_views=120,
        rotation_grid_inplane_step=60,
    )


def run_pose_estimation(
    sequence_folder,
    object_idx,
    est_refine_iter,
    track_refine_iter,
    start_frame,
    end_frame,
    rot_thresh,
    trans_thresh,
):
    sequence_folder = Path(sequence_folder)
    object_idx = object_idx - 1  # 0-based index

    # Load parameters from data_loader
    data_loader = HOCapLoader(sequence_folder)
    rs_width = data_loader.rs_width
    rs_height = data_loader.rs_height
    num_frames = data_loader.num_frames
    object_id = data_loader.object_ids[object_idx]
    rs_serials = data_loader.rs_serials
    cam_Ks = data_loader.rs_Ks
    cam_RTs = data_loader.extr2world
    valid_serials = data_loader.get_valid_seg_serials()
    valid_serial_indices = [rs_serials.index(serial) for serial in valid_serials]
    valid_Ks = data_loader.rs_Ks[valid_serial_indices]
    valid_RTs = data_loader.extr2world[valid_serial_indices]
    valid_RTs_inv = data_loader.extr2world_inv[valid_serial_indices]
    object_mesh_textured = trimesh.load(data_loader.object_textured_files[object_idx], force='mesh')
    # object_mesh_cleaned = trimesh.load(data_loader.object_cleaned_files[object_idx], force='mesh')
    empty_mat_pose = np.full((4, 4), -1.0, dtype=np.float32)

    # Check start and end frame_idx
    start_frame = max(start_frame, 0)
    end_frame = num_frames if end_frame < start_frame else end_frame

    logging.info(f"start_frame: {start_frame}, end_frame: {end_frame}")

    save_folder = sequence_folder / "processed" / "fd_pose_solver"
    save_folder.mkdir(parents=True, exist_ok=True)

    set_seed(0)

    estimator = FoundationPose(
        model_pts=object_mesh_textured.vertices.astype(np.float32),
        model_normals=object_mesh_textured.vertex_normals.astype(np.float32),
        mesh=object_mesh_textured,
        scorer=ScorePredictor(),
        refiner=PoseRefinePredictor(),
        glctx=dr.RasterizeCudaContext(),
        debug=0,
        debug_dir=save_folder / "debug" / object_id,
        rotation_grid_min_n_views=120,
        rotation_grid_inplane_step=60,
    )

    # Initialize poses
    ob_in_world_refined = empty_mat_pose.copy()
    ob_in_cam_poses = [empty_mat_pose.copy()] * len(valid_serials)
    all_poses_w = []
    for frame_id in range(start_frame, end_frame, 1):
        for serial_idx, serial in enumerate(valid_serials):
            color = data_loader.get_color(serial, frame_id)
            depth = data_loader.get_depth(serial, frame_id)
            mask = data_loader.get_mask(serial, frame_id, object_idx)
            K = valid_Ks[serial_idx]

            if mask.sum() < 100:
                ob_in_cam_mat = empty_mat_pose.copy()
            elif is_valid_ob_pose(ob_in_world_refined):
                ob_in_cam_mat = estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=track_refine_iter,
                    prev_pose=valid_RTs_inv[serial_idx] @ ob_in_world_refined,
                )
            elif is_valid_ob_pose(ob_in_cam_poses[serial_idx], valid_RTs[serial_idx]):
                ob_in_cam_mat = estimator.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=track_refine_iter,
                    prev_pose=ob_in_cam_poses[serial_idx],
                )
            else:
                init_ob_pos_center = data_loader.get_init_translation(
                    frame_id, [serial], object_idx, kernel_size=5
                )[0][0]
                if init_ob_pos_center is not None:
                    ob_in_cam_mat = estimator.register(
                        rgb=color,
                        depth=depth,
                        ob_mask=mask,
                        K=K,
                        iteration=est_refine_iter,
                        init_ob_pos_center=init_ob_pos_center,
                    )
                    if not is_valid_ob_pose(ob_in_cam_mat, valid_RTs[serial_idx]):
                        ob_in_cam_mat = empty_mat_pose.copy()
                else:
                    ob_in_cam_mat = empty_mat_pose.copy()

            ob_in_cam_poses[serial_idx] = ob_in_cam_mat

            save_pose_folder = save_folder / object_id / "ob_in_cam" / serial
            save_pose_folder.mkdir(parents=True, exist_ok=True)
            write_pose_to_txt(
                save_pose_folder / f"{frame_id:06d}.txt", mat_to_quat(ob_in_cam_mat)
            )

        # refine object pose in world coordinate system
        curr_pose_w = get_consistent_pose_w(
            mat_poses_c=ob_in_cam_poses,
            cam_RTs=valid_RTs,
            prev_poses_w=all_poses_w,
            rot_thresh=rot_thresh,
            trans_thresh=trans_thresh,
            thresh_factor=2.0,
            outlier_ratio=0.2,
        )

        all_poses_w.append(curr_pose_w)

        # save pose to file
        save_pose_folder = save_folder / object_id / "ob_in_world"
        save_pose_folder.mkdir(parents=True, exist_ok=True)
        write_pose_to_txt(save_pose_folder / f"{frame_id:06d}.txt", curr_pose_w)

        ob_in_world_refined = quat_to_mat(curr_pose_w[:7])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    parser.add_argument(
        "--object_idx",
        type=int,
        default=None,
        choices=[1, 2, 3, 4],
        help="object index",
    )
    parser.add_argument(
        "--est_refine_iter",
        type=int,
        default=15,
        help="number of iterations for estimation",
    )
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=5,
        help="number of iterations for tracking",
    )
    parser.add_argument("--start_frame", type=int, default=0, help="start frame")
    parser.add_argument("--end_frame", type=int, default=-1, help="end frame")
    parser.add_argument(
        "--rot_thresh",
        type=float,
        default=2.0,
        help="rotation threshold, degree",
    )
    parser.add_argument(
        "--trans_thresh",
        type=float,
        default=0.03,
        help="translation threshold, meters",
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please specify the sequence folder.")
    if args.object_idx is None:
        raise ValueError("Please specify the object index.")

    set_logging_format()
    t_start = time.time()

    run_pose_estimation(
        args.sequence_folder,
        args.object_idx,
        args.est_refine_iter,
        args.track_refine_iter,
        args.start_frame,
        args.end_frame,
        args.rot_thresh,
        args.trans_thresh,
    )

    logging.info(f"done!!! time: {time.time() - t_start:.3f}s.")
