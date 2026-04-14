from ..utils import *
from .extrinsics import load_extrinsics


class HOCapLoader:

    def __init__(self, sequence_folder) -> None:
        self._data_folder = Path(sequence_folder)
        self._calib_folder = self._data_folder.parent.parent / "calibration"
        self._models_folder = self._data_folder.parent.parent / "models"
        self._seg_folder = self._data_folder / "processed/segmentation/sam2"

        self._load_metadata()

    def _depth2xyz(self, depth, K, T=None):
        """Convert depth image to xyz point cloud"""
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")

        u_flat = u.flatten()
        v_flat = v.flatten()
        depth_flat = depth.flatten()
        depth_flat[depth_flat < 0] = 0

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x_norm = (u_flat - cx) / fx
        y_norm = (v_flat - cy) / fy

        x = depth_flat * x_norm
        y = depth_flat * y_norm
        z = depth_flat

        xyz = np.stack((x, y, z), axis=1)  # camera space

        if T is not None:
            xyz = xyz @ T[:3, :3].T + T[:3, 3]

        return xyz

    def get_init_translation(self, frame_id, serials, object_idx, kernel_size=3):
        masks = [
            self.get_mask(s, frame_id, object_idx, kernel_size)
            for s in self._rs_serials
        ]
        depths = [self.get_depth(s, frame_id) for s in self._rs_serials]

        pts = [
            self._depth2xyz(depth, K, T)
            for depth, K, T in zip(depths, self._rs_Ks, self._extr2world)
        ]
        pts = [pt[mask.flatten().astype(bool)] for pt, mask in zip(pts, masks)]
        pts = np.concatenate(pts, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        # Remove outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        pts = np.asarray(pcd.points, dtype=np.float32)

        if len(pts) < 100:
            return [None] * len(serials), pcd

        center = np.mean(pts, axis=0)

        center = pcd.get_center()
        # transform to each camera coordinate system
        centers = []
        for serial in serials:
            extr = self._extr2world_inv[self._rs_serials.index(serial)]
            center_cam = center @ extr[:3, :3].T + extr[:3, 3]
            centers.append(center_cam)
        centers = np.stack(centers, axis=0, dtype=np.float32)
        return centers, pcd

    def _load_metadata(self):
        file_path = self._data_folder / "meta.yaml"
        data = read_data_from_yaml(file_path)

        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._subject_id = data["subject_id"]
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        # self._hl_serial = data["hololens"]["serial"]
        # self._hl_height = data["hololens"]["pv_height"]
        # self._hl_width = data["hololens"]["pv_width"]
        self._object_textured_files = [
            self._models_folder / obj_id / "textured_mesh.obj"
            for obj_id in self._object_ids
        ]
        # self._object_cleaned_files = [
        #     self._models_folder / obj_id / "cleaned_mesh_10000.obj"
        #     for obj_id in self._object_ids
        # ]

        # Load camera intrinsics
        self._load_intrinsics(data.get("intrinsics"))

        # Load rs camera extrinsics from per-camera cam2world/world2cam files
        self._load_extrinsics(data.get("extrinsics"))

        # Load MANO shape parameters
        self._load_mano_beta()

    def _load_intrinsics(self, intrinsics_source=None):
        def read_K_from_yaml(serial, cam_type="color", intrinsics_source=None):
            if intrinsics_source is not None:
                intrinsics_source = Path(intrinsics_source)
                file_path = intrinsics_source / f"{serial}.yaml"
            else:
                raise ValueError("No intrinsics source provided in meta.yaml")
            data = read_data_from_yaml(file_path)[cam_type]
            K = np.array(
                [
                    [data["fx"], 0.0, data["ppx"]],
                    [0.0, data["fy"], data["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K

        self._rs_Ks = np.stack(
            [read_K_from_yaml(serial, intrinsics_source=intrinsics_source) for serial in self._rs_serials],
            axis=0,
        )
        # self._hl_K = read_K_from_yaml(self._hl_serial)

    def _load_extrinsics(self, extrinsics_source=None):
        if extrinsics_source is None:
            raise ValueError("No extrinsics source provided in meta.yaml")
        data = load_extrinsics(
            calib_folder=self._calib_folder,
            subject_id=self._subject_id,
            serials=self._rs_serials,
            extrinsics_source=extrinsics_source,
        )

        self._rs_master = data["rs_master"]
        self._tag2master = data["tag_0"]
        self._tag2master_inv = data["tag_0_inv"]
        self._tag2world = data["tag_1_inv"]
        self._tag2world_inv = data["tag_1"]
        self._extr2master = data["extr2master"]
        self._extr2master_inv = data["extr2master_inv"]
        self._extr2world = data["extr2world"]
        self._extr2world_inv = data["extr2world_inv"]

    def _load_mano_beta(self):
        file_path = self._calib_folder / "mano" / f"{self._subject_id}.yaml"
        # 如果不存在则全为10个0
        if not file_path.exists():
            self._mano_beta = np.zeros((10,), dtype=np.float32)
            return
        data = read_data_from_yaml(file_path)
        self._mano_beta = np.array(data["betas"], dtype=np.float32)

    def get_color(self, serial, frame_id):
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]"""
        file_path = self._data_folder / serial / f"color_{frame_id:06d}.jpg"
        return read_rgb_image(file_path)

    def get_depth(self, serial, frame_id):
        """Get depth image in numpy format, dtype=float32, [H, W]"""
        file_path = self._data_folder / serial / f"depth_{frame_id:06d}.png"
        return read_depth_image(file_path, 1000.0)

    def get_mask(self, serial, frame_id, object_idx, kernel_size=1):
        """Get mask image in numpy format, dtype=uint8, [H, W]"""
        file_path = self._seg_folder / serial / "mask" / f"mask_{frame_id:06d}.png"
        if not file_path.exists():
            return np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)
        mask = read_mask_image(file_path)
        mask = (mask == (object_idx + 1)).astype(np.uint8)
        if kernel_size > 1:
            mask = erode_mask(mask, kernel_size)
        return mask

    def get_valid_seg_serials(self):
        valid_serials = []
        for serial in self._rs_serials:
            if (self._seg_folder / serial / "mask" / "mask_000000.png").exists():
                valid_serials.append(serial)
        return valid_serials

    def get_seg_color_index_map(self):
        color_index_map = {color.rgb: idx for idx, color in enumerate(HO_CAP_SEG_COLOR)}
        return color_index_map

    def get_object_seg_color(self, object_id):
        idx = self._object_ids.index(object_id) + 1
        color = HO_CAP_SEG_COLOR[idx].rgb
        return color

    def get_mano_seg_color(self, side):
        idx = 5 if side == "right" else 6
        color = HAND_COLORS[idx].rgb
        return color

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def rs_serials(self):
        return self._rs_serials

    @property
    def rs_width(self):
        return self._rs_width

    @property
    def rs_height(self):
        return self._rs_height

    @property
    def hl_serial(self):
        return self._hl_serial

    @property
    def hl_width(self):
        return self._hl_width

    @property
    def hl_height(self):
        return self._hl_height

    @property
    def rs_Ks(self):
        return self._rs_Ks

    @property
    def hl_K(self):
        return self._hl_K

    @property
    def extr2world(self):
        return self._extr2world

    @property
    def extr2world_inv(self):
        return self._extr2world_inv

    @property
    def cam2world(self):
        return self._extr2world

    @property
    def world2cam(self):
        return self._extr2world_inv

    @property
    def mano_sides(self):
        return self._mano_sides

    @property
    def mano_beta(self):
        return self._mano_beta

    @property
    def subject_id(self):
        return self._subject_id

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def object_textured_files(self):
        return self._object_textured_files

    @property
    def object_cleaned_files(self):
        return self._object_cleaned_files
