from ..layers import MANOGroupLayer, ObjectGroupLayer

from ..utils import *
from .extrinsics import load_extrinsics


class SequenceLoader:
    """
    Class for loading and processing sequence data.

    Supports loading MANO and object layers, along with their poses, intrinsics,
    extrinsics, and metadata required for 3D reconstruction and analysis.
    """

    def __init__(
        self,
        sequence_folder: Union[str, Path],
        load_mano: bool = False,
        load_object: bool = False,
        in_world: bool = True,
        device: str = "cuda",
    ):
        """
        Initializes the SequenceLoader object.

        Args:
            sequence_folder (str): The path to the sequence folder.
            load_mano (bool): Whether to load MANO hand models. Defaults to False.
            load_object (bool): Whether to load object models. Defaults to False.
            in_world (bool): Whether to transform points to the world frame. Defaults to True.
            device (str): The device to run computations on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self._data_folder = Path(sequence_folder)
        self._calib_folder = self._data_folder.parent.parent / "calibration"
        self._models_folder = self._data_folder.parent.parent / "models"
        self._seg_folder = self._data_folder / "processed/segmentation/sam2"
        self._load_mano = load_mano
        self._load_object = load_object
        self._in_world = in_world
        self._device = device

        # Crop limits in world frame, [x_min, x_max, y_min, y_max, z_min, z_max]
        self._crop_lim = [-0.60, +0.60, -0.35, +0.35, -0.01, +0.80]

        # Load metadata
        self._load_metadata()

        # Load MANO and object layers if specified
        self._mano_group_layer = self._init_mano_group_layer()
        self._object_group_layer = self._init_object_group_layer()

        # Create mapping from 2D coordinates to 3D rays
        self._rays = self._create_3d_rays()

        # Create projection matrices from camera to master/world
        self._M2master = torch.bmm(self._rs_Ks, self._extr2master_inv[:, :3, :])
        self._M2world = torch.bmm(self._rs_Ks, self._extr2world_inv[:, :3, :])

        # Initialize points, colors, and masks
        self._frame_id = -1
        self._points = torch.zeros(
            (self.num_cameras, self._rs_height * self._rs_width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._colors = torch.zeros(
            (self.num_cameras, self._rs_height * self._rs_width, 3),
            dtype=torch.float32,
            device=self._device,
        )
        self._masks = torch.zeros(
            (self.num_cameras, self._rs_height * self._rs_width),
            dtype=torch.bool,
            device=self._device,
        )

    def _load_metadata(self):
        data = read_data_from_yaml(self._data_folder / "meta.yaml")

        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._task_id = data["task_id"]
        self._subject_id = data["subject_id"]
        # RealSense camera metadata
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cams = len(self._rs_serials)

        # HoloLens metadata
        # self._hl_serial = data["hololens"]["serial"]
        # self._hl_pv_width = data["hololens"]["pv_width"]
        # self._hl_pv_height = data["hololens"]["pv_height"]

        # Object models file paths
        self._object_textured_files = [
            self._models_folder / obj_id / "textured_mesh.obj"
            for obj_id in self._object_ids
        ]
        self._object_cleaned_files = [
            self._models_folder / obj_id / "cleaned_mesh_10000.obj"
            for obj_id in self._object_ids
        ]

        # Load camera intrinsics
        self._load_intrinsics()

        # Load rs camera extrinsics from per-camera cam2world/world2cam files
        self._load_extrinsics(data.get("extrinsics"))

        # Load MANO shape parameters
        self._mano_beta = self._load_mano_beta()

    def _load_intrinsics(self):
        def read_K_from_yaml(serial, cam_type="color"):
            yaml_file = self._calib_folder / "intrinsics" / f"{serial}.yaml"
            data = read_data_from_yaml(yaml_file)[cam_type]
            K = np.array(
                [
                    [data["fx"], 0.0, data["ppx"]],
                    [0.0, data["fy"], data["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K

        rs_Ks = np.stack(
            [read_K_from_yaml(serial) for serial in self._rs_serials], axis=0
        )
        rs_Ks_inv = np.stack([np.linalg.inv(K) for K in rs_Ks], axis=0)

        # hl_K = read_K_from_yaml(self._hl_serial)
        # hl_K_inv = np.linalg.inv(hl_K)

        # Convert intrinsics to torch tensors
        self._rs_Ks = torch.from_numpy(rs_Ks).to(self._device)
        self._rs_Ks_inv = torch.from_numpy(rs_Ks_inv).to(self._device)
        # self._hl_K = torch.from_numpy(hl_K).to(self._device)
        # self._hl_K_inv = torch.from_numpy(hl_K_inv).to(self._device)

    def _load_extrinsics(self, extrinsics_source=None):
        data = load_extrinsics(
            calib_folder=self._calib_folder,
            subject_id=self._subject_id,
            serials=self._rs_serials,
            extrinsics_source=extrinsics_source,
        )

        self._rs_master = data["rs_master"]

        # Convert extrinsics to torch tensors
        self._tag_0 = torch.from_numpy(data["tag_0"]).to(self._device)
        self._tag_0_inv = torch.from_numpy(data["tag_0_inv"]).to(self._device)
        self._tag_1 = torch.from_numpy(data["tag_1"]).to(self._device)
        self._tag_1_inv = torch.from_numpy(data["tag_1_inv"]).to(self._device)
        self._extr2master = torch.from_numpy(data["extr2master"]).to(self._device)
        self._extr2master_inv = torch.from_numpy(data["extr2master_inv"]).to(
            self._device
        )
        self._extr2world = torch.from_numpy(data["extr2world"]).to(self._device)
        self._extr2world_inv = torch.from_numpy(data["extr2world_inv"]).to(
            self._device
        )

    def _load_mano_beta(self) -> torch.Tensor:
        file_path = self._calib_folder / "mano" / f"{self._subject_id}.yaml"
        # 如果不存在则全为10个0
        if not file_path.exists():
            self._mano_beta = np.zeros(10, dtype=np.float32)
            return
        data = read_data_from_yaml(file_path)
        return torch.tensor(data["betas"], dtype=torch.float32, device=self._device)

    def _create_3d_rays(self) -> torch.Tensor:
        """Creates 3D rays for deprojecting depth images to 3D space."""

        def create_2d_coords() -> torch.Tensor:
            xv, yv = torch.meshgrid(
                torch.arange(self._rs_width),
                torch.arange(self._rs_height),
                indexing="xy",
            )
            coord_2d = torch.stack(
                (xv, yv, torch.ones_like(xv)), dim=0
            ).float()  # (3, H, W)
            coords_2d = (
                coord_2d.unsqueeze(0)
                .repeat(self._num_cams, 1, 1, 1)
                .view(self._num_cams, 3, -1)
            )  # (N, 3, H*W)
            coords_2d = coords_2d.to(self._device)
            return coords_2d

        coords_2d = create_2d_coords()
        return torch.bmm(self._rs_Ks_inv, coords_2d)  # (N, 3, H*W)

    def _init_mano_group_layer(self):
        """Initialize the MANO group layer."""
        if not self._load_mano:
            return None

        mano_group_layer = MANOGroupLayer(
            self._mano_sides, [self._mano_beta.cpu().numpy()] * len(self._mano_sides)
        ).to(self._device)
        return mano_group_layer

    def _init_object_group_layer(self):
        """Initialize the object group layer."""
        if not self._load_object:
            return None

        verts, faces, norms = [], [], []
        for obj_file in self.object_cleaned_mesh_files:
            m = trimesh.load(obj_file, process=False, force='mesh')
            verts.append(m.vertices)
            faces.append(m.faces)
            norms.append(m.vertex_normals)
        object_group_layer = ObjectGroupLayer(verts, faces, norms).to(self._device)
        return object_group_layer

    def _deproject(self, colors, depths) -> tuple:
        """
        Deprojects depth images to 3D points.

        Args:
            colors (np.ndarray): List of color images, [N, H, W, 3], dtype=float32.
            depths (np.ndarray): List of depth images, [N, H, W], dtype=np.float32.

        Returns:
            tuple: Colors, 3D points, and masks.
        """
        # Process color images
        colors = torch.from_numpy(colors.reshape(self._num_cams, -1, 3)).to(
            self._device
        )  # [N, H*W, 3]

        # Process depth images
        depths = torch.from_numpy(depths.reshape(self._num_cams, 1, -1)).to(
            self._device
        )  # [N, 1, H*W]

        # Deproject depth images to 3D points in camera frame
        pts_c = self._rays * depths  # [N, 3, H*W]
        # Transform 3D points from camera frame to world frame
        pts = torch.baddbmm(
            self._extr2world[:, :3, 3].unsqueeze(2),
            self._extr2world[:, :3, :3],
            pts_c,
        ).permute(
            0, 2, 1
        )  # (N, H*W, 3)

        # Crop 3D points
        mx1 = pts[..., 0] > self._crop_lim[0]
        mx2 = pts[..., 0] < self._crop_lim[1]
        my1 = pts[..., 1] > self._crop_lim[2]
        my2 = pts[..., 1] < self._crop_lim[3]
        mz1 = pts[..., 2] > self._crop_lim[4]
        mz2 = pts[..., 2] < self._crop_lim[5]
        masks = mx1 & mx2 & my1 & my2 & mz1 & mz2

        # Transform 3D points from world frame to master frame if necessary
        if not self._in_world:
            pts = torch.baddbmm(
                self._extr2master[:, :3, 3].unsqueeze(2),
                self._extr2master[:, :3, :3],
                pts_c,
            ).permute(
                0, 2, 1
            )  # [N, H*W, 3]

        return colors, pts, masks

    def _update_pcd(self, frame_id: int):
        """Update point cloud data."""
        colors, points, masks = self._deproject(
            np.stack(
                [self.get_rgb_image(frame_id, serial) for serial in self._rs_serials],
                axis=0,
                dtype=np.float32,
            )
            / 255.0,
            np.stack(
                [self.get_depth_image(frame_id, serial) for serial in self._rs_serials],
                axis=0,
                dtype=np.float32,
            ),
        )
        self._points.copy_(points)
        self._colors.copy_(colors)
        self._masks.copy_(masks)

    def get_rgb_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get RGB image in numpy format, dtype=uint8, [H, W, 3]."""
        image_file = self._data_folder / f"{serial}/color_{frame_id:06d}.jpg"
        return read_rgb_image(image_file)

    def get_depth_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get depth image in numpy format, dtype=uint16, [H, W]."""
        image_file = self._data_folder / f"{serial}/depth_{frame_id:06d}.png"
        return read_depth_image(image_file, 1000.0)

    def get_mask_image(self, frame_id: int, serial: str) -> np.ndarray:
        """Get mask image in numpy format, dtype=uint8, [H, W]."""
        image_file = self._seg_folder / f"{serial}/mask/mask_{frame_id:06d}.png"
        if image_file.exists():
            return read_mask_image(image_file)
        else:
            return np.zeros((self._rs_height, self._rs_width), dtype=np.uint8)

    def object_group_layer_forward(
        self, poses: list[torch.Tensor], subset: list[int] = None
    ) -> tuple:
        """Forward pass for the object group layer."""
        p = torch.cat(poses, dim=1)
        v, n = self._object_group_layer(p, inds=subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            n = n.squeeze(0)
        return v, n

    def mano_group_layer_forward(
        self, poses: list[torch.Tensor], subset: list[int] = None
    ) -> tuple:
        """Forward pass for the MANO group layer."""
        p = torch.cat(poses, dim=1)
        v, j = self._mano_group_layer(p, inds=subset)
        if p.size(0) == 1:
            v = v.squeeze(0)
            j = j.squeeze(0)
        return v, j

    def step(self):
        """Step to the next frame."""
        self._frame_id = (self._frame_id + 1) % self._num_frames
        self._update_pcd(self._frame_id)

    def step_by_frame_id(self, frame_id: int):
        """Step to a specific frame."""
        self._frame_id = frame_id % self._num_frames
        self._update_pcd(self._frame_id)

    # Property methods for access to class attributes

    @property
    def sequence_folder(self) -> str:
        return str(self._data_folder)

    @property
    def load_mano(self) -> bool:
        return self._load_mano

    @property
    def load_object(self) -> bool:
        return self._load_object

    @property
    def in_world(self) -> bool:
        return self._in_world

    @property
    def device(self) -> str:
        return self._device

    @property
    def object_ids(self) -> list:
        return self._object_ids

    @property
    def group_id(self) -> str:
        return self._object_ids[0].split("_")[0]

    @property
    def subject_id(self) -> str:
        return self._subject_id

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def rs_width(self) -> int:
        return self._rs_width

    @property
    def rs_height(self) -> int:
        return self._rs_height

    @property
    def rs_serials(self) -> list:
        return self._rs_serials

    @property
    def rs_master(self) -> str:
        return self._rs_master

    @property
    def num_cameras(self) -> int:
        return self._num_cams

    @property
    def holo_serial(self) -> list:
        return self._hl_serial

    @property
    def holo_pv_width(self) -> int:
        return self._hl_pv_width

    @property
    def holo_pv_height(self) -> int:
        return self._hl_pv_height

    @property
    def mano_beta(self) -> torch.Tensor:
        return self._mano_beta

    @property
    def mano_sides(self) -> list:
        return self._mano_sides

    @property
    def intrinsics(self) -> torch.Tensor:
        return self._rs_Ks

    @property
    def intrinsics_inv(self) -> torch.Tensor:
        return self._rs_Ks_inv

    @property
    def extrinsics2master(self) -> torch.Tensor:
        return self._extr2master

    @property
    def extrinsics2master_inv(self) -> torch.Tensor:
        return self._extr2master_inv

    @property
    def extrinsics2world(self) -> torch.Tensor:
        return self._extr2world

    @property
    def extrinsics2world_inv(self) -> torch.Tensor:
        return self._extr2world_inv

    @property
    def cam2world(self) -> torch.Tensor:
        return self._extr2world

    @property
    def world2cam(self) -> torch.Tensor:
        return self._extr2world_inv

    @property
    def tag_0(self) -> torch.Tensor:
        """Legacy compatibility alias. The cam2world format does not store tag_0."""
        return self._tag_0

    @property
    def tag_0_inv(self) -> torch.Tensor:
        """Legacy compatibility alias. The cam2world format does not store tag_0."""
        return self._tag_0_inv

    @property
    def tag_1(self) -> torch.Tensor:
        """Legacy alias for world2cam of the master camera."""
        return self._tag_1

    @property
    def tag_1_inv(self) -> torch.Tensor:
        """Legacy alias for cam2world of the master camera."""
        return self._tag_1_inv

    @property
    def M2master(self) -> torch.Tensor:
        """camera to master transformation matrix"""
        return self._M2master

    @property
    def M2world(self) -> torch.Tensor:
        """camera to world transformation matrix"""
        return self._M2world

    @property
    def frame_id(self) -> int:
        return self._frame_id

    @property
    def mano_group_layer(self):
        return self._mano_group_layer

    @property
    def object_group_layer(self):
        return self._object_group_layer

    @property
    def object_textured_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/textured_mesh.obj")
            for object_id in self._object_ids
        ]

    @property
    def object_cleaned_mesh_files(self) -> list:
        return [
            str(self._models_folder / f"{object_id}/cleaned_mesh_10000.obj")
            for object_id in self._object_ids
        ]

    @property
    def points(self) -> torch.Tensor:
        return self._points

    @property
    def colors(self) -> torch.Tensor:
        return self._colors

    @property
    def masks(self) -> torch.Tensor:
        return self._masks

    @property
    def points_map(self) -> torch.Tensor:
        return self._points.view(self._num_cams, self._rs_height, self._rs_width, 3)

    @property
    def colors_map(self) -> torch.Tensor:
        return self._colors.view(self._num_cams, self._rs_height, self._rs_width, 3)

    @property
    def masks_map(self) -> torch.Tensor:
        return self._masks.view(self._num_cams, self._rs_height, self._rs_width)
