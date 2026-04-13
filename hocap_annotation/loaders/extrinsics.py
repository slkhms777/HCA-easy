from ..utils import *


def _to_matrix4x4(values: Union[list, np.ndarray], field_name: str) -> np.ndarray:
    mat = np.array(values, dtype=np.float32)
    if mat.shape == (4, 4):
        return mat
    if mat.size == 12:
        return np.array(
            [mat[0:4], mat[4:8], mat[8:12], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        )
    raise ValueError(f"Invalid shape for {field_name}: expected 4x4 or 12 values, got {mat.shape}")


def _resolve_source_path(calib_folder: Path, source_path: str) -> Path:
    raw_path = Path(source_path)
    candidates = [
        raw_path,
        PROJ_ROOT / raw_path,
        calib_folder / raw_path,
        calib_folder.parent / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Extrinsics source path not found: {source_path}")


def _infer_master_serial(
    calib_folder: Path,
    serials: list[str],
    legacy_source_path: Optional[str] = None,
) -> str:
    if legacy_source_path:
        legacy_path = _resolve_source_path(calib_folder, legacy_source_path)
        if legacy_path.is_file():
            legacy_data = read_data_from_yaml(legacy_path)
            legacy_master = legacy_data.get("rs_master")
            if legacy_master in serials:
                return legacy_master
    return serials[0]


def load_extrinsics(
    calib_folder: Path,
    subject_id: str,
    serials: list[str],
    extrinsics_source: Optional[str] = None,
) -> dict[str, np.ndarray]:
    if extrinsics_source:
        source_path = _resolve_source_path(calib_folder, extrinsics_source)
        if source_path.is_dir():
            extrinsics_dir = source_path
            legacy_source_path = None
        else:
            extrinsics_dir = calib_folder / "extrinsics" / subject_id
            legacy_source_path = extrinsics_source
    else:
        extrinsics_dir = calib_folder / "extrinsics" / subject_id
        legacy_source_path = None

    if not extrinsics_dir.is_dir():
        raise FileNotFoundError(
            f"Subject extrinsics folder not found: {extrinsics_dir}"
        )

    rs_master = _infer_master_serial(calib_folder, serials, legacy_source_path)

    cam2world_list = []
    world2cam_list = []
    for serial in serials:
        file_path = extrinsics_dir / f"{serial}.yaml"
        data = read_data_from_yaml(file_path)
        cam2world = _to_matrix4x4(data["cam2world"], f"{serial}.cam2world")
        if "world2cam" in data:
            world2cam = _to_matrix4x4(data["world2cam"], f"{serial}.world2cam")
        else:
            world2cam = np.linalg.inv(cam2world).astype(np.float32)
        cam2world_list.append(cam2world)
        world2cam_list.append(world2cam)

    extr2world = np.stack(cam2world_list, axis=0).astype(np.float32)
    extr2world_inv = np.stack(world2cam_list, axis=0).astype(np.float32)

    master_idx = serials.index(rs_master)
    master_cam2world = extr2world[master_idx]
    master_world2cam = extr2world_inv[master_idx]

    extr2master = np.stack(
        [master_world2cam @ cam2world for cam2world in extr2world],
        axis=0,
    ).astype(np.float32)
    extr2master_inv = np.stack(
        [world2cam @ master_cam2world for world2cam in extr2world_inv],
        axis=0,
    ).astype(np.float32)

    # Legacy compatibility aliases. The new format does not store tag transforms.
    identity = np.eye(4, dtype=np.float32)

    return {
        "rs_master": rs_master,
        "extr2world": extr2world,
        "extr2world_inv": extr2world_inv,
        "extr2master": extr2master,
        "extr2master_inv": extr2master_inv,
        "tag_0": identity,
        "tag_0_inv": identity,
        "tag_1": master_world2cam,
        "tag_1_inv": master_cam2world,
    }
