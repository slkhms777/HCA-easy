from ..utils import *
from ..loaders import HOCapLoader
from .renderer_pyrd import OffscreenRenderer


class HOCapRenderer:
    def __init__(self, sequence_folder, debug=False, log_file=None) -> None:
        self._logger = get_logger(
            self.__class__.__name__,
            log_level="DEBUG" if debug else "INFO",
            log_file=log_file,
        )
        self._data_folder = Path(sequence_folder)
        self._reader = HOCapLoader(sequence_folder)
        self._num_frames = self._reader.num_frames
        self._rs_width = self._reader.rs_width
        self._rs_height = self._reader.rs_height
        self._rs_serials = self._reader.rs_serials
        self._rs_Ks = self._reader.rs_Ks
        # self._hl_serial = self._reader.hl_serial
        # self._hl_K = self._reader.hl_K
        # self._hl_width = self._reader.hl_width
        # self._hl_height = self._reader.hl_height
        self._object_ids = self._reader.object_ids
        self._mano_beta = self._reader.mano_beta
        self._mano_sides = self._reader.mano_sides
        self._camera_poses = self._reader.extr2world
        self._seg_color_index_map = self._reader.get_seg_color_index_map()
        self._render_fn = None
        self.render_dict = {}

        # Create renderer
        self._renderer = OffscreenRenderer()
        self._add_cameras_to_renderer()
        self._add_objects_to_renderer()
        self.update_render_dict()

    def _add_cameras_to_renderer(self):
        # Add realsense cameras
        for serial, K in zip(self._rs_serials, self._rs_Ks):
            self._renderer.add_camera(K, serial)
        # Add hololens camera
        # self._renderer.add_camera(self._hl_K, self._hl_serial)

    def _add_objects_to_renderer(self):
        for object_id, mesh_file in zip(
            self._object_ids, self._reader.object_textured_files
        ):
            seg_color = self._reader.get_object_seg_color(object_id)
            self._renderer.add_mesh(str(mesh_file), object_id, seg_color)

    def _get_color_images(self, frame_id):
        serials = (
            self._rs_serials + [self._hl_serial]
            if self.render_dict["pv_poses"] is not None
            else self._rs_serials
        )
        return [self._reader.get_color(serial, frame_id) for serial in serials]

    def update_render_dict(
        self,
        object_poses=None,
        mano_verts=None,
        mano_faces=None,
        mano_colors=None,
        mano_joints=None,
        pv_poses=None,
    ):
        self.render_dict["object_poses"] = object_poses
        self.render_dict["mano_verts"] = mano_verts
        self.render_dict["mano_faces"] = mano_faces
        self.render_dict["mano_colors"] = mano_colors
        self.render_dict["mano_joints"] = mano_joints
        self.render_dict["pv_poses"] = pv_poses
        self._render_fn = (
            draw_image_grid if pv_poses is None else draw_all_camera_images
        )

    def load_poses_m(self, pose_file):
        if Path(pose_file).is_file():
            poses_m = np.load(pose_file).astype(np.float32)
            self._logger.debug(f"MANO poses loaded: {poses_m.shape}, from: {pose_file}")
        else:
            msg = f"MANO poses file not found: {pose_file}"
            self._logger.error(msg)
            raise FileNotFoundError(msg)
        return poses_m

    def load_poses_o(self, pose_file):
        if Path(pose_file).is_file():
            poses_o = np.load(pose_file).astype(np.float32)
            poses_o = np.stack([quat_to_mat(pose) for pose in poses_o], axis=1)
            self._logger.debug(
                f"Object poses loaded: {poses_o.shape}, from: {pose_file}"
            )
        else:
            msg = f"Object poses file not found: {pose_file}"
            self._logger.error(msg)
            raise FileNotFoundError(msg)
        return poses_o

    def get_rendered_colors_by_frame_id(self, frame_id, return_vis=True):
        poses_o = (
            self.render_dict["object_poses"][frame_id]
            if self.render_dict["object_poses"] is not None
            else None
        )
        verts_m = (
            self.render_dict["mano_verts"][frame_id]
            if self.render_dict["mano_verts"] is not None
            else None
        )
        faces_m = self.render_dict["mano_faces"]
        colors_m = self.render_dict["mano_colors"]
        r_colors = self._renderer.get_render_colors(
            width=self._rs_width,
            height=self._rs_height,
            cam_names=self._rs_serials,
            cam_poses=self._camera_poses,
            mesh_names=self._object_ids,
            mesh_poses=poses_o,
            mano_vertices=verts_m,
            mano_faces=faces_m,
            mano_colors=colors_m,
        )
        if self.render_dict["pv_poses"] is not None:
            pv_pose = self.render_dict["pv_poses"][frame_id]
            r_colors += self._renderer.get_render_colors(
                width=self._hl_width,
                height=self._hl_height,
                cam_names=[self._hl_serial],
                cam_poses=[pv_pose],
                mesh_names=self._object_ids,
                mesh_poses=poses_o,
                mano_vertices=verts_m,
                mano_faces=faces_m,
                mano_colors=colors_m,
            )

        if return_vis:
            color_images = self._get_color_images(frame_id)
            vis_image = self._render_fn(
                images=[
                    draw_image_overlay(color_image, r_color, 0.618)
                    for color_image, r_color in zip(color_images, r_colors)
                ],
                names=(
                    self._rs_serials + [self._hl_serial]
                    if self.render_dict["pv_poses"] is not None
                    else self._rs_serials
                ),
                facecolor="black",
                titlecolor="white",
            )
            return r_colors, vis_image
        return r_colors

    def get_rendered_depths_by_frame_id(self, frame_id, return_vis=True):
        poses_o = (
            self.render_dict["object_poses"][frame_id]
            if self.render_dict["object_poses"] is not None
            else None
        )
        verts_m = (
            self.render_dict["mano_verts"][frame_id]
            if self.render_dict["mano_verts"] is not None
            else None
        )
        faces_m = self.render_dict["mano_faces"]
        colors_m = self.render_dict["mano_colors"]
        r_depths = self._renderer.get_render_depths(
            width=self._rs_width,
            height=self._rs_height,
            cam_names=self._rs_serials,
            cam_poses=self._camera_poses,
            mesh_names=self._object_ids,
            mesh_poses=poses_o,
            mano_vertices=verts_m,
            mano_faces=faces_m,
            mano_colors=colors_m,
        )

        if self.render_dict["pv_poses"] is not None:
            pv_pose = self.render_dict["pv_poses"][frame_id]
            r_depths += self._renderer.get_render_depths(
                width=self._hl_width,
                height=self._hl_height,
                cam_names=[self._hl_serial],
                cam_poses=[pv_pose],
                mesh_names=self._object_ids,
                mesh_poses=poses_o,
                mano_vertices=verts_m,
                mano_faces=faces_m,
                mano_colors=colors_m,
            )

        r_depths = [(r_depth * 1000.0).astype(np.uint16) for r_depth in r_depths]
        if return_vis:
            vis_image = self._render_fn(
                images=r_depths,
                names=(
                    self._rs_serials + [self._hl_serial]
                    if self.render_dict["pv_poses"] is not None
                    else self._rs_serials
                ),
                facecolor="black",
                titlecolor="white",
            )
            return r_depths, vis_image
        return r_depths

    def get_rendered_segs_by_frame_id(
        self,
        frame_id,
        return_vis=True,
        seg_obj=True,
        seg_mano=True,
    ):
        poses_o = (
            self.render_dict["object_poses"][frame_id]
            if self.render_dict["object_poses"] is not None
            else None
        )
        verts_m = (
            self.render_dict["mano_verts"][frame_id]
            if self.render_dict["mano_verts"] is not None
            else None
        )
        faces_m = self.render_dict["mano_faces"]
        colors_m = self.render_dict["mano_colors"]
        r_segs = self._renderer.get_render_segs(
            width=self._rs_width,
            height=self._rs_height,
            cam_names=self._rs_serials,
            cam_poses=self._camera_poses,
            mesh_names=self._object_ids,
            mesh_poses=poses_o,
            mano_vertices=verts_m,
            mano_faces=faces_m,
            mano_colors=colors_m,
            seg_obj=seg_obj,
            seg_mano=seg_mano,
        )

        if self.render_dict["pv_poses"] is not None:
            pv_pose = self.render_dict["pv_poses"][frame_id]
            r_segs += self._renderer.get_render_segs(
                width=self._hl_width,
                height=self._hl_height,
                cam_names=[self._hl_serial],
                cam_poses=[pv_pose],
                mesh_names=self._object_ids,
                mesh_poses=poses_o,
                mano_vertices=verts_m,
                mano_faces=faces_m,
                mano_colors=colors_m,
                seg_obj=seg_obj,
                seg_mano=seg_mano,
            )

        r_masks = [
            get_mask_from_seg_image(seg, self._seg_color_index_map) for seg in r_segs
        ]
        if return_vis:
            color_images = self._get_color_images(frame_id)
            vis_image = self._render_fn(
                images=[
                    draw_image_overlay(color_image, r_seg, 0.7)
                    for color_image, r_seg in zip(color_images, r_segs)
                ],
                names=(
                    self._rs_serials + [self._hl_serial]
                    if self.render_dict["pv_poses"] is not None
                    else self._rs_serials
                ),
                facecolor="black",
                titlecolor="white",
            )
            return r_masks, vis_image
        return r_masks

    def render_pose_images(
        self,
        save_folder,
        save_video_path,
        vis_only=True,
        save_vis=False,
        single_process=False,
    ):
        self._logger.info(f"Rendering pose images...")
        vis_images = [None] * self._num_frames
        render_images = {s: [None] * self._num_frames for s in self._rs_serials}
        tqbar = tqdm(total=self._num_frames, ncols=100)
        if single_process:
            for frame_id in range(self._num_frames):
                try:
                    r_images, vis_image = self.get_rendered_colors_by_frame_id(
                        frame_id, return_vis=True
                    )
                    if not vis_only:
                        for s, r_image in zip(self._rs_serials, r_images):
                            render_images[s][frame_id] = r_image
                    vis_images[frame_id] = vis_image
                except Exception as e:
                    self._logger.error(f"Error rendering frame {frame_id}: {e}")
                finally:
                    tqbar.update(1)
        else:
            with concurrent.futures.ProcessPoolExecutor(CFG.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.get_rendered_colors_by_frame_id, frame_id, return_vis=True
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    frame_id = futures[future]
                    try:
                        r_images, vis_image = future.result()
                        if not vis_only:
                            for s, r_image in zip(self._rs_serials, r_images):
                                render_images[s][frame_id] = r_image
                        vis_images[frame_id] = vis_image
                    except Exception as e:
                        self._logger.error(f"Error rendering frame {frame_id}: {e}")
                    finally:
                        tqbar.update(1)
        tqbar.close()
        self._logger.info(f"Saving vis video...")
        self._save_video(Path(save_video_path), vis_images)
        if save_vis:
            self._logger.info(f"Saving vis images...")
            save_image_folder = Path(save_folder) / save_video_path.stem
            self._save_images(save_image_folder, vis_images, image_type="color")
        vis_images.clear()
        del vis_images
        gc.collect()

        if not vis_only:
            self._logger.info(f"Saving render images...")
            for serial in self._rs_serials:
                save_image_folder = Path(save_folder) / serial / "color"
                self._save_images(
                    save_image_folder,
                    render_images[serial],
                    image_type="color",
                    tqdesc=serial,
                )
                del render_images[serial]
        render_images.clear()
        del render_images
        gc.collect()

    def render_depth_images(
        self,
        save_folder,
        save_video_path,
        vis_only=True,
        save_vis=False,
        single_process=False,
    ):
        self._logger.info(f"Rendering depth images...")
        vis_images = [None] * self._num_frames
        render_images = {s: [None] * self._num_frames for s in self._rs_serials}
        tqbar = tqdm(total=self._num_frames, ncols=100)
        if single_process:
            for frame_id in range(self._num_frames):
                try:
                    r_images, vis_image = self.get_rendered_depths_by_frame_id(
                        frame_id, return_vis=True
                    )
                    if not vis_only:
                        for s, r_image in zip(self._rs_serials, r_images):
                            render_images[s][frame_id] = r_image
                    vis_images[frame_id] = vis_image
                except Exception as e:
                    self._logger.error(f"Error rendering frame {frame_id}: {e}")
                finally:
                    tqbar.update(1)
        else:
            with concurrent.futures.ProcessPoolExecutor(CFG.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.get_rendered_depths_by_frame_id, frame_id, return_vis=True
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    frame_id = futures[future]
                    try:
                        r_images, vis_image = future.result()
                        if not vis_only:
                            for s, r_image in zip(self._rs_serials, r_images):
                                render_images[s][frame_id] = r_image
                        vis_images[frame_id] = vis_image
                    except Exception as e:
                        self._logger.error(f"Error rendering frame {frame_id}: {e}")
                    finally:
                        tqbar.update(1)
        tqbar.close()
        self._logger.info(f"Saving vis video...")
        self._save_video(Path(save_video_path), vis_images)
        if save_vis:
            self._logger.info(f"Saving vis images...")
            save_image_folder = Path(save_folder) / save_video_path.stem
            self._save_images(save_image_folder, vis_images, image_type="color")
        vis_images.clear()
        del vis_images
        gc.collect()

        if not vis_only:
            self._logger.info(f"Saving render images...")
            for serial in self._rs_serials:
                save_image_folder = Path(save_folder) / serial / "depth"
                self._save_images(
                    save_image_folder,
                    render_images[serial],
                    image_type="depth",
                    tqdesc=serial,
                )
                del render_images[serial]
        render_images.clear()
        del render_images
        gc.collect()

    def render_mask_images(
        self,
        save_folder,
        save_video_path,
        vis_only=True,
        save_vis=False,
        single_process=False,
    ):
        self._logger.info(f"Rendering mask images...")
        vis_images = [None] * self._num_frames
        render_images = {s: [None] * self._num_frames for s in self._rs_serials}
        tqbar = tqdm(total=self._num_frames, ncols=100)
        if single_process:
            for frame_id in range(self._num_frames):
                try:
                    r_images, vis_image = self.get_rendered_segs_by_frame_id(
                        frame_id, return_vis=True
                    )
                    if not vis_only:
                        for s, r_image in zip(self._rs_serials, r_images):
                            render_images[s][frame_id] = r_image
                    vis_images[frame_id] = vis_image
                except Exception as e:
                    self._logger.error(f"Error rendering frame {frame_id}: {e}")
                finally:
                    tqbar.update(1)
        else:
            with concurrent.futures.ProcessPoolExecutor(CFG.max_workers) as executor:
                futures = {
                    executor.submit(
                        self.get_rendered_segs_by_frame_id, frame_id, return_vis=True
                    ): frame_id
                    for frame_id in range(self._num_frames)
                }
                for future in concurrent.futures.as_completed(futures):
                    frame_id = futures[future]
                    try:
                        r_images, vis_image = future.result()
                        if not vis_only:
                            for s, r_image in zip(self._rs_serials, r_images):
                                render_images[s][frame_id] = r_image
                        vis_images[frame_id] = vis_image
                    except Exception as e:
                        self._logger.error(f"Error rendering frame {frame_id}: {e}")
                    finally:
                        tqbar.update(1)
        tqbar.close()
        self._logger.info(f"Saving vis video...")
        self._save_video(Path(save_video_path), vis_images)
        if save_vis:
            self._logger.info(f"Saving vis images...")
            save_image_folder = Path(save_folder) / save_video_path.stem
            self._save_images(save_image_folder, vis_images, image_type="color")
        vis_images.clear()
        del vis_images
        gc.collect()
        if not vis_only:
            self._logger.info(f"Saving render images...")
            for serial in self._rs_serials:
                save_image_folder = Path(save_folder) / serial / "mask"
                self._save_images(
                    save_image_folder,
                    render_images[serial],
                    image_type="mask",
                    tqdesc=serial,
                )
                del render_images[serial]
        render_images.clear()
        del render_images
        gc.collect()

    def _save_video(self, video_path, images):
        video_path.parent.mkdir(parents=True, exist_ok=True)
        create_video_from_rgb_images(video_path, images, fps=30)

    def _save_images(self, save_folder, images, image_type="color", tqdesc=None):
        make_clean_folder(save_folder)
        if image_type == "color":
            save_image_func = write_rgb_image
            name_template = "color_{:06d}.jpg"
        elif image_type == "depth":
            save_image_func = write_depth_image
            name_template = "depth_{:06d}.png"
        elif image_type == "mask":
            save_image_func = write_mask_image
            name_template = "mask_{:06d}.png"
        else:
            raise ValueError(
                f"Invalid image type: {image_type}, must be 'color', 'depth' or 'mask'"
            )
        tqbar = tqdm(total=len(images), ncols=100, desc=tqdesc)
        with concurrent.futures.ThreadPoolExecutor(CFG.max_workers) as executor:
            futures = {
                executor.submit(
                    save_image_func,
                    save_folder / name_template.format(frame_id),
                    image,
                ): (frame_id, image)
                for frame_id, image in enumerate(images)
            }
            for future in concurrent.futures.as_completed(futures):
                frame_id, image = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._logger.error(f"Error saving image {frame_id}: {e}")
                tqbar.update(1)
        tqbar.close()

    @property
    def mano_beta(self):
        return self._mano_beta

    @property
    def mano_sides(self):
        return self._mano_sides

    @property
    def object_ids(self):
        return self._object_ids

    @property
    def num_frames(self):
        return self._num_frames
