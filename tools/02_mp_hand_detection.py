import os

os.environ["MPLBACKEND"] = "Agg"  # Disable matplotlib GUI backend

from hocap_annotation.utils import *
from hocap_annotation.wrappers.mediapipe import MPHandDetector
from hocap_annotation.loaders import SequenceLoader


def draw_handmarks_results_frame(rgb_images, handmarks, serials, save_path):
    """
    Draws hand landmarks on a list of RGB images and displays them with their respective serials.

    Args:
        rgb_images (list of np.ndarray): List of RGB images.
        handmarks (list of list): List of hand landmarks for each image.
        serials (list of str): List of serial names for each image.

    Returns:
        np.ndarray: Array of images with drawn hand landmarks and other debug information.
    """
    if not (len(rgb_images) == len(handmarks) == len(serials)):
        raise ValueError(
            "The length of rgb_images, handmarks, and serials must be the same."
        )

    vis_images = [
        draw_debug_image(
            rgb_image,
            hand_marks=handmarks[idx],
            draw_boxes=True,
            draw_hand_sides=True,
            reduce_background=True,
        )
        for idx, rgb_image in enumerate(rgb_images)
    ]

    vis_frame = draw_image_grid(
        images=vis_images, names=serials, facecolor="black", titlecolor="white"
    )

    write_rgb_image(save_path, vis_frame)


def run_mp_hand_detector(rgb_images, mp_config):
    detector = MPHandDetector(mp_config)
    marks_result = np.full((len(rgb_images), 2, 21, 2), -1, dtype=np.int64)
    for frame_id, rgb_image in enumerate(rgb_images):
        hand_marks, hand_sides, hand_scores = detector.detect_one(rgb_image)
        if hand_sides:
            # Ensure there are no two same hand sides
            if len(hand_sides) == 2 and hand_sides[0] == hand_sides[1]:
                if hand_scores[0] >= hand_scores[1]:
                    hand_sides[1] = "right" if hand_sides[0] == "left" else "left"
                else:
                    hand_sides[0] = "right" if hand_sides[1] == "left" else "left"
            # Update hand marks result
            for i, hand_side in enumerate(hand_sides):
                side_index = 0 if hand_side == "right" else 1
                marks_result[frame_id][side_index] = hand_marks[i]
    return marks_result.astype(np.int64)


class HandJointsDetector:
    def __init__(self, sequence_folder):
        self._data_folder = Path(sequence_folder)
        self._save_folder = self._data_folder / "processed" / "hand_detection"

        self._logger = get_logger(__class__.__name__, "DEBUG")

        self._loader = SequenceLoader(sequence_folder, device=CFG.device)
        self._rs_serials = self._loader.rs_serials
        self._num_frames = self._loader.num_frames
        self._mano_sides = self._loader.mano_sides
        self._num_cameras = len(self._rs_serials)

    def get_rgb_images_by_frame_id(self, frame_id):
        rgb_images = [None] * self._num_cameras
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._loader.get_rgb_image, frame_id, serial): idx
                for idx, serial in enumerate(self._rs_serials)
            }
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                rgb_images[idx] = future.result()
        return rgb_images

    def get_rgb_images_by_serial(self, serial):
        rgb_images = [None] * self._num_frames
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=CFG.max_workers
        ) as executor:
            futures = {
                executor.submit(self._loader.get_rgb_image, frame_id, serial): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                frame_id = futures[future]
                rgb_images[frame_id] = future.result()
        return rgb_images

    def detect_mp_handmarks(self):
        self._logger.info(">>>>>>>>>> Running MediaPipe Hand Detection <<<<<<<<<<")

        mp_config = CFG.mp_hand
        mp_config.device = CFG.device
        mp_config["max_num_hands"] = len(self._mano_sides)
        mp_handmarks = {serial: None for serial in self._rs_serials}

        tqbar = tqdm(total=self._num_cameras, ncols=100)
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    run_mp_hand_detector,
                    rgb_images=self.get_rgb_images_by_serial(serial),
                    mp_config=mp_config,
                ): serial
                for serial in self._rs_serials
            }
            for future in concurrent.futures.as_completed(futures):
                serial = futures[future]
                mp_handmarks[serial] = future.result()
                tqbar.update()
            futures.clear()
        tqbar.close()

        self._logger.info("Updating Hand Detection Results with 'mano_sides'...")
        if self._mano_sides is not None and len(self._mano_sides) == 1:
            for serial in self._rs_serials:
                for frame_id in range(self._num_frames):
                    if "right" in self._mano_sides:
                        if np.any(mp_handmarks[serial][frame_id][0] == -1) and np.all(
                            mp_handmarks[serial][frame_id][1] != -1
                        ):
                            mp_handmarks[serial][frame_id][0] = mp_handmarks[serial][
                                frame_id
                            ][1]
                        mp_handmarks[serial][frame_id][1] = -1
                    if "left" in self._mano_sides:
                        if np.any(mp_handmarks[serial][frame_id][1] == -1) and np.all(
                            mp_handmarks[serial][frame_id][0] != -1
                        ):
                            mp_handmarks[serial][frame_id][1] = mp_handmarks[serial][
                                frame_id
                            ][0]
                        mp_handmarks[serial][frame_id][0] = -1

        self._logger.info("Saving Hand Detection Results...")
        # swap axis to (2, num_frames, 21, 2)
        for serial in self._rs_serials:
            mp_handmarks[serial] = np.swapaxes(mp_handmarks[serial], 0, 1).astype(
                np.int64
            )
        self._save_folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._save_folder / "mp_handmarks_results.npz", **mp_handmarks
        )

    def render_mp_handmarks(self):
        self._logger.info(">>>>>>>>>> Saving Hand Detection Visualizations <<<<<<<<<<")

        mp_handmarks = np.load(self._save_folder / "mp_handmarks_results.npz")
        save_vis_folder = self._save_folder / "vis" / "mp_handmarks"
        make_clean_folder(save_vis_folder)

        tqbar = tqdm(total=self._num_frames, ncols=100)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=CFG.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    draw_handmarks_results_frame,
                    rgb_images=self.get_rgb_images_by_frame_id(frame_id),
                    handmarks=[
                        mp_handmarks[serial][:, frame_id] for serial in self._rs_serials
                    ],
                    serials=self._rs_serials,
                    save_path=save_vis_folder / f"vis_{frame_id:06d}.png",
                ): frame_id
                for frame_id in range(self._num_frames)
            }
            for future in concurrent.futures.as_completed(futures):
                frame_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    self._logger.error(f"Error: {exc}")
                finally:
                    tqbar.update()
            futures.clear()
        tqbar.close()

        self._logger.info("*** Creating Visualization Video ***")
        save_video_path = self._save_folder / "vis_mp_handmarks.mp4"
        vis_image_files = [
            save_vis_folder / f"vis_{frame_id:06d}.png"
            for frame_id in range(self._num_frames)
        ]
        create_video_from_image_files(save_video_path, vis_image_files, fps=30)

    def run(self):
        self._logger.info("=" * 100)
        self._logger.info(
            f"Starting Hand Landmarks Detection for {self._data_folder.name}"
        )
        self._logger.info("=" * 100)

        t_start = time.time()

        self.detect_mp_handmarks()
        self.render_mp_handmarks()

        self._logger.info(f"Done!!! ({time.time() - t_start:.2f} s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe Hand Detection")
    parser.add_argument(
        "--sequence_folder", type=str, default=None, help="Path to the sequence folder."
    )
    args = parser.parse_args()

    if args.sequence_folder is None:
        raise ValueError("Please provide the sequence folder path.")

    detector = HandJointsDetector(args.sequence_folder)
    detector.run()
