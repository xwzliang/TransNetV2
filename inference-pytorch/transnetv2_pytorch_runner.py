import os
import cv2
import numpy as np
import logging
from typing import Optional, Tuple, Iterable

import torch
from torch import no_grad
from PIL import Image, ImageDraw

from transnetv2_pytorch import TransNetV2 as TransNetV2Torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # make tqdm optional

logger = logging.getLogger(__name__)


class TransNetV2:
    """
    PyTorch TransNetV2 runner/adapter.

    - Loads weights (env TRANSNETV2_TORCH_WEIGHTS or ./transnetv2-pytorch-weights.pth)
    - Uses OpenCV to extract frames (RGB, 48x27)
    - Provides predict_raw / predict_frames / predict_video with the same outputs as the TF version
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        show_progressbar: bool = False,
    ):
        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info("TransNetV2 device: %s", self.device)

        # Model
        self.model = TransNetV2Torch().eval().to(self.device)

        # Resolve weights
        if weights_path is None:
            weights_path = os.getenv("TRANSNETV2_TORCH_WEIGHTS", "")
        if not weights_path:
            base = os.path.dirname(__file__)
            default = os.path.join(base, "transnetv2-pytorch-weights.pth")
            if os.path.isfile(default):
                weights_path = default
            else:
                # fallback: pick any .pth/.pt in the folder
                if os.path.isdir(base):
                    cands = [
                        os.path.join(base, f)
                        for f in os.listdir(base)
                        if f.endswith((".pth", ".pt"))
                    ]
                    if cands:
                        weights_path = sorted(cands)[0]
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "Weights not found. Set TRANSNETV2_TORCH_WEIGHTS or place a .pth/.pt in ./transnetv2-weights/"
            )

        # Load state dict (strip DataParallel prefixes if any)
        sd = torch.load(weights_path, map_location=self.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if isinstance(sd, dict):
            sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        logger.info("Loaded weights from %s", weights_path)

        self._input_size = (27, 48, 3)  # H, W, C
        self._show_progressbar = show_progressbar

    # ---------- Video / frames handling ----------
    @staticmethod
    def _extract_frames_with_ffmpeg(video_path: str) -> np.ndarray:
        import ffmpeg

        out, _ = (
            ffmpeg.input(video_path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27")
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, np.uint8).reshape([-1, 27, 48, 3])

    @staticmethod
    def _extract_frames_with_opencv(
        video_path: str,
        target_height: int = 27,
        target_width: int = 48,
        show_progressbar: bool = False,
    ) -> np.ndarray:
        logger.info("Opening video: %s", video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Failed to open video: %s", video_path)
            raise ValueError("Failed to open video: {}".format(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        pbar = None
        if show_progressbar and tqdm is not None and total_frames > 0:
            pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (target_width, target_height))
            frames.append(frame_resized)
            if pbar:
                pbar.update(1)

        cap.release()
        if pbar:
            pbar.close()

        logger.info("Extracted %d frames", len(frames))
        return np.asarray(frames, dtype=np.uint8)

    @staticmethod
    def _input_iterator(frames: np.ndarray) -> Iterable[np.ndarray]:
        """
        Yields batches shaped [1,100,27,48,3], padded at both ends.
        """
        no_padded_frames_start = 25
        no_padded_frames_end = (
            25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)
        )

        start_frame = np.expand_dims(frames[0], 0)
        end_frame = np.expand_dims(frames[-1], 0)
        padded_inputs = np.concatenate(
            [start_frame] * no_padded_frames_start
            + [frames]
            + [end_frame] * no_padded_frames_end,
            axis=0,
        )

        ptr = 0
        while ptr + 100 <= len(padded_inputs):
            out = padded_inputs[ptr : ptr + 100]
            ptr += 50
            yield out[np.newaxis]  # [1,100,27,48,3]

    # ---------- Inference ----------

    @staticmethod
    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def predict_raw(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        frames: uint8 np.ndarray [B,100,27,48,3]
        returns:
          single_frame_pred: [B,100,1]  (after sigmoid)
          all_frames_pred:   [B,100,1]  (after sigmoid)
        """
        assert (
            len(frames.shape) == 5 and tuple(frames.shape[2:]) == self._input_size
        ), "[TransNetV2] Input must be [B, T, 27, 48, 3]"
        if frames.dtype != np.uint8:
            frames = frames.astype(np.uint8)

        x = torch.from_numpy(frames).to(self.device)  # uint8 → model handles cast
        with no_grad():
            out = self.model(x)
            if isinstance(out, tuple):
                one_hot, extra = out
                many_hot = extra["many_hot"]
            else:
                one_hot = many_hot = out
            one = torch.sigmoid(one_hot).detach().cpu().numpy()
            many = torch.sigmoid(many_hot).detach().cpu().numpy()
        return one, many

    def predict_frames(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        frames: uint8 [T,27,48,3]
        returns:
          single_frame_pred: [T]
          all_frames_pred:   [T]
        """
        assert (
            len(frames.shape) == 4 and tuple(frames.shape[1:]) == self._input_size
        ), "[TransNetV2] Frames must be [T,27,48,3]."

        preds = []
        processed = 0
        for inp in self._input_iterator(frames):
            single_p, all_p = self.predict_raw(inp)
            preds.append((single_p[0, 25:75, 0], all_p[0, 25:75, 0]))
            processed = min(processed + 50, len(frames))
            print(
                "\r[TransNetV2] Processing video frames {}/{}".format(
                    processed, len(frames)
                ),
                end="",
            )
        print("")

        single = np.concatenate([s for s, _ in preds])[: len(frames)]
        many = np.concatenate([a for _, a in preds])[: len(frames)]
        return single, many

    def predict_video(
        self, video_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        frames = self._extract_frames_with_ffmpeg(video_path)  # <— was OpenCV
        single, many = self.predict_frames(frames)
        return frames, single, many

    # ---------- Post / Viz ----------

    @staticmethod
    def predictions_to_scenes(
        predictions: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        predictions = (predictions > threshold).astype(np.uint8)
        scenes = []
        t, t_prev, start = -1, 0, 0
        for i, t in enumerate(predictions):
            if t_prev == 1 and t == 0:
                start = i
            if t_prev == 0 and t == 1 and i != 0:
                scenes.append([start, i])
            t_prev = t
        if t == 0:
            scenes.append([start, i])
        if not scenes:
            return np.array([[0, len(predictions) - 1]], dtype=np.int32)
        return np.array(scenes, dtype=np.int32)

    @staticmethod
    def visualize_predictions(frames: np.ndarray, predictions):
        if isinstance(predictions, np.ndarray):
            predictions = [predictions]

        ih, iw, ic = frames.shape[1:]
        width = 25
        pad_with = width - len(frames) % width if len(frames) % width != 0 else 0
        frames = np.pad(frames, [(0, pad_with), (0, 1), (0, len(predictions)), (0, 0)])

        predictions = [np.pad(x, (0, pad_with)) for x in predictions]
        height = len(frames) // width

        img = frames.reshape([height, width, ih + 1, iw + len(predictions), ic])
        img = np.concatenate(
            np.split(np.concatenate(np.split(img, height), axis=2)[0], width), axis=2
        )[0, :-1]

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for i, pred in enumerate(zip(*predictions)):
            x, y = i % width, i // width
            x, y = x * (iw + len(predictions)) + iw, y * (ih + 1) + ih - 1
            for j, p in enumerate(pred):
                color = [0, 0, 0]
                color[(j + 1) % 3] = 255
                value = round(p * (ih - 1))
                if value != 0:
                    draw.line((x + j, y, x + j, y - value), fill=tuple(color), width=1)
        return img
