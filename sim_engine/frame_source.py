import time
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
import pandas as pd


class FrameSource:
    """
    Simulates real-time frame arrival:
    - mode='replay': Plays back video at exact native FPS. Sleeps if processing is ahead of schedule.
    - mode='webcam': Reads directly from webcam.
    - mode='sequence': Plays a list of images at target FPS to simulate a camera moving into frame.
    """
    def __init__(
        self,
        mode: str = "replay",
        video_path: Optional[str] = None,
        camera_index: int = 0,
        image_sequence: Optional[List[str]] = None,
        target_fps: float = 30.0
    ):
        self.mode = mode
        self.video_path = video_path
        self.camera_index = camera_index
        self.image_sequence = image_sequence or []
        self.seq_idx = 0
        self.cap = None

        if mode == "webcam":
            self.cap = cv2.VideoCapture(camera_index)
            self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or target_fps
            if self.native_fps <= 0 or np.isnan(self.native_fps):
                self.native_fps = target_fps
        elif mode == "replay":
            if video_path is None or not Path(video_path).exists():
                raise FileNotFoundError(f"Video path not found: {video_path}")
            self.cap = cv2.VideoCapture(video_path)
            self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or target_fps
            if self.native_fps <= 0 or np.isnan(self.native_fps):
                self.native_fps = target_fps
        elif mode == "sequence":
            self.native_fps = target_fps
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'webcam', 'replay', or 'sequence'.")

        self.frame_interval = 1.0 / self.native_fps
        self._next_due_time = time.perf_counter()
        self.frame_index = 0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else len(self.image_sequence)

    def get_next_frame(self) -> Optional[Tuple[np.ndarray, int, float]]:
        """
        Returns:
            (frame_bgr, frame_index, arrival_timestamp) or None if stream is finished.
        """
        if self.mode in ["webcam", "replay"]:
            if self.cap is None or not self.cap.isOpened():
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
        elif self.mode == "sequence":
            if self.seq_idx >= len(self.image_sequence):
                return None
            item = self.image_sequence[self.seq_idx]
            self.seq_idx += 1
            if isinstance(item, np.ndarray):
                frame = item.copy()
            elif isinstance(item, (str, Path)):
                frame = cv2.imread(str(item))
            else:
                return None
            if frame is None:
                return None
        else:
            return None

        # Real-time pacing in replay & sequence mode
        if self.mode in ["replay", "sequence"]:
            now = time.perf_counter()
            wait = self._next_due_time - now
            if wait > 0:
                time.sleep(wait)
            self._next_due_time = time.perf_counter() + self.frame_interval

        arrival_ts = time.perf_counter()
        self.frame_index += 1
        return frame, self.frame_index, arrival_ts

    def reset(self):
        """Rewinds the video/sequence to beginning."""
        self.frame_index = 0
        self.seq_idx = 0
        self._next_due_time = time.perf_counter()
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class DatasetSequenceGenerator:
    """
    Generates video sequences or image lists simulating realistic test sessions from dataset:
    - Phase 1 (0.5s - 0.7s): Aiming / Positioning / Entry phase (user finding note, camera moving)
    - Phase 2 (1.0s - 2.0s): Stabilized note in target lighting condition
    """
    def __init__(self, metadata_path: str = "metadata.csv", data_root: str = "."):
        self.metadata_path = Path(metadata_path)
        self.data_root = Path(data_root)
        self.df = None

        # Build file mapping from all images in train/valid/test directories
        self.img_map = {}
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for p in self.data_root.rglob(ext):
                self.img_map[p.name] = str(p.resolve())

        # Build label mapping from all txt files
        self.lbl_map = {}
        for p in self.data_root.rglob("*.txt"):
            if p.name != "classes.txt":
                self.lbl_map[p.stem] = str(p.resolve())

        if self.metadata_path.exists():
            df_raw = pd.read_csv(self.metadata_path)
            # Map filename to actual path on disk
            df_raw['img_path'] = df_raw['filename'].map(self.img_map)
            # Keep only rows where image file exists on disk
            self.df = df_raw.dropna(subset=['img_path']).reset_index(drop=True)

    def _parse_label_boxes(self, img_path: str, filename: str) -> Tuple[Optional[List[float]], List[List[float]]]:
        """Parses YOLO ground-truth bounding boxes for banknote and tears."""
        stem = Path(filename).stem
        lbl_file = self.lbl_map.get(stem)
        if not lbl_file or not Path(lbl_file).exists():
            return None, []
        img = cv2.imread(img_path)
        if img is None:
            return None, []
        h, w = img.shape[:2]
        banknote_box = None
        tear_boxes = []
        try:
            with open(lbl_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                        x1 = max(0.0, (cx - bw / 2.0) * w)
                        y1 = max(0.0, (cy - bh / 2.0) * h)
                        x2 = min(float(w), (cx + bw / 2.0) * w)
                        y2 = min(float(h), (cy + bh / 2.0) * h)
                        if cls_id < 6:
                            banknote_box = [x1, y1, x2, y2]
                        elif cls_id == 6:
                            tear_boxes.append([x1, y1, x2, y2])
        except Exception:
            pass
        return banknote_box, tear_boxes

    @staticmethod
    def _normalize_denom_str(denom: Any) -> str:
        """Converts 100000 / 100k / 100 to standard '100'."""
        s = str(denom).strip().lower().replace("k", "").replace(".0", "")
        mapping = {
            "10000": "10",
            "20000": "20",
            "50000": "50",
            "100000": "100",
            "200000": "200",
            "500000": "500",
            "10": "10",
            "20": "20",
            "50": "50",
            "100": "100",
            "200": "200",
            "500": "500"
        }
        return mapping.get(s, s)

    def create_condition_session(
        self,
        target_condition: str = "overexposed",
        denomination: Optional[str] = None,
        is_torn: Optional[bool] = None,
        total_frames: int = 60,
        fps: float = 30.0,
        sample_idx: Optional[int] = None,
        include_entry_phase: bool = True,
        sample_img_path: Optional[str] = None,
        condition: Optional[str] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Creates a realistic stream of frames (total_frames @ fps) for a simulation session:
        - Picks unique distinct images from dataset based on condition and sample_idx or sample_img_path.
        - Realistic temporal dynamics:
          Phase 1: Entry / Searching phase (~15 frames) simulating hand moving, camera aiming.
          Phase 2: Stable presentation under target lighting condition.
        """
        if condition is not None:
            target_condition = condition
        if self.df is None or len(self.df) == 0:
            test_dir = self.data_root / "test" / "images"
            images = list(test_dir.glob(f"*{target_condition}*.jpg"))
            if not images:
                images = list(self.data_root.rglob("*.jpg"))
            chosen = [str(images[i % len(images)]) for i in range(total_frames)]
            gt = {
                "denomination": self._normalize_denom_str(denomination) if denomination else "unknown",
                "condition": target_condition,
                "is_torn": is_torn if is_torn is not None else False
            }
            return chosen, gt

        # Filter dataset
        sub_df = self.df.copy()
        if target_condition:
            cond_mask = sub_df['condition'].astype(str).str.lower() == target_condition.lower()
            if cond_mask.sum() > 0:
                sub_df = sub_df[cond_mask]

        if denomination and denomination != "Any":
            target_norm = self._normalize_denom_str(denomination)
            sub_df['norm_denom'] = sub_df['denomination_class'].apply(self._normalize_denom_str)
            denom_mask = sub_df['norm_denom'] == target_norm
            if denom_mask.sum() > 0:
                sub_df = sub_df[denom_mask]

        if is_torn is not None:
            torn_mask = sub_df['is_torn'].astype(str).str.lower() == str(is_torn).lower()
            if torn_mask.sum() > 0:
                sub_df = sub_df[torn_mask]

        if len(sub_df) == 0:
            sub_df = self.df

        # Pick distinct sample based on sample_img_path or sample_idx
        idx = sample_idx if sample_idx is not None else 0
        if sample_img_path:
            img_matches = self.df[self.df['img_path'] == str(sample_img_path)]
            if not img_matches.empty:
                row = img_matches.iloc[0]
            else:
                p_name = Path(sample_img_path).name
                fn_matches = self.df[self.df['filename'] == p_name]
                if not fn_matches.empty:
                    row = fn_matches.iloc[0]
                else:
                    row = sub_df.iloc[0]
        else:
            n_samples = len(sub_df)
            idx = idx % n_samples
            row = sub_df.iloc[idx]
        img_p = str(row['img_path'])
        raw_denom = row.get('denomination_class', '')

        gt_banknote_box, gt_tear_boxes = self._parse_label_boxes(img_p, str(row.get('filename', '')))

        # Ground truth info
        gt = {
            "denomination": self._normalize_denom_str(raw_denom),
            "condition": str(row.get('condition', '')),
            "is_torn": str(row.get('is_torn', '')).lower() == 'true',
            "filename": str(row.get('filename', '')),
            "sample_index": idx,
            "img_path": img_p,
            "gt_banknote_box": gt_banknote_box,
            "gt_tear_boxes": gt_tear_boxes
        }

        # Sequence construction with Entry / Searching Phase
        if not include_entry_phase or total_frames <= 15:
            img_list = [img_p] * total_frames
            return img_list, gt

        base_bgr = cv2.imread(img_p)
        if base_bgr is None:
            img_list = [img_p] * total_frames
            return img_list, gt

        h, w = base_bgr.shape[:2]
        entry_frames = min(15, max(5, total_frames // 4))
        frames_sequence = []

        # Sub-phase 1a (First 8 frames): No note / empty background or desk surface
        # Simulated by creating a low-texture desk background crop or solid surface
        corner_crop = base_bgr[:max(10, h//6), :max(10, w//6)]
        avg_color = corner_crop.mean(axis=(0, 1)).astype(np.uint8)
        empty_bg = np.full((h, w, 3), avg_color, dtype=np.uint8)
        # Add subtle natural sensor noise
        noise = np.random.normal(0, 2, (h, w, 3)).astype(np.int16)
        empty_bg = np.clip(empty_bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        for _ in range(8):
            frames_sequence.append(empty_bg.copy())

        # Sub-phase 1b (Next 7 frames): Banknote sliding into frame with slight motion blur
        for step in range(1, entry_frames - 8 + 1):
            ratio = step / float(entry_frames - 8 + 1)
            # Shift note from bottom-right into center
            shift_y = int((1.0 - ratio) * (h * 0.4))
            shift_x = int((1.0 - ratio) * (w * 0.4))
            M = np.float32([[1, 0, -shift_x], [0, 1, shift_y]])
            transition_frame = cv2.warpAffine(base_bgr, M, (w, h), borderValue=tuple(int(c) for c in avg_color))
            # Apply slight motion blur during movement
            ksize = max(3, int(15 * (1.0 - ratio)))
            if ksize % 2 == 0:
                ksize += 1
            if ksize >= 3:
                transition_frame = cv2.GaussianBlur(transition_frame, (ksize, ksize), 0)
            frames_sequence.append(transition_frame)

        # Phase 2: Remaining frames are stable target banknote image
        stable_count = total_frames - len(frames_sequence)
        for _ in range(stable_count):
            frames_sequence.append(base_bgr)

        return frames_sequence, gt


