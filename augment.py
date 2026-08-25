#!/usr/bin/env python3
"""
augment.py — Script tăng cường dữ liệu ảnh tiền polymer (YOLO format)

3 loại augmentation được nâng cấp:
  1. Photometric augmentation (toàn bộ train set, kết hợp biến đổi toàn cục và cục bộ)
  2. Light/Dark pair (cho consistency loss: 50% toàn cục + 50% chói sáng / bóng đổ cục bộ)
  3. Oversampling lớp rách (torn class, kết hợp chói sáng cục bộ mô phỏng torn_bright)

Hỗ trợ chạy độc lập qua CLI hoặc import trực tiếp từ run_c2.py cho từng fold.
"""

import argparse
import csv
import os
import random
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path

# Fix Unicode output on Windows (cp1252 -> utf-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import albumentations as A
import numpy as np
import pandas as pd

DEFAULT_TORN_CLASS_ID = 6
DEFAULT_OUTPUT_DIR    = "train_augmented"
TORN_TARGET_PER_DENOM = 70


def _make_ellipse_mask(h: int, w: int) -> np.ndarray:
    """Tạo 1 mask elip mềm ngẫu nhiên (dùng chung cho glare/shadow)."""
    cx = np.random.randint(int(w * 0.15), int(w * 0.85))
    cy = np.random.randint(int(h * 0.15), int(h * 0.85))
    rx = np.random.randint(int(w * 0.10), int(w * 0.35))
    ry = np.random.randint(int(h * 0.10), int(h * 0.35))
    angle = np.random.randint(0, 180)

    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
    ksize = max(15, (min(rx, ry) // 2) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return mask


def apply_specular_glare(img_bgr: np.ndarray, seed: int = None, n_spots: int = None) -> np.ndarray:
    """
    Mô phỏng vệt chói sáng cục bộ (Specular Glare) trên chất liệu tiền polymer:
    Tạo 1 hoặc nhiều vệt sáng hình elip với tâm ngẫu nhiên, bán kính biến thiên và
    suy giảm Gaussian mượt mà. n_spots=None -> ngẫu nhiên 1-2 vệt (đa dạng hoá hơn
    bản 1-vệt-cố-định trước đây, để Local Grid Head học được nhiều pattern hơn).
    """
    if seed is not None:
        np.random.seed(seed)
    h, w = img_bgr.shape[:2]

    if n_spots is None:
        n_spots = int(np.random.choice([1, 2], p=[0.6, 0.4]))

    combined_mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_spots):
        combined_mask = np.maximum(combined_mask, _make_ellipse_mask(h, w))

    # Cường độ chói (+70 đến +160 pixel)
    intensity = np.random.uniform(70.0, 160.0)
    glare = (combined_mask[:, :, np.newaxis] * intensity).astype(np.float32)

    img_out = np.clip(img_bgr.astype(np.float32) + glare, 0, 255).astype(np.uint8)
    return img_out


def apply_local_shadow(img_bgr: np.ndarray, seed: int = None, n_spots: int = None) -> np.ndarray:
    """
    Mô phỏng bóng đổ cục bộ dạng ELIP (Local Shadow):
    Tạo 1-2 vùng tối cục bộ có cạnh suy giảm mượt mà để mô phỏng góc khuất ánh sáng /
    bàn tay che. n_spots=None -> ngẫu nhiên 1-2 vùng.
    """
    if seed is not None:
        np.random.seed(seed)
    h, w = img_bgr.shape[:2]

    if n_spots is None:
        n_spots = int(np.random.choice([1, 2], p=[0.6, 0.4]))

    combined_mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_spots):
        cx = np.random.randint(int(w * 0.15), int(w * 0.85))
        cy = np.random.randint(int(h * 0.15), int(h * 0.85))
        rx = np.random.randint(int(w * 0.15), int(w * 0.45))
        ry = np.random.randint(int(h * 0.15), int(h * 0.45))
        angle = np.random.randint(0, 180)
        m = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(m, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
        ksize = max(15, (min(rx, ry) // 2) * 2 + 1)
        m = cv2.GaussianBlur(m, (ksize, ksize), 0)
        combined_mask = np.maximum(combined_mask, m)

    # Hệ số giảm sáng (giảm 35% - 65%)
    drop_factor = np.random.uniform(0.35, 0.65)
    shadow_map = 1.0 - (combined_mask[:, :, np.newaxis] * drop_factor)
    img_out = np.clip(img_bgr.astype(np.float32) * shadow_map, 0, 255).astype(np.uint8)
    return img_out


def apply_directional_shadow(img_bgr: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Mô phỏng bóng đổ CÓ HƯỚNG (Directional/Gradient Shadow):
    Backlight thật thường không phải 1 đốm tối tròn ở giữa mà là một dải tối chạy dọc
    theo 1 cạnh của ảnh và mờ dần vào giữa (do nguồn sáng chiếu từ phía sau/1 bên chủ
    thể). Hàm này tạo gradient tuyến tính theo 1 hướng ngẫu nhiên (trên/dưới/trái/phải)
    thay vì luôn dùng elip đối xứng tâm -- giúp Local Grid Head thấy thêm 1 dạng suy
    hao không gian khác với apply_local_shadow, sát với backlight/torn_bright thật hơn.
    """
    if seed is not None:
        np.random.seed(seed)
    h, w = img_bgr.shape[:2]

    edge = np.random.choice(['top', 'bottom', 'left', 'right'])
    coverage = np.random.uniform(0.35, 0.65)  # tỉ lệ chiều dài ảnh bị ảnh hưởng
    drop_factor = np.random.uniform(0.30, 0.60)

    if edge in ('top', 'bottom'):
        ramp = np.clip(1.0 - (np.arange(h) / (h * coverage)), 0.0, 1.0)
        if edge == 'bottom':
            ramp = ramp[::-1]
        grad = np.tile(ramp[:, None], (1, w))
    else:
        ramp = np.clip(1.0 - (np.arange(w) / (w * coverage)), 0.0, 1.0)
        if edge == 'right':
            ramp = ramp[::-1]
        grad = np.tile(ramp[None, :], (h, 1))

    # Làm mượt gradient để tránh viền cứng
    ksize = max(15, (min(h, w) // 8) * 2 + 1)
    grad = cv2.GaussianBlur(grad.astype(np.float32), (ksize, ksize), 0)

    shadow_map = 1.0 - (grad[:, :, np.newaxis] * drop_factor)
    img_out = np.clip(img_bgr.astype(np.float32) * shadow_map, 0, 255).astype(np.uint8)
    return img_out


def get_photo_transform():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.15, p=0.9),
        A.RandomGamma(gamma_limit=(83, 120), p=0.6),
    ])


def get_light_transform():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(0.20, 0.35), contrast_limit=0, p=1.0),
        A.RandomGamma(gamma_limit=(60, 85), p=1.0),
    ])


def get_dark_transform():
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.35, -0.20), contrast_limit=0, p=1.0),
        A.RandomGamma(gamma_limit=(115, 150), p=1.0),
    ])


def get_torn_transform(seed: int):
    random.seed(seed)
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.15, p=0.9),
        A.RandomGamma(gamma_limit=(83, 120), p=0.6),
    ])


def read_label(label_path: Path):
    with open(label_path, "r", encoding="utf-8") as f:
        return f.readlines()


def write_label(label_path: Path, lines: list):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def apply_and_save(transform, img_bgr, out_img_path: Path, seed=None, local_effect=None):
    """
    Áp dụng biến đổi Albumentations kết hợp hiệu ứng cục bộ (nếu có) và lưu ảnh.
    local_effect: None, 'glare', 'shadow', hoặc 'directional_shadow'
    transform=None -> bỏ qua bước Albumentations, chỉ áp hiệu ứng cục bộ (dùng cho
    các sample 'local_only', để tách tín hiệu local khỏi biến đổi global).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if transform is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = transform(image=img_rgb)["image"]
        img_out = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    else:
        img_out = img_bgr.copy()

    if local_effect == 'glare':
        img_out = apply_specular_glare(img_out, seed=seed)
    elif local_effect == 'shadow':
        img_out = apply_local_shadow(img_out, seed=seed)
    elif local_effect == 'directional_shadow':
        img_out = apply_directional_shadow(img_out, seed=seed)

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_img_path), img_out, [cv2.IMWRITE_JPEG_QUALITY, 95])


def has_torn_class(lines: list, torn_class_id: int):
    for line in lines:
        line = line.strip()
        if line and int(line.split()[0]) == torn_class_id:
            return True
    return False


def infer_denomination(label_lines: list, torn_class_id: int):
    class_map = {0: "10k", 1: "100k", 2: "20k", 3: "200k", 4: "50k", 5: "500k"}
    for line in label_lines:
        line = line.strip()
        if line:
            cid = int(line.split()[0])
            if cid != torn_class_id:
                return class_map.get(cid, f"class{cid}")
    return "unknown"


def compute_torn_copies(torn_images_by_denom: dict):
    result = {}
    for denom, stems in torn_images_by_denom.items():
        n_orig = len(stems)
        if n_orig == 0:
            continue
        total_needed = max(0, TORN_TARGET_PER_DENOM - n_orig)
        copies_per_img = max(2, min(5, -(-total_needed // n_orig)))
        for stem in stems:
            result[stem] = copies_per_img
    return result


def merge_to_train(augmented_dir: Path, train_dir: Path):
    print("\n[MERGE] Đang gộp train_augmented/ vào train/ ...")
    src_images = augmented_dir / "images"
    src_labels = augmented_dir / "labels"
    dst_images = train_dir / "images"
    dst_labels = train_dir / "labels"
    n_img = n_lbl = 0
    for f in src_images.glob("*"):
        shutil.copy2(f, dst_images / f.name)
        n_img += 1
    for f in src_labels.glob("*.txt"):
        shutil.copy2(f, dst_labels / f.name)
        n_lbl += 1
    print(f"[MERGE] Đã copy {n_img} ảnh + {n_lbl} nhãn vào train/")


def run_augmentation(
    data_dir: str = ".",
    torn_class_id: int = DEFAULT_TORN_CLASS_ID,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    merge: bool = False,
    image_list: list = None,
    image_list_path: str = None,
    metadata_path: str = None
):
    data_path = Path(data_dir).resolve()
    out_root = Path(output_dir).resolve() if Path(output_dir).is_absolute() else (data_path / output_dir).resolve()
    out_img = out_root / "images"
    out_lbl = out_root / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    log_path = out_root / "augmentation_log.csv"
    combined_meta_path = out_root / "combined_metadata_augmented.csv"

    # Đọc danh sách ảnh cần augment
    target_filenames = set()
    if image_list is not None:
        target_filenames = set(str(x).strip() for x in image_list)
    elif image_list_path is not None and Path(image_list_path).exists():
        with open(image_list_path, 'r', encoding='utf-8') as f:
            target_filenames = set(line.strip() for line in f if line.strip())

    # Tìm kiếm file ảnh và nhãn
    img_map = {}
    lbl_map = {}
    for root, _, files in os.walk(data_path):
        for file in files:
            p = Path(root) / file
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img_map[file] = p
            elif p.suffix.lower() == '.txt' and file != 'classes.txt':
                lbl_map[p.stem] = p

    # Lọc danh sách ảnh
    if len(target_filenames) > 0:
        image_files = [img_map[fname] for fname in target_filenames if fname in img_map]
    else:
        train_img_dir = data_path / "train" / "images"
        if train_img_dir.exists():
            image_files = sorted([f for f in train_img_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        else:
            image_files = sorted(list(img_map.values()))

    print(f"[AUGMENT] Chuẩn bị augment cho {len(image_files)} ảnh -> Output: {out_root}")

    # Đọc metadata gốc nếu có
    meta_df = None
    if metadata_path and Path(metadata_path).exists():
        meta_df = pd.read_csv(metadata_path)
        meta_df['filename'] = meta_df['filename'].astype(str).str.strip()

    stats = {"total_orig": len(image_files), "photo": 0, "light": 0, "dark": 0, "torn_aug": 0, "skipped": 0}
    torn_by_denom = defaultdict(list)
    log_rows = []
    combined_meta_rows = []

    photo_tf = get_photo_transform()
    light_tf = get_light_transform()
    dark_tf  = get_dark_transform()

    for idx, img_path in enumerate(image_files):
        stem = img_path.stem
        lbl_path = lbl_map.get(stem)

        if not lbl_path or not Path(lbl_path).exists():
            stats["skipped"] += 1
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            stats["skipped"] += 1
            continue

        label_lines = read_label(lbl_path)
        denom = infer_denomination(label_lines, torn_class_id)
        is_torn = has_torn_class(label_lines, torn_class_id)

        # Lấy metadata gốc
        orig_cond = "indoor"
        orig_denom_num = 10000
        if meta_df is not None:
            match = meta_df[meta_df['filename'] == img_path.name]
            if len(match) > 0:
                orig_cond = match['condition'].iloc[0]
                orig_denom_num = match['denomination_class'].iloc[0]

        # Thêm ảnh gốc vào combined metadata
        # orig_img_path = chính nó (ảnh gốc không bị degrade) -- dùng làm target cho
        # identity/photometric loss của IC-Net (task 1): IC-Net(ảnh gốc) nên ≈ ảnh gốc.
        combined_meta_rows.append({
            'filename': img_path.name,
            'img_path': str(img_path),
            'lbl_path': str(lbl_path),
            'denomination_class': orig_denom_num,
            'condition': orig_cond,
            'is_torn': str(is_torn).lower(),
            'augmentation_type': 'original',
            'pair_id': stem,
            'orig_img_path': str(img_path)
        })

        if is_torn:
            torn_by_denom[denom].append(stem)

        # 1. Photometric (toàn cục + xen kẽ chói sáng/bóng đổ cục bộ)
        out_photo_img = out_img / f"{stem}_photo.jpg"
        out_photo_lbl = out_lbl / f"{stem}_photo.txt"
        photo_seed = hash(stem + "_photo") & 0xFFFF
        # Trước đây: chỉ 33% có glare, còn lại thuần global -> Local Grid Head thiếu
        # tín hiệu. Giờ ~60% có thêm hiệu ứng cục bộ, luân phiên cả 3 loại để đa dạng.
        _p_roll = idx % 5
        if _p_roll in (0, 1, 2):
            p_effect = ['glare', 'shadow', 'directional_shadow'][_p_roll]
        else:
            p_effect = None
        apply_and_save(photo_tf, img_bgr, out_photo_img, seed=photo_seed, local_effect=p_effect)
        write_label(out_photo_lbl, label_lines)
        stats["photo"] += 1
        log_rows.append({
            "original_filename": img_path.name, "augmented_filename": out_photo_img.name,
            "augmentation_type": "photometric", "denomination_class": denom,
            "is_torn_pair": "false", "pair_id": stem
        })
        # pair_id + orig_img_path giờ luôn trỏ về ảnh gốc (trước đây để trống) -- cần
        # thiết để tính photometric supervision loss (task 1) cho MỌI loại augmentation,
        # không chỉ riêng light/dark pair.
        combined_meta_rows.append({
            'filename': out_photo_img.name,
            'img_path': str(out_photo_img),
            'lbl_path': str(out_photo_lbl),
            'denomination_class': orig_denom_num,
            'condition': orig_cond,
            'is_torn': str(is_torn).lower(),
            'augmentation_type': 'photometric',
            'pair_id': stem,
            'orig_img_path': str(img_path)
        })

        # 2. Light/Dark pair (kết hợp toàn cục + chói sáng/bóng đổ cục bộ)
        out_light_img = out_img / f"{stem}_light.jpg"
        out_light_lbl = out_lbl / f"{stem}_light.txt"
        light_seed = hash(stem + "_light") & 0xFFFF
        # Trước đây chỉ 50% có glare (1 elip cố định). Giờ ~75% có glare (1-2 elip,
        # xem apply_specular_glare) để Local Grid Head thấy nhiều pattern chói hơn,
        # đúng bản chất "light" = có nguồn sáng/phản chiếu cục bộ mạnh chứ không chỉ
        # tăng sáng đều toàn ảnh.
        light_effect = 'glare' if (idx % 4 != 3) else None
        apply_and_save(light_tf, img_bgr, out_light_img, seed=light_seed, local_effect=light_effect)
        write_label(out_light_lbl, label_lines)
        stats["light"] += 1
        log_rows.append({
            "original_filename": img_path.name, "augmented_filename": out_light_img.name,
            "augmentation_type": "light_pair", "denomination_class": denom,
            "is_torn_pair": "true", "pair_id": stem
        })
        combined_meta_rows.append({
            'filename': out_light_img.name,
            'img_path': str(out_light_img),
            'lbl_path': str(out_light_lbl),
            'denomination_class': orig_denom_num,
            'condition': 'light_aug',
            'is_torn': str(is_torn).lower(),
            'augmentation_type': 'light_pair',
            'pair_id': stem,
            'orig_img_path': str(img_path)
        })

        out_dark_img = out_img / f"{stem}_dark.jpg"
        out_dark_lbl = out_lbl / f"{stem}_dark.txt"
        dark_seed = hash(stem + "_dark") & 0xFFFF
        # Trước đây chỉ 50% có shadow (luôn dạng elip tâm). Giờ ~75% có hiệu ứng cục
        # bộ, luân phiên elip (che cục bộ) và directional (mô phỏng backlight/ngược
        # sáng thật -- tối dần theo 1 cạnh chứ không phải 1 đốm tròn ở giữa).
        _d_roll = idx % 4
        if _d_roll == 0:
            dark_effect = 'directional_shadow'
        elif _d_roll in (1, 2):
            dark_effect = 'shadow'
        else:
            dark_effect = None
        apply_and_save(dark_tf, img_bgr, out_dark_img, seed=dark_seed, local_effect=dark_effect)
        write_label(out_dark_lbl, label_lines)
        stats["dark"] += 1
        log_rows.append({
            "original_filename": img_path.name, "augmented_filename": out_dark_img.name,
            "augmentation_type": "dark_pair", "denomination_class": denom,
            "is_torn_pair": "true", "pair_id": stem
        })
        combined_meta_rows.append({
            'filename': out_dark_img.name,
            'img_path': str(out_dark_img),
            'lbl_path': str(out_dark_lbl),
            'denomination_class': orig_denom_num,
            'condition': 'dark_aug',
            'is_torn': str(is_torn).lower(),
            'augmentation_type': 'dark_pair',
            'pair_id': stem,
            'orig_img_path': str(img_path)
        })

        # 2b. Local-only sample (task 2): CHỈ áp hiệu ứng cục bộ (glare/shadow/
        # directional), KHÔNG áp global brightness/gamma. Trước đây mọi hiệu ứng cục
        # bộ đều bị chồng lên nền global (photo/light/dark) nên Local Grid Head luôn
        # học lẫn với tín hiệu global -> tín hiệu yếu, nhiễu. Sample này cho nhánh
        # Local Grid một tín hiệu "sạch": ảnh giữ nguyên phơi sáng tổng thể, chỉ có
        # vùng cục bộ bị chói/tối -- đúng với thực tế banknote chụp trong điều kiện
        # ánh sáng chuẩn nhưng có 1 vùng bị chói đèn / che khuất / ngược sáng cục bộ.
        out_local_img = out_img / f"{stem}_local.jpg"
        out_local_lbl = out_lbl / f"{stem}_local.txt"
        local_seed = hash(stem + "_local") & 0xFFFF
        local_effect_type = ['glare', 'shadow', 'directional_shadow'][idx % 3]
        apply_and_save(None, img_bgr, out_local_img, seed=local_seed, local_effect=local_effect_type)
        write_label(out_local_lbl, label_lines)
        stats["local_only"] = stats.get("local_only", 0) + 1
        log_rows.append({
            "original_filename": img_path.name, "augmented_filename": out_local_img.name,
            "augmentation_type": "local_only", "denomination_class": denom,
            "is_torn_pair": "true", "pair_id": stem
        })
        combined_meta_rows.append({
            'filename': out_local_img.name,
            'img_path': str(out_local_img),
            'lbl_path': str(out_local_lbl),
            'denomination_class': orig_denom_num,
            'condition': 'local_aug',
            'is_torn': str(is_torn).lower(),
            'augmentation_type': 'local_only',
            'pair_id': stem,
            'orig_img_path': str(img_path)
        })

    # 3. Oversampling lớp rách (kết hợp chói sáng cục bộ mô phỏng torn_bright)
    copies_map = compute_torn_copies(torn_by_denom)
    for img_path in image_files:
        stem = img_path.stem
        if stem not in copies_map:
            continue
        n_copies = copies_map[stem]
        lbl_path = lbl_map.get(stem)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None or not lbl_path:
            continue
        label_lines = read_label(lbl_path)
        denom = infer_denomination(label_lines, torn_class_id)
        
        orig_denom_num = 10000
        if meta_df is not None:
            m_r = meta_df[meta_df['filename'] == img_path.name]
            if len(m_r) > 0:
                orig_denom_num = m_r['denomination_class'].iloc[0]

        for i in range(1, n_copies + 1):
            seed = hash(stem + str(i)) & 0xFFFF
            torn_tf = get_torn_transform(seed)
            out_torn_img = out_img / f"{stem}_torn_aug{i}.jpg"
            out_torn_lbl = out_lbl / f"{stem}_torn_aug{i}.txt"
            # Trước đây 50% có glare (mô phỏng torn_bright). Giờ ~75% có hiệu ứng cục
            # bộ, luân phiên cả glare/shadow/directional_shadow để oversample lớp
            # rách cũng thấy đa dạng điều kiện sáng thay vì chỉ mỗi "sáng thêm".
            _t_roll = i % 4
            if _t_roll == 1:
                torn_effect = 'glare'
            elif _t_roll == 2:
                torn_effect = 'shadow'
            elif _t_roll == 3:
                torn_effect = 'directional_shadow'
            else:
                torn_effect = None
            apply_and_save(torn_tf, img_bgr, out_torn_img, seed=seed, local_effect=torn_effect)
            write_label(out_torn_lbl, label_lines)
            stats["torn_aug"] += 1
            log_rows.append({
                "original_filename": img_path.name, "augmented_filename": out_torn_img.name,
                "augmentation_type": "torn_oversample", "denomination_class": denom,
                "is_torn_pair": "false", "pair_id": stem
            })
            combined_meta_rows.append({
                'filename': out_torn_img.name,
                'img_path': str(out_torn_img),
                'lbl_path': str(out_torn_lbl),
                'denomination_class': orig_denom_num,
                'condition': 'torn_aug',
                'is_torn': 'true',
                'augmentation_type': 'torn_oversample',
                'pair_id': stem,
                'orig_img_path': str(img_path)
            })

    # Ghi log CSV
    fieldnames = ["original_filename", "augmented_filename", "augmentation_type",
                  "denomination_class", "is_torn_pair", "pair_id"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    # Ghi combined metadata CSV
    combined_df = pd.DataFrame(combined_meta_rows)
    combined_df.to_csv(combined_meta_path, index=False)
    print(f"[AUGMENT] Đã ghi {len(combined_df)} records vào: {combined_meta_path}")

    if merge:
        merge_to_train(out_root, data_path / "train")

    return combined_meta_path


def main():
    parser = argparse.ArgumentParser(description="Data augmentation for Vietnamese polymer banknote dataset.")
    parser.add_argument("--data_dir", type=str, default=".", help="Root directory")
    parser.add_argument("--torn_class_id", type=int, default=DEFAULT_TORN_CLASS_ID, help="Class ID for torn label")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output subdirectory name")
    parser.add_argument("--image_list_path", type=str, default=None, help="File txt chứa danh sách ảnh cần augment")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Đường dẫn file metadata.csv")
    parser.add_argument("--merge", action="store_true", default=False, help="Gộp kết quả vào train/ gốc")
    args = parser.parse_args()

    run_augmentation(
        data_dir=args.data_dir,
        torn_class_id=args.torn_class_id,
        output_dir=args.output_dir,
        merge=args.merge,
        image_list_path=args.image_list_path,
        metadata_path=args.metadata
    )


if __name__ == "__main__":
    main()