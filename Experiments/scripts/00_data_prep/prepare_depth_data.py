import cv2
import numpy as np
import os
import glob
from pathlib import Path

# --- CONFIG ---
# Raw depth source from the Goh et al. dataset (RealSense z16 exported to 16-bit PNG).
SOURCE_DEPTH_DIR = r"D:\Work\Assisten Dosen\Anylabel\Dataset\gohjinyu-oilpalm-ffb-dataset-d66eb99\ffb-localization\depth_maps"

# Output directory containing depth images converted to 3-channel uint8 PNGs.
OUTPUT_DEPTH_RGB_DIR = r"D:\Work\Assisten Dosen\Anylabel\Experiments\datasets\depth_processed_rgb"

# Normalization range (as required): 0.6m to 6.0m
MIN_DEPTH_M = 0.6  # meters
MAX_DEPTH_M = 6.0  # meters

# Assumption for RealSense z16 PNG: units are millimeters.
# Convert raw uint16 depth to meters by dividing by 1000.
MM_PER_METER = 1000.0

# Invalid depth codes commonly seen in RealSense exports
INVALID_VALUES = (0, 65535)


def depth_uint16_to_uint8_3ch(depth_u16: np.ndarray) -> np.ndarray:
    """
    Convert raw depth (uint16) to a 3-channel uint8 image using:
    - uint16 (assumed millimeters) -> meters (/1000)
    - handle invalid values (0 and 65535)
    - clip to [MIN_DEPTH_M, MAX_DEPTH_M]
    - linear scale to [0, 255]
    - replicate to 3 channels (R=G=B)
    """
    if depth_u16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth, got dtype={depth_u16.dtype}")

    depth_m = depth_u16.astype(np.float32) / MM_PER_METER

    invalid_mask = np.zeros(depth_u16.shape, dtype=bool)
    for v in INVALID_VALUES:
        invalid_mask |= depth_u16 == v

    # Mark invalid as NaN so they don't interfere with clipping/scaling logic.
    depth_m[invalid_mask] = np.nan

    depth_m = np.clip(depth_m, MIN_DEPTH_M, MAX_DEPTH_M)

    scaled = (depth_m - MIN_DEPTH_M) / (MAX_DEPTH_M - MIN_DEPTH_M)  # 0..1 (NaN for invalid)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)  # invalid -> 0 (black)

    depth_u8 = np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)
    depth_3ch = cv2.merge([depth_u8, depth_u8, depth_u8])
    return depth_3ch


def process_depth_maps() -> None:
    os.makedirs(OUTPUT_DEPTH_RGB_DIR, exist_ok=True)

    depth_files = sorted(glob.glob(os.path.join(SOURCE_DEPTH_DIR, "*.png")))
    total_files = len(depth_files)

    print(f"Processing {total_files} depth maps...")
    print(
        "Normalization: uint16(mm)->meters(/1000), invalid {0,65535}, "
        f"clip [{MIN_DEPTH_M}m, {MAX_DEPTH_M}m], scale to uint8 [0,255]."
    )

    written = 0
    for i, file_path in enumerate(depth_files, start=1):
        filename = os.path.basename(file_path)

        # Read depth map as-is (16-bit).
        depth_u16 = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if depth_u16 is None:
            print(f"WARNING: failed to read: {filename}")
            continue
        if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
            raise RuntimeError(
                f"Unexpected depth format for {filename}: dtype={depth_u16.dtype}, shape={depth_u16.shape}"
            )

        depth_3ch = depth_uint16_to_uint8_3ch(depth_u16)

        save_path = os.path.join(OUTPUT_DEPTH_RGB_DIR, filename)
        ok = cv2.imwrite(save_path, depth_3ch)
        if not ok:
            print(f"WARNING: failed to write: {save_path}")
            continue

        written += 1
        if i % 50 == 0 or i == total_files:
            print(f"  ...processed {i}/{total_files} (written={written})")

    print(f"Done. Depth 3-channel images saved to: {OUTPUT_DEPTH_RGB_DIR}")

if __name__ == "__main__":
    process_depth_maps()
