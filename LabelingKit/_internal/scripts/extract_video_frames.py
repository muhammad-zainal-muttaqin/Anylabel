"""
Extract image frames from video files for labeling in AnyLabeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="Dataset/Video",
        help="Input video file or folder (relative from LabelingKit root)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="Dataset/Video",
        help="Output base folder where extracted images are saved",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Target extracted frames per second (default: 2.0)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan input folder recursively for video files",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Image filename prefix (default: frame)",
    )
    parser.add_argument(
        "--image_ext",
        type=str,
        default="jpg",
        choices=["jpg", "png", "webp"],
        help="Image file extension (default: jpg)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG/WebP quality (1-100, default: 95)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing extracted frames in output images folder",
    )
    return parser.parse_args()


def resolve_path(root_dir: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    return candidate if candidate.is_absolute() else (root_dir / candidate).resolve()


def collect_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in VIDEO_EXTS else []

    if not input_path.exists():
        return []

    if recursive:
        videos = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    else:
        videos = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(videos)


def clear_images_dir(images_dir: Path) -> None:
    if not images_dir.exists():
        return
    for p in images_dir.iterdir():
        if p.is_file():
            p.unlink()


def frame_stride(src_fps: float, target_fps: float) -> int:
    if target_fps <= 0:
        return 1
    if src_fps <= 0:
        return 1
    stride = int(round(src_fps / target_fps))
    return max(stride, 1)


def get_rotation_angle(cap: cv2.VideoCapture) -> int:
    """Detect video rotation from metadata (handles phone videos)."""
    try:
        angle = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if angle in (0, 90, 180, 270):
            return angle
    except Exception:
        pass
    return 0


def apply_rotation(frame, angle: int):
    """Rotate frame according to metadata angle."""
    if angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extract_from_video(
    video_path: Path,
    out_images_dir: Path,
    target_fps: float,
    prefix: str,
    image_ext: str,
    quality: int,
    overwrite: bool,
) -> tuple[int, int]:
    out_images_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        clear_images_dir(out_images_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    # Detect rotation metadata (common in phone videos)
    rotation = get_rotation_angle(cap)
    if rotation != 0:
        print(f"  Rotation detected: {rotation} degrees -> will auto-correct")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    stride = frame_stride(src_fps, target_fps)

    saved = 0
    idx = 0

    if image_ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, quality)))]
    elif image_ext == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(max(1, min(100, quality)))]
    else:
        params = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % stride == 0:
            if rotation != 0:
                frame = apply_rotation(frame, rotation)
            filename = f"{prefix}_{saved + 1:06d}.{image_ext}"
            out_path = out_images_dir / filename
            if not cv2.imwrite(str(out_path), frame, params):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            saved += 1
        idx += 1

    cap.release()
    return saved, idx


def default_output_for_video(video_path: Path, output_base: Path) -> Path:
    video_name = video_path.stem
    return output_base / video_name / "images"


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[2]
    input_path = resolve_path(root_dir, args.input)
    output_base = resolve_path(root_dir, args.output)

    videos = collect_videos(input_path, args.recursive)
    if not videos:
        print(f"[ERROR] No video files found in: {input_path}")
        return 1

    print("=" * 70)
    print("Video Frame Extraction")
    print("=" * 70)
    print(f"Input       : {input_path}")
    print(f"Output base : {output_base}")
    print(f"Videos      : {len(videos)}")
    print(f"Target FPS  : {args.fps}")
    print(f"Image ext   : {args.image_ext}")
    print(f"Overwrite   : {args.overwrite}")
    print()

    total_saved = 0
    failed = 0
    for video in videos:
        out_dir = default_output_for_video(video, output_base)
        try:
            saved, total_read = extract_from_video(
                video_path=video,
                out_images_dir=out_dir,
                target_fps=args.fps,
                prefix=args.prefix,
                image_ext=args.image_ext,
                quality=args.quality,
                overwrite=args.overwrite,
            )
            total_saved += saved
            print(f"[OK] {video.name} -> {out_dir} (saved: {saved}, read: {total_read})")
        except Exception as exc:
            failed += 1
            print(f"[ERROR] {video.name}: {exc}")

    print()
    print("=" * 70)
    print("Done")
    print("=" * 70)
    print(f"Saved frames : {total_saved}")
    print(f"Failed videos: {failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
