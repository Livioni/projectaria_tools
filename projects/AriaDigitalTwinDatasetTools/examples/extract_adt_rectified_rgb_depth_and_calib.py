#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extract (rectified/undistorted) RGB images, depth maps, and camera intrinsics/extrinsics
from Aria Digital Twin (ADT) sequences.

This script mirrors the APIs used in:
`projects/AriaDigitalTwinDatasetTools/examples/adt_quickstart_tutorial.ipynb`

Typical usage (your conda env should already have projectaria_tools installed):

  python projects/AriaDigitalTwinDatasetTools/examples/extract_adt_rectified_rgb_depth_and_calib.py \
    --adt_root /Users/livion/Documents/github/projectaria_tools/projectaria_tools_adt_data \
    --output_root /tmp/adt_exports \
    --stream_id 214-1 \
    --every_n 10

Outputs per sequence:
  output_root/<sequence_name>/
    rgb_rectified/0000000001234567890.png
    depth_rectified/0000000001234567890.png     (uint16, depth in millimeters)
    camera_to_world/0000000001234567890.npy     (float64, 3x4, OpenCV camera->world)
    K.npy                                       (float64, 3x3, OpenCV intrinsics)
    camera_calibration_src.json
    camera_calibration_rectified.json
    frames.jsonl   (one line per exported frame: timestamp, dt, poses, file paths)
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from projectaria_tools.core import calibration
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider,
    AriaDigitalTwinDataProvider,
)
from projectaria_tools.utils.calibration_utils import rotate_upright_image_and_calibration


@dataclass(frozen=True)
class ExportFrameRecord:
    timestamp_ns: int
    index: int
    stream_id: str
    image_source: str  # "aria" | "synthetic"
    rgb_dt_ns: Optional[int]
    depth_dt_ns: Optional[int]
    pose_dt_ns: Optional[int]
    rgb_path: Optional[str]
    depth_path: Optional[str]
    camera_to_world_path: Optional[str]
    K_path: str
    # 4x4 transforms as row-major lists
    T_scene_device: Optional[List[List[float]]]
    T_scene_camera: Optional[List[List[float]]]


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_png_uint8(path: Path, image_u8: np.ndarray) -> None:
    """
    Save HxWx{1,3} uint8 image as PNG with minimal dependencies.
    Prefers Pillow if installed; falls back to OpenCV; then imageio.
    """
    if image_u8.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image_u8.dtype}")

    try:
        from PIL import Image

        Image.fromarray(image_u8).save(path)
        return
    except Exception:
        pass

    try:
        import cv2  # type: ignore[import-not-found]

        if image_u8.ndim == 3 and image_u8.shape[2] == 3:
            cv2.imwrite(str(path), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(str(path), image_u8)
        return
    except Exception:
        pass

    import imageio.v2 as iio

    iio.imwrite(path, image_u8)


def _save_png_uint16(path: Path, image_u16: np.ndarray) -> None:
    """
    Save HxW uint16 as 16-bit PNG. Depth is expected to be in millimeters.
    """
    if image_u16.dtype != np.uint16:
        raise ValueError(f"Expected uint16 image, got {image_u16.dtype}")

    try:
        from PIL import Image

        Image.fromarray(image_u16, mode="I;16").save(path)
        return
    except Exception:
        pass

    import imageio.v2 as iio

    iio.imwrite(path, image_u16)


def load_depth_png_to_meters(depth_png_path: str | Path) -> np.ndarray:
    """
    Load a depth PNG saved by this script and convert it to meters.

    Our exported depth is a 16-bit PNG storing depth in **millimeters** (uint16).
    This function returns a float32 HxW array in **meters**.
    """
    depth_png_path = Path(depth_png_path)
    try:
        from PIL import Image

        depth_mm = np.array(Image.open(depth_png_path), dtype=np.uint16)
    except Exception:
        import imageio.v2 as iio

        depth_mm = iio.imread(depth_png_path).astype(np.uint16, copy=False)

    if depth_mm.ndim != 2:
        raise ValueError(
            f"Expected a single-channel depth PNG (HxW). Got shape={depth_mm.shape} from {depth_png_path}"
        )

    depth_m = depth_mm.astype(np.float32) / 1000.0
    return depth_m


def _camera_calib_to_dict(cam_calib: Any) -> Dict[str, Any]:
    """
    Extract a JSON-serializable subset of CameraCalibration.
    Works with projectaria_tools pybind objects.
    """
    def _jsonable(v: Any) -> Any:
        # Numpy arrays / scalars
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()

        # Common pybind / enum patterns
        # (CameraModelType, StreamId-like, etc.)
        if hasattr(v, "name") and isinstance(getattr(v, "name"), str):
            return v.name
        if hasattr(v, "value"):
            try:
                val = v.value
                if isinstance(val, (int, float, str, bool)) or val is None:
                    return val
            except Exception:
                pass

        # Sophus SE3/SO3-like
        if hasattr(v, "to_matrix"):
            try:
                return v.to_matrix().tolist()
            except Exception:
                pass

        # Simple types
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v

        # Lists/tuples
        if isinstance(v, (list, tuple)):
            return [_jsonable(x) for x in v]

        # Fallback
        return str(v)

    out: Dict[str, Any] = {}
    for name, fn in (
        ("label", "get_label"),
        ("image_size", "get_image_size"),
        ("model_name", "get_model_name"),
        ("projection_params", "get_projection_params"),
        ("focal_lengths", "get_focal_lengths"),
        ("principal_point", "get_principal_point"),
        ("serial_number", "get_serial_number"),
    ):
        if hasattr(cam_calib, fn):
            try:
                v = getattr(cam_calib, fn)()
                out[name] = _jsonable(v)
            except Exception:
                # keep best-effort; different versions expose slightly different APIs
                pass

    if hasattr(cam_calib, "get_transform_device_camera"):
        try:
            T = cam_calib.get_transform_device_camera()
            out["T_device_camera"] = _jsonable(T)
        except Exception:
            pass

    return out


def _camera_calib_to_opencv_K(cam_calib: Any) -> np.ndarray:
    """
    Build an OpenCV-style 3x3 intrinsics matrix K from a CameraCalibration.
    Assumes the calibration is pinhole/linear (after rectification).
    """
    focals = cam_calib.get_focal_lengths()
    if isinstance(focals, np.ndarray):
        focals = focals.tolist()
    if isinstance(focals, (list, tuple)) and len(focals) >= 2:
        fx, fy = float(focals[0]), float(focals[1])
    else:
        fx = fy = float(focals[0] if isinstance(focals, (list, tuple)) else focals)

    cx = cy = None
    if hasattr(cam_calib, "get_principal_point"):
        pp = cam_calib.get_principal_point()
        if isinstance(pp, np.ndarray):
            pp = pp.tolist()
        cx, cy = float(pp[0]), float(pp[1])
    if cx is None or cy is None:
        # Fallback to image center
        w, h = cam_calib.get_image_size()
        cx, cy = float(w) / 2.0, float(h) / 2.0

    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return K


def _make_rectified_pinhole_calib(
    src_calib: Any,
    rect_width: int,
    rect_height: int,
    rect_focal: float,
    label: str,
) -> Any:
    # Keep extrinsics consistent with the source camera.
    T_device_camera = None
    if hasattr(src_calib, "get_transform_device_camera"):
        T_device_camera = src_calib.get_transform_device_camera()

    if T_device_camera is None:
        return calibration.get_linear_camera_calibration(
            rect_width, rect_height, rect_focal, label
        )

    return calibration.get_linear_camera_calibration(
        rect_width, rect_height, rect_focal, label, T_device_camera
    )


def _to_numpy_u16_depth(depth_image_obj: Any) -> np.ndarray:
    """
    Convert ADT depth image object to a uint16 numpy array (millimeters).
    """
    if hasattr(depth_image_obj, "to_numpy_array"):
        arr = depth_image_obj.to_numpy_array()
        if isinstance(arr, np.ndarray):
            return arr.astype(np.uint16, copy=False)
        return np.array(arr, dtype=np.uint16)
    # Fallback: visualizable is typically 8-bit; prefer failing loudly.
    raise RuntimeError("Depth image object does not support to_numpy_array()")


def _find_sequences(adt_root: Path) -> List[Path]:
    """
    Sequences are subfolders containing a `video.vrs` (raw data) file.
    """
    seqs: List[Path] = []
    for p in sorted(adt_root.iterdir()):
        if p.is_dir() and (p / "video.vrs").exists():
            seqs.append(p)
    return seqs


def _iter_selected_timestamps(
    timestamps_ns: Iterable[int], every_n: int, max_frames: int
) -> Iterable[Tuple[int, int]]:
    count = 0
    for i, t in enumerate(timestamps_ns):
        if every_n > 1 and (i % every_n) != 0:
            continue
        yield count, int(t)
        count += 1
        if max_frames > 0 and count >= max_frames:
            break


def _export_one_sequence(
    sequence_path: Path,
    output_root: Path,
    stream_id: StreamId,
    every_n: int,
    max_frames: int,
    rect_width: Optional[int],
    rect_height: Optional[int],
    rect_focal: Optional[float],
    rotate_upright: bool,
    write_json: bool,
    image_source: str,
) -> None:
    seq_name = sequence_path.name
    out_seq = output_root / seq_name
    out_rgb = out_seq / "rgb_rectified"
    out_depth = out_seq / "depth_rectified"
    out_extr = out_seq / "camera_to_world"
    _mkdir(out_rgb)
    _mkdir(out_depth)
    _mkdir(out_extr)

    # Load ADT providers
    paths_provider = AriaDigitalTwinDataPathsProvider(str(sequence_path))
    data_paths = paths_provider.get_datapaths()
    gt_provider = AriaDigitalTwinDataProvider(data_paths)

    # Camera calibration from raw aria VRS (video.vrs)
    src_cam_calib = gt_provider.get_aria_camera_calibration(stream_id)
    if src_cam_calib is None:
        raise RuntimeError(f"Could not load aria camera calibration for {stream_id}")

    src_w, src_h = [int(x) for x in src_cam_calib.get_image_size()]
    src_f = float(src_cam_calib.get_focal_lengths()[0])

    rw = int(rect_width) if rect_width is not None else src_w
    rh = int(rect_height) if rect_height is not None else src_h
    rf = float(rect_focal) if rect_focal is not None else src_f

    rect_cam_calib = _make_rectified_pinhole_calib(
        src_cam_calib,
        rect_width=rw,
        rect_height=rh,
        rect_focal=rf,
        label=f"{src_cam_calib.get_label()}_pinhole",
    )

    # Optional: rotate images to upright and update the calibration accordingly.
    # We compute the rotated calibration once (rotation is constant across frames).
    out_cam_calib = rect_cam_calib
    if rotate_upright:
        dummy = np.zeros((rh, rw, 3), dtype=np.uint8)
        _, out_cam_calib = rotate_upright_image_and_calibration(dummy, rect_cam_calib)

    # Persist calibration JSON once per sequence (optional)
    if write_json:
        with (out_seq / "camera_calibration_src.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                _camera_calib_to_dict(src_cam_calib), f, indent=2, ensure_ascii=False
            )
        with (out_seq / "camera_calibration_rectified.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                _camera_calib_to_dict(out_cam_calib), f, indent=2, ensure_ascii=False
            )

    # Persist OpenCV intrinsics once per sequence
    K = _camera_calib_to_opencv_K(out_cam_calib)
    K_rel_path = "intrinsic.npy"
    np.save(out_seq / K_rel_path, K)

    # Timestamps
    img_timestamps_ns = gt_provider.get_aria_device_capture_timestamps_ns(stream_id)

    frame_records: List[ExportFrameRecord] = []
    for out_idx, ts_ns in _iter_selected_timestamps(
        img_timestamps_ns, every_n=every_n, max_frames=max_frames
    ):
        # Image (aria RGB or synthetic)
        rgb_path = None
        rgb_dt = None
        if image_source == "synthetic":
            image_with_dt = gt_provider.get_synthetic_image_by_timestamp_ns(
                int(ts_ns), stream_id
            )
        else:
            image_with_dt = gt_provider.get_aria_image_by_timestamp_ns(ts_ns, stream_id)
        if image_with_dt.is_valid():
            rgb_dt = int(image_with_dt.dt_ns())
            rgb = image_with_dt.data().to_numpy_array()
            if rgb.ndim == 2:
                rgb = np.repeat(rgb[..., np.newaxis], 3, axis=2)
            rgb = rgb.astype(np.uint8, copy=False)

            # Rectify/undistort
            rgb_rect = calibration.distort_by_calibration(
                rgb, rect_cam_calib, src_cam_calib
            )
            rgb_rect = np.asarray(rgb_rect)
            if rotate_upright:
                rgb_rect = np.rot90(rgb_rect, k=3)

            rgb_path = str(Path("rgb_rectified") / f"{ts_ns}.png")
            _save_png_uint8(out_seq / rgb_path, rgb_rect)

        # Depth (GT depth maps are queried using the same stream_id in the ADT tutorials)
        depth_path = None
        depth_dt = None
        depth_with_dt = gt_provider.get_depth_image_by_timestamp_ns(ts_ns, stream_id)
        if depth_with_dt.is_valid():
            depth_dt = int(depth_with_dt.dt_ns())
            depth_raw = _to_numpy_u16_depth(depth_with_dt.data())
            depth_rect = calibration.distort_depth_by_calibration(
                depth_raw, rect_cam_calib, src_cam_calib
            )
            depth_rect = np.asarray(depth_rect).astype(np.uint16, copy=False)
            if rotate_upright:
                depth_rect = np.rot90(depth_rect, k=3)
            depth_path = str(Path("depth_rectified") / f"{ts_ns}.png")
            _save_png_uint16(out_seq / depth_path, depth_rect)

        # Pose (scene -> device). If available, derive scene -> camera.
        pose_dt = None
        T_scene_device = None
        T_scene_camera = None
        extr_path = None
        aria_pose_with_dt = gt_provider.get_aria_3d_pose_by_timestamp_ns(ts_ns)
        if aria_pose_with_dt.is_valid():
            pose_dt = int(aria_pose_with_dt.dt_ns())
            aria_pose = aria_pose_with_dt.data()
            try:
                Tsd = aria_pose.transform_scene_device.to_matrix()
                T_scene_device = Tsd.tolist()
                Tdc = out_cam_calib.get_transform_device_camera().to_matrix()
                Twc = (Tsd @ Tdc)  # world(scene) <- camera
                T_scene_camera = Twc.tolist()

                # Save OpenCV camera-to-world extrinsics as 3x4 matrix (R|t)
                extr = np.asarray(Twc, dtype=np.float64)[:3, :]
                extr_path = str(Path("camera_to_world") / f"{ts_ns}.npy")
                np.save(out_seq / extr_path, extr)
            except Exception:
                pass

        frame_records.append(
            ExportFrameRecord(
                timestamp_ns=int(ts_ns),
                index=int(out_idx),
                stream_id=str(stream_id),
                image_source=str(image_source),
                rgb_dt_ns=rgb_dt,
                depth_dt_ns=depth_dt,
                pose_dt_ns=pose_dt,
                rgb_path=rgb_path,
                depth_path=depth_path,
                camera_to_world_path=extr_path,
                K_path=K_rel_path,
                T_scene_device=T_scene_device,
                T_scene_camera=T_scene_camera,
            )
        )

    # Write per-frame manifest
    with (out_seq / "frames.jsonl").open("w", encoding="utf-8") as f:
        for r in frame_records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    print(f"[OK] Exported sequence '{seq_name}' -> {out_seq}")
    print(f"     frames: {len(frame_records)} (every_n={every_n}, max_frames={max_frames})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export rectified RGB/depth and camera intrinsics/extrinsics from ADT."
    )
    p.add_argument(
        "--adt_root",
        type=str,
        default=str(
            Path(__file__).resolve().parents[3] / "projectaria_tools_adt_data"
        ),
        help="Root folder containing ADT sequences (each subfolder is a sequence).",
    )
    p.add_argument(
        "--sequence_path",
        type=str,
        default="",
        help="If set, export only this sequence path (overrides --adt_root scan).",
    )
    p.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Output directory for exported images and JSON metadata.",
    )
    p.add_argument(
        "--stream_id",
        type=str,
        default="214-1",
        help='Aria RGB stream id (ADT tutorial uses "214-1"). Depth and synthetic are queried with the same stream id.',
    )
    p.add_argument(
        "--image_source",
        type=str,
        default="synthetic",
        choices=["aria", "synthetic"],
        help='Which image stream to export. "synthetic" uses get_synthetic_image_by_timestamp_ns; "aria" uses get_aria_image_by_timestamp_ns.',
    )
    p.add_argument(
        "--every_n",
        type=int,
        default=1,
        help="Export every N-th RGB timestamp (1 = export all).",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="If > 0, export at most this many frames per sequence.",
    )
    p.add_argument(
        "--rectified_width",
        type=int,
        default=512,
        help="Rectified pinhole width. Defaults to the source calibration width.",
    )
    p.add_argument(
        "--rectified_height",
        type=int,
        default=512,
        help="Rectified pinhole height. Defaults to the source calibration height.",
    )
    p.add_argument(
        "--rectified_focal",
        type=float,
        default=280,
        help="Rectified pinhole focal length (pixels). Defaults to source focal length.",
    )
    # Default behavior: rotate images to upright and update calibration accordingly.
    # We keep a CLI option to disable it for users who want raw orientation.
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--rotate_upright",
        dest="rotate_upright",
        action="store_true",
        default=True,
        help="Rotate exported RGB + depth clockwise to upright orientation, and update intrinsics/extrinsics accordingly (default).",
    )
    g.add_argument(
        "--no_rotate_upright",
        dest="rotate_upright",
        action="store_false",
        help="Disable upright rotation; exported images keep original orientation and calibration remains un-rotated.",
    )
    p.add_argument(
        "--no_json",
        action="store_true",
        help="If set, skip writing camera_calibration_*.json (still writes K.npy, camera_to_world/*.npy, and frames.jsonl).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    adt_root = Path(args.adt_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    _mkdir(output_root)

    if args.sequence_path:
        sequences = [Path(args.sequence_path).expanduser().resolve()]
    else:
        sequences = _find_sequences(adt_root)
        if not sequences:
            raise RuntimeError(f"No sequences found under {adt_root} (expected subfolders with video.vrs)")

    stream_id = StreamId(args.stream_id)
    for seq in sequences:
        _export_one_sequence(
            sequence_path=seq,
            output_root=output_root,
            stream_id=stream_id,
            every_n=max(1, int(args.every_n)),
            max_frames=int(args.max_frames),
            rect_width=args.rectified_width,
            rect_height=args.rectified_height,
            rect_focal=args.rectified_focal,
            rotate_upright=bool(args.rotate_upright),
            write_json=not bool(args.no_json),
            image_source=str(args.image_source),
        )


if __name__ == "__main__":
    main()


