"""
Example:
python /projects/zaijia001/FoundationPose/run_realr1_dino_sam.py \
  --data_dir /projects/zaijia001/R1/hand/d_pour_low \
  --video_id 0 \
  --mesh_file /projects/_hdd/zaijia/R1/obj_mesh/bottle.obj \
  --output_dir /projects/zaijia001/R1/object_pose/d_pour_low_bottle_0 \
  --prompt bottle

Notes:
- FoundationPose requires metric depth. This script will prefer raw `depth_{id}/`
  frames when present, and only fall back to `depth_{id}.mp4` or `depth_vis_{id}.mp4`.
- The script initializes pose on `--init_frame_idx` with Grounding DINO + SAM2,
  then tracks subsequent frames with FoundationPose.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
PHANTOM_ROOT = REPO_ROOT / "phantom"
SAM2_ROOT = PHANTOM_ROOT / "submodules" / "sam2"

for path in [SCRIPT_DIR, PHANTOM_ROOT, SAM2_ROOT]:
  path_str = str(path)
  if path_str not in sys.path:
    sys.path.insert(0, path_str)

from estimater import *  # noqa: E402,F403


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Run Grounding DINO + SAM2 + FoundationPose on a RealR1-style RGB-D video."
  )
  parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Directory with rgb_{id}.mp4, params_{id}.json, and optionally depth_{id}/ or depth_vis_{id}.mp4.",
  )
  parser.add_argument("--video_id", type=str, default="0", help="RealR1 video id when using --data_dir")
  parser.add_argument("--rgb_video", type=str, default=None, help="Optional explicit RGB video path")
  parser.add_argument("--depth_video", type=str, default=None, help="Optional explicit depth video path")
  parser.add_argument(
    "--depth_dir",
    type=str,
    default=None,
    help="Optional directory of metric depth frames (.png/.tiff/.npy). Use this if the depth mp4 is only a visualization.",
  )
  parser.add_argument("--camera_json", type=str, default=None, help="Optional explicit camera intrinsics json path")
  parser.add_argument("--mesh_file", type=str, required=True, help="Object mesh for FoundationPose")
  parser.add_argument("--output_dir", type=str, required=True, help="Directory to save masks, poses, and visualization")
  parser.add_argument("--prompt", type=str, default="bottle", help="Grounding DINO text prompt")
  parser.add_argument("--init_frame_idx", type=int, default=0, help="Frame used for DINO+SAM initialization")
  parser.add_argument("--init_mask", type=str, default=None, help="Optional binary init mask path. If set, DINO+SAM is skipped.")
  parser.add_argument("--dino_model_id", type=str, default="IDEA-Research/grounding-dino-base")
  parser.add_argument("--dino_threshold", type=float, default=0.35)
  parser.add_argument("--sam_config", type=str, default="configs/sam2/sam2_hiera_l.yaml")
  parser.add_argument(
    "--sam_checkpoint",
    type=str,
    default=str(SAM2_ROOT / "checkpoints" / "sam2_hiera_large.pt"),
    help="Local SAM2 checkpoint path",
  )
  parser.add_argument(
    "--depth_scale",
    type=float,
    default=None,
    help="Optional override for depth scale in meters per integer unit. If omitted, uses depth_scale_m from camera json or falls back to 0.001.",
  )
  parser.add_argument(
    "--max_depth_m",
    type=float,
    default=5.0,
    help="Depth values larger than this are treated as invalid and set to zero. Set <=0 to disable clipping.",
  )
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--debug", type=int, default=1)
  parser.add_argument("--save_video", type=int, default=1, help="Whether to save the rendered visualization video")
  return parser.parse_args()


@dataclass
class InputPaths:
  rgb_video: Path
  depth_video: Optional[Path]
  depth_dir: Optional[Path]
  camera_json: Path


def resolve_realr1_inputs(args: argparse.Namespace) -> InputPaths:
  data_dir = Path(args.data_dir).resolve() if args.data_dir else None

  def resolve_optional(explicit: Optional[str], default_name: str) -> Optional[Path]:
    if explicit is not None:
      return Path(explicit).resolve()
    if data_dir is None:
      return None
    candidate = data_dir / default_name
    return candidate.resolve() if candidate.exists() else None

  def resolve_optional_dir(explicit: Optional[str], default_name: str) -> Optional[Path]:
    if explicit is not None:
      return Path(explicit).resolve()
    if data_dir is None:
      return None
    candidate = data_dir / default_name
    return candidate.resolve() if candidate.is_dir() else None

  rgb_video = resolve_optional(args.rgb_video, f"rgb_{args.video_id}.mp4")
  depth_video = resolve_optional(args.depth_video, f"depth_{args.video_id}.mp4")
  if depth_video is None:
    depth_video = resolve_optional(args.depth_video, f"depth_vis_{args.video_id}.mp4")
  depth_dir = resolve_optional_dir(args.depth_dir, f"depth_{args.video_id}")
  camera_json = resolve_optional(args.camera_json, f"params_{args.video_id}.json")

  if rgb_video is None or not rgb_video.exists():
    raise FileNotFoundError("RGB video not found. Pass --rgb_video or provide a valid --data_dir/--video_id.")
  if camera_json is None or not camera_json.exists():
    raise FileNotFoundError("Camera intrinsics json not found. Pass --camera_json or provide a valid --data_dir/--video_id.")
  if depth_dir is None and (depth_video is None or not depth_video.exists()):
    raise FileNotFoundError(
      "Depth input not found. Pass --depth_video or --depth_dir. "
      "If your depth mp4 is only a visualization, use --depth_dir with raw uint16 depth frames."
    )
  if depth_dir is not None and not depth_dir.exists():
    raise FileNotFoundError(f"Depth frame directory does not exist: {depth_dir}")

  return InputPaths(
    rgb_video=rgb_video,
    depth_video=depth_video if depth_dir is None else None,
    depth_dir=depth_dir,
    camera_json=camera_json,
  )


def load_camera_intrinsics(camera_json: Path) -> Tuple[np.ndarray, Dict[str, float]]:
  with open(camera_json, "r", encoding="utf-8") as f:
    params = json.load(f)

  cx = params.get("ppx", params.get("cx"))
  cy = params.get("ppy", params.get("cy"))
  if cx is None or cy is None:
    raise KeyError(f"Camera params in {camera_json} are missing ppx/ppy or cx/cy.")

  K = np.eye(3, dtype=np.float32)
  K[0, 0] = float(params["fx"])
  K[1, 1] = float(params["fy"])
  K[0, 2] = float(cx)
  K[1, 2] = float(cy)

  return K, {
    "fx": float(params["fx"]),
    "fy": float(params["fy"]),
    "cx": float(cx),
    "cy": float(cy),
    "width": int(params["width"]),
    "height": int(params["height"]),
    "depth_scale_m": float(params["depth_scale_m"]) if "depth_scale_m" in params else None,
    "depth_aligned_to_color": bool(params["depth_aligned_to_color"]) if "depth_aligned_to_color" in params else None,
    "depth_storage": params.get("depth_storage"),
  }


def ensure_exists(path: Path, description: str) -> None:
  if not path.exists():
    raise FileNotFoundError(f"{description} not found: {path}")


def ensure_foundationpose_weights() -> None:
  required = [
    SCRIPT_DIR / "weights" / "2023-10-28-18-33-37" / "config.yml",
    SCRIPT_DIR / "weights" / "2023-10-28-18-33-37" / "model_best.pth",
    SCRIPT_DIR / "weights" / "2024-01-11-20-02-45" / "config.yml",
    SCRIPT_DIR / "weights" / "2024-01-11-20-02-45" / "model_best.pth",
  ]
  missing = [path for path in required if not path.exists()]
  if missing:
    missing_str = "\n".join(str(path) for path in missing)
    raise FileNotFoundError(
      "FoundationPose weights are missing. Put the official weights under FoundationPose/weights/.\n"
      f"Missing files:\n{missing_str}"
    )


def clip_bbox_xyxy(bbox_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
  bbox = np.asarray(bbox_xyxy, dtype=np.float32).copy()
  bbox[0] = np.clip(bbox[0], 0, width - 1)
  bbox[1] = np.clip(bbox[1], 0, height - 1)
  bbox[2] = np.clip(bbox[2], bbox[0] + 1, width)
  bbox[3] = np.clip(bbox[3], bbox[1] + 1, height)
  return bbox


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
  mask_u8 = mask.astype(np.uint8)
  if mask_u8.sum() == 0:
    return mask_u8.astype(bool)
  num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
  if num_labels <= 1:
    return mask_u8.astype(bool)
  largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
  return labels == largest_label


def refine_mask(mask: np.ndarray) -> np.ndarray:
  mask_u8 = mask.astype(np.uint8)
  kernel = np.ones((5, 5), dtype=np.uint8)
  mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
  mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
  return largest_connected_component(mask_u8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, bbox_xyxy: np.ndarray, prompt: str, score: float) -> np.ndarray:
  overlay = rgb.copy()
  color = np.array([0, 255, 0], dtype=np.uint8)
  overlay[mask] = (0.35 * overlay[mask] + 0.65 * color).astype(np.uint8)
  vis = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
  x1, y1, x2, y2 = bbox_xyxy.astype(int)
  cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
  cv2.putText(
    vis,
    f"{prompt}: {score:.3f}",
    (x1, max(20, y1 - 8)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2,
    cv2.LINE_AA,
  )
  return vis


def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
  ys, xs = np.where(mask > 0)
  if len(xs) == 0:
    raise ValueError("The provided init mask is empty.")
  return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def load_binary_mask(mask_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
  mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
  if mask is None:
    raise RuntimeError(f"Failed to read init mask: {mask_path}")
  if mask.ndim == 3:
    mask = mask[..., 0]
  if mask.shape != image_shape:
    mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
  mask = mask > 0
  mask = refine_mask(mask)
  if mask.sum() < 64:
    raise ValueError(f"Init mask from {mask_path} is empty or too small after cleanup.")
  return mask


class DepthSource:
  def __init__(self, depth_video: Optional[Path], depth_dir: Optional[Path], depth_scale: float, max_depth_m: Optional[float] = 5.0):
    self.depth_scale = float(depth_scale)
    self.max_depth_m = None if max_depth_m is None or max_depth_m <= 0 else float(max_depth_m)
    self.depth_dir = depth_dir
    self.cap: Optional[cv2.VideoCapture] = None
    self.frame_files: Optional[List[Path]] = None

    if depth_dir is not None:
      supported = []
      for pattern in ["*.png", "*.tif", "*.tiff", "*.npy", "*.npz"]:
        supported.extend(sorted(depth_dir.glob(pattern)))
      if not supported:
        raise FileNotFoundError(f"No supported depth frames found under {depth_dir}")
      self.frame_files = supported
      return

    if depth_video is None:
      raise ValueError("Either depth_video or depth_dir must be provided.")
    self.cap = cv2.VideoCapture(str(depth_video))
    if not self.cap.isOpened():
      raise RuntimeError(f"Failed to open depth video: {depth_video}")

  def __len__(self) -> int:
    if self.frame_files is not None:
      return len(self.frame_files)
    assert self.cap is not None
    return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

  def seek(self, frame_idx: int) -> None:
    if self.cap is not None:
      self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

  def read_raw(self) -> np.ndarray:
    if self.frame_files is not None:
      raise RuntimeError("Use read_raw_at for directory-based depth inputs.")
    assert self.cap is not None
    ok, frame = self.cap.read()
    if not ok:
      raise EOFError("Failed to read next depth frame.")
    return frame

  def read_raw_at(self, frame_idx: int) -> np.ndarray:
    if self.frame_files is not None:
      depth_file = self.frame_files[frame_idx]
      if depth_file.suffix == ".npy":
        return np.load(depth_file)
      if depth_file.suffix == ".npz":
        data = np.load(depth_file)
        if "depth" not in data:
          raise KeyError(f"{depth_file} does not contain a 'depth' array.")
        return data["depth"]
      frame = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
      if frame is None:
        raise RuntimeError(f"Failed to read depth frame: {depth_file}")
      return frame

    assert self.cap is not None
    ok, frame = self.cap.read()
    if not ok:
      raise EOFError(f"Failed to read depth frame at index {frame_idx}.")
    return frame

  def to_meters(self, raw_depth: np.ndarray) -> np.ndarray:
    depth = raw_depth
    if depth.ndim == 3 and depth.shape[2] == 1:
      depth = depth[..., 0]
    elif depth.ndim == 3 and depth.dtype == np.uint16:
      if np.all(depth[..., 0] == depth[..., 1]) and np.all(depth[..., 0] == depth[..., 2]):
        depth = depth[..., 0]
    elif depth.ndim == 3 and depth.dtype == np.uint8:
      mean_diffs = [
        float(np.mean(np.abs(depth[..., 0].astype(np.int16) - depth[..., 1].astype(np.int16)))),
        float(np.mean(np.abs(depth[..., 0].astype(np.int16) - depth[..., 2].astype(np.int16)))),
        float(np.mean(np.abs(depth[..., 1].astype(np.int16) - depth[..., 2].astype(np.int16)))),
      ]
      if max(mean_diffs) < 3.0:
        mode = "gray8_3ch"
      else:
        mode = "pseudocolor8_3ch"
      raise ValueError(
        "Depth input does not look metric. "
        f"Observed {mode} uint8 video, but FoundationPose needs raw metric depth. "
        "Provide --depth_dir with uint16 depth frames or another metric depth source."
      )

    if depth.ndim != 2:
      raise ValueError(f"Unsupported depth shape {depth.shape}. Expected a single-channel depth map.")

    if np.issubdtype(depth.dtype, np.floating):
      depth_m = depth.astype(np.float32)
    elif np.issubdtype(depth.dtype, np.integer):
      depth_m = depth.astype(np.float32) * self.depth_scale
    else:
      raise ValueError(f"Unsupported depth dtype {depth.dtype}.")

    depth_m[~np.isfinite(depth_m)] = 0
    depth_m[depth_m <= 0] = 0
    if self.max_depth_m is not None:
      depth_m[depth_m > self.max_depth_m] = 0
    return depth_m.astype(np.float32)

  def release(self) -> None:
    if self.cap is not None:
      self.cap.release()


def build_dino_detector(model_id: str):
  try:
    from phantom.detectors.detector_dino import DetectorDino
  except Exception as exc:
    raise RuntimeError(
      "Failed to import Grounding DINO wrapper from phantom. "
      "Install the needed packages into the FoundationPose environment, especially transformers."
    ) from exc
  return DetectorDino(detector_id=model_id)


def build_sam2_predictor(config_name: str, checkpoint_path: Path):
  ensure_exists(checkpoint_path, "SAM2 checkpoint")
  try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
  except Exception as exc:
    raise RuntimeError(
      "Failed to import SAM2. Make sure the SAM2 submodule is installed in the same environment "
      "or added to PYTHONPATH."
    ) from exc

  sam_model = build_sam2(config_file=config_name, ckpt_path=str(checkpoint_path), device="cuda")
  return SAM2ImagePredictor(sam_model)


def detect_mask_with_dino_sam(
  rgb: np.ndarray,
  prompt: str,
  dino_detector,
  sam_predictor,
  dino_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
  bbox_xyxy = dino_detector.get_best_bbox(rgb, prompt, threshold=dino_threshold, visualize=False)
  if bbox_xyxy is None:
    raise RuntimeError(f"Grounding DINO failed to detect '{prompt}' in the init frame.")

  h, w = rgb.shape[:2]
  bbox_xyxy = clip_bbox_xyxy(np.asarray(bbox_xyxy, dtype=np.float32), width=w, height=h)

  sam_predictor.set_image(rgb)
  with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    masks, scores, _ = sam_predictor.predict(box=bbox_xyxy, multimask_output=True)
  best_id = int(np.argmax(scores))
  mask = masks[best_id] > 0
  score = float(scores[best_id])

  if mask.sum() < 64:
    center_x = 0.5 * (bbox_xyxy[0] + bbox_xyxy[2])
    center_y = 0.5 * (bbox_xyxy[1] + bbox_xyxy[3])
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
      masks, scores, _ = sam_predictor.predict(
        box=bbox_xyxy,
        point_coords=np.array([[center_x, center_y]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
      )
    best_id = int(np.argmax(scores))
    mask = masks[best_id] > 0
    score = float(scores[best_id])

  mask = refine_mask(mask)
  if mask.sum() < 64:
    raise RuntimeError("SAM2 returned an empty or tiny mask for the init frame.")

  return bbox_xyxy, mask.astype(bool), score


def open_rgb_capture(rgb_video: Path, init_frame_idx: int) -> cv2.VideoCapture:
  cap = cv2.VideoCapture(str(rgb_video))
  if not cap.isOpened():
    raise RuntimeError(f"Failed to open RGB video: {rgb_video}")
  cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame_idx)
  return cap


def make_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
  fourcc = cv2.VideoWriter_fourcc(*"mp4v")
  writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
  if not writer.isOpened():
    raise RuntimeError(f"Failed to create video writer: {output_path}")
  return writer


def main() -> None:
  args = parse_args()
  set_logging_format()
  set_seed(0)

  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for Grounding DINO, SAM2, and FoundationPose.")

  inputs = resolve_realr1_inputs(args)
  ensure_exists(Path(args.mesh_file), "Mesh file")
  ensure_foundationpose_weights()

  output_dir = Path(args.output_dir).resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  poses_dir = output_dir / "ob_in_cam"
  poses_dir.mkdir(parents=True, exist_ok=True)
  debug_dir = output_dir / "foundationpose_debug"
  debug_dir.mkdir(parents=True, exist_ok=True)

  K, camera_meta = load_camera_intrinsics(inputs.camera_json)
  depth_scale = args.depth_scale if args.depth_scale is not None else camera_meta.get("depth_scale_m")
  if depth_scale is None:
    depth_scale = 0.001

  mesh = trimesh.load(args.mesh_file, force="mesh")
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox_3d = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)

  rgb_cap = open_rgb_capture(inputs.rgb_video, args.init_frame_idx)
  rgb_frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
  fps = float(rgb_cap.get(cv2.CAP_PROP_FPS))
  fps = fps if fps > 0 else 30.0

  depth_source = DepthSource(inputs.depth_video, inputs.depth_dir, depth_scale, max_depth_m=args.max_depth_m)
  depth_source.seek(args.init_frame_idx)
  if rgb_frame_count != len(depth_source):
    logging.warning("RGB/depth length mismatch: rgb=%d, depth=%d. Using the shorter stream.", rgb_frame_count, len(depth_source))

  total_frames = min(rgb_frame_count, len(depth_source))
  if args.init_frame_idx >= total_frames:
    raise ValueError(f"init_frame_idx={args.init_frame_idx} is out of range for a {total_frames}-frame sequence.")

  # Fail fast on non-metric depth before loading FoundationPose/SAM2 weights.
  init_raw_depth = depth_source.read_raw_at(args.init_frame_idx) if depth_source.frame_files is not None else depth_source.read_raw()
  init_depth_m = depth_source.to_meters(init_raw_depth)
  if depth_source.frame_files is None:
    depth_source.seek(args.init_frame_idx)

  dino_detector = None
  sam_predictor = None
  if args.init_mask is None:
    logging.info("Initializing detectors...")
    dino_detector = build_dino_detector(args.dino_model_id)
    sam_predictor = build_sam2_predictor(args.sam_config, Path(args.sam_checkpoint).resolve())

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  estimator = FoundationPose(
    model_pts=mesh.vertices,
    model_normals=mesh.vertex_normals,
    mesh=mesh,
    scorer=scorer,
    refiner=refiner,
    debug_dir=str(debug_dir),
    debug=args.debug,
    glctx=glctx,
  )

  video_writer: Optional[cv2.VideoWriter] = None
  pose_list: List[np.ndarray] = []
  frame_indices: List[int] = []

  init_bbox_xyxy: Optional[np.ndarray] = None
  init_mask: Optional[np.ndarray] = None
  init_sam_score: Optional[float] = None

  logging.info("Starting pose estimation from frame %d", args.init_frame_idx)
  for frame_idx in range(args.init_frame_idx, total_frames):
    ok, rgb_bgr = rgb_cap.read()
    if not ok:
      logging.warning("RGB stream ended at frame %d", frame_idx)
      break

    if frame_idx == args.init_frame_idx:
      depth_m = init_depth_m
      if depth_source.frame_files is None:
        depth_source.read_raw()
    else:
      raw_depth = depth_source.read_raw_at(frame_idx) if depth_source.frame_files is not None else depth_source.read_raw()
      depth_m = depth_source.to_meters(raw_depth)
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    if frame_idx == args.init_frame_idx:
      if args.init_mask is not None:
        init_mask = load_binary_mask(Path(args.init_mask).resolve(), image_shape=rgb.shape[:2])
        init_bbox_xyxy = bbox_from_mask(init_mask)
        init_sam_score = None
      else:
        assert dino_detector is not None and sam_predictor is not None
        init_bbox_xyxy, init_mask, init_sam_score = detect_mask_with_dino_sam(
          rgb=rgb,
          prompt=args.prompt,
          dino_detector=dino_detector,
          sam_predictor=sam_predictor,
          dino_threshold=args.dino_threshold,
        )
      cv2.imwrite(str(output_dir / "init_mask.png"), (init_mask.astype(np.uint8) * 255))
      if init_sam_score is not None:
        init_overlay = overlay_mask(rgb, init_mask, init_bbox_xyxy, args.prompt, init_sam_score)
        cv2.imwrite(str(output_dir / "init_detection_overlay.png"), init_overlay)
      pose = estimator.register(
        K=K,
        rgb=rgb,
        depth=depth_m,
        ob_mask=init_mask.astype(bool),
        iteration=args.est_refine_iter,
      )
    else:
      pose = estimator.track_one(
        rgb=rgb,
        depth=depth_m,
        K=K,
        iteration=args.track_refine_iter,
      )

    pose = np.asarray(pose, dtype=np.float32).reshape(4, 4)
    pose_list.append(pose)
    frame_indices.append(frame_idx)
    np.savetxt(poses_dir / f"{frame_idx:06d}.txt", pose)

    center_pose = pose @ np.linalg.inv(to_origin)
    vis_rgb = draw_posed_3d_box(K, img=rgb.copy(), ob_in_cam=center_pose, bbox=bbox_3d)
    vis_rgb = draw_xyz_axis(
      vis_rgb,
      ob_in_cam=center_pose,
      scale=max(0.05, float(np.max(extents)) * 0.6),
      K=K,
      thickness=3,
      transparency=0,
      is_input_rgb=True,
    )
    if frame_idx == args.init_frame_idx and init_mask is not None and init_bbox_xyxy is not None and init_sam_score is not None:
      vis_bgr = overlay_mask(vis_rgb, init_mask, init_bbox_xyxy, args.prompt, init_sam_score)
    else:
      vis_bgr = cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR)

    if args.save_video:
      if video_writer is None:
        video_writer = make_video_writer(output_dir / "pose_vis.mp4", fps=fps, width=vis_bgr.shape[1], height=vis_bgr.shape[0])
      video_writer.write(vis_bgr)

  rgb_cap.release()
  depth_source.release()
  if video_writer is not None:
    video_writer.release()

  if not pose_list:
    raise RuntimeError("No poses were produced.")

  pose_array = np.stack(pose_list, axis=0)
  np.savez_compressed(
    output_dir / "poses.npz",
    poses=pose_array,
    frame_indices=np.asarray(frame_indices, dtype=np.int32),
    K=K.astype(np.float32),
    init_bbox_xyxy=np.asarray(init_bbox_xyxy, dtype=np.float32) if init_bbox_xyxy is not None else np.zeros(4, dtype=np.float32),
    init_mask=np.asarray(init_mask, dtype=np.uint8) if init_mask is not None else np.zeros((1, 1), dtype=np.uint8),
  )

  meta = {
    "prompt": args.prompt,
    "mesh_file": str(Path(args.mesh_file).resolve()),
    "rgb_video": str(inputs.rgb_video),
    "depth_video": str(inputs.depth_video) if inputs.depth_video is not None else None,
    "depth_dir": str(inputs.depth_dir) if inputs.depth_dir is not None else None,
    "camera_json": str(inputs.camera_json),
    "init_frame_idx": args.init_frame_idx,
    "init_mask": str(Path(args.init_mask).resolve()) if args.init_mask is not None else None,
    "dino_threshold": args.dino_threshold,
    "sam_config": args.sam_config,
    "sam_checkpoint": str(Path(args.sam_checkpoint).resolve()),
    "depth_scale": depth_scale,
    "max_depth_m": args.max_depth_m,
    "camera": camera_meta,
    "num_output_frames": len(frame_indices),
    "first_output_frame": frame_indices[0],
    "last_output_frame": frame_indices[-1],
    "init_sam_score": init_sam_score,
  }
  with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

  logging.info("Saved %d poses to %s", len(frame_indices), output_dir)


if __name__ == "__main__":
  main()
