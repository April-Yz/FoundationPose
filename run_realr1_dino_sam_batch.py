"""
Batch wrapper for run_realr1_dino_sam.py.

Example:
python /projects/zaijia001/FoundationPose/run_realr1_dino_sam_batch.py \
  --data_dir /projects/zaijia001/R1/hand/d_pour_low \
  --mesh_file /projects/_hdd/zaijia/R1/obj_mesh/bottle.obj \
  --output_root /projects/zaijia001/R1/object_pose \
  --prompt bottle \
  --save_video 1 \
  --save_mesh_overlay_video 1 \
  --save_bbox_overlay_video 1
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SINGLE_RUN_SCRIPT = SCRIPT_DIR / "run_realr1_dino_sam.py"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Batch run Grounding DINO + SAM2 + FoundationPose on a RealR1-style folder.")
  parser.add_argument("--data_dir", type=str, required=True, help="Directory with rgb_{id}.mp4, params_{id}.json, and optionally depth_{id}/.")
  parser.add_argument("--mesh_file", type=str, required=True, help="Object mesh for FoundationPose.")
  parser.add_argument("--output_root", type=str, required=True, help="Root directory for per-video outputs.")
  parser.add_argument("--prompt", type=str, default="bottle")
  parser.add_argument("--video_ids", type=str, nargs="*", default=None, help="Optional subset of video IDs. If omitted, process all IDs found in data_dir.")
  parser.add_argument("--output_name_pattern", type=str, default="{data_name}_{prompt}_{video_id}", help="Pattern for per-video output folder names.")
  parser.add_argument("--skip_existing", type=int, default=1, help="Skip a video if output_dir/poses.npz already exists.")
  parser.add_argument("--continue_on_error", type=int, default=1, help="Continue processing remaining IDs after a failure.")
  parser.add_argument("--init_frame_idx", type=int, default=0)
  parser.add_argument("--init_mask", type=str, default=None)
  parser.add_argument("--dino_model_id", type=str, default="IDEA-Research/grounding-dino-base")
  parser.add_argument("--dino_threshold", type=float, default=0.35)
  parser.add_argument("--sam_config", type=str, default="configs/sam2/sam2_hiera_l.yaml")
  parser.add_argument("--sam_checkpoint", type=str, default=None)
  parser.add_argument("--depth_scale", type=float, default=None)
  parser.add_argument("--max_depth_m", type=float, default=5.0)
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--debug", type=int, default=1)
  parser.add_argument("--save_video", type=int, default=1)
  parser.add_argument("--save_mesh_overlay_video", type=int, default=0)
  parser.add_argument("--save_bbox_overlay_video", type=int, default=0)
  parser.add_argument("--mesh_overlay_alpha", type=float, default=0.45)
  return parser.parse_args()


def discover_video_ids(data_dir: Path) -> List[str]:
  ids = set()
  for path in data_dir.glob("rgb_*.mp4"):
    ids.add(path.stem.split("_", 1)[1])
  for path in data_dir.glob("params_*.json"):
    ids.add(path.stem.split("_", 1)[1])
  for path in data_dir.glob("depth_*"):
    if path.is_dir():
      ids.add(path.name.split("_", 1)[1])
  return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def build_output_dir(output_root: Path, data_dir: Path, prompt: str, video_id: str, pattern: str) -> Path:
  safe_prompt = prompt.replace(" ", "_")
  name = pattern.format(data_name=data_dir.name, prompt=safe_prompt, video_id=video_id)
  return output_root / name


def build_single_command(args: argparse.Namespace, video_id: str, output_dir: Path) -> List[str]:
  cmd = [
    sys.executable,
    str(SINGLE_RUN_SCRIPT),
    "--data_dir",
    str(Path(args.data_dir).resolve()),
    "--video_id",
    str(video_id),
    "--mesh_file",
    str(Path(args.mesh_file).resolve()),
    "--output_dir",
    str(output_dir.resolve()),
    "--prompt",
    args.prompt,
    "--init_frame_idx",
    str(args.init_frame_idx),
    "--dino_model_id",
    args.dino_model_id,
    "--dino_threshold",
    str(args.dino_threshold),
    "--sam_config",
    args.sam_config,
    "--max_depth_m",
    str(args.max_depth_m),
    "--est_refine_iter",
    str(args.est_refine_iter),
    "--track_refine_iter",
    str(args.track_refine_iter),
    "--debug",
    str(args.debug),
    "--save_video",
    str(args.save_video),
    "--save_mesh_overlay_video",
    str(args.save_mesh_overlay_video),
    "--save_bbox_overlay_video",
    str(args.save_bbox_overlay_video),
    "--mesh_overlay_alpha",
    str(args.mesh_overlay_alpha),
  ]
  if args.init_mask is not None:
    cmd.extend(["--init_mask", args.init_mask])
  if args.sam_checkpoint is not None:
    cmd.extend(["--sam_checkpoint", args.sam_checkpoint])
  if args.depth_scale is not None:
    cmd.extend(["--depth_scale", str(args.depth_scale)])
  return cmd


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
  args = parse_args()

  data_dir = Path(args.data_dir).resolve()
  output_root = Path(args.output_root).resolve()
  output_root.mkdir(parents=True, exist_ok=True)

  if args.video_ids is None or len(args.video_ids) == 0:
    video_ids = discover_video_ids(data_dir)
  else:
    video_ids = [str(video_id) for video_id in args.video_ids]

  if not video_ids:
    raise RuntimeError(f"No video IDs found under {data_dir}")

  logging.info("Found %d video ids: %s", len(video_ids), video_ids)

  results: List[Dict[str, Optional[str]]] = []
  for video_id in video_ids:
    output_dir = build_output_dir(output_root=output_root, data_dir=data_dir, prompt=args.prompt, video_id=video_id, pattern=args.output_name_pattern)
    poses_path = output_dir / "poses.npz"
    if args.skip_existing and poses_path.exists():
      logging.info("Skipping video %s because %s already exists", video_id, poses_path)
      results.append({"video_id": video_id, "status": "skipped", "output_dir": str(output_dir), "error": None})
      continue

    cmd = build_single_command(args=args, video_id=video_id, output_dir=output_dir)
    logging.info("Running video %s -> %s", video_id, output_dir)
    logging.info("Command: %s", " ".join(cmd))

    try:
      subprocess.run(cmd, check=True)
      results.append({"video_id": video_id, "status": "success", "output_dir": str(output_dir), "error": None})
    except subprocess.CalledProcessError as exc:
      logging.error("Video %s failed with exit code %s", video_id, exc.returncode)
      results.append({"video_id": video_id, "status": "failed", "output_dir": str(output_dir), "error": f"exit_code={exc.returncode}"})
      if not args.continue_on_error:
        break

  summary = {
    "data_dir": str(data_dir),
    "mesh_file": str(Path(args.mesh_file).resolve()),
    "prompt": args.prompt,
    "results": results,
  }
  summary_path = output_root / f"{data_dir.name}_{args.prompt.replace(' ', '_')}_batch_summary.json"
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
  logging.info("Saved batch summary to %s", summary_path)

  failed = [item for item in results if item["status"] == "failed"]
  if failed:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
