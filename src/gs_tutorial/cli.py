from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .colmap import ColmapPaths, colmap_database_summary, colmap_stage_status, run_reconstruction
from .config import load_config
from .dataset import load_colmap_text, scene_summary
from .environment import print_environment
from .preprocessing import extract_video_frames, reports_as_dicts, select_images


def _prepare(config_path: str, overwrite: bool) -> None:
    cfg = load_config(config_path)
    root = cfg.project_dir
    raw_frames = root / "raw_frames"
    image_dir = root / "images"
    source = Path(cfg.input.source)
    root.mkdir(parents=True, exist_ok=True)
    if cfg.input.kind == "video":
        extract_video_frames(
            source,
            raw_frames,
            fps=cfg.input.fps,
            max_width=cfg.input.max_width,
            overwrite=overwrite,
        )
        selection_source = raw_frames
    else:
        if not source.is_dir():
            raise NotADirectoryError(source)
        selection_source = source
    if selection_source.resolve() == image_dir.resolve():
        raise ValueError("The input image directory must differ from the generated image directory")
    if image_dir.exists() and any(image_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{image_dir} is not empty; pass --overwrite explicitly")
        shutil.rmtree(image_dir)
    reports = select_images(
        selection_source,
        image_dir,
        max_width=cfg.input.max_width,
        blur_threshold=cfg.input.blur_threshold,
        duplicate_hamming_threshold=cfg.input.duplicate_hamming_threshold,
    )
    report_path = root / "frame_report.json"
    report_path.write_text(json.dumps(reports_as_dicts(reports), indent=2), encoding="utf-8")
    kept = sum(item.kept for item in reports)
    print(f"Prepared {kept}/{len(reports)} images in {image_dir}")
    if kept < len(reports) / 2:
        print(
            "Warning: fewer than half of the frames were kept. "
            "Inspect frame_report.json and consider lowering input.blur_threshold."
        )
    print(f"Quality report: {report_path}")


def _reconstruct(config_path: str, backend: str) -> None:
    cfg = load_config(config_path)
    paths = run_reconstruction(
        cfg.project_dir / "images",
        cfg.project_dir / "colmap",
        backend=backend,
        matcher=cfg.colmap.matcher,
        camera_model=cfg.colmap.camera_model,
        single_camera=cfg.colmap.single_camera,
        use_gpu=cfg.colmap.use_gpu,
    )
    print(f"Undistorted images: {paths.processed_dir / 'images'}")
    print(f"COLMAP text model: {paths.text_model_dir}")


def _colmap_step(config_path: str, stage: str, backend: str) -> None:
    from .colmap import (
        export_processed_text,
        export_sparse_text,
        extract_features,
        map_sparse,
        match_features,
        select_sparse_model,
        sparse_model_summary,
        undistort_images,
    )

    cfg = load_config(config_path)
    path_factory = ColmapPaths.from_workspace if stage == "status" else ColmapPaths.create
    paths = path_factory(cfg.project_dir / "colmap", cfg.project_dir / "images")
    options = cfg.colmap
    if stage == "features":
        extract_features(
            paths,
            backend=backend,
            camera_model=options.camera_model,
            single_camera=options.single_camera,
            use_gpu=options.use_gpu,
        )
        print(json.dumps(colmap_database_summary(paths.database), indent=2))
    elif stage == "matching":
        match_features(
            paths,
            backend=backend,
            matcher=options.matcher,
            use_gpu=options.use_gpu,
        )
        print(json.dumps(colmap_database_summary(paths.database), indent=2))
    elif stage == "mapping":
        model_dir = map_sparse(paths, backend=backend)
        print(json.dumps(sparse_model_summary(model_dir), indent=2))
    elif stage == "sparse-text":
        print(export_sparse_text(paths, backend=backend))
    elif stage == "undistort":
        print(undistort_images(paths, backend=backend))
    elif stage == "export-text":
        print(export_processed_text(paths, backend=backend))
    elif stage == "status":
        if paths.database.is_file():
            print("Database:")
            print(json.dumps(colmap_database_summary(paths.database), indent=2))
        if colmap_stage_status(paths)["mapping"]:
            print("Sparse model:")
            print(json.dumps(sparse_model_summary(select_sparse_model(paths)), indent=2))
    print("Stages:")
    print(json.dumps(colmap_stage_status(paths), indent=2))


def _inspect(config_path: str) -> None:
    cfg = load_config(config_path)
    scene = load_colmap_text(cfg.project_dir / "colmap" / "processed" / "sparse_txt")
    print(json.dumps(scene_summary(scene), indent=2))


def _train(config_path: str, overwrite: bool) -> None:
    from .training import train

    cfg = load_config(config_path)
    processed = cfg.project_dir / "colmap" / "processed"
    scene = load_colmap_text(processed / "sparse_txt")

    def progress(event: dict[str, object]) -> None:
        if "loss" in event:
            print(
                f"step={event['step']:>6} loss={event['loss']:.5f} "
                f"ssim={event['ssim']:.4f} psnr={event['psnr']:.2f} "
                f"N={event['gaussians']:,}"
            )

    train(
        scene,
        processed / "images",
        cfg.project_dir / "training",
        cfg.training,
        progress,
        overwrite=overwrite,
    )


def _render_views(config_path: str, checkpoint: str | None, output: str | None, overwrite: bool):
    from .evaluation import render_registered_views

    cfg = load_config(config_path)
    processed = cfg.project_dir / "colmap" / "processed"
    scene = load_colmap_text(processed / "sparse_txt")
    checkpoint_path = (
        Path(checkpoint)
        if checkpoint
        else (
            cfg.project_dir / "training" / "checkpoints" / f"step_{cfg.training.max_steps:06d}.pt"
        )
    )
    output_dir = Path(output) if output else cfg.project_dir / "evaluation"

    def progress(event: dict[str, object]) -> None:
        print(
            f"[{event['index']:>3}/{event['total']}] {event['name']} "
            f"PSNR={event['psnr']:.2f} SSIM={event['ssim']:.4f}"
        )

    result = render_registered_views(
        scene,
        processed / "images",
        checkpoint_path,
        output_dir,
        cfg.training,
        overwrite=overwrite,
        callback=progress,
    )
    aggregate = result["aggregate"]
    print(
        f"Mean over {result['views']} views: PSNR={aggregate['psnr']:.2f}, "
        f"SSIM={aggregate['ssim']:.4f}, L1={aggregate['l1']:.5f}"
    )
    print(f"Gallery: {output_dir / 'index.html'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Educational COLMAP → gsplat pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("env", help="check the local environment")
    prepare = subparsers.add_parser("prepare", help="extract and select input frames")
    prepare.add_argument("--config", default="configs/video.yaml")
    prepare.add_argument("--overwrite", action="store_true")
    reconstruct = subparsers.add_parser("reconstruct", help="run COLMAP stages")
    reconstruct.add_argument("--config", default="configs/video.yaml")
    reconstruct.add_argument("--backend", choices=["auto", "cli", "pycolmap"], default="auto")
    colmap_step = subparsers.add_parser("colmap-step", help="run one inspectable COLMAP stage")
    colmap_step.add_argument(
        "stage",
        choices=[
            "features",
            "matching",
            "mapping",
            "sparse-text",
            "undistort",
            "export-text",
            "status",
        ],
    )
    colmap_step.add_argument("--config", default="configs/video.yaml")
    colmap_step.add_argument("--backend", choices=["auto", "cli", "pycolmap"], default="auto")
    inspect = subparsers.add_parser("inspect", help="summarize a COLMAP model")
    inspect.add_argument("--config", default="configs/video.yaml")
    training = subparsers.add_parser("train", help="optimize Gaussians with gsplat")
    training.add_argument("--config", default="configs/video.yaml")
    training.add_argument("--overwrite", action="store_true")
    render_views = subparsers.add_parser(
        "render-views", help="render original/render comparisons without training"
    )
    render_views.add_argument("--config", default="configs/video.yaml")
    render_views.add_argument("--checkpoint")
    render_views.add_argument("--output")
    render_views.add_argument("--overwrite", action="store_true")
    viewer = subparsers.add_parser("viewer", help="open a Gaussian PLY in Viser")
    viewer.add_argument("path")
    viewer.add_argument("--config", help="overlay COLMAP cameras from this scene config")
    viewer.add_argument("--port", type=int, default=8080)
    viewer.add_argument("--watch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "env":
        print_environment()
    elif args.command == "prepare":
        _prepare(args.config, args.overwrite)
    elif args.command == "reconstruct":
        _reconstruct(args.config, args.backend)
    elif args.command == "colmap-step":
        _colmap_step(args.config, args.stage, args.backend)
    elif args.command == "inspect":
        _inspect(args.config)
    elif args.command == "train":
        _train(args.config, args.overwrite)
    elif args.command == "render-views":
        _render_views(args.config, args.checkpoint, args.output, args.overwrite)
    elif args.command == "viewer":
        from .visualization import serve_ply

        viewer_scene = None
        viewer_images = None
        if args.config:
            cfg = load_config(args.config)
            viewer_processed = cfg.project_dir / "colmap" / "processed"
            viewer_scene = load_colmap_text(viewer_processed / "sparse_txt")
            viewer_images = viewer_processed / "images"
        serve_ply(
            args.path,
            port=args.port,
            watch=args.watch,
            scene=viewer_scene,
            image_dir=viewer_images,
        )


if __name__ == "__main__":
    main()
