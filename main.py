#!/usr/bin/env python
"""Command-line driver for the IMERG 10 km -> 1 km downscaling pipeline.

    python main.py synthetic                  # synthetic raw data (no downloads)
    python main.py download [--imerg --aorc --nlcd]
    python main.py preprocess                 # parse IMERG granules + align grids
    python main.py features                   # stage-1 training matrices
    python main.py train [--stage both|coarse|fine]
    python main.py predict [--method residual|bilinear|bilinear+lapse]
    python main.py validate
    python main.py all [--synthetic]          # everything, end to end
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import LOG, ensure_dirs, grids_from_config, load_config, setup_logging  # noqa: E402


def cmd_synthetic(cfg, grids, args):
    from src import synthetic

    ensure_dirs(cfg)
    info = synthetic.generate(cfg, grids, seed=args.seed)
    LOG.info("Synthetic summary: %s", info)


def cmd_download(cfg, grids, args):
    from src import data_pipeline as dp

    ensure_dirs(cfg)
    which = [w for w in ("imerg", "aorc", "nlcd") if getattr(args, w)]
    if not which:
        which = ["nlcd", "aorc", "imerg"]
    for w in which:
        if w == "nlcd":
            dp.download_nlcd(cfg, grids)
        elif w == "aorc":
            dp.download_aorc(cfg, grids)
        elif w == "imerg":
            dp.download_imerg(cfg, grids)


def cmd_preprocess(cfg, grids, args):
    from src import data_pipeline as dp

    ensure_dirs(cfg)
    dp.parse_imerg_dir(cfg, grids)
    outputs = dp.align_grids(cfg, grids, force=getattr(args, "force", False))
    for k, v in outputs.items():
        LOG.info("  %-16s %s", k, v)


def cmd_features(cfg, grids, args):
    from src.feature_engineering import prepare_training_data

    meta = prepare_training_data(cfg, grids)
    LOG.info("Features: %s", meta["feature_names"])


def cmd_train(cfg, grids, args):
    from src.train_model import train_coarse_stage, train_fine_stage

    stage = args.stage
    if stage in ("both", "coarse"):
        meta = train_coarse_stage(cfg, grids)
        LOG.info("Stage 1 top features: %s", meta["top10_features"])
    if stage in ("both", "fine") and cfg["upsampling"].get("method", "auto") in ("auto", "residual"):
        meta = train_fine_stage(cfg, grids)
        LOG.info("Stage 2 top features: %s", meta["top10_features"])


def cmd_predict(cfg, grids, args):
    from src.predict import downscale_timeseries

    downscale_timeseries(cfg, grids, method=args.method)


def cmd_validate(cfg, grids, args):
    from src.validation import validate

    validate(cfg, grids)


def cmd_all(cfg, grids, args):
    if args.synthetic:
        cmd_synthetic(cfg, grids, args)
    else:
        cmd_download(cfg, grids, args)
    cmd_preprocess(cfg, grids, args)
    cmd_features(cfg, grids, args)
    args.stage = "both"
    cmd_train(cfg, grids, args)
    args.method = None
    cmd_predict(cfg, grids, args)
    cmd_validate(cfg, grids, args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(Path(__file__).resolve().parent / "config.yaml"))
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synthetic", help="generate synthetic raw data")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_synthetic)

    d = sub.add_parser("download", help="download raw data (all sources unless flags given)")
    d.add_argument("--imerg", action="store_true")
    d.add_argument("--aorc", action="store_true")
    d.add_argument("--nlcd", action="store_true")
    d.set_defaults(func=cmd_download)

    pp = sub.add_parser("preprocess", help="parse granules and align grids")
    pp.add_argument("--force", action="store_true", help="rebuild aligned products even if they exist")
    pp.set_defaults(func=cmd_preprocess)
    sub.add_parser("features", help="build training/validation matrices").set_defaults(func=cmd_features)

    t = sub.add_parser("train", help="train stage-1 (10 km) and stage-2 (1 km residual) models")
    t.add_argument("--stage", choices=["both", "coarse", "fine"], default="both")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="downscale the full timeline to 1 km")
    pr.add_argument("--method", choices=["auto", "residual", "bilinear", "bilinear+lapse"], default=None)
    pr.set_defaults(func=cmd_predict)

    sub.add_parser("validate", help="evaluate against AORC 1 km").set_defaults(func=cmd_validate)

    a = sub.add_parser("all", help="run every phase end to end")
    a.add_argument("--synthetic", action="store_true", help="use synthetic data instead of downloading")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--imerg", action="store_true")
    a.add_argument("--aorc", action="store_true")
    a.add_argument("--nlcd", action="store_true")
    a.set_defaults(func=cmd_all)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    cfg = load_config(args.config)
    grids = grids_from_config(cfg)
    LOG.info("Domain %s: coarse %s, fine %s", cfg["domain"]["name"], grids.coarse.shape, grids.fine.shape)
    args.func(cfg, grids, args)


if __name__ == "__main__":
    main()
