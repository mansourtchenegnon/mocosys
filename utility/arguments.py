#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:16:14 2022

@author: mansour
"""

import yaml

def common_args(parser):
    parser.add_argument("-r", "--resume", type=str, help="path to checkpoint (default: None)")

    # dataset
    parser.add_argument(
        "-d", "--dataset", type=str, default="h36m", help='Dataset to use. "h36m" by default')
    parser.add_argument(
        "-k", "--keypoints", default="gt", type=str,
        help='2D input source for estimation. Can be "gt" or "cpn" or "detectron"...')
    parser.add_argument(
        "-estim", "--estimation", type=str,
        help='3D estimation source for correction. Can be "aanet" or "mnet" or "poseformer"...')
    parser.add_argument(
        "-o", "--output", default="./output", type=str,
        help="Path to the directory where to save all outputs",
    )
    parser.add_argument(
        "-cor",
        "--correction",
        action="store_true",
        default=False,
        help="Tells whether it is a motion correction task or not.",
    )

    # Run instructions (evaluation)
    parser.add_argument(
        "--metrics", action="store_true", default=False, help="Evaluate with metrics"
    )
    parser.add_argument(
        "--curves", action="store_true", default=False, help="Draw curves"
    )
    parser.add_argument(
        "--rendering", action="store_true", default=False, help="Generate random gifs"
    )
    parser.add_argument(
        "--versus_rendering", action="store_true", default=False, help="Generate random comparison gifs"
    )
    parser.add_argument(
        "--animation", action="store_true", default=False, help="Generate animation"
    )
    parser.add_argument(
        "--export", action="store_true", default=False, help="Export animation gifs"
    )

    return parser

def update_config(args):
    config = None
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
            # config = tools.dict_to_namespace(config)
    else:
        raise Exception("Missing configuration file !")
    if not config:
        raise Exception("Failed to load configuration file !")
    
    # Dataset configuration
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.keypoints:
        config["dataset"]["keypoints"] = args.keypoints
    if args.estimation:
        config["dataset"]["estimation"] = args.estimation

    # Architecture configurations
    if args.channels:
        config["model"]["arch"]["channels"] = args.channels
    if args.window:
        config["model"]["arch"]["window"] = config.arch.window
    if args.stages:
        config["model"]["arch"]["stages"] = args.stages
    if args.channels_in:
        config["model"]["arch"]["channels_in"] = args.channels_in
    if args.channels_out:
        config["model"]["arch"]["channels_out"] = args.channels_out

    # Running configurations
    if args.batch_size:
        config["running"]["batch_size"] = args.batch_size
    if args.frames:
        config["running"]["frames"] = args.frames
    if args.epochs:
        config["running"]["epochs"] = args.epochs
    if args.frames:
        config["running"]["data_cut"] = args.frames

    # # only in local for testing
    # config.dataset.location = "/home/mansour/Workspace/phd/codes/npz_data"

    return config
