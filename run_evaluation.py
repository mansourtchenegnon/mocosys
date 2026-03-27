#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 14.04.2025
"""

import datetime
import argparse
import tensorflow as tf
from utility import arguments
from utility.datasets.loaders.h36m_dataloader import Human36mSotaDatasetLoader
from model.models import MotionFineTuningModel
from training.trainers import MFTModelTrainer

# %% Arguments parsing and configuration
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation : parse arguments to configure evaluation session."
    )

    # common arguments parsing
    parser = arguments.common_args(parser)

    # configuration and neural networks model
    parser.add_argument(
        "-c", "--config", default="./config/config.yaml", type=str,
        help="Path to the default configuration file")
    parser.add_argument(
        "-a", "--arch", type=str, default="mftmodel",# required=True,
        help='Model architecture to use. For example "mftmodel" or "skelmodel"')
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="enables CUDA training")

    # Architecture parameters
    parser.add_argument("--channels_in", type=int, help="Number of input channels (2 for 2d, 3 for 3d).")
    parser.add_argument("--channels_out", type=int, help="Number of output channels (2 for 2d, 3 for 3d).")
    parser.add_argument("--channels", type=int, help="Number of channels for intermediate layers.")
    parser.add_argument("--window", type=int, help="Window size for graph convolution.")
    parser.add_argument("--stages", type=int, help="Number of times to loop in the intermediate layers.")

    # Running parameters
    parser.add_argument("--batch_size", default=64, type=int, help="The batch size")
    parser.add_argument("--frames", type=int, help="Size of data cut in number of frames")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")

    # Reproducibility measure
    parser.add_argument("--seed", default=97, type=int, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args

def get_config():
    args = parse_args()
    args.config = "./config/config.yaml"
    config = arguments.update_config(args)
    # version = datetime.datetime.now().strftime("%Y%m%d")
    version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config["running"]["version"] = version
    return config, args

# %% Evaluation methods
def evaluate(config, args):
    # Load dataset
    if args.dataset == "h36m":
        dataset = Human36mSotaDatasetLoader(training_set=False, batch_size=config.running.batch_size, keypoints="cpn", chunk_size=243)
    else:
        raise Exception(f"Dataset {args.dataset} not recognized !")

# %% Main execution
if __name__ == "__main__":
    # parse args
    config, args = get_config()
    # reproducibility measure
    if args.seed:
        tf.random.set_seed(args.seed)
    else:
        tf.random.set_seed(97)
    
    # evaluation