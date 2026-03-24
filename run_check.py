#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: author
@version: version
""" 

import datetime
import argparse

from utility import arguments
from utility.datasets.loaders.h36m_dataloader import Human36mDatasetLoader, Human36mSotaDatasetLoader, get_bones, h36m_17_get_bones

# %% Arguments parsing and configuration
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training : parse arguments to configure training session."
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
    version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config.running.version = version
    return config, args

# %% Test
def test(config, args):
    # Load dataset
    dataset = Human36mDatasetLoader(training_set=False, batch_size=config.running.batch_size, keypoints="gt", chunk_size=243)
    iterator = iter(dataset.get_dataset())
    _, gt, _ = next(iterator)
    from view import display, animation
    # display.generate_3d_animation(gt[0] * 1000, "sample")
    # animation.animate_motion(gt[0] * 1000)
    animation.save_animate_motion(gt[0] * 1000, "sample.avi")
    # bones_1 = h36m_17_get_bones(gt.numpy()[0])
    # bones_2 = get_bones(gt.numpy()[0].reshape([-1, 17*3]))
    # assert (((bones_1 - bones_2) == 0).all())
    # print(bones_1)
    
# %% Main execution
if __name__ == "__main__":
    # parse args
    config, args = get_config()    
    # training
    test(config, args)