#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: author
@version: version
""" 

import datetime
import argparse
import tensorflow as tf
import keras.ops as kops
from model.graph import skeleton
from model.models import MotionFineTuningModel
from model.solvers import DeltaConverter, PosesConverter
from model import ops
from utility import arguments
from utility.datasets.loaders.h36m_dataloader import Human36mDatasetLoader, Human36mSotaDatasetLoader

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
    ## Test dataset
    # h36m default data
    # dataset = Human36mDatasetLoader(training_set=False, batch_size=config.running.batch_size, keypoints="gt", chunk_size=243)
    # iterator = iter(dataset.get_dataset())
    # _, gt, _ = next(iterator)
    # h36m sota dataset
    dataset = Human36mSotaDatasetLoader(
        training_set=False,
        batch_size=config.running.batch_size,
        keypoints="cpn", chunk_size=243, fused=False,
        location="data/human36m")
    iterator = iter(dataset.get_dataset())
    inp, est, gt, _ = next(iterator)
    # print("input", inp.shape)
    # print("estimation", est.shape)
    # print("gt", gt.shape)
    
    ## Test pose and delta converter
    skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
    delta_converter = DeltaConverter(3, skel, 3)
    pose_converter = PosesConverter(3, skel, 3, [0, 0, 0])
    delta = delta_converter(gt, format=True, keepdims=True)
    u = ops.format_inputs(gt[..., 0:1, :], 3)
    vec = ops.vectorize(gt, 17, 3)
    gamma = pose_converter.lgs.D @ vec
    pose_converter.set_constraints(u, gamma)
    pose = pose_converter(delta)
    diff = kops.max(pose - gt)
    print(diff)
    
    # from rendering import display, animation
    # display.generate_3d_animation(gt[0] * 1000, "sample")
    # animation.animate_motions_vs(gt[0] * 1000, est[0] * 1000)
    # animation.save_animate_motion(gt[0] * 1000, "sample.avi")

    ## Test model
    # model = MotionFineTuningModel(config)
    # sample_data = tf.random.uniform((64, 81, 51))
    # model(sample_data)
    # model.summary()
    
# %% Main execution
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # parse args
    config, args = get_config()    
    # training
    test(config, args)