#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 2025.04
"""

import datetime
import argparse
import tensorflow as tf
import keras
import sys

import yaml
from utility import arguments, tools
from utility.datasets.loaders.h36m_dataloader import Human36mBoneDatasetLoader, Human36mSotaDatasetLoader
from model.models import MotionFineTuningModel, SkeletonModel
from training.trainers import MFTModelTrainer, SkeletonModelTrainer
import numpy as np

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
    parser.add_argument("--epochs", default=10, type=int, help="Number of training epochs")

    # Reproducibility measure
    parser.add_argument("--seed", default=97, type=int, help="Random seed for reproducibility.")

    args = parser.parse_args()
    return args

def get_config():
    args = parse_args()
    args.config = "./config/config.yaml"
    config = arguments.update_config(args)
    version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config["running"]["version"] = version
    return config, args

# %% Training methods
def train_motion_fine_tuning_model(config, args):
    # Load dataset
    if args.dataset == "h36m":
        trainset = Human36mSotaDatasetLoader(
            keypoints="cpn",
            batch_size=config["running"]["batch_size"],
            chunk_size=config["running"]["data_cut"],
            fused=False,
            location="data/human36m"
        )
    else:
        raise Exception(f"Dataset {args.dataset} not recognized!")
    # Create model and trainer
    model = MotionFineTuningModel(config)
    trainer = MFTModelTrainer(config, model, trainset)
    # Start training
    trainer.train()

def train_skeleton_model(config, args):
    # Load dataset
    if args.dataset == "h36m":
        trainset = Human36mBoneDatasetLoader(
            keypoints="gt",
            batch_size=config["running"]["batch_size"],
            chunk_size=config["running"]['data_cut'],
            fused=True)
    else:
        raise Exception(f"Dataset {args.dataset} not recognized !")
    # Create model and trainer
    model = SkeletonModel(config)
    trainer = SkeletonModelTrainer(config, model, trainset)
    # Start training
    trainer.train()

def evaluate_skeleton_model(config, args):
    # Load dataset
    if args.dataset == "h36m":
        testset = Human36mBoneDatasetLoader(
            training_set=False,
            keypoints="gt",
            batch_size=config["running"]["batch_size"],
            chunk_size=config["running"]['data_cut'],
            fused=True)
    else:
        raise Exception(f"Dataset {args.dataset} not recognized !")
    # Load model and evaluate
    # model = keras.saving.load_model(f"{args.resume}/best.keras")
    model = SkeletonModel(config)
    model(tf.random.uniform((1, 27, 34)))
    model.load_weights(f"{args.resume}/best.weights.h5")
    # Load model parameters
    parameters = None
    with np.load(f"{args.resume}/normalization.npz") as data:
        parameters = []
        parameters.append(data["inputs_mean"])
        parameters.append(data["inputs_std"])
        parameters.append(data["bones_mean"])
        parameters.append(data["bones_std"])
    if not parameters:
        print("Loading normalization testdataset")
        parameters = [i for i in testset.get_parameters()]
    bone_errors = []
    bone_error_list = {}
    print("parameters", len(parameters))
    for video_idx, datas in enumerate(testset.get_dataset()):
        inputs_2d, bones, video_name = datas
        video_name = str(video_name[0].numpy(), "utf-8")
        bones_prediction = model(inputs_2d)
        # bones = f.bones_to_skeleton(bones)
        # bone length
        bone_error = tf.reduce_mean(tf.abs(
            tools.denormalise(
                keras.ops.max(bones, axis=1, keepdims=True),
                parameters[2], parameters[3])
            - tools.denormalise(bones_prediction, parameters[2], parameters[3]))
        ) * 1000
        bone_errors.append(bone_error)

        action_name = video_name.split("_")[1].split(" ")[0]

        if action_name in bone_error_list.keys():
            bone_error_list[action_name].append(bone_error)
        else:
            bone_error_list[action_name] = [bone_error]
        # sys.stdout.write(f"\r{video_idx+1}/{len(eval_dataset)}")
        sys.stdout.write(f"\r{video_idx + 1}")
        sys.stdout.flush()

    print("MBLE")
    mean_bone_error_file = "%s/bone_errors.txt" % args.resume
    with open(mean_bone_error_file, "w") as f:
        f.writelines("=====Action===== ==mm==\n")
        total = []
        print("\n=====Action===== ==mm==\n")
        for key in bone_error_list.keys():
            mean_error = np.mean(np.array(bone_error_list[key]))
            total.append(mean_error)
            print(f"{key:>16} {mean_error:.2f}")
            f.writelines(f"{key:>16} {mean_error:.2f} \n")
        print(f"{'Average':>16} {np.mean(np.array(bone_errors)):.2f}")
        f.writelines(f"{'Average':>16} {np.mean(np.array(bone_errors)):.2f} \n")
        f.close()

# %% Main execution
if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # parse args
    config, args = get_config()
    # reproducibility measure
    if args.seed:
        keras.utils.set_random_seed(args.seed)
        config["seed"] = args.seed
    else:
        keras.utils.set_random_seed(97)
        config["seed"] = 97
    
    # training
    if args.arch == "mftmodel":
        train_motion_fine_tuning_model(config, args)
    elif args.arch == "skelmodel":
        train_skeleton_model(config, args)
    elif args.arch == "skelmodel.eval":
        evaluate_skeleton_model(config, args)
    else:
        raise Exception("Unknown model type !")