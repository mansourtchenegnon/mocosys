#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 2026.04
"""

import os
import datetime
import argparse
import sys
import numpy
import tensorflow as tf
from model.modules import MoCoSys
from utility import arguments
from utility.datasets.loaders.h36m_dataloader import Human36mSotaDatasetLoader
from model.models import MotionFineTuningModel
from model import metrics
from training.trainers import MFTModelTrainer

# %% Arguments parsing and configuration
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation : parse arguments to configure evaluation session."
    )

    # common arguments parsing
    parser = arguments.common_args(parser)

    # model loading paths
    parser.add_argument("--skel_model_path", required=True, type=str, help="Path to the skeleton model checkpoints")
    parser.add_argument("--mft_model_path", required=True, type=str, help="Path to the motion fine tuning model checkpoints")

    # Running parameters
    parser.add_argument("--frames", default=0, type=int, help="Size of data cut in number of frames")
    parser.add_argument("--epochs", default=5, type=int, help="Number of training epochs")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="enables CUDA training")

    # Reproducibility measure
    parser.add_argument("--seed", default=97, type=int, help="Random seed for reproducibility.")

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

def load_dataset(config, args):
    if args.dataset == "h36m":
        dataset = Human36mSotaDatasetLoader(
            training_set=False,
            estimation=args.estimator,
            keypoints=args.keypoints,
            chunk_size=args.frames,
            fused=False
        )
    else:
        raise Exception(f"Dataset {args.dataset} not recognized !")
    return dataset

# %% Evaluation methods
def evaluate_with_metrics(model, dataset : Human36mSotaDatasetLoader, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    error_list = {}
    velocity_error_list = {}
    errors = []
    velocity_errors = []
    acceleration_error_list = {}
    acceleration_errors = []
    bone_errors = []
    bone_variances = []
    bone_error_list = {}
    bone_variance_list = {}

    for video_idx, datas in enumerate(dataset.get_dataset()):
        poses_2d, poses_3d_estimation, poses_3d_gt, video_name = datas
        video_name = str(video_name[0].numpy(), "utf-8")
        poses_3d_prediction = model([poses_3d_estimation, poses_2d])
        # poses_3d_prediction = network(poses_3d_estimation)

        # poses_3d_prediction = poses_3d_prediction - tf.tile(
        #     poses_3d_prediction[:, :, :3], [1, 1, 17]
        # )
        # MPJPE
        error = (
            metrics.mean_position_error(poses_3d_gt, poses_3d_prediction)
            * 1000
        )
        errors.append(error.numpy())
        # velocity
        velocity_error = (
            metrics.mean_velocity_error(poses_3d_gt, poses_3d_prediction)
            * 1000
        )
        velocity_errors.append(velocity_error.numpy())

        # acceleration
        acceleration_error = (
            metrics.mean_acceleration_error(poses_3d_gt, poses_3d_prediction)
            * 1000
        )
        acceleration_errors.append(acceleration_error.numpy())

        # bone length
        bone_error = (
            metrics.mean_bone_length_error(poses_3d_gt, poses_3d_prediction)
            * 1000
        )
        bone_errors.append(bone_error.numpy())

        # bone length
        bone_variance = (
            metrics.bone_length_variance(poses_3d_prediction[0] * 1000)
        )
        bone_variances.append(bone_variance.numpy())

        action_name = video_name.split("_")[1].split(" ")[0]

        if action_name in error_list.keys():
            error_list[action_name].append(error)
            velocity_error_list[action_name].append(velocity_error)
            acceleration_error_list[action_name].append(acceleration_error)
            bone_error_list[action_name].append(bone_error)
            bone_variance_list[action_name].append(bone_variance)
        else:
            error_list[action_name] = [error]
            velocity_error_list[action_name] = [velocity_error]
            acceleration_error_list[action_name] = [acceleration_error]
            bone_error_list[action_name] = [bone_error]
            bone_variance_list[action_name] = [bone_variance]
        sys.stdout.write(f"\r{video_idx+1}/{len(dataset)}")
        sys.stdout.flush()

    print("MPJPE")
    metrics_output_folder = "%s/metrics" % (output_folder)
    if not os.path.exists(metrics_output_folder):
        os.makedirs(metrics_output_folder)
    error_file = "%s/mpjpe.txt" % metrics_output_folder
    error_file_md = "%s/errors.md" % metrics_output_folder
    error_str = " Action |" + " | ".join(error_list.keys()) + " | Average |\n"
    error_str += " :---: | " * (len(error_list.keys()) + 2) + "\n MPJPE |"
    with open(error_file, "w") as f:
        f.writelines("=====Action===== ==mm==\n")
        total = []
        print("\n=====Action===== ==mm==\n")
        for key in error_list.keys():
            mean_error = numpy.mean(numpy.array(error_list[key]))
            total.append(mean_error)
            print(f"{key:>16} {mean_error:.2f}")
            f.writelines(f"{key:>16} {mean_error:.2f} \n")
            error_str += " %.2f |" % mean_error
        print(f"{'Average':>16} {numpy.mean(numpy.array(errors)):.2f}")
        f.writelines(f"{'Average':>16} {numpy.mean(numpy.array(errors)):.2f} \n")
        error_str += " %.2f |\n" % numpy.mean(numpy.array(errors))
        f.close()

    print("MPJVE")
    velocity_error_file = "%s/mpjve.txt" % metrics_output_folder
    error_str += " MPJVE |"
    with open(velocity_error_file, "w") as f:
        f.writelines("=====Action===== ==mm==\n")
        total = []
        print("\n=====Action===== ==mm==\n")
        for key in error_list.keys():
            mean_error = numpy.mean(numpy.array(velocity_error_list[key]))
            total.append(mean_error)
            print(f"{key:>16} {mean_error:.2f}")
            f.writelines(f"{key:>16} {mean_error:.2f} \n")
            # velocity_error_str += " %.2f |" % mean_error
            error_str += " %.2f |" % mean_error
        print(f"{'Average':>16} {numpy.mean(numpy.array(velocity_errors)):.2f}")
        f.writelines(f"{'Average':>16} {numpy.mean(numpy.array(velocity_errors)):.2f} \n")
        error_str += " %.2f |\n" % numpy.mean(numpy.array(velocity_errors))
        f.close()

    print("MPJAccE")
    mean_acceleration_error_file = "%s/acceleration_errors.txt" % metrics_output_folder
    error_str += " MPJAccE |"
    with open(mean_acceleration_error_file, "w") as f:
        f.writelines("=====Action===== ==mm==\n")
        total = []
        print("\n=====Action===== ==mm==\n")
        for key in error_list.keys():
            mean_error = numpy.mean(numpy.array(acceleration_error_list[key]))
            total.append(mean_error)
            print(f"{key:>16} {mean_error:.2f}")
            f.writelines(f"{key:>16} {mean_error:.2f} \n")
            error_str += " %.2f |" % mean_error
        print(f"{'Average':>16} {numpy.mean(numpy.array(acceleration_errors)):.2f}")
        f.writelines(f"{'Average':>16} {numpy.mean(numpy.array(acceleration_errors)):.2f} \n")
        error_str += " %.2f |\n" % numpy.mean(numpy.array(acceleration_errors))
        f.close()

    print("MBLE")
    mean_bone_error_file = "%s/bone_errors.txt" % metrics_output_folder
    error_str += " MBLE |"
    with open(mean_bone_error_file, "w") as f:
        f.writelines("=====Action===== ==mm==\n")
        total = []
        print("\n=====Action===== ==mm==\n")
        for key in error_list.keys():
            mean_error = numpy.mean(numpy.array(bone_error_list[key]))
            total.append(mean_error)
            print(f"{key:>16} {mean_error:.2f}")
            f.writelines(f"{key:>16} {mean_error:.2f} \n")
            error_str += " %.2f |" % mean_error
        print(f"{'Average':>16} {numpy.mean(numpy.array(bone_errors)):.2f}")
        f.writelines(f"{'Average':>16} {numpy.mean(numpy.array(bone_errors)):.2f} \n")
        error_str += " %.2f |\n" % numpy.mean(numpy.array(bone_errors))
        f.close()

    print("Bone Variance")
    mean_bone_variance_file = "%s/bone_variances.txt" % metrics_output_folder
    error_str += " Bone Var |"
    with open(mean_bone_variance_file, "w") as f:
        f.writelines("=====Action===== ==mm==\n")
        total = []
        print("\n=====Action===== ==mm==\n")
        for key in error_list.keys():
            mean_error = numpy.mean(numpy.array(bone_variance_list[key]))
            total.append(mean_error)
            print(f"{key:>16} {mean_error:.2f}")
            f.writelines(f"{key:>16} {mean_error:.2f} \n")
            error_str += " %.2f |" % mean_error
        print(f"{'Average':>16} {numpy.mean(numpy.array(bone_variances)):.2f}")
        f.writelines(f"{'Average':>16} {numpy.mean(numpy.array(bone_variances)):.2f} \n")
        error_str += " %.2f |\n" % numpy.mean(numpy.array(bone_variances))
        f.close()

    with open(error_file_md, "w") as md:
        md.writelines(error_str)
    
def rendering_animation(model, dataset : Human36mSotaDatasetLoader, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    from rendering import animation

    sampling_export = numpy.random.choice(235, size=1, replace=False)

    for video_idx, datas in enumerate(dataset.get_dataset()):
        if video_idx in sampling_export:
            poses_2d_gt, poses_3d_estimation, poses_3d_gt, video_name = datas
            video_name = str(video_name[0].numpy(), "utf-8")
            poses_3d_prediction = model([poses_3d_estimation, poses_2d_gt])

            # poses_3d_prediction = poses_3d_prediction - tf.tile(
            #     poses_3d_prediction[:, :, :3], [1, 1, 17]
            # )

            animation.animate_motions_vs(
                poses_3d_prediction[0].numpy() * 1000,
                poses_3d_estimation[0].numpy() * 1000
            )


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
    # Load dataset
    dataset = load_dataset(config, args)
    # Load model
    module = MoCoSys(args.skel_model_path, args.mft_model_path)  
    module.set_normalization_parameters(dataset.get_parameters())  
    
    if args.metrics:
        evaluate_with_metrics(module, dataset, "evaluation")
    if args.animation:
        rendering_animation(module, dataset, "evaluation")