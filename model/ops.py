#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
from keras import ops, KerasTensor, layers
import numpy as np

# from nn import graph


def pad(inputs, padding):
    """
    Pads a motion sequence on sequence length axis.

    Args:
        inputs: Tensor, a 4+D tensor with shape: :math:`(..., L, V, C)`.
        padding: int, length of the padding.

    Returns:
        A 4+D tensor with shape: :math:`(..., padding + L + padding, V, C)`.
    """
    return ops.concatenate(
        (
            ops.repeat(inputs[..., 0:1, :, :], padding, axis=-3),
            inputs,
            ops.repeat(inputs[..., -1:, :, :], padding, axis=-3)
        ),
        axis=-3
    )


def format_inputs(inputs : KerasTensor, window_size : int = 3, stride : int = 1):
    """Formats motion sequential data to realize a sliding window operation for temporal graph convolution.

    Parameters
    ----------
    inputs: KerasTensor,
        A 4+D tensor with shape: :math:`(..., L, V, C)` where `L` is the length of the sequence, `V` the number
        of vertices (joints) and `C` the coordinates for each joint (features dimension).
    window_size: int,
        Size of the window `W`.
    stride: int, optional
        Stride to apply while sliding the window, by default 1.

    Returns
    -------
        A 4+D tensor with shape: :math:`(..., L, W * V, C)`.
    """
    def sliding_window(x, window_size=3, stride=1, axis=1):
        outputs = ops.take(
            x,
            ops.extract_sequences(
                ops.arange(ops.shape(x)[axis]),
                window_size, stride),
            axis=axis)
        return ops.concatenate(
            [outputs[..., i, :, :] for i in range(window_size)], axis=-2
        )

    return sliding_window(inputs, window_size, stride)

def vectorize(inputs : KerasTensor, num_of_joints : int = 17, window : int = None, stride : int = 1):
    """_summary_

    Args:
        inputs (KerasTensor): Tensor of the of the motion sequence.
        num_of_joints (int, optional): Number of joint of the skeleton in motion. Defaults to 17.
        window (int, optional): Size of the sliding window. Defaults to None.
        stride (int, optional): The stride for the sliding window. Defaults to 1.

    Returns:
        KerasTensor: A vectorize tensor ot the motion with features seperated from joints.
    """
    if window:
        outputs = format_inputs(inputs, window, stride)
    else:
        outputs = ops.reshape(inputs, (..., num_of_joints, ops.shape(inputs)[-1] // num_of_joints))
    return outputs


def absolute_to_relative(poses, space="2d"):
    """Convert a sequence of human poses from absolute coordinates to relative root coordinates

    Parameters
    ----------
    poses : KerasTensor
        A 3D-tensor of dimension [L, J, 2] or [L, J, 3] representing the sequence of poses. 
    space : str, optional
        An indication of the dimension, by default "2d"

    Returns
    -------
    KerasTensor
        The sequence of poses with relative root coordinates.
    """
    if space == "2d":
        root = poses[:, :, 0:2]
        root = ops.tile(root, [1, 1, 17])
    else:
        root = poses[:, :, 0:3]
        root = ops.tile(root, [1, 1, 17])
    return poses - root

def distance(p, q):
    return ops.norm(p - q, axis=-1, keepdims=True)

def retrieve_skeleton(poses):

    assert ops.ndim(poses) == 3, "Expected positions to be of rank 3 but got rank {} instead".format(
        ops.shape(poses))
    hips = ((distance(poses[..., 3 * 0:3 * 0 + 3], poses[..., 3 * 1:3 * 1 + 3]) + distance(
        poses[..., 3 * 0:3 * 0 + 3], poses[..., 3 * 4:3 * 4 + 3])) / 2)
    femur = ((distance(poses[..., 3 * 1:3 * 1 + 3], poses[..., 3 * 2:3 * 2 + 3]) + distance(
        poses[..., 3 * 4:3 * 4 + 3], poses[..., 3 * 5:3 * 5 + 3])) / 2)
    tibia = ((distance(poses[..., 3 * 2:3 * 2 + 3], poses[..., 3 * 3:3 * 3 + 3]) + distance(
        poses[..., 3 * 5:3 * 5 + 3], poses[..., 3 * 6:3 * 6 + 3])) / 2)
    spine_back = (distance(poses[..., 3 * 0:3 * 0 + 3], poses[..., 3 * 7:3 * 7 + 3]))
    spine_top = (distance(poses[..., 3 * 7:3 * 7 + 3], poses[..., 3 * 8:3 * 8 + 3]))
    neck = (distance(poses[..., 3 * 8:3 * 8 + 3], poses[..., 3 * 9:3 * 9 + 3]))
    head = (distance(poses[..., 3 * 9:3 * 9 + 3], poses[..., 3 * 10:3 * 10 + 3]))
    clavicle = ((distance(poses[..., 3 * 8:3 * 8 + 3], poses[..., 3 * 11:3 * 11 + 3]) + distance(
        poses[..., 3 * 8:3 * 8 + 3], poses[..., 3 * 14:3 * 14 + 3])) / 2)
    humerus = ((distance(poses[..., 3 * 14:3 * 14 + 3], poses[..., 3 * 15:3 * 15 + 3]) + distance(
        poses[..., 3 * 11:3 * 11 + 3], poses[..., 3 * 12:3 * 12 + 3])) / 2)
    radius = ((distance(poses[..., 3 * 15:3 * 15 + 3], poses[..., 3 * 16:3 * 16 + 3]) + distance(
        poses[..., 3 * 12:3 * 12 + 3], poses[..., 3 * 13:3 * 13 + 3])) / 2)

    skeleton = ops.concatenate((hips, femur, tibia, spine_back, spine_top, neck, head, clavicle, humerus, radius),
                         axis=-1)
    skeleton = ops.max(skeleton, axis=-2, keepdims=True)
    return skeleton


# @tf.function
def retrieve_normalized_skeleton(poses):

    assert ops.ndim(poses) == 3, "Expected positions to be of rank 3 but got rank {} instead".format(
        ops.shape(poses))
    hips = ((distance(poses[..., 3 * 0:3 * 0 + 3], poses[..., 3 * 1:3 * 1 + 3]) + distance(
        poses[..., 3 * 0:3 * 0 + 3], poses[..., 3 * 4:3 * 4 + 3])) / 2)
    femur = ((distance(poses[..., 3 * 1:3 * 1 + 3], poses[..., 3 * 2:3 * 2 + 3]) + distance(
        poses[..., 3 * 4:3 * 4 + 3], poses[..., 3 * 5:3 * 5 + 3])) / 2)
    tibia = ((distance(poses[..., 3 * 2:3 * 2 + 3], poses[..., 3 * 3:3 * 3 + 3]) + distance(
        poses[..., 3 * 5:3 * 5 + 3], poses[..., 3 * 6:3 * 6 + 3])) / 2)
    spine_back = (distance(poses[..., 3 * 0:3 * 0 + 3], poses[..., 3 * 7:3 * 7 + 3]))
    spine_top = (distance(poses[..., 3 * 7:3 * 7 + 3], poses[..., 3 * 8:3 * 8 + 3]))
    neck = (distance(poses[..., 3 * 8:3 * 8 + 3], poses[..., 3 * 9:3 * 9 + 3]))
    head = (distance(poses[..., 3 * 9:3 * 9 + 3], poses[..., 3 * 10:3 * 10 + 3]))
    clavicle = ((distance(poses[..., 3 * 8:3 * 8 + 3], poses[..., 3 * 11:3 * 11 + 3]) + distance(
        poses[..., 3 * 8:3 * 8 + 3], poses[..., 3 * 14:3 * 14 + 3])) / 2)
    humerus = ((distance(poses[..., 3 * 14:3 * 14 + 3], poses[..., 3 * 15:3 * 15 + 3]) + distance(
        poses[..., 3 * 11:3 * 11 + 3], poses[..., 3 * 12:3 * 12 + 3])) / 2)
    radius = ((distance(poses[..., 3 * 15:3 * 15 + 3], poses[..., 3 * 16:3 * 16 + 3]) + distance(
        poses[..., 3 * 12:3 * 12 + 3], poses[..., 3 * 13:3 * 13 + 3])) / 2)

    skeleton = ops.concatenate((
        hips, femur, tibia,
        hips, femur, tibia,
        spine_back, spine_top,neck, head,
        clavicle, humerus, radius,
        clavicle, humerus, radius
    ),
        axis=-1)
    skeleton = ops.mean(skeleton, axis=1)
    return skeleton


# @tf.function
def retrieve_skeleton_normalized_vectors(poses: KerasTensor):
    assert ops.ndim(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        ops.shape(poses))
    hips_right = ops.mean(poses[..., 3 * 0:3 * 0 + 3] - poses[..., 3 * 1:3 * 1 + 3], axis=1, keepdims=True)
    femur_right = ops.mean(poses[..., 3 * 1:3 * 1 + 3] - poses[..., 3 * 2:3 * 2 + 3], axis=1, keepdims=True)
    tibia_right = ops.mean(poses[..., 3 * 2:3 * 2 + 3] - poses[..., 3 * 3:3 * 3 + 3], axis=1, keepdims=True)

    hips_left = ops.mean(poses[..., 3 * 0:3 * 0 + 3] - poses[..., 3 * 4:3 * 4 + 3], axis=1, keepdims=True)
    femur_left = ops.mean(poses[..., 3 * 4:3 * 4 + 3] - poses[..., 3 * 5:3 * 5 + 3], axis=1, keepdims=True)
    tibia_left = ops.mean(poses[..., 3 * 5:3 * 5 + 3] - poses[..., 3 * 6:3 * 6 + 3], axis=1, keepdims=True)

    spine_back = ops.mean(poses[..., 3 * 0:3 * 0 + 3] - poses[..., 3 * 7:3 * 7 + 3], axis=1, keepdims=True)
    spine_top = ops.mean(poses[..., 3 * 7:3 * 7 + 3] - poses[..., 3 * 8:3 * 8 + 3], axis=1, keepdims=True)
    neck = ops.mean(poses[..., 3 * 8:3 * 8 + 3] - poses[..., 3 * 9:3 * 9 + 3], axis=1, keepdims=True)
    head = ops.mean(poses[..., 3 * 9:3 * 9 + 3] - poses[..., 3 * 10:3 * 10 + 3], axis=1, keepdims=True)

    clavicle_left = ops.mean(poses[..., 3 * 8:3 * 8 + 3] - poses[..., 3 * 11:3 * 11 + 3], axis=1, keepdims=True)
    humerus_left = ops.mean(poses[..., 3 * 11:3 * 11 + 3] - poses[..., 3 * 12:3 * 12 + 3], axis=1,
                                  keepdims=True)
    radius_left = ops.mean(poses[..., 3 * 12:3 * 12 + 3] - poses[..., 3 * 13:3 * 13 + 3], axis=1, keepdims=True)

    clavicle_right = ops.mean(poses[..., 3 * 8:3 * 8 + 3] - poses[..., 3 * 14:3 * 14 + 3], axis=1,
                                    keepdims=True)
    humerus_right = ops.mean(poses[..., 3 * 14:3 * 14 + 3] - poses[..., 3 * 15:3 * 15 + 3], axis=1,
                                   keepdims=True)
    radius_right = ops.mean(poses[..., 3 * 15:3 * 15 + 3] - poses[..., 3 * 16:3 * 16 + 3], axis=1,
                                  keepdims=True)

    skeleton = ops.concatenate((hips_right,
                          femur_right,
                          tibia_right,
                          hips_left,
                          femur_left,
                          tibia_left,
                          spine_back,
                          spine_top,
                          neck,
                          head,
                          clavicle_left,
                          humerus_left,
                          radius_left,
                          clavicle_right,
                          humerus_right,
                          radius_right
                          ), axis=-2)
    return skeleton


def retrieve_skeleton_vectors(poses: KerasTensor, norm=False):
    assert ops.ndim(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        ops.shape(poses))
    batch, length, _ = ops.shape(poses)
    hips_right = poses[..., 3 * 0:3 * 0 + 3] - poses[..., 3 * 1:3 * 1 + 3]
    femur_right = poses[..., 3 * 1:3 * 1 + 3] - poses[..., 3 * 2:3 * 2 + 3]
    tibia_right = poses[..., 3 * 2:3 * 2 + 3] - poses[..., 3 * 3:3 * 3 + 3]

    hips_left = poses[..., 3 * 0:3 * 0 + 3] - poses[..., 3 * 4:3 * 4 + 3]
    femur_left = poses[..., 3 * 4:3 * 4 + 3] - poses[..., 3 * 5:3 * 5 + 3]
    tibia_left = poses[..., 3 * 5:3 * 5 + 3] - poses[..., 3 * 6:3 * 6 + 3]

    spine_back = poses[..., 3 * 0:3 * 0 + 3] - poses[..., 3 * 7:3 * 7 + 3]
    spine_top = poses[..., 3 * 7:3 * 7 + 3] - poses[..., 3 * 8:3 * 8 + 3]
    neck = poses[..., 3 * 8:3 * 8 + 3] - poses[..., 3 * 9:3 * 9 + 3]
    head = poses[..., 3 * 9:3 * 9 + 3] - poses[..., 3 * 10:3 * 10 + 3]

    clavicle_left = poses[..., 3 * 8:3 * 8 + 3] - poses[..., 3 * 11:3 * 11 + 3]
    humerus_left = poses[..., 3 * 11:3 * 11 + 3] - poses[..., 3 * 12:3 * 12 + 3]
    radius_left = poses[..., 3 * 12:3 * 12 + 3] - poses[..., 3 * 13:3 * 13 + 3]

    clavicle_right = poses[..., 3 * 8:3 * 8 + 3] - poses[..., 3 * 14:3 * 14 + 3]
    humerus_right = poses[..., 3 * 14:3 * 14 + 3] - poses[..., 3 * 15:3 * 15 + 3]
    radius_right = poses[..., 3 * 15:3 * 15 + 3] - poses[..., 3 * 16:3 * 16 + 3]

    skeleton = ops.concatenate((hips_right,
                          femur_right,
                          tibia_right,
                          hips_left,
                          femur_left,
                          tibia_left,
                          spine_back,
                          spine_top,
                          neck,
                          head,
                          clavicle_left,
                          humerus_left,
                          radius_left,
                          clavicle_right,
                          humerus_right,
                          radius_right
                          ), axis=-1)
    if norm:
        skeleton = ops.reshape(skeleton, [batch, length, 16, -1])
        skeleton = ops.norm(skeleton, axis=-1)
    else:
        skeleton = ops.reshape(skeleton, [batch, length, 16, -1])
    return skeleton


def get_skeleton_bones(poses: KerasTensor, dim=3):
    assert ops.ndim(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        ops.shape(poses))
    batch, length, _ = ops.shape(poses)
    hips_right = ops.norm(poses[..., dim * 0:dim * 0 + dim] - poses[..., dim * 1:dim * 1 + dim], axis=-1,
                         keepdims=True)
    femur_right = ops.norm(poses[..., dim * 1:dim * 1 + dim] - poses[..., dim * 2:dim * 2 + dim], axis=-1,
                          keepdims=True)
    tibia_right = ops.norm(poses[..., dim * 2:dim * 2 + dim] - poses[..., dim * 3:dim * 3 + dim], axis=-1,
                          keepdims=True)

    hips_left = ops.norm(poses[..., dim * 0:dim * 0 + dim] - poses[..., dim * 4:dim * 4 + dim], axis=-1, keepdims=True)
    femur_left = ops.norm(poses[..., dim * 4:dim * 4 + dim] - poses[..., dim * 5:dim * 5 + dim], axis=-1,
                         keepdims=True)
    tibia_left = ops.norm(poses[..., dim * 5:dim * 5 + dim] - poses[..., dim * 6:dim * 6 + dim], axis=-1,
                         keepdims=True)

    spine_back = ops.norm(poses[..., dim * 0:dim * 0 + dim] - poses[..., dim * 7:dim * 7 + dim], axis=-1,
                         keepdims=True)
    spine_top = ops.norm(poses[..., dim * 7:dim * 7 + dim] - poses[..., dim * 8:dim * 8 + dim], axis=-1, keepdims=True)
    neck = ops.norm(poses[..., dim * 8:dim * 8 + dim] - poses[..., dim * 9:dim * 9 + dim], axis=-1, keepdims=True)
    head = ops.norm(poses[..., dim * 9:dim * 9 + dim] - poses[..., dim * 10:dim * 10 + dim], axis=-1, keepdims=True)

    clavicle_left = ops.norm(poses[..., dim * 8:dim * 8 + dim] - poses[..., dim * 11:dim * 11 + dim], axis=-1,
                            keepdims=True)
    humerus_left = ops.norm(poses[..., dim * 11:dim * 11 + dim] - poses[..., dim * 12:dim * 12 + dim], axis=-1,
                           keepdims=True)
    radius_left = ops.norm(poses[..., dim * 12:dim * 12 + dim] - poses[..., dim * 13:dim * 13 + dim], axis=-1,
                          keepdims=True)

    clavicle_right = ops.norm(poses[..., dim * 8:dim * 8 + dim] - poses[..., dim * 14:dim * 14 + dim], axis=-1,
                             keepdims=True)
    humerus_right = ops.norm(poses[..., dim * 14:dim * 14 + dim] - poses[..., dim * 15:dim * 15 + dim], axis=-1,
                            keepdims=True)
    radius_right = ops.norm(poses[..., dim * 15:dim * 15 + dim] - poses[..., dim * 16:dim * 16 + dim], axis=-1,
                           keepdims=True)

    skeleton = ops.concatenate((hips_right,
                          femur_right,
                          tibia_right,
                          hips_left,
                          femur_left,
                          tibia_left,
                          spine_back,
                          spine_top,
                          neck,
                          head,
                          clavicle_left,
                          humerus_left,
                          radius_left,
                          clavicle_right,
                          humerus_right,
                          radius_right
                          ), axis=-1)
    return skeleton


# @tf.function
def bones_to_skeleton(bones: KerasTensor):
    assert ops.ndim(bones) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        ops.shape(bones))
    batch, length, _ = ops.shape(bones)
    skeleton = ops.concatenate((bones[..., 0:1],
                          bones[..., 1:2],
                          bones[..., 2:3],
                          bones[..., 0:1],
                          bones[..., 1:2],
                          bones[..., 2:3],
                          bones[..., 3:4],
                          bones[..., 4:5],
                          bones[..., 5:6],
                          bones[..., 6:7],
                          bones[..., 7:8],
                          bones[..., 8:9],
                          bones[..., 9:10],
                          bones[..., 7:8],
                          bones[..., 8:9],
                          bones[..., 9:10],
                          ), axis=-1)
    skeleton = ops.reshape(skeleton, [batch, length, -1])
    # skeleton = ops.mean(skeleton, axis=-2)
    return skeleton


# def retrieve_skeleton_vectors_v2(poses: KerasTensor):
#
#     assert ops.ndim(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
#         ops.shape(poses))
#     batch, length, _ = ops.shape(poses)
#     dist_matrix = graph.distance_matrix(length, 17)
#
#     skeleton = dist_matrix @ ops.reshape(poses, [batch, length * 17, -1])
#     return skeleton


SKEL_NAMES = ["hips", "femur", "tibia", "spine back", "spine top", "neck", "head", "clavicle", "humerus", "radius"]
SKEL_LR_NAMES = ["hips right",
                 "femur right",
                 "tibia right",
                 "hips left",
                 "femur left",
                 "tibia left",
                 "spine back",
                 "spine top",
                 "neck",
                 "head",
                 "clavicle left",
                 "humerus left",
                 "radius left"
                 "clavicle right",
                 "humerus right",
                 "radius right"
                 ]

def get_bones_length_variations(positions):
    def _get_bones(poses: KerasTensor, dim=3):
        length, _ = ops.shape(poses)
        hips_right = ops.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 1:dim * 1 + dim], axis=-1,
                            keepdims=True)
        femur_right = ops.norm(poses[:, dim * 1:dim * 1 + dim] - poses[:, dim * 2:dim * 2 + dim], axis=-1,
                            keepdims=True)
        tibia_right = ops.norm(poses[:, dim * 2:dim * 2 + dim] - poses[:, dim * 3:dim * 3 + dim], axis=-1,
                            keepdims=True)

        hips_left = ops.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 4:dim * 4 + dim], axis=-1, keepdims=True)
        femur_left = ops.norm(poses[:, dim * 4:dim * 4 + dim] - poses[:, dim * 5:dim * 5 + dim], axis=-1,
                            keepdims=True)
        tibia_left = ops.norm(poses[:, dim * 5:dim * 5 + dim] - poses[:, dim * 6:dim * 6 + dim], axis=-1,
                            keepdims=True)

        spine_back = ops.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 7:dim * 7 + dim], axis=-1,
                            keepdims=True)
        spine_top = ops.norm(poses[:, dim * 7:dim * 7 + dim] - poses[:, dim * 8:dim * 8 + dim], axis=-1, keepdims=True)
        neck = ops.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 9:dim * 9 + dim], axis=-1, keepdims=True)
        head = ops.norm(poses[:, dim * 9:dim * 9 + dim] - poses[:, dim * 10:dim * 10 + dim], axis=-1, keepdims=True)

        clavicle_left = ops.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 11:dim * 11 + dim], axis=-1,
                                keepdims=True)
        humerus_left = ops.norm(poses[:, dim * 11:dim * 11 + dim] - poses[:, dim * 12:dim * 12 + dim], axis=-1,
                            keepdims=True)
        radius_left = ops.norm(poses[:, dim * 12:dim * 12 + dim] - poses[:, dim * 13:dim * 13 + dim], axis=-1,
                            keepdims=True)

        clavicle_right = ops.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 14:dim * 14 + dim], axis=-1,
                                keepdims=True)
        humerus_right = ops.norm(poses[:, dim * 14:dim * 14 + dim] - poses[:, dim * 15:dim * 15 + dim], axis=-1,
                                keepdims=True)
        radius_right = ops.norm(poses[:, dim * 15:dim * 15 + dim] - poses[:, dim * 16:dim * 16 + dim], axis=-1,
                            keepdims=True)

        skeleton = ops.concatenate((hips_right,
                            femur_right,
                            tibia_right,
                            hips_left,
                            femur_left,
                            tibia_left,
                            spine_back,
                            spine_top,
                            neck,
                            head,
                            clavicle_left,
                            humerus_left,
                            radius_left,
                            clavicle_right,
                            humerus_right,
                            radius_right
                            ), axis=-1)
        return skeleton.numpy()
    bones = _get_bones(positions)
    return bones


# Other functions
def interpolate(input_tensor, output_size):
    # Calculate the interpolation factor
    factor = (output_size - 1) / (ops.shape(input_tensor)[0] - 1)

    # Generate the interpolated values
    interpolated = ops.linspace(0., (output_size - 1) / factor, output_size)

    # Expand the dimensions of the interpolated values
    interpolated = ops.expand_dims(interpolated, -1)

    # Cast the input tensor and interpolated values to the same dtype
    input_tensor = ops.cast(input_tensor, interpolated.dtype)
    interpolated = ops.cast(interpolated, input_tensor.dtype)

    # Calculate the indices where the interpolated values should be inserted
    indices = ops.cast(interpolated, "int32")

    # Use scatter_nd to insert the interpolated values into the input tensor
    output = ops.tensor_scatter_nd_update(input_tensor, ops.expand_dims(indices, -1), interpolated)

    return output

# Example 2
def adaptive_avg_pool1d(inputs, output_size):
    input_size = inputs.shape[1]
    window_size = input_size // output_size
    windows = [inputs[:, i*window_size : (i+1)*window_size] for i in range(output_size)]
    pool = layers.AveragePooling1D(window_size)
    return ops.concatenate([pool(w) for w in windows], axis=1)

# Example 2
# def adaptive_avg_pool1d(x, output_size):
#     shape = ops.shape(x)
#     in_length = shape[1]
#     if output_size == in_length:
#         return x
#     kernel_size = in_length // output_size
#     stride = kernel_size
#     kernel = kernel_size
#     x = tf.nn.avg_pool1d(x, [1, kernel, 1], [1, stride, 1], "VALID")
#     return x

def simplify_skeleton(poses, joints=17):
    b, t, _ = ops.shape(poses)
    skeleton = ops.reshape(poses, (b, t, joints, -1))
    lwrist = skeleton[..., 13, :]
    rwrist = skeleton[..., 16, :]
    lankle = skeleton[..., 6, :]
    rankle = skeleton[..., 3, :]
    head = skeleton[..., 10, :]
    skeleton = ops.concatenate([
        head, rankle, lankle, lwrist, rwrist
    ], axis=-1)
    return skeleton


def get_motion_dsc_per_part(descriptor, part):
    if part == "leftleg":
        result = descriptor[:, 1:4]
        result = np.mean(result, axis=-1)
    if part == "rightleg":
        result = descriptor[:, 4:7]
        result = np.mean(result, axis=-1)
    if part == "rightarm":
        result = descriptor[:, 11:14]
        result = np.mean(result, axis=-1)
    if part == "leftarm":
        result = descriptor[:, 14:17]
        result = np.mean(result, axis=-1)
    if part == "head":
        result = descriptor[:, 8:11]
        result = np.mean(result, axis=-1)
    if part == "shoulder":
        result = descriptor[:, [11,14]]
        result = np.mean(result, axis=-1)
    else:
        result = np.mean(descriptor, axis=-1)
    return result


def get_skeleton_dsc_per_part(descriptor, part):
    if part == "leftleg":
        result = descriptor[:, 1]
    if part == "rightleg":
        result = descriptor[:, 4]
    if part == "rightarm":
        result = descriptor[:, 14]
    if part == "leftarm":
        result = descriptor[:, 11]
    if part == "head":
        result = descriptor[:, 8:10]
    else:
        result = np.mean(descriptor, axis=-1)
    return result
