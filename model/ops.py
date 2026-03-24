#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
import tensorflow as tf
import numpy as np

# from nn import graph


def pad(inputs, padding):
    """
    Pads a motion sequence.

    Args:
        inputs: Tensor, a 4D tensor with shape: :math:`(B, L, V, C)`.
        padding: int, length of the padding.

    Returns:
        inputs: A 4D tensor with shape: :math:`(B, padding + L + padding, V, C)`.
    """
    b, steps, joints, features = tf.shape(inputs)
    vb = tf.reshape(inputs[:, 0, :, :], [b, 1, joints, features])
    vb = tf.tile(vb, (1, padding, 1, 1))
    ve = tf.reshape(inputs[:, -1, :, :], [b, 1, joints, features])
    ve = tf.tile(ve, (1, padding, 1, 1))
    inputs = tf.concat((vb, inputs, ve), axis=1)
    return inputs


def format_inputs(inputs, window_size, stride=1):
    """Formats motion sequential data to realize a sliding window operation
    for temporal graph convolution.
    Applies padding if necessary
    Parameters
    ----------
    inputs: Tensor,
        A 4D tensor with shape: :math:`(B, L, V, C)` where
        `B` is the batch size, `L` is the length of the sequence, `V` the number
        of vertices (joints) and `C` the coordinates for each joint
        (features dimension).
    window_size: int,
        Size of the window `W`.
    stride: int, optional
        Stride to apply while sliding the window, by default 1.

    Returns
    -------
        A 4D tensor with shape: :math:`(B, L, W * V, C)`.
    """

    def sliding_window(x, axis=1):
        n_in = tf.shape(x)[axis]
        n_out = (n_in - window_size) // stride + 1
        # Just in case n_in < window_size
        n_out = tf.math.maximum(n_out, 0)
        r = tf.expand_dims(tf.range(n_out), 1)
        idx = r * stride + tf.range(window_size)
        return tf.gather(x, idx, axis=axis)

    pads = window_size // 2
    outputs =  sliding_window(pad(inputs, pads))
    b, t, w, v, c = tf.shape(outputs)
    outputs = tf.reshape(outputs, [b, t, w * v, c])
    return outputs


def vectorize(inputs, num_of_joints=17, window=None, stride=1):
    """Vectorize motion sequence.

    Parameters
    ----------
    inputs : tf.Tensor
        _description_
    num_of_joints : int, optional
        _description_, by default 17
    window : int, optional
        _description_, by default None
    stride : int, optional
        _description_, by default 1

    Returns
    -------
    tf.Tensor
        _description_
    """
    if window:
        outputs = format_inputs(inputs, window, stride)
        # b, t, _, vc = tf.shape(outputs)
        # c = vc // num_of_joints
        # ts = t // window
        # outputs = tf.reshape(outputs, (b, t, window * num_of_joints, c))
    else:
        b, t, vc = tf.shape(inputs)
        c = vc // num_of_joints
        outputs = tf.reshape(inputs, (b, t, num_of_joints, c))
    return outputs


def absolute_to_relative(poses, space="2d"):
    """Convert a sequence of human poses from absolute coordinates to relative root coordinates

    Parameters
    ----------
    poses : tf.Tensor
        A 3D-tensor of dimension [L, J, 2] or [L, J, 3] representing the sequence of poses. 
    space : str, optional
        An indication of the dimension, by default "2d"

    Returns
    -------
    tf.Tensor
        The sequence of poses with relative root coordinates.
    """
    if space == "2d":
        root = poses[:, :, 0:2]
        root = tf.tile(root, [1, 1, 17])
    else:
        root = poses[:, :, 0:3]
        root = tf.tile(root, [1, 1, 17])
    return poses - root


# @tf.function
def leaky_relu(x):
    return tf.keras.activations.relu(x, alpha=.3)


# @tf.function
def swish(x):
    return tf.keras.activations.sigmoid(x) * x


# @tf.function
def retrieve_skeleton(poses):
    def distance(p, q):
        return tf.norm(p - q, axis=-1, keepdims=True)

    assert tf.rank(poses) == 3, "Expected positions to be of rank 3 but got rank {} instead".format(
        tf.shape(poses))
    hips = ((distance(poses[:, :, 3 * 0:3 * 0 + 3], poses[:, :, 3 * 1:3 * 1 + 3]) + distance(
        poses[:, :, 3 * 0:3 * 0 + 3], poses[:, :, 3 * 4:3 * 4 + 3])) / 2)
    femur = ((distance(poses[:, :, 3 * 1:3 * 1 + 3], poses[:, :, 3 * 2:3 * 2 + 3]) + distance(
        poses[:, :, 3 * 4:3 * 4 + 3], poses[:, :, 3 * 5:3 * 5 + 3])) / 2)
    tibia = ((distance(poses[:, :, 3 * 2:3 * 2 + 3], poses[:, :, 3 * 3:3 * 3 + 3]) + distance(
        poses[:, :, 3 * 5:3 * 5 + 3], poses[:, :, 3 * 6:3 * 6 + 3])) / 2)
    spine_back = (distance(poses[:, :, 3 * 0:3 * 0 + 3], poses[:, :, 3 * 7:3 * 7 + 3]))
    spine_top = (distance(poses[:, :, 3 * 7:3 * 7 + 3], poses[:, :, 3 * 8:3 * 8 + 3]))
    neck = (distance(poses[:, :, 3 * 8:3 * 8 + 3], poses[:, :, 3 * 9:3 * 9 + 3]))
    head = (distance(poses[:, :, 3 * 9:3 * 9 + 3], poses[:, :, 3 * 10:3 * 10 + 3]))
    clavicle = ((distance(poses[:, :, 3 * 8:3 * 8 + 3], poses[:, :, 3 * 11:3 * 11 + 3]) + distance(
        poses[:, :, 3 * 8:3 * 8 + 3], poses[:, :, 3 * 14:3 * 14 + 3])) / 2)
    humerus = ((distance(poses[:, :, 3 * 14:3 * 14 + 3], poses[:, :, 3 * 15:3 * 15 + 3]) + distance(
        poses[:, :, 3 * 11:3 * 11 + 3], poses[:, :, 3 * 12:3 * 12 + 3])) / 2)
    radius = ((distance(poses[:, :, 3 * 15:3 * 15 + 3], poses[:, :, 3 * 16:3 * 16 + 3]) + distance(
        poses[:, :, 3 * 12:3 * 12 + 3], poses[:, :, 3 * 13:3 * 13 + 3])) / 2)

    skeleton = tf.concat((hips, femur, tibia, spine_back, spine_top, neck, head, clavicle, humerus, radius),
                         axis=-1)
    skeleton = tf.reduce_max(skeleton, axis=1, keepdims=True)
    return skeleton


# @tf.function
def retrieve_normalized_skeleton(poses):
    def distance(p, q):
        return tf.norm(p - q, axis=-1)

    assert tf.rank(poses) == 3, "Expected positions to be of rank 3 but got rank {} instead".format(
        tf.shape(poses))
    hips = ((distance(poses[:, :, 3 * 0:3 * 0 + 3], poses[:, :, 3 * 1:3 * 1 + 3]) + distance(
        poses[:, :, 3 * 0:3 * 0 + 3], poses[:, :, 3 * 4:3 * 4 + 3])) / 2)
    femur = ((distance(poses[:, :, 3 * 1:3 * 1 + 3], poses[:, :, 3 * 2:3 * 2 + 3]) + distance(
        poses[:, :, 3 * 4:3 * 4 + 3], poses[:, :, 3 * 5:3 * 5 + 3])) / 2)
    tibia = ((distance(poses[:, :, 3 * 2:3 * 2 + 3], poses[:, :, 3 * 3:3 * 3 + 3]) + distance(
        poses[:, :, 3 * 5:3 * 5 + 3], poses[:, :, 3 * 6:3 * 6 + 3])) / 2)
    spine_back = (distance(poses[:, :, 3 * 0:3 * 0 + 3], poses[:, :, 3 * 7:3 * 7 + 3]))
    spine_top = (distance(poses[:, :, 3 * 7:3 * 7 + 3], poses[:, :, 3 * 8:3 * 8 + 3]))
    neck = (distance(poses[:, :, 3 * 8:3 * 8 + 3], poses[:, :, 3 * 9:3 * 9 + 3]))
    head = (distance(poses[:, :, 3 * 9:3 * 9 + 3], poses[:, :, 3 * 10:3 * 10 + 3]))
    clavicle = ((distance(poses[:, :, 3 * 8:3 * 8 + 3], poses[:, :, 3 * 11:3 * 11 + 3]) + distance(
        poses[:, :, 3 * 8:3 * 8 + 3], poses[:, :, 3 * 14:3 * 14 + 3])) / 2)
    humerus = ((distance(poses[:, :, 3 * 14:3 * 14 + 3], poses[:, :, 3 * 15:3 * 15 + 3]) + distance(
        poses[:, :, 3 * 11:3 * 11 + 3], poses[:, :, 3 * 12:3 * 12 + 3])) / 2)
    radius = ((distance(poses[:, :, 3 * 15:3 * 15 + 3], poses[:, :, 3 * 16:3 * 16 + 3]) + distance(
        poses[:, :, 3 * 12:3 * 12 + 3], poses[:, :, 3 * 13:3 * 13 + 3])) / 2)

    skeleton = tf.concat((
        hips,
        femur,
        tibia,
        hips,
        femur,
        tibia,
        spine_back,
        spine_top,
        neck,
        head,
        clavicle,
        humerus,
        radius,
        clavicle,
        humerus,
        radius
    ),
        axis=-1)
    skeleton = tf.reduce_mean(skeleton, axis=1)
    return skeleton


# @tf.function
def retrieve_skeleton_normalized_vectors(poses: tf.Tensor):
    assert tf.rank(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        tf.shape(poses))
    hips_right = tf.reduce_mean(poses[:, :, 3 * 0:3 * 0 + 3] - poses[:, :, 3 * 1:3 * 1 + 3], axis=1, keepdims=True)
    femur_right = tf.reduce_mean(poses[:, :, 3 * 1:3 * 1 + 3] - poses[:, :, 3 * 2:3 * 2 + 3], axis=1, keepdims=True)
    tibia_right = tf.reduce_mean(poses[:, :, 3 * 2:3 * 2 + 3] - poses[:, :, 3 * 3:3 * 3 + 3], axis=1, keepdims=True)

    hips_left = tf.reduce_mean(poses[:, :, 3 * 0:3 * 0 + 3] - poses[:, :, 3 * 4:3 * 4 + 3], axis=1, keepdims=True)
    femur_left = tf.reduce_mean(poses[:, :, 3 * 4:3 * 4 + 3] - poses[:, :, 3 * 5:3 * 5 + 3], axis=1, keepdims=True)
    tibia_left = tf.reduce_mean(poses[:, :, 3 * 5:3 * 5 + 3] - poses[:, :, 3 * 6:3 * 6 + 3], axis=1, keepdims=True)

    spine_back = tf.reduce_mean(poses[:, :, 3 * 0:3 * 0 + 3] - poses[:, :, 3 * 7:3 * 7 + 3], axis=1, keepdims=True)
    spine_top = tf.reduce_mean(poses[:, :, 3 * 7:3 * 7 + 3] - poses[:, :, 3 * 8:3 * 8 + 3], axis=1, keepdims=True)
    neck = tf.reduce_mean(poses[:, :, 3 * 8:3 * 8 + 3] - poses[:, :, 3 * 9:3 * 9 + 3], axis=1, keepdims=True)
    head = tf.reduce_mean(poses[:, :, 3 * 9:3 * 9 + 3] - poses[:, :, 3 * 10:3 * 10 + 3], axis=1, keepdims=True)

    clavicle_left = tf.reduce_mean(poses[:, :, 3 * 8:3 * 8 + 3] - poses[:, :, 3 * 11:3 * 11 + 3], axis=1, keepdims=True)
    humerus_left = tf.reduce_mean(poses[:, :, 3 * 11:3 * 11 + 3] - poses[:, :, 3 * 12:3 * 12 + 3], axis=1,
                                  keepdims=True)
    radius_left = tf.reduce_mean(poses[:, :, 3 * 12:3 * 12 + 3] - poses[:, :, 3 * 13:3 * 13 + 3], axis=1, keepdims=True)

    clavicle_right = tf.reduce_mean(poses[:, :, 3 * 8:3 * 8 + 3] - poses[:, :, 3 * 14:3 * 14 + 3], axis=1,
                                    keepdims=True)
    humerus_right = tf.reduce_mean(poses[:, :, 3 * 14:3 * 14 + 3] - poses[:, :, 3 * 15:3 * 15 + 3], axis=1,
                                   keepdims=True)
    radius_right = tf.reduce_mean(poses[:, :, 3 * 15:3 * 15 + 3] - poses[:, :, 3 * 16:3 * 16 + 3], axis=1,
                                  keepdims=True)

    skeleton = tf.concat((hips_right,
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


def retrieve_skeleton_vectors(poses: tf.Tensor, norm=False):
    assert tf.rank(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        tf.shape(poses))
    batch, length, _ = tf.shape(poses)
    hips_right = poses[:, :, 3 * 0:3 * 0 + 3] - poses[:, :, 3 * 1:3 * 1 + 3]
    femur_right = poses[:, :, 3 * 1:3 * 1 + 3] - poses[:, :, 3 * 2:3 * 2 + 3]
    tibia_right = poses[:, :, 3 * 2:3 * 2 + 3] - poses[:, :, 3 * 3:3 * 3 + 3]

    hips_left = poses[:, :, 3 * 0:3 * 0 + 3] - poses[:, :, 3 * 4:3 * 4 + 3]
    femur_left = poses[:, :, 3 * 4:3 * 4 + 3] - poses[:, :, 3 * 5:3 * 5 + 3]
    tibia_left = poses[:, :, 3 * 5:3 * 5 + 3] - poses[:, :, 3 * 6:3 * 6 + 3]

    spine_back = poses[:, :, 3 * 0:3 * 0 + 3] - poses[:, :, 3 * 7:3 * 7 + 3]
    spine_top = poses[:, :, 3 * 7:3 * 7 + 3] - poses[:, :, 3 * 8:3 * 8 + 3]
    neck = poses[:, :, 3 * 8:3 * 8 + 3] - poses[:, :, 3 * 9:3 * 9 + 3]
    head = poses[:, :, 3 * 9:3 * 9 + 3] - poses[:, :, 3 * 10:3 * 10 + 3]

    clavicle_left = poses[:, :, 3 * 8:3 * 8 + 3] - poses[:, :, 3 * 11:3 * 11 + 3]
    humerus_left = poses[:, :, 3 * 11:3 * 11 + 3] - poses[:, :, 3 * 12:3 * 12 + 3]
    radius_left = poses[:, :, 3 * 12:3 * 12 + 3] - poses[:, :, 3 * 13:3 * 13 + 3]

    clavicle_right = poses[:, :, 3 * 8:3 * 8 + 3] - poses[:, :, 3 * 14:3 * 14 + 3]
    humerus_right = poses[:, :, 3 * 14:3 * 14 + 3] - poses[:, :, 3 * 15:3 * 15 + 3]
    radius_right = poses[:, :, 3 * 15:3 * 15 + 3] - poses[:, :, 3 * 16:3 * 16 + 3]

    skeleton = tf.concat((hips_right,
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
        skeleton = tf.reshape(skeleton, [batch, length, 16, -1])
        skeleton = tf.norm(skeleton, axis=-1)
    else:
        skeleton = tf.reshape(skeleton, [batch, length, 16, -1])
    return skeleton


def get_skeleton_bones(poses: tf.Tensor, dim=3):
    assert tf.rank(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        tf.shape(poses))
    batch, length, _ = tf.shape(poses)
    hips_right = tf.norm(poses[:, :, dim * 0:dim * 0 + dim] - poses[:, :, dim * 1:dim * 1 + dim], axis=-1,
                         keepdims=True)
    femur_right = tf.norm(poses[:, :, dim * 1:dim * 1 + dim] - poses[:, :, dim * 2:dim * 2 + dim], axis=-1,
                          keepdims=True)
    tibia_right = tf.norm(poses[:, :, dim * 2:dim * 2 + dim] - poses[:, :, dim * 3:dim * 3 + dim], axis=-1,
                          keepdims=True)

    hips_left = tf.norm(poses[:, :, dim * 0:dim * 0 + dim] - poses[:, :, dim * 4:dim * 4 + dim], axis=-1, keepdims=True)
    femur_left = tf.norm(poses[:, :, dim * 4:dim * 4 + dim] - poses[:, :, dim * 5:dim * 5 + dim], axis=-1,
                         keepdims=True)
    tibia_left = tf.norm(poses[:, :, dim * 5:dim * 5 + dim] - poses[:, :, dim * 6:dim * 6 + dim], axis=-1,
                         keepdims=True)

    spine_back = tf.norm(poses[:, :, dim * 0:dim * 0 + dim] - poses[:, :, dim * 7:dim * 7 + dim], axis=-1,
                         keepdims=True)
    spine_top = tf.norm(poses[:, :, dim * 7:dim * 7 + dim] - poses[:, :, dim * 8:dim * 8 + dim], axis=-1, keepdims=True)
    neck = tf.norm(poses[:, :, dim * 8:dim * 8 + dim] - poses[:, :, dim * 9:dim * 9 + dim], axis=-1, keepdims=True)
    head = tf.norm(poses[:, :, dim * 9:dim * 9 + dim] - poses[:, :, dim * 10:dim * 10 + dim], axis=-1, keepdims=True)

    clavicle_left = tf.norm(poses[:, :, dim * 8:dim * 8 + dim] - poses[:, :, dim * 11:dim * 11 + dim], axis=-1,
                            keepdims=True)
    humerus_left = tf.norm(poses[:, :, dim * 11:dim * 11 + dim] - poses[:, :, dim * 12:dim * 12 + dim], axis=-1,
                           keepdims=True)
    radius_left = tf.norm(poses[:, :, dim * 12:dim * 12 + dim] - poses[:, :, dim * 13:dim * 13 + dim], axis=-1,
                          keepdims=True)

    clavicle_right = tf.norm(poses[:, :, dim * 8:dim * 8 + dim] - poses[:, :, dim * 14:dim * 14 + dim], axis=-1,
                             keepdims=True)
    humerus_right = tf.norm(poses[:, :, dim * 14:dim * 14 + dim] - poses[:, :, dim * 15:dim * 15 + dim], axis=-1,
                            keepdims=True)
    radius_right = tf.norm(poses[:, :, dim * 15:dim * 15 + dim] - poses[:, :, dim * 16:dim * 16 + dim], axis=-1,
                           keepdims=True)

    skeleton = tf.concat((hips_right,
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
def bones_to_skeleton(bones: tf.Tensor):
    assert tf.rank(bones) == 3, "Expected positions to be of rank 3 but got {} instead".format(
        tf.shape(bones))
    batch, length, _ = tf.shape(bones)
    skeleton = tf.concat((bones[:, :, 0:1],
                          bones[:, :, 1:2],
                          bones[:, :, 2:3],
                          bones[:, :, 0:1],
                          bones[:, :, 1:2],
                          bones[:, :, 2:3],
                          bones[:, :, 3:4],
                          bones[:, :, 4:5],
                          bones[:, :, 5:6],
                          bones[:, :, 6:7],
                          bones[:, :, 7:8],
                          bones[:, :, 8:9],
                          bones[:, :, 9:10],
                          bones[:, :, 7:8],
                          bones[:, :, 8:9],
                          bones[:, :, 9:10],
                          ), axis=-1)
    skeleton = tf.reshape(skeleton, [batch, length, -1])
    # skeleton = tf.reduce_mean(skeleton, axis=-2)
    return skeleton


# def retrieve_skeleton_vectors_v2(poses: tf.Tensor):
#
#     assert tf.rank(poses) == 3, "Expected positions to be of rank 3 but got {} instead".format(
#         tf.shape(poses))
#     batch, length, _ = tf.shape(poses)
#     dist_matrix = graph.distance_matrix(length, 17)
#
#     skeleton = dist_matrix @ tf.reshape(poses, [batch, length * 17, -1])
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
    def _get_bones(poses: tf.Tensor, dim=3):
        length, _ = tf.shape(poses)
        hips_right = tf.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 1:dim * 1 + dim], axis=-1,
                            keepdims=True)
        femur_right = tf.norm(poses[:, dim * 1:dim * 1 + dim] - poses[:, dim * 2:dim * 2 + dim], axis=-1,
                            keepdims=True)
        tibia_right = tf.norm(poses[:, dim * 2:dim * 2 + dim] - poses[:, dim * 3:dim * 3 + dim], axis=-1,
                            keepdims=True)

        hips_left = tf.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 4:dim * 4 + dim], axis=-1, keepdims=True)
        femur_left = tf.norm(poses[:, dim * 4:dim * 4 + dim] - poses[:, dim * 5:dim * 5 + dim], axis=-1,
                            keepdims=True)
        tibia_left = tf.norm(poses[:, dim * 5:dim * 5 + dim] - poses[:, dim * 6:dim * 6 + dim], axis=-1,
                            keepdims=True)

        spine_back = tf.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 7:dim * 7 + dim], axis=-1,
                            keepdims=True)
        spine_top = tf.norm(poses[:, dim * 7:dim * 7 + dim] - poses[:, dim * 8:dim * 8 + dim], axis=-1, keepdims=True)
        neck = tf.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 9:dim * 9 + dim], axis=-1, keepdims=True)
        head = tf.norm(poses[:, dim * 9:dim * 9 + dim] - poses[:, dim * 10:dim * 10 + dim], axis=-1, keepdims=True)

        clavicle_left = tf.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 11:dim * 11 + dim], axis=-1,
                                keepdims=True)
        humerus_left = tf.norm(poses[:, dim * 11:dim * 11 + dim] - poses[:, dim * 12:dim * 12 + dim], axis=-1,
                            keepdims=True)
        radius_left = tf.norm(poses[:, dim * 12:dim * 12 + dim] - poses[:, dim * 13:dim * 13 + dim], axis=-1,
                            keepdims=True)

        clavicle_right = tf.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 14:dim * 14 + dim], axis=-1,
                                keepdims=True)
        humerus_right = tf.norm(poses[:, dim * 14:dim * 14 + dim] - poses[:, dim * 15:dim * 15 + dim], axis=-1,
                                keepdims=True)
        radius_right = tf.norm(poses[:, dim * 15:dim * 15 + dim] - poses[:, dim * 16:dim * 16 + dim], axis=-1,
                            keepdims=True)

        skeleton = tf.concat((hips_right,
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
    factor = (output_size - 1) / (tf.shape(input_tensor)[0] - 1)

    # Generate the interpolated values
    interpolated = tf.linspace(0., (output_size - 1) / factor, output_size)

    # Expand the dimensions of the interpolated values
    interpolated = tf.expand_dims(interpolated, -1)

    # Cast the input tensor and interpolated values to the same dtype
    input_tensor = tf.cast(input_tensor, interpolated.dtype)
    interpolated = tf.cast(interpolated, input_tensor.dtype)

    # Calculate the indices where the interpolated values should be inserted
    indices = tf.cast(interpolated, tf.int32)

    # Use scatter_nd to insert the interpolated values into the input tensor
    output = tf.tensor_scatter_nd_update(input_tensor, tf.expand_dims(indices, -1), interpolated)

    return output

# Example 2
def adaptive_avg_pool1d(inputs, output_size):
    input_size = inputs.shape[1]
    window_size = input_size // output_size
    windows = [inputs[:, i*window_size : (i+1)*window_size] for i in range(output_size)]
    pool = tf.keras.layers.AveragePooling1D(window_size)
    return tf.concat([pool(w) for w in windows], axis=1)

# Example 2
# def adaptive_avg_pool1d(x, output_size):
#     shape = tf.shape(x)
#     in_length = shape[1]
#     if output_size == in_length:
#         return x
#     kernel_size = in_length // output_size
#     stride = kernel_size
#     kernel = kernel_size
#     x = tf.nn.avg_pool1d(x, [1, kernel, 1], [1, stride, 1], "VALID")
#     return x

def simplify_skeleton(poses, joints=17):
    b, t, _ = tf.shape(poses)
    skeleton = tf.reshape(poses, (b, t, joints, -1))
    lwrist = skeleton[:, :, 13, :]
    rwrist = skeleton[:, :, 16, :]
    lankle = skeleton[:, :, 6, :]
    rankle = skeleton[:, :, 3, :]
    head = skeleton[:, :, 10, :]
    skeleton = tf.concat([
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
