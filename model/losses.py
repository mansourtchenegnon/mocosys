#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
# %% Imports
import tensorflow as tf

# 16 joints
SKELETON_PAIRS_16 = [[0, 1], [0, 4], [0, 7],  # hips and spine
                     [1, 2], [2, 3],  # left leg
                     [4, 5], [5, 6],  # right leg
                     [7, 10], [7, 13],  # shoulders
                     [7, 8], [8, 9],  # head
                     [10, 11], [11, 12],  # left arm
                     [13, 14], [14, 15]  # right arm
                     ]

# 17 joints
SKELETON_PAIRS_17 = [[0, 1], [0, 4], [0, 7], [7, 8],  # hips and spine
                     [1, 2], [2, 3],  # left leg
                     [4, 5], [5, 6],  # right leg
                     [8, 11], [8, 14],  # shoulders
                     [8, 9], [9, 10],  # head
                     [11, 12], [12, 13],  # left arm
                     [14, 15], [15, 16]  # right arm
                     ]


# %% Distance Loss

def mean_distance_loss(target, output):
    """
    Computes the sum of distance loss values over joints position and then the mean over sequence.
    Case of batched frames features.

    Parameters
    ----------
    target : tf.Tensor
        The ground truth values.
    output : tf.Tensor
        The predicted values.

    Returns
    -------
    loss : tf.Tensor
        Computed mean per joint distance lost value. It returns a 1D-Tensor.

    """
    target_reshaped = tf.reshape(target, [-1, 3])
    output_reshaped = tf.reshape(output, [-1, 3])
    loss = tf.reduce_mean(
        tf.norm(
            target_reshaped - output_reshaped, axis=-1
        )
    )
    return loss


def norm_loss(target, output):
    loss = tf.reduce_mean(
        tf.abs(tf.norm(output, axis=-1) - tf.norm(target, axis=-1))
    )
    return loss


def ed_loss(target, output):
    loss = tf.reduce_mean(
        tf.norm(target - output, axis=-1)
    )
    # loss = tf.reduce_mean(
    #     tf.norm(target - output, axis=-1)
    # )
    return loss


class DistanceLoss(tf.keras.losses.Loss):
    def __init__(self, name='distance_loss', reduction=tf.keras.losses.Reduction.NONE):
        """
        DistanceLoss class defines a loss computed as the average Euclidean distance
        loss. It can be used both for JointPositionLoss (on joint positions) or
        LaplacianLoss (on joint differential coordinates) depending on
        the nature of the target and the prediction.

        Returns
        -------
        None.

        """
        super().__init__(name=name, reduction=reduction)

    def call(self, target, output):
        """
        Computes the sum of distance loss values over joints position and then
        the mean over sequence. Case of batched frames features.

        Parameters
        ----------
        target : tf.Tensor
            The ground truth values.
        output : tf.Tensor
            The predicted values.

        Returns
        -------
        loss : tf.Tensor
            Computed mean per joint distance lost value. It returns a 1D-Tensor.

        """
        loss = tf.reduce_mean(
            tf.norm(
                target - output, axis=len(target.shape) - 1
            ),
            axis=-1
        )
        return loss


# %% Velocity Loss
def mean_velocity_loss(output, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative).

    Parameters
    ----------
    output : tf.Tensor
        The predicted values.
    target : tf.Tensor
        The ground truth values.

    Returns
    -------
    error : float32
        Computed mean velocity error.
    """
    shape = tf.shape(target)
    target_reshaped = tf.reshape(target, [shape[0], shape[1], 17, 3])
    output_reshaped = tf.reshape(output, [shape[0], shape[1], 17, 3])
    velocity_output = output_reshaped[:, 1:, :] - output_reshaped[:, :-1, :]
    velocity_target = target_reshaped[:, 1:, :] - target_reshaped[:, :-1, :]
    loss = tf.reduce_mean(
        tf.norm(velocity_output - velocity_target, axis=-1)
    )

    return loss


# %% Bone Length Loss
def get_bone_length(positions):
    def distance(p, q):
        return tf.norm(p - q, axis=-1)

    assert tf.rank(positions) == 4, "Expected positions to be of rank 4 but got rank {} instead".format(
        tf.shape(positions))
    hips = ((distance(positions[:, :, 0, :], positions[:, :, 1, :]) + distance(
        positions[:, :, 0, :], positions[:, :, 4, :])) / 2)
    femur = ((distance(positions[:, :, 1, :], positions[:, :, 2, :]) + distance(
        positions[:, :, 4, :], positions[:, :, 5, :])) / 2)
    tibia = ((distance(positions[:, :, 2, :], positions[:, :, 3, :]) + distance(
        positions[:, :, 5, :], positions[:, :, 6, :])) / 2)
    spine_back = (distance(positions[:, :, 0, :], positions[:, :, 7, :]))
    spine_up = (distance(positions[:, :, 7, :], positions[:, :, 8, :]))
    neck = (distance(positions[:, :, 8, :], positions[:, :, 9, :]))
    head = (distance(positions[:, :, 9, :], positions[:, :, 10, :]))
    clavicle = ((distance(positions[:, :, 8, :], positions[:, :, 11, :]) + distance(
        positions[:, :, 8, :], positions[:, :, 14, :])) / 2)
    humerus = ((distance(positions[:, :, 14, :], positions[:, :, 15, :]) + distance(
        positions[:, :, 11, :], positions[:, :, 12, :])) / 2)
    radius = ((distance(positions[:, :, 15, :], positions[:, :, 16, :]) + distance(
        positions[:, :, 12, :], positions[:, :, 13, :])) / 2)
    return tf.concat((hips, femur, tibia, spine_back, spine_up, neck, head, clavicle, humerus, radius),
                     axis=-1)

def mean_bone_length_loss(output, target):
    loss = tf.reduce_mean(
        tf.abs(get_bone_length(target) - get_bone_length(output))
    )
    return loss


# %% Motion Loss
class MotionLoss(tf.keras.losses.Loss):

    def __init__(self):
        """
        MotionLoss class represents objects for computing the motion loss proposed in
        -Motion Guided 3D Poses Estimation from Videos- by Wang et al., 2020-.

        Returns
        -------
        None.

        """
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, target, output):
        """
        Computes motion loss

        Parameters
        ----------
        target : tf.Tensor
            The ground truth values.
        output : tf.Tensor
            The predicted values.

        Returns
        -------
        mean : float32
            Computed mean distance lost value.
        """
        target = tf.reshape(target, [-1, 17, 3])
        output = tf.reshape(output, [-1, 17, 3])
        t_list = [8, 12, 16]
        m_output = encode_motion(output, t_list)
        m_target = encode_motion(target, t_list)
        mean = tf.keras.metrics.Mean()
        for t in t_list:
            mean.update_state(
                tf.keras.losses.mean_absolute_error(m_target[f"{t}"], m_output[f"{t}"])
            )
        loss = mean.result()
        return loss


def encode_motion(s, t_list):
    """
    Encodes a sequence into a motion.

    Parameters
    ----------
    s : tf.Tensor
        The sequence to encode. s is of shape [T, 17, 3] where T represents the sequence length.
    t_list : list,
        List of time intervals to use to encode de sequence.

    Returns
    -------
    m : dict
        A dictionary of encoded motions for each time intervals.

    """
    # shape = tf.shape(s).numpy()
    m = {}
    for ti in t_list:
        mti = tf.linalg.cross(s[:-ti, :, :], s[ti:, :, :])
        m[f"{ti}"] = mti
    return m

# %% Vector based
def vector_linearity_loss(targets, outputs):
    loss = tf.reduce_mean(
        tf.norm(
            tf.linalg.cross(
                tf.reshape(targets, [-1, 3]), tf.reshape(outputs, [-1, 3])
            ), axis=-1
        )
    )
    return loss
