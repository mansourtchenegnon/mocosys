#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import numpy as np
import tensorflow as tf


# %% Distance errors
def mean_points_error(output, target):
    # if not isinstance(output, np.ndarray):
    #     output = output.numpy()
    #     target = target.numpy()
    # output_reshape = output.reshape((-1, 3))
    # target_reshape = target.reshape((-1, 3))
    # error = np.mean(
    #     np.sqrt(np.sum(np.square((output_reshape - target_reshape)), axis=1))
    # )
    # return error
    output_reshape = tf.reshape(output, (-1, 3))
    target_reshape = tf.reshape(target, (-1, 3))
    error = tf.reduce_mean(
        tf.norm(output_reshape - target_reshape, axis=1)
    )
    return error.numpy()


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0 ** 2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0 ** 2, axis=(1, 2), keepdims=True))

    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * norm_x / norm_y  # Scale
    t = mu_x - a * np.matmul(mu_y, r)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, r) + t

    # Return MPJPE
    return np.mean(
        np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
    )


def mean_bone_length_error(output, target):
    def get_bones(position_3d):
        def distance(position1, position2):
            return np.sqrt(np.sum(np.square(position1 - position2), axis=-1))

        length = np.zeros((position_3d.shape[0], 10))
        length[:, 0] = ((distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 1:3 * 1 + 3]) + distance(
            position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 4:3 * 4 + 3])) / 2)
        length[:, 1] = ((distance(position_3d[:, 3 * 1:3 * 1 + 3], position_3d[:, 3 * 2:3 * 2 + 3]) + distance(
            position_3d[:, 3 * 4:3 * 4 + 3], position_3d[:, 3 * 5:3 * 5 + 3])) / 2)
        length[:, 2] = ((distance(position_3d[:, 3 * 2:3 * 2 + 3], position_3d[:, 3 * 3:3 * 3 + 3]) + distance(
            position_3d[:, 3 * 5:3 * 5 + 3], position_3d[:, 3 * 6:3 * 6 + 3])) / 2)
        length[:, 3] = (distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 7:3 * 7 + 3]))
        length[:, 4] = (distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 8:3 * 8 + 3]))
        length[:, 5] = (distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 9:3 * 9 + 3]))
        length[:, 6] = (distance(position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 10:3 * 10 + 3]))
        length[:, 7] = ((distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 11:3 * 11 + 3]) + distance(
            position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 14:3 * 14 + 3])) / 2)
        length[:, 8] = ((distance(position_3d[:, 3 * 14:3 * 14 + 3], position_3d[:, 3 * 15:3 * 15 + 3]) + distance(
            position_3d[:, 3 * 11:3 * 11 + 3], position_3d[:, 3 * 12:3 * 12 + 3])) / 2)
        length[:, 9] = ((distance(position_3d[:, 3 * 15:3 * 15 + 3], position_3d[:, 3 * 16:3 * 16 + 3]) + distance(
            position_3d[:, 3 * 12:3 * 12 + 3], position_3d[:, 3 * 13:3 * 13 + 3])) / 2)
        return length

    assert output.shape == target.shape, "output and target should have same shape"
    if not isinstance(output, np.ndarray):
        output = output.numpy()
        target = target.numpy()

    output_bones = get_bones(output.squeeze(0))
    target_bones = get_bones(target.squeeze(0))
    error = np.abs(output_bones - target_bones)
    error = np.mean(error)
    return error


def bone_length_variance(positions, per_bone=False):
    def get_bones(position_3d):
        def distance(position1, position2):
            return np.sqrt(np.sum(np.square(position1 - position2), axis=-1))

        length = np.zeros((position_3d.shape[0], 10))
        length[:, 0] = ((distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 1:3 * 1 + 3]) + distance(
            position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 4:3 * 4 + 3])) / 2)
        length[:, 1] = ((distance(position_3d[:, 3 * 1:3 * 1 + 3], position_3d[:, 3 * 2:3 * 2 + 3]) + distance(
            position_3d[:, 3 * 4:3 * 4 + 3], position_3d[:, 3 * 5:3 * 5 + 3])) / 2)
        length[:, 2] = ((distance(position_3d[:, 3 * 2:3 * 2 + 3], position_3d[:, 3 * 3:3 * 3 + 3]) + distance(
            position_3d[:, 3 * 5:3 * 5 + 3], position_3d[:, 3 * 6:3 * 6 + 3])) / 2)
        length[:, 3] = (distance(position_3d[:, 3 * 0:3 * 0 + 3], position_3d[:, 3 * 7:3 * 7 + 3]))
        length[:, 4] = (distance(position_3d[:, 3 * 7:3 * 7 + 3], position_3d[:, 3 * 8:3 * 8 + 3]))
        length[:, 5] = (distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 9:3 * 9 + 3]))
        length[:, 6] = (distance(position_3d[:, 3 * 9:3 * 9 + 3], position_3d[:, 3 * 10:3 * 10 + 3]))
        length[:, 7] = ((distance(position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 11:3 * 11 + 3]) + distance(
            position_3d[:, 3 * 8:3 * 8 + 3], position_3d[:, 3 * 14:3 * 14 + 3])) / 2)
        length[:, 8] = ((distance(position_3d[:, 3 * 14:3 * 14 + 3], position_3d[:, 3 * 15:3 * 15 + 3]) + distance(
            position_3d[:, 3 * 11:3 * 11 + 3], position_3d[:, 3 * 12:3 * 12 + 3])) / 2)
        length[:, 9] = ((distance(position_3d[:, 3 * 15:3 * 15 + 3], position_3d[:, 3 * 16:3 * 16 + 3]) + distance(
            position_3d[:, 3 * 12:3 * 12 + 3], position_3d[:, 3 * 13:3 * 13 + 3])) / 2)
        return length

    bones = get_bones(positions)
    mean_bone = np.mean(bones, axis=0)
    variance = np.sqrt(np.mean((bones - mean_bone) ** 2, axis=0))
    if not per_bone:
        variance = np.mean(variance)
    return variance


# %% Velocity error
def mean_velocity_error(output, target):
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
    assert output.shape == target.shape, "output and target should have same shape"
    if not isinstance(output, np.ndarray):
        output = output.numpy()
        target = target.numpy()
    velocity_output = np.diff(output.reshape([-1, 17, 3]), axis=0)
    velocity_target = np.diff(target.reshape([-1, 17, 3]), axis=0)
    error = np.mean(
        np.linalg.norm(velocity_output - velocity_target, axis=len(target.shape) - 1)
    )

    return error


def mean_velocity_error_t2(output, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 2nd derivative).

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
    assert output.shape == target.shape, "output and target should have same shape"
    target_reshaped = tf.reshape(target, [-1, 17, 3])
    output_reshaped = tf.reshape(output, [-1, 17, 3])
    velocity_output = (
            (output_reshaped[2:, :, :] - output_reshaped[:-2, :, :]) / 2
    )
    velocity_target = (
            (target_reshaped[2:, :, :] - target_reshaped[:-2, :, :]) / 2
    )
    error = tf.reduce_mean(
        tf.norm(velocity_output - velocity_target, axis=len(velocity_target.shape) - 1)
    )

    return error


# %% Motion Error
def motion_error(target, output):
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
    tis = [2, 4, 8, 12, 16]
    m_output = encode_motion(output, tis)
    m_target = encode_motion(target, tis)
    errors = {}
    for i in range(len(tis)):
        errors[tis[i]] = tf.reduce_mean(tf.abs((m_target[i] - m_output[i]))).numpy()

    return errors


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
    m : list of tf.Tensor
        A list of encoded motions for each time intervals.

    """
    # shape = tf.shape(s).numpy()
    m = []
    for ti in t_list:
        mti = tf.linalg.cross(s[:-ti, :, :], s[ti:, :, :])
        m.append(mti)
    return m


# %% Mean acceleration error
def mean_acceleration_error(output, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 2nd derivative).

    Parameters
    ----------
    output : tf.Tensor
        The predicted values.
    target : tf.Tensor
        The ground truth values.

    Returns
    -------
    error : float32
        Computed mean acceleration error.
    """
    assert output.shape == target.shape, "output and target should have same shape"
    target_reshaped = tf.reshape(target, [-1, 17, 3])
    output_reshaped = tf.reshape(output, [-1, 17, 3])
    acceleration_output = (
            output_reshaped[2:, :, :] - 2 * output_reshaped[1:-1, :, :] + output_reshaped[:-2, :, :]
    )
    acceleration_target = (
            target_reshaped[2:, :, :] - 2 * target_reshaped[1:-1, :, :] + target_reshaped[:-2, :, :]
    )
    error = tf.reduce_mean(
        tf.norm(acceleration_target - acceleration_output, axis=len(acceleration_target.shape) - 1)
    )

    return error.numpy()


def mean_acceleration_error_np(output, target):
    """
    Mean per-joint acceleration error (i.e. mean Euclidean distance of the 2nd derivative).
    """
    assert output.shape == target.shape, "output and target should have same shape"
    if not isinstance(output, np.ndarray):
        output = output.numpy()
        target = target.numpy()
    output_reshape = output.reshape((-1, 17, 3))
    target_reshape = target.reshape((-1, 17, 3))
    acc_predicted = (output_reshape[2:, :, :] - 2 * output_reshape[1:-1, :, :] + output_reshape[:-2, :, :])
    acc_target = (target_reshape[2:, :, :] - 2 * target_reshape[1:-1, :, :] + target_reshape[:-2, :, :])

    return np.mean(np.linalg.norm(acc_predicted - acc_target, axis=len(target.shape) - 1))


# %% Per frame errors
def per_frame_mean_acceleration_error(output, target):
    assert output.shape == target.shape, "output and target should have same shape"
    target_reshaped = tf.reshape(target, [-1, 17, 3])
    output_reshaped = tf.reshape(output, [-1, 17, 3])
    acceleration_output = (
            output_reshaped[2:, :, :] - 2 * output_reshaped[1:-1, :, :] + output_reshaped[:-2, :, :]
    )
    acceleration_target = (
            target_reshaped[2:, :, :] - 2 * target_reshaped[1:-1, :, :] + target_reshaped[:-2, :, :]
    )
    error = tf.reduce_mean(
        tf.norm(acceleration_target - acceleration_output, axis=len(acceleration_target.shape) - 1),
        axis=-1
    )

    return error.numpy()


def absolute_error(output, target):
    assert output.shape == target.shape, "output and target should have same shape"
    error = tf.abs(target - output)
    # error = tf.reduce_mean(error)

    return error.numpy()


def r_squared_on_position_error(targets, outputs):
    assert outputs.shape == targets.shape, "outputs and targets should have same shape"
    y = tf.reshape(targets, (-1, 17, 3))
    y_mean = tf.reduce_mean(y, axis=0, keepdims=True)
    y_hat = tf.reshape(targets, (-1, 17, 3))

    var_mean = tf.reduce_mean(tf.square(y - y_mean))
    var_line = tf.reduce_mean(tf.square(y - y_hat))
    r_score = (var_mean - var_line) / var_mean
    return r_score
