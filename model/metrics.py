#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 1.0
"""
from keras import ops


def mean_position_error(y_true, y_pred):
    return ops.mean(
        ops.norm(y_true - y_pred, axis=-1),  axis=[-1, -2]
    )


def mean_velocity_error(y_true, y_pred):
    return ops.mean(
        ops.norm(
            (y_true[..., 1:, :, :] - y_true[..., :-1, :, :]) - (y_pred[..., 1:, :, :] - y_pred[..., :-1, :, :]),
            axis=-1
        ), axis=[-1, -2]
    )


def mean_acceleration_error(y_true, y_pred):
    y_true_acc = y_true[..., 2:, :, :] - 2 * y_true[..., 1:-1, :, :] + y_true[..., :-2, :, :]
    y_pred_acc = y_pred[..., 2:, :, :] - 2 * y_pred[..., 1:-1, :, :] + y_pred[..., :-2, :, :]
    return ops.mean(
        ops.norm(
            y_true_acc - y_pred_acc, axis=-1
            ), axis=[-1, -2]
        )


def distance(position1, position2):
    return ops.norm(position1 - position2, axis=-1)


def get_bones(positions):
    length = ops.concatenate(
        ((distance(positions[..., :, 0, :], positions[..., :, 1, :]) + distance(positions[..., :, 0, :], positions[..., :, 4, :])) / 2,
        (distance(positions[..., :, 1, :], positions[..., :, 2, :]) + distance(positions[..., :, 4, :], positions[..., :, 5, :])) / 2,
        (distance(positions[..., :, 2, :], positions[..., :, 3, :]) + distance(positions[..., :, 5, :], positions[..., :, 6, :])) / 2,
        distance(positions[..., :, 0, :], positions[..., :, 7, :]),
        distance(positions[..., :, 7, :], positions[..., :, 8, :]),
        distance(positions[..., :, 8, :], positions[..., :, 9, :]),
        distance(positions[..., :, 9, :], positions[..., :, 10, :]),
        (distance(positions[..., :, 8, :], positions[..., :, 11, :]) + distance(positions[..., :, 8, :], positions[..., :, 14, :])) / 2,
        (distance(positions[..., :, 14, :], positions[..., :, 15, :]) + distance(positions[..., :, 11, :], positions[..., :, 12, :])) / 2,
        (distance(positions[..., :, 15, :], positions[..., :, 16, :]) + distance(positions[..., :, 12, :], positions[..., :, 13, :])) / 2)
        , axis=-1)
    return length


def mean_bone_length_error(y_true, y_pred):
    return ops.mean(
        ops.abs(
            get_bones(y_true) - get_bones(y_pred)
        )
    )


def bone_length_variance(positions, per_bone=False):
    bones = get_bones(positions)
    mean_bone = ops.mean(bones, axis=1)
    variance = ops.sqrt(ops.mean((bones - mean_bone) ** 2, axis=0))
    if not per_bone:
        variance = ops.mean(variance)
    return variance



# %% Per frame errors
def per_frame_mean_acceleration_error(output, target):
    assert output.shape == target.shape, "output and target should have same shape"
    target_reshaped = ops.reshape(target, [-1, 17, 3])
    output_reshaped = ops.reshape(output, [-1, 17, 3])
    acceleration_output = (
            output_reshaped[2:, :, :] - 2 * output_reshaped[1:-1, :, :] + output_reshaped[:-2, :, :]
    )
    acceleration_target = (
            target_reshaped[2:, :, :] - 2 * target_reshaped[1:-1, :, :] + target_reshaped[:-2, :, :]
    )
    error = ops.mean(
        ops.norm(acceleration_target - acceleration_output, axis=len(acceleration_target.shape) - 1),
        axis=-1
    )

    return error.numpy()


def absolute_error(output, target):
    assert output.shape == target.shape, "output and target should have same shape"
    error = ops.abs(target - output)
    # error = tf.reduce_mean(error)

    return error.numpy()


def r_squared_on_position_error(targets, outputs):
    assert outputs.shape == targets.shape, "outputs and targets should have same shape"
    y = ops.reshape(targets, (-1, 17, 3))
    y_mean = ops.mean(y, axis=0, keepdims=True)
    y_hat = ops.reshape(targets, (-1, 17, 3))

    var_mean = ops.mean(ops.square(y - y_mean))
    var_line = ops.mean(ops.square(y - y_hat))
    r_score = (var_mean - var_line) / var_mean
    return r_score
