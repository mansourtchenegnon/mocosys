#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 1.0
"""
# %% Imports
import keras
from model import ops
from model.graph import laplacian, skeleton


class DistanceLoss(keras.losses.Loss):
    def __init__(self, name="position_loss"):
        super().__init__(name)

    def call(self, y_true, y_pred):
        """Computes the distance loss values over joints position and then over sequence.

        Args:
            y_true (Tensor): The ground truth tensor.
            y_pred (Tensor): The predicted tensor.

        Returns:
            A **Tensor**, corresponding to the computed joint distance loss value.
        """
        loss = keras.ops.mean(
            keras.ops.norm(
                keras.ops.reshape(y_true, [-1, 3]) - keras.ops.reshape(y_pred, [-1, 3]), axis=-1
            )
        )
        return loss

class VelocityLoss(keras.losses.Loss):
    def __init__(self, name="velocity_loss"):
        super().__init__(name)

    def call(self, y_true, y_pred):
        """Computes the velocity loss values over joints and then over sequence.

        Args:
            y_true (Tensor): The ground truth tensor.
            y_pred (Tensor): The predicted tensor.

        Returns:
            A **Tensor**, corresponding to the computed joint distance loss value.
        """
        loss = ops.mean(
        ops.norm(
            (y_true[..., 1:, :, :] - y_true[..., :-1, :, :]) - (y_pred[..., 1:, :, :] - y_pred[..., :-1, :, :]),
                axis=-1
            ), axis=[-1, -2]
        )
        return loss

class AccelerationLoss(keras.losses.Loss):
    def __init__(self, name="acceleration_loss"):
        super().__init__(name)

    def call(self, y_true, y_pred):
        """Computes the acceleration loss values over joints and then over sequence.

        Args:
            y_true (Tensor): The ground truth tensor.
            y_pred (Tensor): The predicted tensor.

        Returns:
            A **Tensor**, corresponding to the computed joint distance loss value.
        """
        y_true_acc = y_true[..., 2:, :, :] - 2 * y_true[..., 1:-1, :, :] + y_true[..., :-2, :, :]
        y_pred_acc = y_pred[..., 2:, :, :] - 2 * y_pred[..., 1:-1, :, :] + y_pred[..., :-2, :, :]
        loss = ops.mean(
            ops.norm(
                y_true_acc - y_pred_acc, axis=-1
                ), axis=[-1, -2]
            )
        return loss

class BoneLengthLoss(keras.losses.Loss):
    def __init__(self, name="bone_length_loss"):
        super().__init__(name)

    def call(self, y_true, y_pred):
        """Computes the bone length loss between `y_true` and `y_pred`.
        Args:
            y_true (Tensor): Tensor of the target.
            y_pred (Tensor): Tensor of the prediction.

        Returns:
            A **Tensor**, results of the computed Laplacian loss.
        """
        return keras.ops.mean(
            keras.ops.abs(self.get_bone_length(y_true) - self.get_bone_length(y_pred))
        )
    
    @staticmethod
    def get_bone_length(positions):
        def distance(p, q):
            return keras.ops.norm(p - q, axis=-1)

        assert keras.ops.ndim(positions) == 4, "Expected positions to be of rank 4 but got rank {} instead".format(
            keras.ops.shape(positions))
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
        return keras.ops.concatenate((hips, femur, tibia, spine_back, spine_up, neck, head, clavicle, humerus, radius),
                        axis=-1)


class MotionLoss(keras.losses.Loss):
    def __init__(self, name="motion_loss", reduction="sum_over_batch_size", dtype=None):
        super().__init__(name, reduction, dtype)

    def call(self, y_true, y_pred):
        """Computes motion loss.

        Args:
            y_true (Tensor): Tensor of ground truth values
            y_pred (Tensor): Tensor of predicted values

        Returns:
            A **Tensor**, result of the computed motion loss.
        """
        tis = [8, 12, 16]
        m_y_pred = self.encode_motion(y_pred, tis)
        m_y_true = self.encode_motion(y_true, tis)
        mean = 0.0
        for i in range(len(tis)):
            mean += keras.losses.mean_absolute_error(m_y_true[i], m_y_pred[i])
        loss = mean / len(tis)
        return loss
    
    @staticmethod
    def encode_motion(s, tis):
        """Encodes a sequence into a motion following Motion loss description.

        Args:
            s (Tensor): Tensor of joint positions to encode. Shape is [..., T, 17, 3] where T represents the sequence length.
            tis (list): List of time intervals to use to encode de sequence.

        Returns:
            list: A list of encoded motions for each time intervals in `tis`.
        """     
        # shape = tf.shape(s).numpy()
        m = []
        for ti in tis:
            mti = keras.ops.cross(s[..., :-ti, :, :], s[..., ti:, :, :])
            m.append(mti)
        return m


class LaplacianLoss(keras.losses.Loss):
    """
    Computes loss from laplacian representation of sequences of poses of same length.
    """

    def __init__(self, skeleton_graph:skeleton.SkeletonGraph, sequence_length=3, name="laplacian_loss"):
        """Creates a new instance of LaplacianLoss.

        Args:
            skeleton_graph (skeleton.SkeletonGraph): Graph representation in Laplacian Motion.
            sequence_length (int): Length of the window.
        """
        super().__init__(name)
        self.number_of_joints = skeleton_graph.get_num_of_joints()
        self.sequence_length = sequence_length
        self.skeleton = skeleton.SkeletonGraph(self.number_of_joints, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        self.L = laplacian.create_matrix_L(self.skeleton, sequence_length)

    def call(self, y_true, y_pred):
        """Computes the Laplacian loss between target and output.
        Args:
            y_true (Tensor): Tensor of the target.
            y_pred (Tensor): Tensor of the prediction.

        Returns:
            A **Tensor**, result of the computed Laplacian loss.
        """
        # Complete call function
        assert len(keras.ops.shape(y_true)) == len(keras.ops.shape(y_pred))
        center = self.sequence_length // 2
        delta_y_true = keras.ops.matmul(
            self.L, 
            ops.format_inputs(y_true, self.sequence_length)
        )
        delta_y_true = delta_y_true[..., center * self.number_of_joints:center * self.number_of_joints + self.number_of_joints, :]
        delta_y_pred = keras.ops.matmul(
            self.L, 
            ops.format_inputs(y_pred, self.sequence_length)
        )
        delta_y_pred = delta_y_pred[..., center * self.number_of_joints:center * self.number_of_joints + self.number_of_joints, :]
        loss = keras.ops.mean(keras.ops.norm(delta_y_true - delta_y_pred, axis=-1))
        return loss


class CombinedLoss(keras.losses.Loss):
    """ A combined loss made of the position loss and an additional loss.
    """
    def __init__(self, additional_loss, loss_weight=[1.0, 1.0], name="combined_loss", reduction="sum_over_batch_size", dtype=None):
        cname = name + f".position-{additional_loss.name}"
        super().__init__(cname, reduction, dtype)
        self.position_loss = DistanceLoss()
        self.additional_loss = additional_loss
        self.loss_weight = loss_weight

    def call(self, y_true, y_pred):
        """Computes the Laplacian loss between target and output.
        Args:
            y_true (Tensor): Tensor of the target.
            y_pred (Tensor): Tensor of the prediction.

        Returns:
            A **Tensor**, result of the computed combined loss.
        """
        return self.position_loss(y_true, y_pred) * self.loss_weight[0] \
            + self.additional_loss(y_true, y_pred) * self.loss_weight[1]
