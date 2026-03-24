#!/usr/bin/env python3
# -*- coding: unp.linalg-8 -*-
"""
@author: Mansour Tchenegnon
@version: 15.03.2025
"""

# %% Imports
import numpy as np


# %% Functions 
def get_velocity(positions, frame_rate=1):
    p = np.reshape(positions, [-1, 17, 3])
    velocity = p[1:, :, :] - p[:-1, :, :]
    velocity = np.sum(np.square(velocity), axis=-1)
    return velocity


def get_acceleration(positions, frame_rate=1):
    p = np.reshape(positions, [-1, 17, 3])
    acceleration = p[2:, :, :] - 2 * p[1:-1, :, :] + p[:-2, :, :]
    acceleration = np.sum(np.square(acceleration), axis=-1)
    return acceleration

def get_bones_length_variations(positions):
    def _get_bones(poses: np.array, dim=3):
        length, _ = np.shape(poses)
        hips_right = np.linalg.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 1:dim * 1 + dim], axis=-1,
                            keepdims=True)
        femur_right = np.linalg.norm(poses[:, dim * 1:dim * 1 + dim] - poses[:, dim * 2:dim * 2 + dim], axis=-1,
                            keepdims=True)
        tibia_right = np.linalg.norm(poses[:, dim * 2:dim * 2 + dim] - poses[:, dim * 3:dim * 3 + dim], axis=-1,
                            keepdims=True)

        hips_left = np.linalg.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 4:dim * 4 + dim], axis=-1, keepdims=True)
        femur_left = np.linalg.norm(poses[:, dim * 4:dim * 4 + dim] - poses[:, dim * 5:dim * 5 + dim], axis=-1,
                            keepdims=True)
        tibia_left = np.linalg.norm(poses[:, dim * 5:dim * 5 + dim] - poses[:, dim * 6:dim * 6 + dim], axis=-1,
                            keepdims=True)

        spine_back = np.linalg.norm(poses[:, dim * 0:dim * 0 + dim] - poses[:, dim * 7:dim * 7 + dim], axis=-1,
                            keepdims=True)
        spine_top = np.linalg.norm(poses[:, dim * 7:dim * 7 + dim] - poses[:, dim * 8:dim * 8 + dim], axis=-1, keepdims=True)
        neck = np.linalg.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 9:dim * 9 + dim], axis=-1, keepdims=True)
        head = np.linalg.norm(poses[:, dim * 9:dim * 9 + dim] - poses[:, dim * 10:dim * 10 + dim], axis=-1, keepdims=True)

        clavicle_left = np.linalg.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 11:dim * 11 + dim], axis=-1,
                                keepdims=True)
        humerus_left = np.linalg.norm(poses[:, dim * 11:dim * 11 + dim] - poses[:, dim * 12:dim * 12 + dim], axis=-1,
                            keepdims=True)
        radius_left = np.linalg.norm(poses[:, dim * 12:dim * 12 + dim] - poses[:, dim * 13:dim * 13 + dim], axis=-1,
                            keepdims=True)

        clavicle_right = np.linalg.norm(poses[:, dim * 8:dim * 8 + dim] - poses[:, dim * 14:dim * 14 + dim], axis=-1,
                                keepdims=True)
        humerus_right = np.linalg.norm(poses[:, dim * 14:dim * 14 + dim] - poses[:, dim * 15:dim * 15 + dim], axis=-1,
                                keepdims=True)
        radius_right = np.linalg.norm(poses[:, dim * 15:dim * 15 + dim] - poses[:, dim * 16:dim * 16 + dim], axis=-1,
                            keepdims=True)

        skeleton = np.concat((hips_right,
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
    bones = _get_bones(positions)
    return bones


def get_mean_motion_desc_per_skeleton_part(descriptor, part):
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