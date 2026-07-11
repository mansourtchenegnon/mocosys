#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import tensorflow as tf
import keras
from types import SimpleNamespace
from model.graph import laplacian, skeleton
from model import layers, models, ops
import utility.ops


class SkeletonConstraintsComputation(tf.Module):
    """ Class for the Skeleton Constraints Computation.
        It uses a SkeletonModel for bone lengths estimation.
        It computes skeletal constraints `Γ` and also implements
        the skeleton adjustement algorithm (method __call__()).
    """    
    def __init__(self, resume, bones=skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS, name="skel_constrained_computation"):
        """Creates new instance of SkeletonConstraintsComputation.

        Parameters
        ----------
        resume : str
            Path to the module checkpoints data.
        name : str, optional
            Name for the module, by default "skel_constrained_computation"
        """        
        super().__init__(name)
        self.skeleton_model = keras.saving.load_model(resume)
        config = self.skeleton_model.configuration
        self.window = config["model"]["arch"]["window"]
        joints = config["dataset"]["graph"]["skeleton"]["number_of_joints"]
        skel = skeleton.SkeletonGraph(joints, bones)
        self.D = laplacian.create_matrix_D(skel, 1)
        self.U = laplacian.create_matrix_U(skel, 1, [0])
        self.DU = keras.ops.concatenate((self.D, self.U), axis=1)
        self.DU = keras.ops.expand_dims(self.DU, axis=0)
        
        self.data_parameters = self.skeleton_model.get_normalization_parameters()

    def __call__(self, inputs):
        # print(inputs[0].shape, inputs[1].shape)
        b, t, _, _ = inputs[0].shape
        # Compute gamma
        bones = self.skeleton_model(inputs[1])
        gamma, bones = self.get_gamma(
            inputs[0], bones, self.data_parameters[2], self.data_parameters[3]
        )

        # Linear resolution
        u = inputs[0][..., 0:1, :]
        rhs = keras.ops.concatenate((gamma, u), axis=2)
        du = keras.ops.tile(self.DU, [b, t, 1, 1])
        # du = keras.ops.swapaxes(du, -1, -2) @ du
        outputs = keras.ops.linalg.solve(du, rhs)

        return outputs, gamma, bones
    
    def set_normalization_parameters(self, parameters):
        self.data_parameters = [keras.ops.convert_to_tensor(parameters[i]) for i in range(len(parameters))] 

    @staticmethod
    def get_gamma(inputs, bones, bone_mean, bone_std):
        batch, length, joints, _ = inputs.shape
        un_norm_bones = utility.ops.denormalise(bones, bone_mean, bone_std)

        # Gets bone lengths
        hips_norm = un_norm_bones[:, :, 0:1]
        femur_norm = un_norm_bones[:, :, 1:2]
        tibia_norm = un_norm_bones[:, :, 2:3]
        spine_back_norm = un_norm_bones[:, :, 3:4]
        spine_top_norm = un_norm_bones[:, :, 4:5]
        neck_norm = un_norm_bones[:, :, 5:6]
        head_norm = un_norm_bones[:, :, 6:7]
        clavicle_norm = un_norm_bones[:, :, 7:8]
        humerus_norm = un_norm_bones[:, :, 8:9]
        radius_norm = un_norm_bones[:, :, 9:10]

        # Get bones vectors
        hips_right = inputs[:, :, 0, :] - inputs[:, :, 1, :]
        femur_right = inputs[:, :, 1, :] - inputs[:, :, 2, :]
        tibia_right = inputs[:, :, 2, :] - inputs[:, :, 3, :]

        hips_left = inputs[:, :, 0, :] - inputs[:, :, 4, :]
        femur_left = inputs[:, :, 4, :] - inputs[:, :, 5, :]
        tibia_left = inputs[:, :, 5, :] - inputs[:, :, 6, :]

        spine_back = inputs[:, :, 0, :] - inputs[:, :, 7, :]
        spine_top = inputs[:, :, 7, :] - inputs[:, :, 8, :]
        neck = inputs[:, :, 8, :] - inputs[:, :, 9, :]
        head = inputs[:, :, 9, :] - inputs[:, :, 10, :]

        clavicle_left = inputs[:, :, 8, :] - inputs[:, :, 11, :]
        humerus_left = inputs[:, :, 11, :] - inputs[:, :, 12, :]
        radius_left = inputs[:, :, 12, :] - inputs[:, :, 13, :]

        clavicle_right = inputs[:, :, 8, :] - inputs[:, :, 14, :]
        humerus_right = inputs[:, :, 14, :] - inputs[:, :, 15, :]
        radius_right = inputs[:, :, 15, :] - inputs[:, :, 16, :]

        hips_right = (hips_norm / keras.ops.norm(hips_right, axis=-1, keepdims=True)) * hips_right
        hips_left = (hips_norm / keras.ops.norm(hips_left, axis=-1, keepdims=True)) * hips_left
        femur_right = (femur_norm / keras.ops.norm(femur_right, axis=-1, keepdims=True)) * femur_right
        femur_left = (femur_norm / keras.ops.norm(femur_left, axis=-1, keepdims=True)) * femur_left
        tibia_left = (tibia_norm / keras.ops.norm(tibia_left, axis=-1, keepdims=True)) * tibia_left
        tibia_right = (tibia_norm / keras.ops.norm(tibia_right, axis=-1, keepdims=True)) * tibia_right
        spine_back = (spine_back_norm / keras.ops.norm(spine_back, axis=-1, keepdims=True)) * spine_back
        spine_top = (spine_top_norm / keras.ops.norm(spine_top, axis=-1, keepdims=True)) * spine_top
        neck = (neck_norm / keras.ops.norm(neck, axis=-1, keepdims=True)) * neck
        head = (head_norm / keras.ops.norm(head, axis=-1, keepdims=True)) * head
        clavicle_left = (clavicle_norm / keras.ops.norm(clavicle_left, axis=-1, keepdims=True)) * clavicle_left
        clavicle_right = (clavicle_norm / keras.ops.norm(clavicle_right, axis=-1, keepdims=True)) * clavicle_right
        humerus_left = (humerus_norm / keras.ops.norm(humerus_left, axis=-1, keepdims=True)) * humerus_left
        humerus_right = (humerus_norm / keras.ops.norm(humerus_right, axis=-1, keepdims=True)) * humerus_right
        radius_left = (radius_norm / keras.ops.norm(radius_left, axis=-1, keepdims=True)) * radius_left
        radius_right = (radius_norm / keras.ops.norm(radius_right, axis=-1, keepdims=True)) * radius_right

        skeleton_bones = keras.ops.concatenate((hips_right,
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

        skeleton_bones = keras.ops.reshape(skeleton_bones, [batch, length, joints - 1, -1])
        return skeleton_bones, un_norm_bones


class MoCoSys(tf.Module):
    """A tensorflow module representing the motion correction system.
    """    
    def __init__(
            self,
            skm_path,
            mft_path,
            name="mocosys"
    ):
        """Constructor.

        Parameters
        ----------
        skm_path : str
            Path to the checkpoints of the skeleton model for bone lengths estimation.
        mft_path : str
            Path to the checkpoint of the motion fine tuning model.
        name : str, optional
            Name of the module for tensorflow initialisation, by default "mocosys"
        """        
        super().__init__(name)
        # Motion Fine Tuning Model
        self.motion_fine_tuning = keras.saving.load_model(mft_path)
        # Skeleton Constraints Computation
        self.skeleton_correction = SkeletonConstraintsComputation(skm_path)
        

    def __call__(self, inputs):
        """Call method for the motion correction process (mocosys).

        Parameters
        ----------
        inputs : tuple or list
            A tuple or list of two values minimum.
            1) inputs[0]: estimated 3D poses sequence to correct
            2) inputs[1]: input 2D poses sequence

        Returns
        -------
        KerasTensor
            The corrected 3D poses sequence.
        """        
        # Motion fine tuning (with skeleton) + skeleton adjustement
        # corrected, gamma, _ = self.skeleton_correction(inputs)
        # gamma = ops.format_inputs(gamma, self.motion_fine_tuning.configuration["model"]["arch"]["window"])
        # print("gamma", gamma.shape)
        # _, corrected = self.motion_fine_tuning(inputs[0], gamma)
        _, corrected = self.motion_fine_tuning(inputs[0])
        corrected, _, _ = self.skeleton_correction([corrected, inputs[1]])
        return corrected

    def set_normalization_parameters(self, parameters):
        self.skeleton_correction.set_normalization_parameters(parameters)

class MotionFineTuner(tf.Module):
    """A tensorflow module for motion fine tuning process in correction system.
    """    
    def __init__(
            self,
            mft_path,
            name="correction_modules"
    ):
        """Constructor.

        Parameters
        ----------
        mft_path : str
            Path to the checkpoint of the motion fine tuning model.
        name : str, optional
            Name of the module for tensorflow initialisation, by default "motion-fine-tuner".
        """        
        super().__init__(name)
        # Motion Fine Tuning Model
        self.motion_correction = keras.saving.load_model(mft_path)

    def __call__(self, inputs):
        """Call method for the motion fine tuning process.

        Parameters
        ----------
        inputs : Tensor
            A tensor of the estimated 3D poses sequence to correct

        Returns
        -------
        Tensor
            The corrected 3D poses sequence.
        """        
        # Motion correction
        return self.motion_correction(inputs)
    
class SkeletonCorrector(tf.Module):
    """A tensorflow module for skeleton adjustement process in correction system.
    """    
    def __init__(
            self,
            config:SimpleNamespace,
            bones_pairs:list,
            name="skeleton_corrector"
    ):
        """_summary_

        Parameters
        ----------
        config : SimpleNamespace
            Configuration data.
        bones_pairs : list
            List of bone pairs to create skeleton graph.
        name : str, optional
            Name of the module, by default "skeleton_corrector".
        """
        super().__init__(name)
        self.joints = config.dataset.graph.skeleton.number_of_joints
        self.skeleton = skeleton.SkeletonGraph(self.joints, bones_pairs)
        sk_conf = SimpleNamespace(number_of_joints=self.joints, bones=bones_pairs)
        config.skeleton = sk_conf
        constraints = []
        for i in range(self.window):
            constraints[i * self.joints] = [0, 0, 0]
        self.skeleton_adjustement = layers.SkeletonAdjustementLayer()
        self.pose_solver = layers.PoseSolver(self.skeleton, 3, self.window, constraints)

    def __call__(self, inputs):
        """Call method for the motion correction process (mocosys).

        Parameters
        ----------
        inputs : tf.Tensor
            A tensor of the estimated 3D poses sequence to correct

        Returns
        -------
        tensorflow.Tensor
            The corrected 3D poses sequence.
        """        
        gamma = self.skeleton_adjustement()
        gamma = ops.format_inputs(gamma, self.window)
        poses = self.pose_solver(inputs, gamma)
        return poses
