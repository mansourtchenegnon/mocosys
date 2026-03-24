#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import pickle

import tensorflow as tf
from types import SimpleNamespace
from utility.graph import laplacian, skeleton
from model import layers, models, ops
import utility.tools as tools


class SkeletonConstraintsComputation(tf.Module):
    """ Class for the Skeleton Constraints Computation.
        It uses a SkeletonModel for bone lengths estimation.
        It computes skeletal constraints `Γ` and also implements
        the skeleton adjustement algorithm (__call__ method).
    """    
    def __init__(self, resume, name="skel_constrained_computation"):
        """_summary_

        Parameters
        ----------
        params : types.SimpleNamespace
            Contains the configuration parameters for the module.
        resume : str, optional
            Path to the module checkpoints data, by default "./checkpoints". The directory
            should contain a "saved.pkl" file and a "weights.keras" file.
        name : str, optional
            Name for the module, by default "skel_constrained_computation"
        """        
        super().__init__(name)
        if resume:
            import pickle
            with open(f"{resume}/saved.pkl", "rb") as fd:
                state = pickle.load(fd)
            self.params = state['config']
    
        self.window = self.params.skelmodel.window
        joints = self.params.graph.skeleton.number_of_joints
        edges = self.params.graph.skeleton.bones
        skel = skeleton.SkeletonGraph(joints, edges)
        self.D = laplacian.create_matrix_D(skel, 1)
        self.U = laplacian.create_matrix_U(skel, 1, [0])
        # self.wU = 0.3 * self.U
        # self.wD = 0.7 * self.D
        self.DU = tf.concat((self.D, self.U), axis=1)
        self.DU = tf.expand_dims(self.DU, axis=0)
        
        # arch = state['arch']
        config = state['config']
        self.skeleton_model = models.SkeletonModel(config)
        # self.skeleton_model = SkeletonModel()
        self.skeleton_model.build([None, None, 34])
        self.skeleton_model.load_weights(f"{resume}/weights.keras")
        self.data_parameters = state['data_params']

        # self.smoother = layer.SmoothingLayer(5)

    def __call__(self, inputs):
        # print(inputs[0].shape, inputs[1].shape)
        b, t, _ = inputs[0].shape
        # outputs = self.smoother(inputs)
        # outputs = inputs[0]
        # Compute desired gamma
        bones = self.skeleton_model(inputs[1], training=False)
        gamma, bones = self.get_gamma(ops.vectorize(inputs[0]), bones,
                                      self.data_parameters[2],
                                      self.data_parameters[3])

        # Linear resolution
        u = tf.zeros((b, t, 1, 3))
        rhs = tf.concat((gamma, u), axis=2)
        outputs = tf.linalg.solve(tf.tile(self.DU, [b, t, 1, 1]),
                                  rhs)

        outputs = tf.reshape(outputs, (b, t, -1))
        return outputs, gamma, bones

    @staticmethod
    def get_gamma(inputs, bones, bone_mean, bone_std):
        batch, length, joints, _ = inputs.shape
        un_norm_bones = tools.un_normalise(bones, bone_mean, bone_std)
        # un_norm_bones = bones * tf.expand_dims(bone_std, 0)
        # un_norm_bones += tf.reshape(bone_mean, (bones.shape[0], 1, bone_mean.shape[0]))

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

        hips_right = (hips_norm / tf.norm(hips_right, axis=-1, keepdims=True)) * hips_right
        hips_left = (hips_norm / tf.norm(hips_left, axis=-1, keepdims=True)) * hips_left
        femur_right = (femur_norm / tf.norm(femur_right, axis=-1, keepdims=True)) * femur_right
        femur_left = (femur_norm / tf.norm(femur_left, axis=-1, keepdims=True)) * femur_left
        tibia_left = (tibia_norm / tf.norm(tibia_left, axis=-1, keepdims=True)) * tibia_left
        tibia_right = (tibia_norm / tf.norm(tibia_right, axis=-1, keepdims=True)) * tibia_right
        spine_back = (spine_back_norm / tf.norm(spine_back, axis=-1, keepdims=True)) * spine_back
        spine_top = (spine_top_norm / tf.norm(spine_top, axis=-1, keepdims=True)) * spine_top
        neck = (neck_norm / tf.norm(neck, axis=-1, keepdims=True)) * neck
        head = (head_norm / tf.norm(head, axis=-1, keepdims=True)) * head
        clavicle_left = (clavicle_norm / tf.norm(clavicle_left, axis=-1, keepdims=True)) * clavicle_left
        clavicle_right = (clavicle_norm / tf.norm(clavicle_right, axis=-1, keepdims=True)) * clavicle_right
        humerus_left = (humerus_norm / tf.norm(humerus_left, axis=-1, keepdims=True)) * humerus_left
        humerus_right = (humerus_norm / tf.norm(humerus_right, axis=-1, keepdims=True)) * humerus_right
        radius_left = (radius_norm / tf.norm(radius_left, axis=-1, keepdims=True)) * radius_left
        radius_right = (radius_norm / tf.norm(radius_right, axis=-1, keepdims=True)) * radius_right

        skeleton_bones = tf.concat((hips_right,
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

        skeleton_bones = tf.reshape(skeleton_bones, [batch, length, joints - 1, -1])
        return skeleton_bones, un_norm_bones


class MoCoSys(tf.Module):
    """A tensorflow module representing the motion correction system.
    """    
    def __init__(
            self,
            skm_path=".checkpoints/best/model_skm",
            mft_path=".checkpoints/best/model_mft",
            name="correction_modules"
    ):
        """Constructor.

        Parameters
        ----------
        skm_path : str, optional
            Path to the checkpoints of the skeleton model for bone lengths estimation, by default ".checkpoints/best/model_skm"
        mft_path : str, optional
            Path to the checkpoint of the motion fine tuning model, by default ".checkpoints/best/model_mft"
        name : str, optional
            Name of the module for tensorflow initialisation, by default "mocosys"
        """        
        super().__init__(name)
        # Motion Fine Tuning Model
        with open(f"{mft_path}/state.pkl", "rb") as fd:
            state_mft = pickle.load(fd)
        mft_config = state_mft['config']
        self.mft_params = mft_config
        bones_pairs = mft_config.skeleton.bones
        self.motion_correction = models.MotionFineTuningModel(mft_config, bones_pairs)
        self.motion_correction(tf.random.uniform((1, 27, 51)))
        self.motion_correction.load_weights(f"{mft_path}/weights.keras")

        self.skeleton_correction = models.SkeletonConstraintsComputation(None, skm_path)
        

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
        tensorflow.Tensor
            The corrected 3D poses sequence.
        """        
        ## Skeleton adjustement + motion correction
        # corrected, _, _ = self.skeleton_correction(inputs)
        # _, corrected = self.motion_correction(corrected)

        ## Motion correction + skeleton adjustement
        # _, corrected = self.motion_correction(inputs[0])
        # corrected, _, _ = self.skeleton_correction([corrected, inputs[1]])

        # Motion correction (with skeleton) + skeleton adjustement
        corrected, gamma, _ = self.skeleton_correction(inputs)
        gamma = ops.format_inputs(gamma, self.mft_params.mftmodel.arch.window)
        batch, length, _, _, features = tf.shape(gamma)
        gamma = tf.reshape(gamma, (batch, length, -1, features))
        _, corrected = self.motion_correction(inputs[0], gamma)
        # _, corrected = self.motion_correction(corrected, gamma)
        corrected, _, _ = self.skeleton_correction([corrected, inputs[1]])
        return corrected


class MotionFineTuner(tf.Module):
    """A tensorflow module for motion fine tuning process in correction system.
    """    
    def __init__(
            self,
            mft_path=".checkpoints/best/model_mft",
            name="correction_modules"
    ):
        """Constructor.

        Parameters
        ----------
        mft_path : str, optional
            Path to the checkpoint of the motion fine tuning model, by default ".checkpoints/best/model_mft"
        name : str, optional
            Name of the module for tensorflow initialisation, by default "mocosys"
        """        
        super().__init__(name)
        # Motion Fine Tuning Model
        with open(f"{mft_path}/state.pkl", "rb") as fd:
            state_mft = pickle.load(fd)
        mft_config = state_mft['config']
        self.mft_params = mft_config
        bones_pairs = mft_config.skeleton.bones
        self.motion_correction = models.MotionFineTuningModel(mft_config, bones_pairs)
        self.motion_correction(tf.random.uniform((1, 27, 51)))
        self.motion_correction.load_weights(f"{mft_path}/weights.keras")


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
