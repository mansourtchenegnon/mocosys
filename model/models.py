#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
# %% Imports
import keras

from model.graph import laplacian as matrix
from model.graph.skeleton import H36M_17_JOINTS_SKELETON_BONES_PAIRS, SkeletonGraph
from model import layers, ops
from types import SimpleNamespace


class MotionFineTuningModel(keras.Model):
    """ Neural Network for the Motion Fine-Tuning stage.
    """
    def __init__(self, config, bones_pairs : list=H36M_17_JOINTS_SKELETON_BONES_PAIRS, name="motion-fine-tuning-model", *args, **kwargs):
        kwargs['name'] = name
        super().__init__(*args, **kwargs)
        self.joints = config.dataset.graph.skeleton.number_of_joints
        self.skeleton = SkeletonGraph(self.joints, bones_pairs)
        sk_conf = SimpleNamespace(number_of_joints=self.joints, bones=bones_pairs)
        config.skeleton = sk_conf
        self.params = config
        self.window = config.model.arch.window
        self.num_stages = config.model.arch.stages
        self.residual = config.model.arch.residual
        self.channels = config.model.arch.channels
        self.out_features = config.model.arch.channels_out
        self.activation = keras.activations.silu

        # Positional constraints will be on root.
        self.constraints = [i * self.joints for i in range(self.window)]
        self.pose_solver = layers.PoseSolver(self.skeleton, 3, self.window, self.constraints)
        self.adj = matrix.create_normalized_matrix_A(self.skeleton, self.window, True)
        self.delta_converter = layers.DeltaConverter(self.skeleton, 3, self.window, self.pose_solver.lgs.L)
        self.graph_conv_seq = keras.Sequential([
            layers.GraphConv(self.channels,
                            self.adj,
                            residual=self.residual,
                            activation=self.activation,
                            name=f"{self.name}.hidden_graph_conv_0"),
            keras.layers.LayerNormalization(axis=[-1, -2],
                                               name=f"{self.name}_norm_0"),
            layers.GraphConv(self.channels,
                            self.adj,
                            residual=self.residual,
                            activation=self.activation,
                            name=f"{self.name}.hidden_graph_conv_1"),
            keras.layers.LayerNormalization(axis=[-1, -2],
                                               name=f"{self.name}_norm_1"),
            layers.GraphConv(self.out_features,
                            self.adj,
                            residual=self.residual,
                            activation=self.activation,
                            name=f"{self.name}.hidden_graph_conv_2")
        ], name=f"{self.name}.st_poses_correction")

    def call(self, inputs, gamma=None):
        if gamma is None:
            vec = ops.vectorize(inputs, self.joints, self.window)
            gamma = self.pose_solver.lgs.D @ vec
        u = ops.format_inputs(ops.pad(inputs[..., 0:1, :]), self.window)
        self.pose_solver.set_constraints(u, gamma)
        outputs = self.delta_converter(inputs, keepdims=False, format=True)
        for s in range(self.num_stages):
            outputs = self.graph_conv_seq(outputs)
            outputs = self.pose_solver(outputs)
            if s == self.num_stages - 1:
                outputs = self.delta_converter(outputs, format=True, keepdims=True)
            else:
                outputs = self.delta_converter(outputs, format=True, keepdims=False)
        poses = self.pose_solver(outputs)
        return outputs, poses
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "params": self.params
        })


class SkeletonModel(keras.Model):
    """ Neural network model that estimates bone lengths from a sequence of poses.
    """    
    def __init__(self, params, name="skeleton-model", *args, **kwargs):
        """_summary_

        Parameters
        ----------
        params : types.SimpleNamespace
            Contains the configuration parameters for the module.
        name : str, optional
            Name for the neural network, by default "sk_model"
        """        
        kwargs["name"] = name
        super().__init__(*args, **kwargs)
        self.params = params
        if params:
            self.channels = params.model.arch.channels
            self.joints = params.dataset.graph.skeleton.number_of_joints
            self.window = params.model.arch.window
        else:
            self.channels = 16
            self.joints = 17
            self.window = 3
        self.features_out = 10
        self.dropout_rate = 0.2

        self.spatial_encoder = keras.Sequential([
            layers.ConvolutionBlock(self.channels, 1, dropout_rate=self.dropout_rate),
            layers.Residual(layers.ConvolutionBlock(self.channels, 1, dropout_rate=self.dropout_rate))
        ], name=f"{self.name}.spatial_encoder")

        self.temporal_encoder = keras.Sequential([
            layers.ConvolutionBlock(self.channels, self.window, dropout_rate=self.dropout_rate),
            layers.Residual(
                layers.ConvolutionBlock(self.channels, self.window, dropout_rate=self.dropout_rate)
            ),
            layers.ConvolutionBlock(self.channels, 1, dropout_rate=self.dropout_rate)
        ], name=f"{self.name}.temporal_encoder")
        self.regression = keras.Sequential([
            keras.layers.BatchNormalization(),
            layers.AdaptiveAvgPool(1),
            keras.layers.Conv1D(self.features_out, 1)
        ], name=f"{self.name}.regression")

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        kwargs['training'] = training
        kwargs['mask'] = mask
        outputs = self.spatial_encoder(inputs)
        outputs = self.temporal_encoder(outputs)
        outputs = self.regression(outputs)
        return outputs
