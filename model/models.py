#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
# %% Imports
import keras
import keras.ops as kops
from model.graph import laplacian as matrix
from model.graph.skeleton import H36M_17_JOINTS_SKELETON_BONES_PAIRS, SkeletonGraph
from model import layers, ops


class MotionFineTuningModel(keras.Model):
    """ Neural Network for the Motion Fine-Tuning stage.
    """
    def __init__(self, config, bones_pairs : list=H36M_17_JOINTS_SKELETON_BONES_PAIRS, name="motion-fine-tuning-model", *args, **kwargs):
        kwargs['name'] = name
        super().__init__(*args, **kwargs)
        self.configuration = config
        self.joints = self.configuration["dataset"]["graph"]["skeleton"]["number_of_joints"]
        self.bones_pairs = bones_pairs
        self.skeleton = SkeletonGraph(self.joints, self.bones_pairs)
        self.window = self.configuration["model"]["arch"]["window"]
        self.num_stages = self.configuration["model"]["arch"]["stages"]
        residual = self.configuration["model"]["arch"]["residual"]
        channels = self.configuration["model"]["arch"]["channels"]
        out_features = self.configuration["model"]["arch"]["channels_out"]
        self.activation = keras.activations.silu

        # Positional constraints will be on root.
        constraints = [i * self.joints for i in range(self.window)]
        self.pose_solver = layers.PoseSolver(self.skeleton, 3, self.window, constraints)
        self.adj = matrix.create_normalized_matrix_A(self.skeleton, self.window, True)
        self.delta_converter = layers.DeltaConverter(self.skeleton, 3, self.window, self.pose_solver.lgs.L)
        self.graph_conv_seq = keras.Sequential([
            layers.GraphConv(channels,
                            self.adj,
                            residual=residual,
                            activation=self.activation,
                            name=f"{self.name}.hidden_graph_conv_0"),
            keras.layers.LayerNormalization(axis=[-1, -2],
                                               name=f"{self.name}_norm_0"),
            layers.GraphConv(channels,
                            self.adj,
                            residual=residual,
                            activation=self.activation,
                            name=f"{self.name}.hidden_graph_conv_1"),
            keras.layers.LayerNormalization(axis=[-1, -2],
                                               name=f"{self.name}_norm_1"),
            layers.GraphConv(out_features,
                            self.adj,
                            residual=residual,
                            activation=self.activation,
                            name=f"{self.name}.hidden_graph_conv_2")
        ], name=f"{self.name}.st_poses_correction")

    def call(self, inputs, gamma=None):
        if gamma is None:
            vec = ops.vectorize(inputs, self.joints, self.window)
            gamma = self.pose_solver.lgs.D @ vec
        u = ops.format_inputs(inputs[..., 0:1, :], self.window)
        self.pose_solver.set_constraints(u, gamma)
        outputs = self.delta_converter(inputs, keepdims=False, format=True)  # [..., T, W*V, C]
        for s in range(self.num_stages):
            outputs = self.graph_conv_seq(outputs)  # [..., T, W*V, C]
            outputs = self.pose_solver(outputs, format=False)
            if s == self.num_stages - 1:
                outputs = self.delta_converter(outputs, format=True, keepdims=True)  # [..., T, V, C]
            else:
                outputs = self.delta_converter(outputs, format=True, keepdims=False)  # [..., T, W*V, C]
        poses = self.pose_solver(outputs)  # [..., T, W*V, C]
        return outputs, poses
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "configuration": self.configuration,
            "window": self.window,
            "num_stages": self.num_stages,
            "bones_pairs" : self.bones_pairs
        }
        return {**base_config, **config}


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
            Name for the neural network, by default "skeleton-model"
        """        
        kwargs["name"] = name
        super().__init__(*args, **kwargs)
        self.params = params
        if self.params:
            channels = self.params["model"]["arch"]["channels"]
            # joints = self.params["dataset"]["graph"]["skeleton"]["number_of_joints"]
            window = self.params["model"]["arch"]["window"]
        else:
            channels = 256
            # joints = 17
            window = 3
        features_out = 10
        dropout_rate = 0.2
        self._bones_mean = []
        self._bones_std = []
        self._inputs_mean = []
        self._inputs_std = []

        self.spatial_encoder = keras.Sequential([
            layers.ConvolutionBlock(channels, 1, dropout_rate=dropout_rate),
            layers.Residual(layers.ConvolutionBlock(channels, 1, dropout_rate=dropout_rate))
        ], name=f"{self.name}.spatial_encoder")

        self.temporal_encoder = keras.Sequential([
            layers.ConvolutionBlock(channels, window, dropout_rate=dropout_rate),
            layers.Residual(
                layers.ConvolutionBlock(channels, window, dropout_rate=dropout_rate)
            ),
            layers.ConvolutionBlock(channels, 1, dropout_rate=dropout_rate)
        ], name=f"{self.name}.temporal_encoder")
        self.regression = keras.Sequential([
            keras.layers.BatchNormalization(),
            layers.AdaptiveAvgPool(1),
            keras.layers.Conv1D(features_out, 1)
        ], name=f"{self.name}.regression")

    def call(self, inputs, training=False):
        kwargs = {}
        kwargs['training'] = training
        outputs = self.spatial_encoder(inputs, training=training)
        outputs = self.temporal_encoder(outputs, training=training)
        outputs = self.regression(outputs)
        return outputs
    
    def forward_denormalized(self, inputs):
        outputs = self.call(inputs)
        return self._denormalize_data(outputs)
    
    def set_normalization_parameters(self, parameters):
        self._inputs_mean = parameters[0]
        self._inputs_std = parameters[1]
        self._bones_mean = parameters[2]
        self._bones_std = parameters[3]
    
    def _normalize_data(self, data):
        mean = kops.ones(data.shape, dtype='float32') * self._normalization_mean
        std = kops.ones(data.shape, dtype='float32') * self._normalization_std
        normalized_data = kops.divide(data - mean, std)
        return normalized_data

    def _denormalize_data(self, data):
        D = self.normalization_mean.shape[0]  # Dimensionality
        shape = [1 for i in range(len(data.shape)-1)] + [D]
        repeat = list(data.shape[:-1]) + [1]
        std = kops.tile(
            kops.reshape(self._normalization_std, shape),
            repeat
        )
        mean = kops.tile(
            kops.reshape(self._normalization_mean, shape),
            repeat
        )
        return kops.multiply(data * std) + mean
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "params": self.params,
            "norm_mean": self._normalization_mean,
            "norm_std": self._normalization_std
        }
        return {**base_config, **config}