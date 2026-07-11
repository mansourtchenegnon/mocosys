#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 2026.07
"""
# %% Imports
import keras
import keras.ops as kops
from model.graph import laplacian as matrix
from model.graph.skeleton import H36M_17_JOINTS_SKELETON_BONES_PAIRS, SkeletonGraph
from model import layers, ops

@keras.saving.register_keras_serializable()
class MotionFineTuningModel(keras.Model):
    """ Neural Network for the Motion Fine-Tuning stage.
    """
    def __init__(self, configuration, bones_pairs : list=H36M_17_JOINTS_SKELETON_BONES_PAIRS, name="motion-fine-tuning-model", *args, **kwargs):
        kwargs['name'] = name
        super().__init__(*args, **kwargs)
        self.configuration = configuration
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
            "bones_pairs" : self.bones_pairs
        }
        return {**base_config, **config}

@keras.saving.register_keras_serializable()
class BaseSkeletonModel(keras.Model):
    """ Neural network model that estimates bone lengths from a sequence of poses.
    """    
    def __init__(self, configuration, name="base-skeleton-model", *args, **kwargs):
        """_summary_

        Parameters
        ----------
        params : dict
            Contains the configuration parameters for the module.
        name : str, optional
            Name for the neural network, by default "skeleton-model"
        """        
        kwargs["name"] = name
        super().__init__(*args, **kwargs)
        self.configuration = configuration
        # self.configuration["normalization"] = [
        #     keras.ops.ones((1,)),  #0 inputs mean
        #     keras.ops.ones((1,)),  #1 inputs std
        #     keras.ops.ones((1,)),  #2 bones mean
        #     keras.ops.ones((1,))   #3 bones std
        # ]
        
    def forward_denormalized(self, inputs):
        outputs = self.call(inputs)
        return self._denormalize_data(outputs)
    
    def set_normalization_parameters(self, parameters):
        self.configuration["normalization"] = [keras.ops.convert_to_tensor(parameters[i]) for i in range(len(parameters))] 

    def load_normalization(self):
        self.configuration["normalization"] = [keras.ops.convert_to_tensor(conf['config']['value']) for conf in self.configuration["normalization"]] 
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        model = super().from_config(config, custom_objects)
        model.load_normalization()
        return model
    
    def _normalize_data(self, data):
        mean = kops.ones(data.shape, dtype='float32') * self.configuration["normalization"][2]
        std = kops.ones(data.shape, dtype='float32') * self.configuration["normalization"][3]
        normalized_data = kops.divide(data - mean, std)
        return normalized_data

    def _denormalize_data(self, data):
        D = self.configuration["normalization"][2].shape[0]  # Dimensionality
        shape = [1 for _ in range(len(data.shape)-1)] + [D]
        repeat = list(data.shape[:-1]) + [1]
        std = kops.tile(
            kops.reshape(self.configuration["normalization"][3], shape),
            repeat
        )
        mean = kops.tile(
            kops.reshape(self.configuration["normalization"][2], shape),
            repeat
        )
        return kops.multiply(data, std) + mean
    
    def get_normalization_parameters(self):
        return self.configuration["normalization"]
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "configuration": self.configuration
        }
        return {**base_config, **config}

@keras.saving.register_keras_serializable()
class SkeletonModel(BaseSkeletonModel):
    """ Neural network model that estimates bone lengths from a sequence of poses.
    """    
    def __init__(self, configuration, name="skeleton-model", *args, **kwargs):
        """_summary_

        Parameters
        ----------
        params : dict
            Contains the configuration parameters for the module.
        name : str, optional
            Name for the neural network, by default "skeleton-model"
        """        
        kwargs["name"] = name
        super().__init__(configuration, *args, **kwargs)
        if self.configuration:
            channels = self.configuration["model"]["arch"]["units"]
        else:
            channels = 32
        self.features_out = 10
        dropout_rate = 0.2
        
        # spatial encoder
        self.expand_channels = keras.layers.Conv1D(channels, 1, 1, padding="same", activation="relu")
        self.sp_enc_block1 = layers.ResidualConvolutionBlock(channels, 1, dropout_rate=dropout_rate)
        self.sp_enc_block2 = layers.ResidualConvolutionBlock(channels, 1, dropout_rate=dropout_rate)

        # Temporal encoder
        self.tp_enc_block1 = layers.ResidualConvolutionBlock(channels, 3, dropout_rate=dropout_rate)
        self.tp_enc_block2 = layers.ResidualConvolutionBlock(channels, 3, dropout_rate=dropout_rate)
        
        # Regression block
        self.reg_norm = keras.layers.BatchNormalization()
        self.reg_pool = keras.layers.GlobalAveragePooling1D(keepdims=True)
        self.reg_out = keras.layers.Conv1D(self.features_out, 1)


    def call(self, inputs):
        outputs = keras.ops.reshape(inputs, (*inputs.shape[:-2], -1))
        outputs = self.expand_channels(outputs)
        outputs = self.sp_enc_block1(outputs)
        outputs = self.sp_enc_block2(outputs)
        outputs = self.tp_enc_block1(outputs)
        outputs = self.tp_enc_block2(outputs)
        outputs = self.reg_norm(outputs)
        outputs = self.reg_pool(outputs)
        outputs = self.reg_out(outputs)
        return outputs

@keras.saving.register_keras_serializable()
class SkeletonGraphModel(BaseSkeletonModel):
    """ Neural network model that estimates bone lengths from a sequence of poses.
    """    
    def __init__(self, configuration, name="skeleton-model", *args, **kwargs):
        """_summary_

        Parameters
        ----------
        params : dict
            Contains the configuration parameters for the module.
        name : str, optional
            Name for the neural network, by default "skeleton-model"
        """        
        kwargs["name"] = name
        super().__init__(configuration, *args, **kwargs)
        if self.configuration:
            channels = self.configuration["model"]["arch"]["units"]
        else:
            channels = 32
        self.features_out = 10
        dropout_rate = 0.2
        
        # spatial encoder
        self.bones_enc = layers.PosesToBones()
        self.expand_channels = keras.layers.Dense(channels, activation='relu')
        self.sp_enc_block1 = layers.ResidualDenseBlock(channels, dropout_rate=dropout_rate)
        self.sp_enc_block2 = layers.ResidualDenseBlock(channels, dropout_rate=dropout_rate)

        # Temporal encoder
        self.tp_enc_block1 = layers.ResidualConvolutionBlock(channels, 5, dropout_rate=dropout_rate)
        self.tp_enc_block2 = layers.ResidualConvolutionBlock(channels, 3, dropout_rate=dropout_rate)
        
        # Regression block
        self.reg_norm = keras.layers.LayerNormalization()
        self.reg_pool = keras.layers.GlobalAveragePooling1D(keepdims=True)
        self.reg_out = keras.layers.Dense(self.features_out)

    def call(self, inputs):
        outputs = self.bones_enc(inputs)
        outputs = self.expand_channels(outputs)
        outputs = self.sp_enc_block1(outputs)
        outputs = self.sp_enc_block2(outputs)
        outputs = self.tp_enc_block1(outputs)
        outputs = self.tp_enc_block2(outputs)
        outputs = self.reg_norm(outputs)
        outputs = self.reg_pool(outputs)
        outputs = self.reg_out(outputs)
        return outputs