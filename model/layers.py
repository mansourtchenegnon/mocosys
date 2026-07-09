#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
# %% Imports
import keras 
from model.graph.skeleton import SkeletonGraph
from model.graph.laplacian import create_matrix_L, create_matrix_D
from model import ops, solvers

SEED = 97

# %% Simple functions
def zero(x):
    return 0 * x


def identity(x):
    return x


# %% Custom Layers
@keras.saving.register_keras_serializable()
class DeltaConverter(keras.layers.Layer):
    def __init__(self, skel : SkeletonGraph, features, t, l_mat=None, sw=1.0, tw=1.0, **kwargs):
        super(DeltaConverter, self).__init__(**kwargs)
        self.features = features
        self.t = t
        self.v = skel.get_num_of_joints()
        if l_mat is None:
            self.l_mat = keras.ops.expand_dims(create_matrix_L(skel, t, sw, tw), 0)
        else:
            self.l_mat = l_mat

    def call(self, inputs, *args, **kwargs):
        """
        Computes Δ from 3D joint positions.

        Args:
            inputs (Tensor): Represents the 3D joint positions.
                It is a 3+D tensor with shape `(..., T, V, C)`.
            kwargs (dict): A dictionary of others parameters

        Returns:
            outputs (Tensor): A 3+D tensor with shape `(..., T, W*V, C)` or `(..., T, V, C)`.
        """
        if "format" in kwargs and kwargs["format"] is True:
            outputs = ops.format_inputs(inputs, self.t)
        else:
            outputs = inputs
        outputs = self.l_mat @ outputs
        if "keepdims" in kwargs and kwargs["keepdims"] is True:
            center = self.t // 2
            outputs = outputs[..., center * self.v:center * self.v + self.v, :]
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "t": self.t,
                "v": self.v,
                "features": self.features,
            }
        )
        return config

@keras.saving.register_keras_serializable()
class PoseSolver(keras.layers.Layer):
    def __init__(self, skel : SkeletonGraph, features, t, constraints, l_mat=None, sw=1.0, tw=1.0, **kwargs):
        super(PoseSolver, self).__init__(**kwargs)
        self.features = features
        self.t = t
        self.v = skel.get_num_of_joints()
        self.lgs = solvers.LaplacianGraphSolver(skel, self.t, constraints)
        self.smoother = keras.layers.AveragePooling2D(pool_size=(3, 1), strides=1, padding="same")

    def call(self, inputs, *args, **kwargs):
        """
        Converts `Δ` to `P` using Cholesky resolution.

        Args:
            inputs (Tensor): Represents the 3D joint differential coordinates Δ. It is a 4+D tensor with shape `(B, T, W*V, C)`.
            gamma (Tensor): A 4+D tensor representing the constraints on Es segments.
            *args:
            **kwargs:

        Returns:
            outputs (Tensor): A 4+D tensor with shape `(B, T, V, C)`.
        """
        if "format" in kwargs:
            outputs = self.lgs.solve(inputs, format=kwargs["format"])
        else:
            outputs = self.lgs.solve(inputs, format=True)

        # retrieve central frames
        center = self.t // 2
        outputs = outputs[..., center * self.v:center * self.v + self.v, :]
        outputs = self.smoother(outputs)
        return outputs
    
    def set_constraints(self, vec_u, vec_gamma):
        self.lgs.set_positional_constraints(vec_u)
        self.lgs.set_distance_constraints(vec_gamma)

    def get_config(self):
        base_config = super().get_config()
        config = {
                "t": self.t,
                "v": self.v,
                "features": self.features,
            }
        return {**base_config, **config}


# %% Graph convolution layer
@keras.saving.register_keras_serializable()
class GraphConv(keras.layers.Layer):
    def __init__(
            self,
            channels,
            adjacency_matrix,
            use_bias=True,
            residual = False,
            trainable=True,
            use_mapper=False,
            activation=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.A = adjacency_matrix
        self.channels = channels
        self.use_bias = use_bias
        self.use_mapper = use_mapper
        self.residual = residual
        self.trainable = trainable
        self.kernel = None
        self.bias = None
        self.mapper = None
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_shape[-1], self.channels),
            initializer="random_normal",
            trainable=True,
        )
        if self.use_mapper:
            self.mapper = self.add_weight(
                name="element_wise_kernel",
                shape=(self.channels, self.channels),
                initializer="random_normal",
                trainable=True,
            )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.channels,),
                initializer="random_normal",
                trainable=True,
            )
        if self.residual:
            if self.channels != input_shape[-1]:
                self.residual = False

    def call(self, inputs, *args, **kwargs):
        assert (
                self.A.shape[-1] == inputs.shape[-2]
        ), "Shapes don't match between A matrix {} and inputs {}".format(
            self.A.shape, inputs.shape
        )

        outputs = keras.ops.matmul(inputs, self.kernel)
        if self.use_mapper:
            outputs = keras.ops.matmul(outputs, self.mapper)

        outputs = keras.ops.matmul(self.A, outputs)
        outputs = self.activation(outputs)

        if self.use_bias:
            outputs = outputs + self.bias

        if self.residual:
            outputs = outputs + inputs
        return outputs

    def get_config(self):
        base_config = super().get_config()
        config = {
                "channels": self.channels,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "use_mapper": self.use_mapper,
                "residual": self.residual,
                "trainable": self.trainable
            }
        return {**base_config, **config}


@keras.saving.register_keras_serializable()
class SkeletonGraphConv(keras.layers.Layer):
    def __init__(
            self,
            skel : SkeletonGraph,
            channels_out,
            channels,
            use_bias=True,
            trainable=True,
            activation=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        # self.skeleton = skel
        self.joints = skel.get_num_of_joints()
        self.D = create_matrix_D(skel, 1)
        self.channels = channels
        self.channels_out = channels_out
        self.use_bias = use_bias
        self.trainable = trainable
        self.kernel = None
        self.tuner = None
        self.bias = None
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.channels),
            name="kernel",
            initializer="random_normal",
            trainable=True,
        )
        self.tuner = self.add_weight(
            name="tuner",
            shape=(self.channels, self.channels_out),
            initializer="random_normal",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.channels_out,),
                name="bias",
                initializer="random_normal",
                trainable=True,
            )
        

    def call(self, inputs, *args, **kwargs):
        assert (
                self.D.shape[-1] == inputs.shape[-2]
        ), "Shapes don't match between D matrix {} and inputs {}".format(
            self.D.shape, inputs.shape
        )

        outputs = keras.ops.matmul(inputs, self.kernel)
        outputs = keras.ops.matmul(outputs, self.tuner)
        outputs = keras.ops.matmul(self.D, outputs)
        outputs = self.activation(outputs)

        if self.use_bias:
            outputs = outputs + self.bias

        return outputs

    def get_config(self):
        base_config = super().get_config()
        config = {
                "channels": self.channels,
                "activation": keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "joints": self.joints,
                "trainable": self.trainable
            }
        return {**base_config, **config}


@keras.saving.register_keras_serializable()
class SkeletonSymmetrisationLayer(keras.layers.Layer):

    def __init__(self, op_type="avg", **kwargs):
        super().__init__(**kwargs)
        self.op_type = op_type
        if op_type == "max":
            self.unifier = keras.layers.Maximum()
        elif op_type == "min":
            self.unifier = keras.layers.Minimum()
        else:
            self.unifier = keras.layers.Average()

    def call(self, inputs, *args, **kwargs):
        hips = self.unifier([inputs[..., 0:1], inputs[..., 3:4]])
        femur = self.unifier([inputs[..., 1:2], inputs[..., 4:5]])
        tibia = self.unifier([inputs[..., 2:3], inputs[..., 5:6]])
        spine_back = inputs[..., 6:7]
        spine_top = inputs[..., 7:8]
        neck = inputs[..., 8:9]
        head = inputs[..., 9:10]
        clavicle = self.unifier([inputs[..., 10:11], inputs[..., 13:14]])
        humerus = self.unifier([inputs[..., 11:12], inputs[..., 14:15]])
        radius = self.unifier([inputs[..., 12:13], inputs[..., 15:16]])
        outputs = keras.ops.concatenate((
            hips, femur, tibia,
            hips, femur, tibia,
            spine_back, spine_top, neck, head,
            clavicle, humerus, radius,
            clavicle, humerus, radius
        ),
            axis=-1)

        return outputs
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "unifier" : self.unifier
        }
        return {**base_config, **config}


class PosesToSkeletonLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        shape = keras.ops.shape(inputs)
        batch, length, joints, _ = shape
        hips_right = inputs[..., 0, :] - inputs[..., 1, :]
        femur_right = inputs[..., 1, :] - inputs[..., 2, :]
        tibia_right = inputs[..., 2, :] - inputs[..., 3, :]

        hips_left = inputs[..., 0, :] - inputs[..., 4, :]
        femur_left = inputs[..., 4, :] - inputs[..., 5, :]
        tibia_left = inputs[..., 5, :] - inputs[..., 6, :]

        spine_back = inputs[..., 0, :] - inputs[..., 7, :]
        spine_top = inputs[..., 7, :] - inputs[..., 8, :]
        neck = inputs[..., 8, :] - inputs[..., 9, :]
        head = inputs[..., 9, :] - inputs[..., 10, :]

        clavicle_left = inputs[..., 8, :] - inputs[..., 11, :]
        humerus_left = inputs[..., 11, :] - inputs[..., 12, :]
        radius_left = inputs[..., 12, :] - inputs[..., 13, :]

        clavicle_right = inputs[..., 8, :] - inputs[..., 14, :]
        humerus_right = inputs[..., 14, :] - inputs[..., 15, :]
        radius_right = inputs[..., 15, :] - inputs[..., 16, :]

        skeleton = keras.ops.concatenate((
            hips_right, femur_right, tibia_right,
            hips_left, femur_left, tibia_left,
            spine_back, spine_top, neck, head,
            clavicle_left, humerus_left, radius_left,
            clavicle_right, humerus_right, radius_right
            ), axis=-1)

        skeleton = keras.ops.reshape(skeleton, [batch, length, joints - 1, -1])
        return skeleton


class SkeletonAdjustementLayer(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        batch, length, joints, _ = keras.ops.shape(inputs)
        hips_right = inputs[..., 0, :] - inputs[..., 1, :]
        femur_right = inputs[..., 1, :] - inputs[..., 2, :]
        tibia_right = inputs[..., 2, :] - inputs[..., 3, :]

        hips_left = inputs[..., 0, :] - inputs[..., 4, :]
        femur_left = inputs[..., 4, :] - inputs[..., 5, :]
        tibia_left = inputs[..., 5, :] - inputs[..., 6, :]

        spine_back = inputs[..., 0, :] - inputs[..., 7, :]
        spine_top = inputs[..., 7, :] - inputs[..., 8, :]
        neck = inputs[..., 8, :] - inputs[..., 9, :]
        head = inputs[..., 9, :] - inputs[..., 10, :]

        clavicle_left = inputs[..., 8, :] - inputs[..., 11, :]
        humerus_left = inputs[..., 11, :] - inputs[..., 12, :]
        radius_left = inputs[..., 12, :] - inputs[..., 13, :]

        clavicle_right = inputs[..., 8, :] - inputs[..., 14, :]
        humerus_right = inputs[..., 14, :] - inputs[..., 15, :]
        radius_right = inputs[..., 15, :] - inputs[..., 16, :]

        hips_norm = (keras.ops.norm(hips_left, axis=-1, keepdims=True) + keras.ops.norm(hips_right, axis=-1, keepdims=True)) / 2
        hips_norm = keras.ops.max(hips_norm, axis=1, keepdims=True)
        femur_norm = (keras.ops.norm(femur_left, axis=-1, keepdims=True) + keras.ops.norm(femur_right, axis=-1, keepdims=True)) / 2
        femur_norm = keras.ops.max(femur_norm, axis=1, keepdims=True)
        tibia_norm = (keras.ops.norm(tibia_left, axis=-1, keepdims=True) + keras.ops.norm(tibia_right, axis=-1, keepdims=True)) / 2
        tibia_norm = keras.ops.max(tibia_norm, axis=1, keepdims=True)
        clavicle_norm = (keras.ops.norm(clavicle_left, axis=-1, keepdims=True) + keras.ops.norm(clavicle_right, axis=-1,
                                                                                  keepdims=True)) / 2
        clavicle_norm = keras.ops.max(clavicle_norm, axis=1, keepdims=True)
        humerus_norm = (keras.ops.norm(humerus_left, axis=-1, keepdims=True) + keras.ops.norm(humerus_right, axis=-1,
                                                                                keepdims=True)) / 2
        humerus_norm = keras.ops.max(humerus_norm, axis=1, keepdims=True)
        radius_norm = (keras.ops.norm(radius_left, axis=-1, keepdims=True) + keras.ops.norm(radius_right, axis=-1, keepdims=True)) / 2
        radius_norm = keras.ops.max(radius_norm, axis=1, keepdims=True)

        spine_back_norm = keras.ops.max(keras.ops.norm(spine_back, axis=-1, keepdims=True), axis=1, keepdims=True)
        spine_top_norm = keras.ops.max(keras.ops.norm(spine_top, axis=-1, keepdims=True), axis=1, keepdims=True)
        neck_norm = keras.ops.max(keras.ops.norm(neck, axis=-1, keepdims=True), axis=1, keepdims=True)
        head_norm = keras.ops.max(keras.ops.norm(head, axis=-1, keepdims=True), axis=1, keepdims=True)

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

        skeleton = keras.ops.concatenate((
            hips_right, femur_right, tibia_right,
            hips_left, femur_left, tibia_left,
            spine_back, spine_top, neck, head,
            clavicle_left, humerus_left, radius_left,
            clavicle_right, humerus_right, radius_right
            ), axis=-1)

        skeleton = keras.ops.reshape(skeleton, [batch, length, joints - 1, -1])
        return skeleton

    def get_config(self):
        return super().get_config()


# %% Sampling layer
class Sampling(keras.layers.Layer):
    """A sampling layer that uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, seed=SEED, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = keras.ops.shape(z_mean)[0]
        steps = keras.ops.shape(z_mean)[1]
        dim = keras.ops.shape(z_mean)[2]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed)
        epsilon = keras.ops.expand_dims(epsilon, axis=1)
        epsilon = keras.ops.tile(epsilon, [1, steps, 1])
        return z_mean + keras.ops.exp(0.5 * z_log_var) * epsilon
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "seed": self.seed
        }
        return {**base_config, **config}



class Projection(keras.layers.Layer):
    def __init__(self, direction, **kwargs):
        super().__init__(**kwargs)
        self.direction = direction

    def call(self, inputs, *args, **kwargs):
        output = keras.ops.einsum('b t j c, ...c -> b t j', inputs, self.direction) / keras.ops.norm(self.direction)
        return keras.ops.expand_dims(output, axis=-1)


def Residual(inputs, operation):
    x = operation(inputs)
    x = keras.layers.Add()([inputs, x])
    return x


class SkeletonSimplificationLayer(keras.layers.Layer):

    def __init__(self, op_type="avg", **kwargs):
        super().__init__(**kwargs)
        self.op_type = op_type
        if op_type == "max":
            self.unifier = keras.layers.Maximum()
        elif op_type == "min":
            self.unifier = keras.layers.Minimum()
        else:
            self.unifier = keras.layers.Average()

    def call(self, inputs, *args, **kwargs):
        hips = self.unifier([inputs[..., 0:1], inputs[..., 3:4]])
        femur = self.unifier([inputs[..., 1:2], inputs[..., 4:5]])
        tibia = self.unifier([inputs[..., 2:3], inputs[..., 5:6]])
        spine_back = inputs[..., 6:7]
        spine_top = inputs[..., 7:8]
        neck = inputs[..., 8:9]
        head = inputs[..., 9:10]
        clavicle = self.unifier([inputs[..., 10:11], inputs[..., 13:14]])
        humerus = self.unifier([inputs[..., 11:12], inputs[..., 14:15]])
        radius = self.unifier([inputs[..., 12:13], inputs[..., 15:16]])
        outputs = keras.ops.concatenate((
            hips, femur, tibia,
            spine_back, spine_top, neck, head,
            clavicle, humerus, radius
        ),
            axis=-1)

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "op_type": self.op_type,
            }
        )
        return config


def ResidualConvolutionBlock(inputs, channels, kernel, dropout_rate=None, padding="SAME", normalize=False,):
    x = keras.layers.Conv1D(channels, kernel, 1, padding=padding)(inputs)
    if dropout_rate:
        x = keras.layers.Dropout(rate=dropout_rate)(x)
    x = keras.layers.LeakyReLU()(x)
    if normalize:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([inputs, x])
    return x

def ResidualDenseBlock(inputs, units, kernel, dropout_rate=None, padding="SAME", normalize=False,):
    x = keras.layers.Dense(units)(inputs)
    if dropout_rate:
        x = keras.layers.Dropout(rate=dropout_rate)(x)
    x = keras.layers.LeakyReLU()(x)
    if normalize:
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Add()([inputs, x])
    return x

@keras.saving.register_keras_serializable()
class AdaptiveAvgPool(keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        if size == 1:
            self.pool = keras.layers.GlobalAveragePooling1D(keepdims=True)
        else:
            self.pool = keras.layers.AveragePooling1D(size)

    def call(self, inputs, *args, **kwargs):
        outputs = self.pool(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size
            }
        )
        return config


class AdaptiveMaxPool(keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        if size == 1:
            self.pool = keras.layers.GlobalMaxPool1D(keepdims=True)
        else:
            self.pool = keras.layers.MaxPool1D(size)

    def call(self, inputs, *args, **kwargs):
        outputs = self.pool(inputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size
            }
        )
        return config


class GraphFeedForward(keras.layers.Layer):
    def __init__(self, adj, dff, dim_out, dropout_rate=0.1,
                 activation='linear', **kwargs):
        super().__init__(**kwargs)
        if activation == 'relu':
            self.activation = keras.activations.relu
        elif activation == 'swish':
            self.activation = keras.activations.swish
        else:
            self.activation = keras.activations.linear
        # self.seq = keras.Sequential([
        #     GraphConv(dff,
        #               adj,
        #               activation=self.activation,
        #               name=f"{self.name}.hidden_graph_conv"),
        #     keras.layers.Dropout(dropout_rate),
        #     GraphConv(dim_out,
        #               adj,
        #               activation=self.activation,
        #               name=f"{self.name}.graph_conv_out")
        # ])
        self.hidden = GraphConv(dff,
                      adj,
                      activation=self.activation,
                      name=f"{self.name}.hidden_graph_conv")
        self.out = GraphConv(dim_out,
                      adj,
                      activation=self.activation,
                      name=f"{self.name}.graph_conv_out")
        self.drop = keras.layers.Dropout(dropout_rate)
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()


def call(self, inputs, *args, **kwargs):
    x = self.add([inputs, self.out(self.drop(self.hidden(inputs)))])
    x = self.layer_norm(x)
    return x
