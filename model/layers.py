#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
# %% Imports
import tensorflow as tf
import keras 
from model.graph.skeleton import SkeletonGraph
from model.graph.laplacian import create_matrix_L, create_matrix_D
from model import ops, solvers


# %% Simple functions
def zero(x):
    return 0 * x


def identity(x):
    return x


# %% Custom Layers
# @tf.keras.saving.register_keras_serializable()
class DeltaConverter(keras.layers.Layer):
    def __init__(self, skel : SkeletonGraph, features, t, l_mat=None, sw=1.0, tw=1.0, **kwargs):
        super(DeltaConverter, self).__init__(**kwargs)
        self.features = features
        self.t = t
        self.v = skel.get_num_of_joints()
        if l_mat is None:
            self.l_mat = tf.expand_dims(create_matrix_L(skel, t, sw, tw), 0)
        else:
            self.l_mat = l_mat

    def call(self, inputs, *args, **kwargs):
        """
        Computes Δ from 3D joint positions.

        Args:
            inputs (Tensor): Represents the 3D joint positions.
                It is a 3+D tensor with shape `(B, T, V*C)`.
            kwargs (dict): A dictionary of others parameters

        Returns:
            outputs (Tensor): A 4+D tensor with shape `(B, T, W*V, C)`.
        """
        outputs = ops.format_inputs(inputs, self.t)
        outputs = self.l_mat @ outputs
        if "keepdims" in kwargs and kwargs["keepdims"] is True:
            center = self.t // 2
            outputs = outputs[:, :, center * self.v:center * self.v + self.v, :]
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


# @tf.keras.saving.register_keras_serializable()
class PoseSolver(keras.layers.Layer):
    def __init__(self, skel : SkeletonGraph, features, t, constraints, l_mat=None, sw=1.0, tw=1.0, **kwargs):
        super(PoseSolver, self).__init__(**kwargs)
        self.features = features
        self.t = t
        self.v = skel.get_num_of_joints()
        self.lgs = solvers.LaplacianGraphSolver(skel, self.t, constraints)
        self.smoother = keras.layers.AveragePooling2D(pool_size=(3, 1), strides=1, padding="same")

    def call(self, inputs, gamma=None, *args, **kwargs):
        """
        Converts Δ to P using Cholesky resolution.

        Args:
            inputs (Tensor): Represents the 3D joint differential coordinates Δ.
                It is a 4+D tensor with shape `(B, T, W*V, C)`.
            gamma (Tensor): A 4+D tensor representing the constraints on Es segments.
            *args:
            **kwargs:

        Returns:
            outputs (Tensor): A 4+D tensor with shape `(B, T, V, C)`.
        """
        outputs = self.lgs.solve(inputs, gamma)
        # retrieve central frames
        center = self.t // 2
        outputs = outputs[:, :, center * self.v:center * self.v + self.v, :]
        outputs = self.smoother(outputs)
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


# %% Graph convolution layer
# @tf.keras.saving.register_keras_serializable()
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
        self.activation = tf.keras.activations.get(activation)

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

        outputs = tf.matmul(inputs, self.kernel)
        if self.use_mapper:
            outputs = tf.matmul(outputs, self.mapper)

        outputs = tf.matmul(self.A, outputs)
        outputs = self.activation(outputs)

        if self.use_bias:
            outputs = outputs + self.bias

        if self.residual:
            outputs = outputs + inputs
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "use_mapper": self.use_mapper,
                "residual": self.residual
            }
        )
        return config


# @tf.keras.saving.register_keras_serializable()
class SkeletonGraphConv(tf.keras.layers.Layer):
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
        self.kernel = None
        self.tuner = None
        self.bias = None
        self.activation = tf.keras.activations.get(activation)

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

        outputs = tf.matmul(inputs, self.kernel)
        outputs = tf.matmul(outputs, self.tuner)
        outputs = tf.matmul(self.D, outputs)
        outputs = self.activation(outputs)

        if self.use_bias:
            outputs = outputs + self.bias

        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "activation": tf.keras.activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "joints": self.joints
            }
        )
        return config


class SkeletonSymmetrisationLayer(tf.keras.layers.Layer):

    def __init__(self, op_type="avg", **kwargs):
        super().__init__(**kwargs)
        self.op_type = op_type
        if op_type == "max":
            self.unifier = tf.keras.layers.Maximum()
        elif op_type == "min":
            self.unifier = tf.keras.layers.Minimum()
        else:
            self.unifier = tf.keras.layers.Average()

    def call(self, inputs, *args, **kwargs):
        hips = self.unifier([inputs[:, :, 0:1], inputs[:, :, 3:4]])
        femur = self.unifier([inputs[:, :, 1:2], inputs[:, :, 4:5]])
        tibia = self.unifier([inputs[:, :, 2:3], inputs[:, :, 5:6]])
        spine_back = inputs[:, :, 6:7]
        spine_top = inputs[:, :, 7:8]
        neck = inputs[:, :, 8:9]
        head = inputs[:, :, 9:10]
        clavicle = self.unifier([inputs[:, :, 10:11], inputs[:, :, 13:14]])
        humerus = self.unifier([inputs[:, :, 11:12], inputs[:, :, 14:15]])
        radius = self.unifier([inputs[:, :, 12:13], inputs[:, :, 15:16]])
        outputs = tf.concat((
            hips,
            femur,
            tibia,
            hips,
            femur,
            tibia,
            spine_back,
            spine_top,
            neck,
            head,
            clavicle,
            humerus,
            radius,
            clavicle,
            humerus,
            radius
        ),
            axis=-1)

        return outputs


class PosesToSkeletonLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        shape = tf.shape(inputs)
        batch, length, joints, _ = shape
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

        skeleton = tf.concat((hips_right,
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

        skeleton = tf.reshape(skeleton, [batch, length, joints - 1, -1])
        return skeleton


# @tf.keras.saving.register_keras_serializable()
class SkeletonAdjustementLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        batch, length, joints, _ = tf.shape(inputs)
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

        hips_norm = (tf.norm(hips_left, axis=-1, keepdims=True) + tf.norm(hips_right, axis=-1, keepdims=True)) / 2
        hips_norm = tf.reduce_max(hips_norm, axis=1, keepdims=True)
        femur_norm = (tf.norm(femur_left, axis=-1, keepdims=True) + tf.norm(femur_right, axis=-1, keepdims=True)) / 2
        femur_norm = tf.reduce_max(femur_norm, axis=1, keepdims=True)
        tibia_norm = (tf.norm(tibia_left, axis=-1, keepdims=True) + tf.norm(tibia_right, axis=-1, keepdims=True)) / 2
        tibia_norm = tf.reduce_max(tibia_norm, axis=1, keepdims=True)
        clavicle_norm = (tf.norm(clavicle_left, axis=-1, keepdims=True) + tf.norm(clavicle_right, axis=-1,
                                                                                  keepdims=True)) / 2
        clavicle_norm = tf.reduce_max(clavicle_norm, axis=1, keepdims=True)
        humerus_norm = (tf.norm(humerus_left, axis=-1, keepdims=True) + tf.norm(humerus_right, axis=-1,
                                                                                keepdims=True)) / 2
        humerus_norm = tf.reduce_max(humerus_norm, axis=1, keepdims=True)
        radius_norm = (tf.norm(radius_left, axis=-1, keepdims=True) + tf.norm(radius_right, axis=-1, keepdims=True)) / 2
        radius_norm = tf.reduce_max(radius_norm, axis=1, keepdims=True)

        spine_back_norm = tf.reduce_max(tf.norm(spine_back, axis=-1, keepdims=True), axis=1, keepdims=True)
        spine_top_norm = tf.reduce_max(tf.norm(spine_top, axis=-1, keepdims=True), axis=1, keepdims=True)
        neck_norm = tf.reduce_max(tf.norm(neck, axis=-1, keepdims=True), axis=1, keepdims=True)
        head_norm = tf.reduce_max(tf.norm(head, axis=-1, keepdims=True), axis=1, keepdims=True)

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

        skeleton = tf.concat((hips_right,
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

        skeleton = tf.reshape(skeleton, [batch, length, joints - 1, -1])
        return skeleton

    def get_config(self):
        config = super().get_config()
        return config


# %% Sampling layer
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        shape = tf.shape(z_mean)
        if len(shape) == 2:
            batch = shape[0]
            dim = shape[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        elif len(shape) == 3:
            batch = shape[0]
            steps = shape[1]
            dim = shape[2]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            epsilon = tf.expand_dims(epsilon, axis=1)
            epsilon = tf.tile(epsilon, [1, steps, 1])
        else:
            raise ValueError('Unsupported Case!')
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Projection(tf.keras.layers.Layer):
    def __init__(self, direction, **kwargs):
        super().__init__(**kwargs)
        self.direction = direction

    def call(self, inputs, *args, **kwargs):
        output = tf.einsum('b t j c, ...c -> b t j', inputs, self.direction) / tf.norm(self.direction)
        return tf.expand_dims(output, axis=-1)


# @tf.keras.saving.register_keras_serializable()
class Residual(tf.keras.layers.Layer):
    def __init__(self, op, **kwargs):
        super().__init__(**kwargs)
        self.op = op

    def call(self, inputs, *args, **kwargs):
        return self.op(inputs) + inputs

    def get_config(self):
        config = super().get_config()
        return config


# @tf.keras.saving.register_keras_serializable()
class SkeletonSimplificationLayer(tf.keras.layers.Layer):

    def __init__(self, op_type="avg", **kwargs):
        super().__init__(**kwargs)
        self.op_type = op_type
        if op_type == "max":
            self.unifier = tf.keras.layers.Maximum()
        elif op_type == "min":
            self.unifier = tf.keras.layers.Minimum()
        else:
            self.unifier = tf.keras.layers.Average()

    def call(self, inputs, *args, **kwargs):
        hips = self.unifier([inputs[:, :, 0:1], inputs[:, :, 3:4]])
        femur = self.unifier([inputs[:, :, 1:2], inputs[:, :, 4:5]])
        tibia = self.unifier([inputs[:, :, 2:3], inputs[:, :, 5:6]])
        spine_back = inputs[:, :, 6:7]
        spine_top = inputs[:, :, 7:8]
        neck = inputs[:, :, 8:9]
        head = inputs[:, :, 9:10]
        clavicle = self.unifier([inputs[:, :, 10:11], inputs[:, :, 13:14]])
        humerus = self.unifier([inputs[:, :, 11:12], inputs[:, :, 14:15]])
        radius = self.unifier([inputs[:, :, 12:13], inputs[:, :, 15:16]])
        outputs = tf.concat((
            hips,
            femur,
            tibia,
            spine_back,
            spine_top,
            neck,
            head,
            clavicle,
            humerus,
            radius
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


# @tf.keras.saving.register_keras_serializable()
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, strides=1,
                 padding="SAME", normalize=False, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(channels, kernel_size, strides, padding)
        if normalize:
            self.norm = tf.keras.layers.BatchNormalization()
        else:
            self.norm = None
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.01)
        # self.activation = tf.keras.activations.relu
        if dropout_rate:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None

        self.channels = channels
        self.dropout_rate = dropout_rate
        self.normalise = normalize
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs, *args, **kwargs):
        outputs = self.activation(self.conv(inputs))
        if self.norm:
            outputs = self.norm(outputs)
        if self.dropout:
            outputs = self.dropout(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "normalise": self.normalise,
                "dropout_rate": self.dropout_rate,
                "strides": self.strides,
                "padding": self.padding,
                "activation": tf.keras.activations.serialize(self.activation),
            }
        )
        return config


# @tf.keras.saving.register_keras_serializable()
class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, normalize=False, dropout_rate=None, **kwargs):
        super().__init__(**kwargs)
        self.linear = tf.keras.layers.Dense(units)
        if normalize:
            self.norm = tf.keras.layers.BatchNormalization()
        else:
            self.norm = None
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.01)
        # self.activation = tf.keras.activations.relu
        if dropout_rate:
            self.dropout = tf.keras.layers.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.units = units
        self.dropout_rate = dropout_rate
        self.normalise = normalize

    def call(self, inputs, *args, **kwargs):
        outputs = self.activation(self.linear(inputs))
        if self.norm:
            outputs = self.norm(outputs)
        if self.dropout:
            outputs = self.dropout(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "normalise": self.normalise,
                "dropout_rate": self.dropout_rate,
                "activation": tf.keras.activations.serialize(self.activation),
            }
        )
        return config


# @tf.keras.saving.register_keras_serializable()
class AdaptiveAvgPool(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        if size == 1:
            self.pool = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)
        else:
            # self.pool = tf.nn.avg_pool1d()
            self.pool = tf.keras.layers.AveragePooling1D(size)

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


class AdaptiveMaxPool(tf.keras.layers.Layer):
    def __init__(self, size, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        if size == 1:
            self.pool = tf.keras.layers.GlobalMaxPool1D(keepdims=True)
        else:
            # self.pool = tf.nn.avg_pool1d()
            self.pool = tf.keras.layers.MaxPool1D(size)

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


class GraphFeedForward(tf.keras.layers.Layer):
    def __init__(self, adj, dff, dim_out, dropout_rate=0.1,
                 activation='linear', **kwargs):
        super().__init__(**kwargs)
        if activation == 'relu':
            self.activation = tf.keras.activations.relu
        elif activation == 'swish':
            self.activation = tf.keras.activations.swish
        else:
            self.activation = tf.keras.activations.linear
        # self.seq = tf.keras.Sequential([
        #     GraphConv(dff,
        #               adj,
        #               activation=self.activation,
        #               name=f"{self.name}.hidden_graph_conv"),
        #     tf.keras.layers.Dropout(dropout_rate),
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
        self.drop = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()


def call(self, inputs, *args, **kwargs):
    x = self.add([inputs, self.out(self.drop(self.hidden(inputs)))])
    x = self.layer_norm(x)
    return x
