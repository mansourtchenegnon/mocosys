#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""
from math import pi
# %% Imports
from random import randrange

import tensorflow as tf
from einops import rearrange, repeat
from einops.layers.tensorflow import Rearrange, Reduce


# %% Functions
def exists(val):
    return val is not None


def pair(val):
    return (val, val) if not isinstance(val, tuple) else val


# TODO: Learn how to use it in tensorflow
def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = tf.greater(tf.random.uniform(num_layers), prob_survival)

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers


def shift(t, amount, mask=None):
    if amount == 0:
        return t
    return tf.pad(t, (0, 0, amount, -amount), constant_values=0.)


# %% Helper classes
class Identity(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return inputs


class Residual(tf.keras.layers.Layer):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn

    def call(self, inputs, *args, **kwargs):
        return self.fn(inputs) + inputs


class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        return (inputs / 2.) * (
                1 + tf.math.tanh(tf.math.sqrt(2. / tf.constant(pi)) * (inputs + 0.044715 * tf.math.pow(inputs, 3))
                                 ))


class PreShiftTokens(tf.keras.layers.Layer):
    def __init__(self, shifts, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.shifts = tuple(shifts)

    def call(self, inputs, **kwargs):
        if self.shifts == (0,):
            return self.fn(inputs, **kwargs)

        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = inputs.shape[-1] // segments
        split = tf.split(inputs, feats_per_shift, axis=-1)
        segments_to_shift, rest = split[:segments], split[segments:]
        segments_to_shift = list(
            map(lambda args: shift(*args), zip(segments_to_shift, shifts)))
        x = tf.concat((*segments_to_shift, *rest), axis=-1)
        return self.fn(x, **kwargs)


class PreNorm(tf.keras.layers.Layer):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.norm = None

    def build(self, input_shape):
        self.norm = tf.keras.layers.LayerNormalization(input_shape=input_shape)

    def call(self, inputs, *args, **kwargs):
        return self.fn(self.norm(inputs))


class Attention(tf.keras.layers.Layer):
    def __init__(self, dim_out, dim_inner, causal=False, **kwargs):
        super().__init__(**kwargs)
        self.scale = dim_inner ** -0.5
        self.causal = causal
        self.to_qkv = tf.keras.layers.Dense(dim_inner * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim_out)

    def call(self, inputs, *args, **kwargs):
        q, k, v = tf.split(self.to_qkv(inputs), 3, axis=-1)
        sim = tf.einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = tf.ones(sim.shape[1:])

            # band_part and set_diag replace triu(1) in lucidrains' implementation
            mask = tf.linalg.band_part(mask, 0, -1)
            mask_diag = tf.linalg.diag_part(mask)
            mask = tf.linalg.set_diag(mask, tf.zeros_like(mask_diag))

            mask = tf.cast(mask, dtype=tf.bool)
            sim = tf.where(mask[None, ...], tf.zeros_like(sim), sim)

        attn = tf.nn.softmax(sim, axis=-1)
        out = tf.einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)


class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(
            self,
            dim_seq,
            causal=False,
            activation=None,
            heads=1,
            init_eps=1e-3,
            circulant_matrix=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.circulant_pos_y = None
        self.circulant_pos_x = None
        self.weight = None
        self.bias = None
        self.heads = heads
        self.causal = causal
        self.norm = tf.keras.layers.LayerNormalization()
        self.activation = activation

        self.dim_seq = dim_seq
        self.init_eps = init_eps / dim_seq

        self.circulant_matrix = circulant_matrix

    def build(self, _):

        self.bias = self.add_weight(
            name="bias",
            shape=[self.heads, self.dim_seq],
            initializer=tf.ones
        )

        self.weight = self.add_weight(
            name="kernel",
            shape=(self.heads, self.dim_seq, self.dim_seq),
            initializer=tf.keras.initializers.RandomUniform(
                minval=-self.init_eps, maxval=self.init_eps)
        )

        if self.circulant_matrix:
            self.circulant_pos_x = self.add_weight(
                name="circulant_pos_x",
                shape=[self.heads, self.dim_seq],
                initializer=tf.ones
            )

            self.circulant_pos_y = self.add_weight(
                name="circulant_pos_y",
                shape=[self.heads, self.dim_seq],
                initializer=tf.ones
            )

    def call(self, inputs, gate_res=None, *args, **kwargs):
        n, h = inputs.shape[1], self.heads

        res, gate = tf.split(inputs, 2, axis=-1)
        gate = self.norm(gate)

        weight, bias = self.weight, self.bias

        if self.circulant_matrix:
            # build the circulant matrix

            dim_seq = weight.shape[-1]
            weight = tf.pad(weight, (0, dim_seq), value=0)
            weight = repeat(weight, '... n -> ... (r n)', r=dim_seq)
            weight = weight[:, :-dim_seq].reshape(h, dim_seq, 2 * dim_seq - 1)
            weight = weight[:, :, (dim_seq - 1):]

            # give circulant matrix absolute position awareness

            pos_x, pos_y = self.circulant_pos_x, self.circulant_pos_y
            weight = weight * rearrange(pos_x, 'h i -> h i ()') * rearrange(pos_y, 'h j -> h () j')

        if self.causal:
            weight, bias = weight[:, :n, :n], bias[:, :n]
            mask = tf.ones(weight.shape[-2:])
            # band_part and set_diag replace triu(1) in lucidrains' implementation
            mask = tf.linalg.band_part(mask, 0, -1)
            mask_diag = tf.linalg.diag_part(mask)
            mask = tf.linalg.set_diag(mask, tf.zeros_like(mask_diag))

            mask = tf.cast(mask, dtype=tf.bool)
            weight = tf.where(mask[None, ...], tf.zeros_like(weight), weight)

        gate = rearrange(gate, 'b n (h d) -> b h n d', h=h)

        gate = tf.einsum('b h n d, h m n -> b h m d', gate, weight)
        gate = gate + rearrange(bias, 'h n -> () h n ()')

        gate = rearrange(gate, 'b h n d -> b n (h d)')

        if exists(gate_res):
            gate = gate + gate_res

        return self.activation(gate) * res


class GMLPBlock(tf.keras.layers.Layer):
    def __init__(
            self,
            dim,
            dim_ff,
            seq_len,
            heads=1,
            attn_dim=None,
            causal=False,
            activation=tf.keras.activations.linear,
            circulant_matrix=False
    ):
        super().__init__()
        self.proj_in = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_ff),
            GELU()
        ])

        self.attn = Attention(dim_ff // 2, attn_dim, causal) if exists(attn_dim) else None

        self.sgu = SpatialGatingUnit(
            seq_len,
            causal=causal,
            activation=activation,
        )

        self.proj_out = tf.keras.layers.Dense(dim)

    def call(self, inputs, *args, **kwargs):
        gate_res = self.attn(inputs) if exists(self.attn) else None
        outputs = self.proj_in(inputs)
        outputs = self.sgu(outputs, gate_res=gate_res)
        outputs = self.proj_out(outputs)
        return outputs


# Main classes
class GMLP(tf.keras.Model):
    def __init__(
            self,
            *,
            num_tokens=None,
            dim,
            depth,
            seq_len,
            heads=1,
            ff_mult=4,
            attn_dim=None,
            prob_survival=1.,
            causal=False,
            circulant_matrix=False,
            shift_tokens=0,
            activation=tf.keras.activations.linear,
            dropout_ratio=0.2
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        self.to_embed = tf.keras.layers.Embedding(num_tokens, dim) if exists(
            num_tokens) else Identity()

        token_shifts = tuple(range(0 if causal else -shift_tokens, shift_tokens + 1))
        layers = []
        for _ in range(depth):
            layers.append(
                Residual(
                    PreNorm(
                        PreShiftTokens(
                            token_shifts,
                            GMLPBlock(
                                dim=dim,
                                heads=heads,
                                dim_ff=dim_ff,
                                seq_len=seq_len,
                                attn_dim=attn_dim,
                                causal=causal,
                                activation=activation,
                                circulant_matrix=circulant_matrix
                            )
                        )
                    )
                )
            )
            layers.append(tf.keras.layers.Dropout(dropout_ratio))

        self.module_layers = tf.keras.Sequential(layers)

        self.to_logits = tf.keras.Sequential(
            tf.keras.layers.LayerNormalization(dim),
            tf.keras.layers.Dense(num_tokens)
        ) if exists(num_tokens) else Identity()

    def call(self, inputs, *args, **kwargs):
        outputs = self.to_embed(inputs)
        outputs = self.module_layers(outputs)
        return self.to_logits(outputs)


class GMLPVision(tf.keras.layers.Layer):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            depth,
            heads=1,
            ff_mult=4,
            channels=3,
            attn_dim=None,
            dropout_ratio=0.2
    ):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by number of heads'

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert (image_height % patch_height) == 0 and (
                image_width % patch_width) == 0, 'image height and width must be divisible by patch size'
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        dim_ff = dim * ff_mult

        self.to_patch_embed = tf.keras.Sequential([
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=patch_height, p2=patch_width),
            tf.keras.layers.Dense(dim)
        ])

        layers = []
        for _ in range(depth):
            layers.append(
                Residual(
                    PreNorm(
                        GMLPBlock(
                            dim=dim,
                            heads=heads,
                            dim_ff=dim_ff,
                            seq_len=num_patches,
                            attn_dim=attn_dim
                        )
                    )
                )
            )
            layers.append(tf.keras.layers.Dropout(dropout_ratio))

        self.module_layers = tf.keras.Sequential(layers)

        self.to_logits = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            Reduce('b n d -> b d', 'mean'),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs, *args, **kwargs):
        outputs = self.to_patch_embed(inputs)
        outputs = self.module_layers(outputs)
        return self.to_logits(outputs)

# Based on Tensorflow tutorial Neural machine translation
# class Attention(tf.keras.layers.Layer):
#     def __init__(self, causal=False, **kwargs):
#         super().__init__(**kwargs)
#         self.attention = tf.keras.layers.Attention(**kwargs)
#         self.norm = tf.keras.layers.LayerNormalization()
#         self.add = tf.keras.layers.Add()
#         self.causal = causal

#     def call(self, inputs, *args, **kwargs):
#         attn_output = self.attention(
#             query=inputs,
#             value=inputs,
#             key=inputs,
#             use_causal_mask=self.causal)
#         outputs = self.add([inputs, attn_output])
#         outputs = self.layernorm(outputs)
#         return outputs


# NydiaAI implementation
# class SpatialGatingUnit(tf.keras.layers.Layer):
#     def __init__(self,
#                 dim_seq,
#                 causal = False,
#                 activation = None,
#                 init_eps = 1e-3,
#                 kernel_regularizer=None,
#                 bias_regularizer=None):

#         self.dim_seq = dim_seq
#         self.causal = causal
#         self.activation = activation
#         self.init_eps = init_eps / dim_seq

#         self.kernel_regularizer = kernel_regularizer
#         self.bias_regularizer = bias_regularizer

#         return super(SpatialGatingUnit, self).__init__()

#     def build(self, _):

#         self.conv1d_bias = self.add_weight(
#             name="sgu_conv1d_bias",
#             regularizer=self.bias_regularizer,
#             shape=[self.dim_seq],
#             initializer=tf.ones
#         )

#         self.conv1d_kernel = self.add_weight(
#             name="sgu_conv1d_kernel",
#             regularizer=self.kernel_regularizer,
#             shape=(1, self.dim_seq, self.dim_seq),
#             initializer=tf.keras.initializers.RandomUniform(minval=-self.init_eps, maxval=self.init_eps)
#         )

#         self.norm = tf.keras.layers.LayerNormalization()

#     def call(self, x):
#         n = x.shape[1]
#         weight, bias = self.conv1d_kernel, self.conv1d_bias
#         if(self.causal):
#             weight, bias = weight[:, :n, :n], bias[:n]

#             mask = tf.ones(weight.shape[1:])

#             # band_part and set_diag replace triu(1) in lucidrains' implementation
#             mask = tf.linalg.band_part(mask, 0, -1)
#             mask_diag = tf.linalg.diag_part(mask)
#             mask = tf.linalg.set_diag(mask, tf.zeros_like(mask_diag))

#             mask = tf.cast(mask, dtype=tf.bool)
#             weight = tf.where(mask[None, ...], tf.zeros_like(weight), weight)


#         res, gate = tf.split(x, 2, axis=-1)
#         gate = self.norm(gate)
#         data_format = "NWC"
#         conv1d_kwargs = {
#             "stride": 1,
#             "use_cudnn_on_gpu": True,
#             "data_format": data_format,
#             "padding": "VALID"
#         }

#         gate = tf.transpose(gate, (0,2,1))
#         gate = tf.nn.conv1d(gate, filters=self.conv1d_kernel, **conv1d_kwargs)
#         gate = tf.nn.bias_add(gate, self.conv1d_bias, data_format=data_format) # Now add bias
#         gate = tf.transpose(gate, (0,2,1))

#         if(self.activation is not None):
#             gate = self.activation(gate)
#         return gate * res
