#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:21:06 2022

@author: mansour
"""

import tensorflow as tf
from model.graph import laplacian
from model.graph.skeleton import SkeletonGraph
from model import ops

class LaplacianGraphSolver:
    """A class to define and solve a Laplacian problem based on the graph 3D+t representation
        of the motion.
        
        Attributes
        ----------
        T : int
            Sequence length.
        J : int
            Number of skeleton joints.
        L : numpy.array
            Represents the Laplacian matrix. L = I - D^{-1} * A.
        Lc : numpy.array
            Represents the constrained Laplacian matrix.
    """    
    def __init__(
            self,
            skel : SkeletonGraph,
            seq_length,
            constraints,
            sw=1.0,
            tw=1.0,
    ) -> None:
        """Constructor for LaplacianGraphSolver

        Parameters
        ----------
        skel : common.graph.StaticSkeleton
            The skeleton representation in the motion.
        seq_length : int
            Length of the motion sequence
        constraints : dict
            Dictionnary of positional constraints
        sw : float, optional
            Weight value for spatial edges, by default 1.0
        tw : float, optional
            Weight value for temporal edges, by default 1.0
        """        
        self.skeleton = skel
        self.J = skel.num_of_joints
        self.L = tf.convert_to_tensor(laplacian.create_matrix_L(skel, seq_length, sw, tw))
        self.T = seq_length
        self.LUD = None
        self.LU = None
        self.LRows = self.L_cols = self.L.shape[1]

        # Constraints parameters
        self.u = None  # positional constraints `(1, 1, Nu, 3)`
        self.U = None  # positional constraints matrix `(1, 1, Nu, N)`
        self.U_old = None  # positional constraints matrix `(1, 1, Nu, N)`
        self.D = None  # distance matrix `(1, 1, card(Es), N)`

        self.set_positional_constraints(constraints)
        self.set_distance_matrix()

    def set_positional_constraints(self, constraints):
        """Add positional constraints to the Laplacian constrained matrix.

        Parameters
        ----------
        constraints : dict, optional
            List of constraints, by default {}.
        """
        # build U matrix and add to L to form LU matrix
        self.U = tf.expand_dims(
            laplacian.create_matrix_U(self.skeleton, self.T, list(constraints.keys())),
            axis=0
        )
        if self.LUD is None:
            self.LUD = tf.expand_dims(tf.identity(self.L), axis=0)
        self.LUD = tf.concat((self.LUD, self.U), axis=2)
        self.LU = self.LUD

        # build u vector
        self.u = None
        for u in constraints.values():
            u = tf.convert_to_tensor(u, tf.float32)
            u = tf.expand_dims(u, 0)
            if self.u is None:
                self.u = u
            else:
                self.u = tf.concat((self.u, u), axis=0)
        self.u = tf.expand_dims(self.u, 0)
        self.u = tf.expand_dims(self.u, 0)

    def set_distance_matrix(self):
        """ Compute the distance matrix for D for the resolution
        based on .
        """
        d = tf.expand_dims(
            laplacian.create_matrix_D(self.skeleton, self.T),
            axis=0
        )
        self.D = d

        self.LUD = tf.concat((self.LUD, d), axis=2)

    def apply_constraints(self, delta, gamma):
        """Apply constraints of the LaplacianProblem to delta.

        Parameters
        ----------
        delta : Tensor
            Laplacian coordinates to constrain.
        gamma : Tensor
            A 4D+Tensor representing the constraints on Es segments.
            Computed from pose or estimated. Shape = `(B, T, N, 3)`

        Returns
        -------
        delta_c : Tensor
            Constrained Laplacian coordinates.

        """
        delta_c = tf.identity(delta)

        # apply positional constraints
        vec_u = tf.tile(self.u, [delta_c.shape[0], delta_c.shape[1], 1, 1])
        delta_c = tf.concat((delta_c, vec_u), axis=-2)

        # apply distance constraints
        if gamma is not None:
            delta_c = tf.concat((delta_c, gamma), axis=-2)
        return delta_c

    def solve(self, delta, gamma):
        """Resolve laplacian problem {self} from joint differential coordinates
        {delta} to compute the joint positions.

        Parameters
        ----------
        delta : Tensor
            Joint differential coordinates from the sequence.
        gamma : Tensor
            Distance constraints for Es segments.

        Returns
        -------
        y : Tensor
            Joint position coordinates of the sequence obtained from the
            differential coordinates {delta}.

        """
        assert tf.rank(delta) == 4, "Expected delta to be of rank 4 but got {}".format(
            tf.shape(delta)
        )
        # Pad if necessary
        n, seq_l, tv, c = tf.shape(delta)
        assert (
                tv == tf.shape(self.L)[1]
        ), "Expected shape to be [{}, {}, {}, {}] but got {}".format(
            n, seq_l, tf.shape(self.L)[1], c, delta.shape
        )
        delta = tf.reshape(delta, (n, -1, tf.shape(self.L)[1], c))
        dc = self.apply_constraints(delta, gamma)

        if gamma is not None:
            m = tf.linalg.matrix_transpose(self.LUD, conjugate=True) @ self.LUD
            b = tf.linalg.matrix_transpose(self.LUD, conjugate=True) @ dc
        else:
            m = tf.linalg.matrix_transpose(self.LU, conjugate=True) @ self.LU
            b = tf.linalg.matrix_transpose(self.LU, conjugate=True) @ dc
        # print("M is symmetric ?", is_symmetric(m))
        u = tf.linalg.cholesky(m)  # upper
        y = tf.linalg.cholesky_solve(u, b)
        return y

    def to_delta(self, x):
        n, t, vc = tf.shape(x)
        ts = t // self.T
        c = vc // self.V
        y = x.view(n, ts, self.T * self.V, c)  # N, Ts, T*V, C
        y = tf.expand_dims(self.L, 0) @ y
        y = y.view(n, -1, self.V * c)  # N, L+pad, V*C
        return y


class DeltaConverter:
    def __init__(self, features, skel : SkeletonGraph, t, sw=1.0, tw=1.0):
        self.features = features
        self.t = t
        self.v = skel.num_of_joints
        self.l_mat = tf.expand_dims(laplacian.create_matrix_L(skel, t, sw, tw), 0)

    def __call__(self, inputs, *args, **kwargs):
        """
        Computes Δ from 3D joint positions.

        Parameters
        ----------
        inputs: Tensor, represents the 3D joint positions.
        It is either a 3D tensor with shape `(B, L, V*C)`
                or a 4D tensor with shape `(B, Ts, T*V, C)`.

        Returns
        -------

        """
        outputs = ops.format_inputs(inputs, self.t)
        outputs = self.l_mat @ outputs
        if "keepdims" in kwargs and kwargs["keepdims"] is True:
            center = self.t // 2
            outputs = outputs[:, :, center * self.v:center * self.v + self.v, :]
        return outputs


class PosesConverter:
    def __init__(self, features : int, skel : SkeletonGraph, t : int, constraints : dict, sw=1.0, tw=1.0, **kwargs):
        """Converts laplacian cordinates `Δ` to cartesian coordinates `P` using graph solver.

        Args:
            features (int): Number of features for each node of the graph.
            skel (SkeletonGraph): Graph of skeleton.
            t (int): Length of the motion graph sequence.
            constraints (dict): Dictionnary of constraints for resolution.
            sw (float, optional): Weight of spatial edge. Defaults to 1.0.
            tw (float, optional): Weight or temporal edge. Defaults to 1.0.
        """
        super(PosesConverter, self).__init__(**kwargs)
        self.features = features
        self.t = t
        self.v = skel.num_of_joints
        self.lgs = LaplacianGraphSolver(skel, self.t, constraints, sw, tw)

    def __call__(self, inputs, gamma, *args, **kwargs):
        """Converts laplacian coordinates `Δ` to cartesian coordinates `P` using laplacian solver.

        Args:
            inputs (Tensor): Tensor representing the laplacian coordinates of the motion graph.
            gamma (Tensor): Tensor representing the skeletal constraints for the laplacian resolution.

        Returns:
            Tensor: Tensor representing the computed cartesian coordinates `P`.
        """
        outputs = self.lgs.solve(inputs, gamma)
        # retrieve central frames
        center = self.t // 2
        outputs = outputs[:, :, center * self.v:center * self.v + self.v, :]
        outputs = self.smoother(outputs)
        return outputs
