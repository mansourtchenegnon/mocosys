#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:21:06 2022

@author: mansour
"""

from logging import raiseExceptions

import keras.ops as kops
from keras import KerasTensor
from model.graph import laplacian
from model.graph.skeleton import SkeletonGraph
from model import ops

class LaplacianGraphSolver:
    """A class to define and solve a Laplacian problem based on the graph 3D+t representation
        of the motion. To reduce the complexity of the calculation, the 3D+t graph is built over
        a fixed number of frames `T` (3 for example), and the resolution operates like a sliding window.
        
        Attributes
        ----------
        T : int
            Sequence length.
        J : int
            Number of skeleton joints.
        L : Tensor
            Represents the Laplacian matrix. `L = I - D^{-1} * A`.
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
        constraints : list
            List of joints to constrain with positional constraints
        sw : float, optional
            Weight value for spatial edges, by default 1.0
        tw : float, optional
            Weight value for temporal edges, by default 1.0
        """        
        self.skeleton = skel
        self.J = skel.get_num_of_joints()
        self.L = kops.convert_to_tensor(laplacian.create_matrix_L(skel, seq_length, sw, tw))
        self.T = seq_length
        self.constraints = constraints
        self.LUD = None
        self.LU = None
        self.LRows = self.L_cols = self.L.shape[1]

        # Constraints matrices and vectors
        self.U = None  # positional constraints matrix `(1, 1, Nu, N)`
        self.D = None  # distance matrix `(1, 1, card(Es), N)`
        self.constraints_U = None  # positional constraints vector `(?, ?, Nu, 3)`
        self.constraints_Gamma = None  # distance (skeleton) constraints vector `(?, ?, card(Es), 3)`

        self.build_positional_constraints_matrix(constraints)
        self.build_distance_matrix()

    def build_positional_constraints_matrix(self, constraints : list):
        """Add positional constraints to the Laplacian constrained matrix.

        Parameters
        ----------
        constraints : list, optional
            Dictionary of constraints, by default [].
        """
        # build U matrix and concatenate it with L matrix to form LU matrix
        if len(constraints) != 0:
            self.U = kops.expand_dims(
                laplacian.create_matrix_U(self.skeleton, self.T, constraints),
                axis=0
            )
            if self.LUD is None:
                self.LUD = kops.expand_dims(self.L, axis=0)
            self.LUD = kops.concatenate((self.LUD, self.U), axis=2)
            self.LU = self.LUD

            # build u vector
            # self.u = None
            # for u in constraints.values():
            #     u = kops.convert_to_tensor(u, dtype="float32")
            #     u = kops.expand_dims(u, 0)
            #     if self.u is None:
            #         self.u = u
            #     else:
            #         self.u = kops.concatenate((self.u, u), axis=0)
            # self.u = kops.expand_dims(self.u, 0)
            # self.u = kops.expand_dims(self.u, 0)

    def build_distance_matrix(self):
        """ Compute the distance matrix for D for the resolution
        based on .
        """
        d = kops.expand_dims(
            laplacian.create_matrix_D(self.skeleton, self.T),
            axis=0
        )
        self.D = d

        self.LUD = kops.concatenate((self.LUD, d), axis=2)

    def set_positional_constraints(self, constraints_U):
        self.constraints_U = constraints_U

    def set_distance_constraints(self, distance_constraints):
        self.constraints_Gamma = distance_constraints


    def apply_constraints(self, delta):
        """Apply constraints of the LaplacianProblem to delta.

        Parameters
        ----------
        delta : Tensor
            Laplacian coordinates to constrain.

        Returns
        -------
        delta_c : Tensor
            Constrained Laplacian coordinates.

        """
        delta_c = delta

        # apply positional constraints
        if self.constraints_U is not None:
            delta_c = kops.concatenate((delta_c, self.constraints_U), axis=-2)
        else:
            raise ValueError("Positional constraints <u> should not be None")

        # apply distance constraints
        if self.constraints_Gamma is not None:
            delta_c = kops.concatenate((delta_c, self.constraints_Gamma), axis=-2)
        return delta_c

    def solve(self, delta, **kwargs):
        """Resolve laplacian problem `self` from joint differential coordinates
        `delta` to compute the joint positions.

        Parameters
        ----------
        delta : Tensor
            Joint differential coordinates from the sequence.

        Returns
        -------
        y : Tensor
            Joint position coordinates of the sequence obtained from the
            differential coordinates `delta`.

        """
        assert kops.ndim(delta) == 4, "Expected delta to be of rank 4 but got {}".format(
            kops.shape(delta)
        )
        if "format" in kwargs and kwargs["format"] is True:
            delta = ops.format_inputs(delta, self.T)
        n, seq_l, tv, c = kops.shape(delta)

        assert (
                tv == kops.shape(self.L)[1]
        ), "Expected shape to be [{}, {}, {}, {}] but got {}".format(
            n, seq_l, kops.shape(self.L)[1], c, delta.shape
        )
        delta = kops.reshape(delta, (n, -1, kops.shape(self.L)[1], c))
        dc = self.apply_constraints(delta)
        # print("delta", dc.shape)
        # print("LUD", self.LUD.shape)
        if self.constraints_Gamma is not None:
            m = kops.swapaxes(self.LUD, -1, -2) @ self.LUD
            b = kops.swapaxes(self.LUD, -1, -2) @ dc
            # m = kops.conjugate(kops.swapaxes(self.LUD, -1, -2)) @ self.LUD
            # b = kops.conjugate(kops.swapaxes(self.LUD, -1, -2)) @ dc
        else:
            m = kops.swapaxes(self.LU, -1, -2) @ self.LU
            b = kops.swapaxes(self.LU, -1, -2) @ dc
            # m = kops.conjugate(kops.swapaxes(self.LU, -1, -2)) @ self.LU
            # b = kops.conjugate(kops.swapaxes(self.LU, -1, -2)) @ dc
        # print("M is symmetric ?", is_symmetric(m))
        # print("m", m.shape)
        # print("b", b.shape)
        m = kops.repeat(m, b.shape[1], axis=1)
        m = kops.repeat(m, b.shape[0], axis=0)
        y = kops.linalg.solve(m, b)

        return y

    def pose_to_delta(self, x : KerasTensor):
        n, t, vc = kops.shape(x)
        ts = t // self.T
        c = vc // self.V
        y = x.view(n, ts, self.T * self.V, c)  # N, Ts, T*V, C
        y = kops.expand_dims(self.L, 0) @ y
        y = y.view(n, -1, self.V * c)  # N, L+pad, V*C
        return y


class DeltaConverter:
    def __init__(self, skel : SkeletonGraph, features, t, sw=1.0, tw=1.0):
        self.features = features
        self.t = t
        self.v = skel.get_num_of_joints()
        self.l_mat = kops.expand_dims(laplacian.create_matrix_L(skel, t, sw, tw), 0)

    def __call__(self, inputs, *args, **kwargs):
        """
        Computes Δ from 3D joint positions.

        Parameters
        ----------
        inputs: Tensor, represents the 3D joint positions. It is either a 3D tensor with shape `(B, L, V*C)` or a 4D tensor with shape `(B, Ts, T*V, C)`.

        Returns
        -------
        out: Tensor

        """
        if "format" in kwargs and kwargs["format"] is True:
            outputs = ops.format_inputs(inputs, self.t)
        else:
            outputs = inputs
        outputs = self.l_mat @ outputs
        if "keepdims" in kwargs and kwargs["keepdims"] is True:
            center = self.t // 2
            outputs = outputs[:, :, center * self.v:center * self.v + self.v, :]
        return outputs


class PosesConverter:
    def __init__(self, skel : SkeletonGraph, features : int, t : int, constraints : dict, sw=1.0, tw=1.0, **kwargs):
        """Converts laplacian cordinates `Δ` to cartesian coordinates `P` using graph solver.

        Args:
            skel (SkeletonGraph): Graph of skeleton.
            features (int): Number of features for each node of the graph.
            t (int): Length of the motion graph sequence.
            constraints (dict): Dictionnary of constraints for resolution.
            sw (float, optional): Weight of spatial edge. Defaults to 1.0.
            tw (float, optional): Weight or temporal edge. Defaults to 1.0.
        """
        super(PosesConverter, self).__init__(**kwargs)
        self.features = features
        self.t = t
        self.v = skel.get_num_of_joints()
        self.lgs = LaplacianGraphSolver(skel, self.t, constraints, sw, tw)

    def set_constraints(self, vec_u, vec_gamma):
        self.lgs.set_positional_constraints(vec_u)
        self.lgs.set_distance_constraints(vec_gamma)

    def __call__(self, inputs, *args, **kwargs):
        """Converts laplacian coordinates `Δ` to cartesian coordinates `P` using laplacian solver.

        Args:
            inputs (KerasTensor): Tensor representing the laplacian coordinates of the motion graph.

        Returns:
            Tensor: Tensor representing the computed cartesian coordinates `P`.
        """
        outputs = self.lgs.solve(inputs)
        # retrieve central frames
        center = self.t // 2
        outputs = outputs[:, :, center * self.v:center * self.v + self.v, :]
        # outputs = self.smoother(outputs)
        return outputs
