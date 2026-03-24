#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 19.03.2025
""" 

import math
import numpy as np
from enum import Enum

class LaplacianType(Enum):
    LT_UNIFORM = 1
    LT_COTANGENT = 2
    LT_GAUSSIAN = 3


class ConstraintType(Enum):
    CT_PENALISATION = 1
    CT_SUBSTITUTION = 2


EDGES = [
    [1, 4, 7],  # 0
    [0, 2],  # 1
    [1, 3],  # 2
    [2],  # 3
    [0, 5],  # 4
    [4, 6],  # 5
    [5],  # 6
    [0, 8],  # 7
    [7, 9, 11, 14],  # 8
    [8, 10],  # 9
    [9],  # 10
    [8, 12],  # 11
    [11, 13],  # 12
    [12],  # 13
    [8, 15],  # 14
    [14, 16],  # 15
    [15],  # 16
]

PAIRS = [
    [0, 1],
    [0, 4],
    [0, 7],
    [7, 8],  # hips and spine
    [1, 2],
    [2, 3],  # right leg
    [4, 5],
    [5, 6],  # left leg
    [8, 11],
    [8, 14],  # shoulders
    [8, 9],
    [9, 10],  # head
    [11, 12],
    [12, 13],  # left arm
    [14, 15],
    [15, 16],  # right arm
]

ES = [
    [0, 1],  # hips r
    [1, 2],  # femur r
    [2, 3],  # tibia r
    [0, 4],  # hips l
    [4, 5],  # femur l
    [5, 6],  # tibia l
    [0, 7],  # spine back
    [7, 8],  # spine top
    [8, 9],  # neck
    [9, 10],  # head
    [8, 11],  # clavicle l
    [11, 12],  # humerus l
    [12, 13],  # radius l
    [8, 14],  # clavicle r
    [14, 15],  # humerus r
    [15, 16],  # radius r
]

SIMPLIFIED_ES = [
    [0, 10],
    [0, 3],  # spine back
    [0, 6],
    [0, 13],  # wrist l
    [0, 16],  # wrist r
]

GAMMA_CST = [
    [[0, 1], [0, 4]],  # hips
    [[0, 4]],  # head
    [[0, 3]],  # hips-rightankle
    [[0, 6]],  # hips-leftankle
    [[0, 13]],  # hips-leftwrist
    [[0, 16]],  # hips-rightwrist
]


def adjacency_matrix(T, J, plus_I=False):
    """
    Computes Adjacency Matrix L from T and J.

    Parameters
    ----------
    T : int
        Sequence length.
    J : int
        Number of joints.
    plus_I : bool, optional
        If True, the identity matrix is added to the adjacency matrix. Default: False.

    Returns
    -------
    numpy.ndarray
        The adjacency Matrix A. Dims = (1, TxJ, TxJ).

    """
    n = T * J
    a = np.zeros(shape=(n, n), dtype="float32")
    for t in range(0, n, J):  # loop over sequence with step "num joints"
        for i in range(J):
            # Adjacency from skeleton
            for j in EDGES[i]:
                a[t + i][t + j] = 1.0
                a[t + j][t + i] = 1.0
            # Link to same joint for next and previous frames
            if T > 1:
                if t == 0:
                    a[t + i][t + J + i] = 1.0
                    a[t + J + i][t + i] = 1.0
                elif t == (n - J):
                    a[t + i][t - J + i] = 1.0
                    a[t - J + i][t + i] = 1.0
                else:
                    a[t + i][t - J + i] = 1.0
                    a[t - J + i][t + i] = 1.0
                    a[t + i][t + J + i] = 1.0
                    a[t + J + i][t + i] = 1.0
    if plus_I:
        a = a + np.eye(n, dtype="float32")
    a = np.expand_dims(a, axis=0)
    return a


def is_symmetric(A):
    """
    Verifies if A is symmetric.

    Parameters
    ----------
    A : array or Tensor
        The matrix to verify.

    Returns
    -------
    bool
        True if A is symmetric, False otherwise.
    """
    assert len(np.shape(A)) >= 2, "Expected rank greater than 2 but got {} instead".format(len(np.shape(A)))
    return np.sum(A - np.transpose(A)) == 0


def normalize_adjacency(A):
    """
    Normalises Adjacency Matrix A.

    Parameters
    ----------
    A : np.ndarray:
        Adjacency matrix.

    Returns
    -------
    numpy.ndarray
        The normalised adjacency matrix. Dims = (1, TxJ, TxJ).

    """
    dl = np.sum(A[0], 0)
    n = A.shape[1]
    dn = np.zeros((n, n), dtype="float32")
    for i in range(n):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-0.5)
    dn = np.expand_dims(dn, 0)
    dad = dn @ A @ dn
    return dad


def build_motion_graph_edges(T, J):
    """
    Builds the graph 3D+t.

    Parameters
    ----------
    T : int
        Length of the motion sequence.
    J : int
        Number of skeleton joints.

    Returns
    -------
    list
        List of the edges of the graph.
    """

    def frame_edges(t):
        return [[v + t * J for v in e] for e in EDGES]

    edges = [x for xs in [frame_edges(t) for t in range(T)] for x in xs]
    n = len(edges)
    if T > 1:
        for i in range(n):
            if i < J:
                edges[i].append(i + J)
            elif i >= n - J:
                edges[i].append(i - J)
            else:
                edges[i].append(i - J)
                edges[i].append(i + J)

    return edges


def create_gaussian_weights(T, J, edges, d_max, alpha, beta):
    """
    Generates gaussian weights for graph 3D+t

    Parameters
    ----------
    T : int
        Length of the motion sequence.
    J : int
        Number of skeleton joints.
    edges: list,
        List of edges of the graph.
    d_max: float,
        Maximum distance between edges.
    alpha : float
        Parameter for gaussian weights.
    beta : float
        Parameter for gaussian weights.

    Returns
    -------
    numpy.ndarray
        Gaussian weights for graph 3D+t.
    """
    wij = np.zeros((T * J, T * J))
    size = len(edges)
    for i in range(size):
        for j in edges[i]:
            if np.abs(i - j) % J != 0:
                wij[i][j] = 1.0
            else:
                dij = 0
                wij[i][j] = 1 + alpha * math.exp(-1 * beta * (dij ** 2 / d_max ** 2))

    return wij


def basic_laplacian_matrix(
        T,
        J,
        sw=1.0,
        tw=1.0,
        laplacianType: LaplacianType = LaplacianType.LT_UNIFORM,
        gaussianWeights=None,
):
    """
    Computes Laplacian Matrix L from T and J.

    Parameters
    ----------
    T : int
        Sequence length.
    J : int
        Number of skeleton joints.
    sw : float
        Weight of direct skeleton edges.
    tw : float
        Weight of temporal edges.
    laplacianType : LaplacianType, optional
        Laplacian type to create. The default is LaplacianType.LT_UNIFORM.
    gaussianWeights : array, optional
        The gaussian weights to build the Laplacian matrix if Gaussian type. Default is None.

    Returns
    -------
    numpy.ndarray
        The Laplacian Matrix L. Dims = (1, TxJ, TxJ).

    """
    n = T * J
    l_mat = np.zeros(shape=(n, n), dtype="float32")
    # Computes init laplacian matrix base on skeleton joints and sequence length
    if laplacianType == LaplacianType.LT_UNIFORM:
        for t in range(0, n, J):  # loop over sequence with step "num joints"
            for i in range(J):
                l_mat[t + i][t + i] = float(len(EDGES[i])) * sw
                # incidence from skeleton
                for j in EDGES[i]:
                    l_mat[t + i][t + j] = -1.0 * sw
                # Link to same joint for next and previous frames
                if T > 1:
                    if t == 0:
                        l_mat[t + i][t + i] += tw
                        l_mat[t + i][t + J + i] = -tw
                    elif t == (n - J):
                        l_mat[t + i][t + i] += tw
                        l_mat[t + i][t - J + i] = -tw
                    else:
                        l_mat[t + i][t + i] += 2.0 * tw
                        l_mat[t + i][t + J + i] = -tw
                        l_mat[t + i][t - J + i] = -tw
    elif laplacianType == LaplacianType.LT_GAUSSIAN:
        # Build edges
        graph_edges = build_motion_graph_edges(T, J)
        # Computes Laplacian matrix weights
        for i in range(n):
            i_sum = 0.0
            for j in range(len(graph_edges[i])):
                l_mat[i][graph_edges[i][j]] = -gaussianWeights[i][graph_edges[i][j]]
                i_sum += gaussianWeights[i][graph_edges[i][j]]
            l_mat[i, i] = i_sum
    l_mat = np.expand_dims(l_mat, axis=0)
    return l_mat


def laplacian_matrix_2_order(
        T,
        J,
        sw=1.0,
        tw1=1.0,
        tw2=0.6,
        laplacianType: LaplacianType = LaplacianType.LT_UNIFORM,
        gaussianWeights=None,
):
    """
    Computes Laplacian Matrix L from T and J.

    Parameters
    ----------
    T : int
        Sequence length.
    J : int
        Number of skeleton joints.
    sw : float
        Weight of direct skeleton edges.
    tw1 : float
        Weight of 1st order temporal edges.
    tw1 : float
        Weight of 2nd order temporal edges.

    Returns
    -------
    numpy.ndarray
        The Laplacian Matrix L. Dims = (1, TxJ, TxJ).

    """
    assert T == 5, "T should be five but received {}.".format(T)
    n = T * J
    l_mat = np.zeros(shape=(n, n), dtype="float32")
    # Computes init laplacian matrix base on skeleton joints and sequence length
    for t in range(0, n, J):  # loop over sequence with step "num joints"
        for i in range(J):
            l_mat[t + i][t + i] = float(len(EDGES[i])) * sw
            # incidence from skeleton
            for j in EDGES[i]:
                l_mat[t + i][t + j] = -1.0 * sw
            # Link to same joint for next and previous frames
            if t == 0:
                l_mat[t + i][t + i] += tw1 + tw2
                l_mat[t + i][t + J + i] = -tw1
                l_mat[t + i][t + 2 * J + i] = -tw2
            if t == J:
                l_mat[t + i][t + i] += 2 * tw1 + tw2
                l_mat[t + i][t + J + i] = -tw1
                l_mat[t + i][t - J + i] = -tw1
                l_mat[t + i][t + 2 * J + i] = -tw2
            if t == J * 2:
                l_mat[t + i][t + i] += 2 * tw1 + tw2
                l_mat[t + i][t + J + i] = -tw1
                l_mat[t + i][t - J + i] = -tw1
                l_mat[t + i][t + 2 * J + i] = -tw2
                l_mat[t + i][t - 2 * J + i] = -tw2
            if t == J * 3:
                l_mat[t + i][t + i] += 2 * tw1 + tw2
                l_mat[t + i][t + J + i] = -tw1
                l_mat[t + i][t - J + i] = -tw1
                l_mat[t + i][t - 2 * J + i] = -tw2
            if t == J * 4:
                l_mat[t + i][t + i] += tw1 + tw2
                l_mat[t + i][t - J + i] = -tw1
                l_mat[t + i][t - 2 * J + i] = -tw1
            # else:
            #     l_mat[t + i][t + i] += 2.0 * tw1
            #     l_mat[t + i][t + J + i] = -tw1
            #     l_mat[t + i][t - J + i] = -tw1

    l_mat = np.expand_dims(l_mat, axis=0)
    return l_mat


def normalised_laplacian_matrix(
        T,
        J,
        laplacianType: LaplacianType = LaplacianType.LT_UNIFORM,
        gaussianWeights=None,
):
    """
    Computes Laplacian Matrix L from T and J.

    Parameters
    ----------
    T : int
        Sequence length.
    J : int
        Number of skeleton joints.
    laplacianType : LaplacianType, optional
        Laplacian type to create. The default is LaplacianType.LT_UNIFORM.
    gaussianWeights : numpy.ndarray, optional
        The gaussian weights to build the Laplacian matrix if Gaussian type. Default is None.

    Returns
    -------
    l_mat : numpy.ndarray
        The Laplacian Matrix L. Dims = (1, TxJ, TxJ).

    """

    def get_degree(vi, ts, V):
        di = float(len(EDGES[vi]))
        if ts == 0 or ts == (V - J):
            di += 1.0
        else:
            di += 2.0
        return di

    n = T * J
    l_mat = np.ones(shape=(n, n), dtype="float32")
    # Computes init laplacian matrix base on skeleton joints and sequence length
    if laplacianType == LaplacianType.LT_UNIFORM:
        for t in range(0, n, J):  # loop over sequence with step "num joints"
            for i in range(J):
                dii = get_degree(i, t, n)
                l_mat[t + i][t + i] = 1
                # incidence from skeleton
                for j in EDGES[i]:
                    l_mat[t + i][t + j] = -1.0 / dii
                # Link to same joint for next and previous frames
                if T > 1:
                    if t == 0:
                        l_mat[t + i][t + J + i] = -1.0 / dii
                    elif t == (n - J):
                        l_mat[t + i][t - J + i] = -1.0 / dii
                    else:
                        l_mat[t + i][t + J + i] = -1.0 / dii
                        l_mat[t + i][t - J + i] = -1.0 / dii
    elif laplacianType == LaplacianType.LT_GAUSSIAN:
        # Build edges
        graph_edges = build_motion_graph_edges(T, J)
        # Computes Laplacian matrix weights
        for i in range(n):
            i_sum = 0.0
            for j in range(len(graph_edges[i])):
                l_mat[i][graph_edges[i][j]] = -gaussianWeights[i][graph_edges[i][j]]
                i_sum += gaussianWeights[i][graph_edges[i][j]]
            l_mat[i, i] = i_sum
    l_mat = np.expand_dims(l_mat, axis=0)
    return l_mat


def complex_laplacian_matrix(  # TODO: Build laplacian matrix with all skeleton connections and different weights
        T,
        J,
        sw,
        tw,
        laplacianType: LaplacianType = LaplacianType.LT_UNIFORM,
        gaussianWeights=None
):
    """

    Args:
        T (int): Sequence length.
        J (int): Number of skeleton joints.
        sw (float) : Weight of direct skeleton edges.
        tw (float) : Weight of temporal edges.
        laplacianType (LaplacianType): Optional; Laplacian type to create. The default is LaplacianType.LT_UNIFORM.
        gaussianWeights (numpy.ndarray): Optional; The gaussian weights to build the Laplacian matrix if Gaussian type.
            (Default is None).

    Returns:
        l_mat (numpy.ndarray): The Laplacian Matrix L. Dims = (1, TxJ, TxJ).
    """

    def get_degree(vi, ts, V):
        di = float(len(EDGES[vi]))
        if ts == 0 or ts == (V - J):
            di += tw
        else:
            di += 2.0 * tw
        return di

    n = T * J
    l_mat = np.zeros(shape=(n, n), dtype="float32")
    # Computes init laplacian matrix base on skeleton joints and sequence length
    if laplacianType == LaplacianType.LT_UNIFORM:
        for t in range(0, n, J):  # loop over sequence with step "num joints"
            for i in range(J):
                l_mat[t + i][t + i] = get_degree(i, t, n)
                # incidence from skeleton
                for j in EDGES[i]:
                    l_mat[t + i][t + j] = -sw
                # Link to same joint for next and previous frames
                if T > 1:
                    if t == 0:
                        l_mat[t + i][t + J + i] = -tw
                    elif t == (n - J):
                        l_mat[t + i][t - J + i] = -tw
                    else:
                        l_mat[t + i][t + J + i] = -tw
                        l_mat[t + i][t - J + i] = -tw
    elif laplacianType == LaplacianType.LT_GAUSSIAN:
        # Build edges
        graph_edges = build_motion_graph_edges(T, J)
        # Computes Laplacian matrix weights
        for i in range(n):
            i_sum = 0.0
            for j in range(len(graph_edges[i])):
                l_mat[i][graph_edges[i][j]] = -gaussianWeights[i][graph_edges[i][j]]
                i_sum += gaussianWeights[i][graph_edges[i][j]]
            l_mat[i, i] = i_sum
    l_mat = np.expand_dims(l_mat, axis=0)
    return l_mat


def distance_matrix(T, J):
    """Create the distance matrix D between.

    Parameters
    ----------
    T : int
        The sequence length.
    J : int
        The number of joints

    Returns
    -------
    numpy.array
        The distance matrix D for the given T and J.
    """
    s = len(ES)
    d = np.zeros(shape=(T * (J - 1), T * J), dtype="float32")
    # distances from skeletons
    for t in range(T):
        for i in range(s):
            d[s * t + i][t * J + ES[i][0]] = 1.0
            d[s * t + i][t * J + ES[i][1]] = -1.0

    d = np.expand_dims(d, axis=0)
    return d


def u_matrix(T, J):
    # s = len(ES)
    u = np.zeros(shape=(T, T * J), dtype="float32")
    for t in range(T):
        u[t][t * J] = 1.0
    u = np.expand_dims(u, axis=0)
    return u