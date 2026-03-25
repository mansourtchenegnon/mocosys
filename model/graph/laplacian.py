#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 03.2025
"""

# %% Imports
from enum import Enum
import numpy as np
from model.graph.skeleton import SkeletonGraph

# %% Constants
class LaplacianType(Enum):
    UNIFORM = 1
    COTANGENT = 2
    GAUSSIAN = 3


class ConstraintType(Enum):
    PENALISATION = 1
    SUBSTITUTION = 2

# %% Functions

def create_matrix_A(skeleton_graph : SkeletonGraph, sequence_length : int, plus_identity=False):
    """Computes the adjacency matrix **A** of a graph.
    If `plus_identity` is `True`, then the matrix is `Ã = A + I`.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        The skeleton representation in the motion sequence.
    sequence_length : int
        The length of the motion sequence.
    plus_identity : bool, optional
        Tells whether to add identity matrix or not, by default False.

    Returns
    -------
    numpy.ndarray
        The adjacency matrix of shape :math:`(1, n, n)`.
    """    
    num_of_vertices = skeleton_graph.get_num_of_joints() * sequence_length
    mat_A = np.zeros(shape=(num_of_vertices, num_of_vertices), dtype="float32")
    edges = skeleton_graph.motion_graph_edges_pairs(sequence_length)
    for edge in edges["spatial"]:
        mat_A[edge[0], edge[1]] = 1.0
        mat_A[edge[1], edge[0]] = 1.0
    for edge in edges["temporal"]:
        mat_A[edge[0], edge[1]] = 1.0
        mat_A[edge[1], edge[0]] = 1.0
    if plus_identity:
        mat_A = mat_A + np.eye(num_of_vertices, dtype="float32")
    mat_A = np.expand_dims(mat_A, axis=0)
    return mat_A


def create_normalized_matrix_A(skeleton_graph : SkeletonGraph, sequence_length : int, plus_identity=False):
    """Computes normalized adjacency matrix with the formula `D̃^(-.5) x Ã x D̃^(-.5)`

    Args:
        skeleton_graph (SkeletonGraph): The skeleton representation in the motion sequence.
        sequence_length (int): The length of the motion sequence.
        plus_identity (bool, optional): Tells whether to add identity matrix or not. Defaults to False.

    Returns:
        numpy.ndarray: The adjacency matrix of shape :math:`(1, n, n)`.
    """
    a = create_matrix_A(skeleton_graph, sequence_length, plus_identity)
    dl = np.sum(a[0], 0)
    n = a.shape[1]
    dn = np.zeros((n, n), dtype="float32")
    for i in range(n):
        if dl[i] > 0:
            dn[i, i] = dl[i] ** (-0.5)
    dn = np.expand_dims(dn, 0)
    dad = dn @ a @ dn
    return dad


def is_symmetric(matrix):
    pass


def create_matrix_D(skeleton_graph : SkeletonGraph, sequence_length : int):
    """Creates the distance matrix **D** between according to a given Laplacian motion correction problem.
    The problem is represented by the skeleton representation and the sequence length.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        Skeleton representation in the motion.
    sequence_length : int
        The length of the motion sequence.

    Returns
    -------
    numpy.ndarray
        The distance matrix **D** for the given Laplacian motion correction problem.
    """    
    num_of_bones = len(skeleton_graph.get_skeleton_bones())
    mat_D = np.zeros(shape=(sequence_length * (skeleton_graph.get_num_of_joints() - 1),
                        sequence_length * skeleton_graph.get_num_of_joints()),
                        dtype="float32")
    # distances from skeletons
    for t in range(sequence_length):
        for i in range(num_of_bones):
            mat_D[num_of_bones * t + i][t * skeleton_graph.get_num_of_joints() + skeleton_graph.get_skeleton_bones()[i][0]] = 1.0
            mat_D[num_of_bones * t + i][t * skeleton_graph.get_num_of_joints() + skeleton_graph.get_skeleton_bones()[i][1]] = -1.0

    mat_D = np.expand_dims(mat_D, axis=0)
    return mat_D

def create_matrix_L(skeleton_graph : SkeletonGraph,
                    sequence_length : int,
                    spatial_weight : float = 1.0,
                    temporal_weight : float = 1.0):
    """Creates the Laplacian matrix **L** of a motion graph according to the
    Laplacian motion correction problem following the formula `L = I - D^(-1) - A`.
    The problem is represented by skeleton representation and the sequence length.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        Skeleton representation in the motion.
    sequence_length : int
        The length of the motion sequence.
    spatial_weight : float
        Weight of direct skeleton edges.
    temporal_weight : float
        Weight of temporal edges.

    Returns
    -------
    numpy.ndarray
        The Laplacian matrix **L**.
    """    
    num_of_vertices = skeleton_graph.get_num_of_joints() * sequence_length
    mat_L = np.zeros(shape=(num_of_vertices, num_of_vertices), dtype="float32")
    edges = skeleton_graph.motion_graph_edges_list(sequence_length)
    for i in range(num_of_vertices):
        # Spatial edges weights
        for j in edges["spatial"][i]:
            mat_L[i][j] = -spatial_weight
        # temporal edges weights
        for j in edges["temporal"][i]:
            mat_L[i][j] = -temporal_weight
        # weights values on diagonal
        mat_L[i][i] = float(len(edges["spatial"][i])) * spatial_weight \
                        + float(len(edges["temporal"][i])) * temporal_weight
    mat_L = np.expand_dims(mat_L, axis=0)
    return mat_L


def create_matrix_U(skeleton_graph : SkeletonGraph, sequence_length : int, constrained_joints : list):
    """Creates the corresponding constraints matrix **U** according to the 
    Laplacian motion correction problem.

    Parameters
    ----------
    skeleton_graph : SkeletonGraph
        Skeleton representation of the motion.
    sequence_length : int
        Length of the motion sequence.
    constrained_joints : list, optional
        List of constrained joints.

    Returns
    -------
    numpy.ndarray
        The constraints matrix **U**.
    """    
    num_of_constraints = len(constrained_joints)
    mat_U = np.zeros(
        shape=(num_of_constraints, skeleton_graph.get_num_of_joints() * sequence_length), #(T*C, T*J)
        dtype="float32"
    )
    for i, constraint in enumerate(constrained_joints):
        mat_U[i][constraint] = 1.0 # U_ti = 1.0
    mat_U = np.expand_dims(mat_U, axis=0)
    return mat_U


def create_kernel_L(skeleton_graph : SkeletonGraph,
                    sequence_length : int = 3,
                    spatial_weight : float = 1.0,
                    temporal_weight : float = 1.0):
    """Creates the Laplacian kernel **L** of a motion graph according to the
    Laplacian motion correction problem following the formula `L = I - D^(-1) - A`
    The problem is represented by skeleton representation and the sequence length.

    Args:
        sequence_length (SkeletonGraph): Skeleton representation in the motion.
        sequence_length (int): The length of the motion sequence.
        spatial_weight (float): Weight of direct skeleton edges.
        temporal_weight (float): Weight of temporal edges.

    Returns:
        A numpy.ndarray, the Laplacian kernel **L**.
    """
    kernel_L = np.zeros(shape=(
        skeleton_graph.num_of_joints,
        sequence_length,
        skeleton_graph.num_of_joints
        ), dtype="float32")
    for i in range(skeleton_graph.num_of_joints):
        # spatial edges
        for j in skeleton_graph.get_joints_incidence()[i]:
            kernel_L[i][1][j] = -spatial_weight
        # temporal edges
        kernel_L[i][0][i] = -temporal_weight
        kernel_L[i][2][i] = -temporal_weight

        kernel_L[i][1][i] = 2 * (temporal_weight) + len(skeleton_graph.get_joints_incidence()[i]) * spatial_weight
    return kernel_L
        