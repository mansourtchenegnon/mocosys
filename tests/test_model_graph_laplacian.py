#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 18.03.2025
"""

import unittest
from model.graph import skeleton
from model.graph import laplacian
from tests.functions import adjacency_matrix, basic_laplacian_matrix, distance_matrix, u_matrix

class TestModelGraphLaplacian(unittest.TestCase):

    def test_matrix_A(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        actual_matrix_A = laplacian.create_matrix_A(skel, 3, True).tolist()
        expected_matrix_A = adjacency_matrix(3, 17, True).tolist()
        self.assertEqual(expected_matrix_A, actual_matrix_A)

    def test_matrix_D(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        actual_matrix_D = laplacian.create_matrix_D(skel, 3).tolist()
        expected_matrix_D = distance_matrix(3, 17).tolist()
        self.assertEqual(expected_matrix_D, actual_matrix_D)

    def test_matrix_L(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        actual_matrix_L = laplacian.create_matrix_L(skel, 3).tolist()
        expected_matrix_L = basic_laplacian_matrix(3, 17).tolist()
        self.assertEqual(expected_matrix_L, actual_matrix_L)

    def test_matrix_U(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        actual_matrix_U = laplacian.create_matrix_U(skel, 3, [0, 17, 34]).tolist()
        expected_matrix_U = u_matrix(3, 17).tolist()
        self.assertEqual(expected_matrix_U, actual_matrix_U)

if __name__ == "__main__":
    unittest.main()