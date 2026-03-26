#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 2026.03
"""

import unittest
from model import solvers
from model.graph import skeleton

class TestModelSolvers(unittest.TestCase):
    
    def test_laplacian_graph_solver(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        # constraints = {}
        # for i in range(3):
        #     constraints[i * skel.get_num_of_joints()] = [0, 0, 0]
        constraints = [i for i in range(3)]
        lgs = solvers.LaplacianGraphSolver(skel, 3, constraints)
        