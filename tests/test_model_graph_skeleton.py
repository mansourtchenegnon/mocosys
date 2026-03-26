#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 03.2025
"""

import unittest
from model.graph import skeleton


def build_motion_graph_edges_list(T, J):
    def frame_edges(t):
        return [[v + t * J for v in e] for e in skeleton.H36M_17_JOINTS_SKELETON_INCIDENT_EDGES]

    s_edges = [x for xs in [frame_edges(t) for t in range(T)] for x in xs]
    n = len(s_edges)
    t_edges = []
    if T > 1:
        for i in range(n):
            if i < J:
                t_edges.append([i + J])
            elif i >= n - J:
                t_edges.append([i - J])
            else:
                t_edges.append([i - J, i + J])
    return {'spatial' : s_edges, 'temporal': t_edges}

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

class TestModelGraphSkeleton(unittest.TestCase):

    def test_skeleton(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        self.assertEqual(skel.get_joints_incidence(), skeleton.H36M_17_JOINTS_SKELETON_INCIDENT_EDGES)

    def test_motion_graph_edges_list_t1(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        actual_graph = skel.motion_graph_edges_list(1)
        expected_graph = build_motion_graph_edges_list(1, 17)
        self.assertEqual(actual_graph["spatial"], expected_graph['spatial'])
        self.assertEqual(actual_graph["temporal"], expected_graph['temporal'])

    def test_motion_graph_edges_pairs_t2(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        actual_graph = skel.motion_graph_edges_list(2)
        expected_graph = build_motion_graph_edges_list(2, 17)
        self.assertEqual(actual_graph["spatial"], expected_graph['spatial'])
        self.assertEqual(actual_graph["temporal"], expected_graph['temporal'])

    def test_motion_graph_edges_pairs_t3(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        actual_graph = skel.motion_graph_edges_list(3)
        expected_graph = build_motion_graph_edges_list(3, 17)
        self.assertEqual(actual_graph["spatial"], expected_graph['spatial'])
        self.assertEqual(actual_graph["temporal"], expected_graph['temporal'])

    def test_motion_graph_edges_pairs_vs_es(self):
        skel_pairs = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        skel_es = skeleton.SkeletonGraph(17, ES)
        actual_graph = skel_pairs.motion_graph_edges_list(3)
        expected_graph = skel_es.motion_graph_edges_list(3)
        self.assertEqual(actual_graph["spatial"], expected_graph['spatial'])
        self.assertEqual(actual_graph["temporal"], expected_graph['temporal'])

    @unittest.skip("HierarchicalSkeletonGraph not implemented yet")
    def test_motion_graph_skeletons(self):
        skel = skeleton.SkeletonGraph(17, skeleton.H36M_17_JOINTS_SKELETON_BONES_PAIRS)
        hskel = skeleton.HierarchicalSkeletonGraph(skeleton.H36M_17_JOINTS_SKELETON_PARENTS)
        actual_graph = hskel.motion_graph_edges_list(3)
        expected_graph = skel.motion_graph_edges_list(3)
        self.assertEqual(hskel.bones, skel.bones)
        self.assertEqual(hskel.get_joints_incidence(), skel.get_joints_incidence())
        self.assertEqual(actual_graph["spatial"], expected_graph['spatial'])
        self.assertEqual(actual_graph["temporal"], expected_graph['temporal'])

if __name__ == "__main__":
    unittest.main()