#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 03.2025
"""

# %% Imports


# %% Constants

H36M_17_JOINTS_SKELETON_BONES_PAIRS_NO_LR = [
    [0, 1], [0, 4], [0, 7], [7, 8],  # hips and spine
    [1, 2], [2, 3],  # left leg
    [4, 5], [5, 6],  # right leg
    [8, 11], [8, 14],  # shoulders
    [8, 9], [9, 10],  # head
    [11, 12], [12, 13],  # right arm
    [14, 15], [15, 16]  # left arm
]

H36M_17_JOINTS_SKELETON_BONES_PAIRS = [
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

H36M_17_JOINTS_SKELETON_INCIDENT_EDGES = [
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

H36M_17_JOINTS_SKELETON_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# %% Skeleton Class
class SkeletonGraph:

    def __init__(self, num_of_joints, edges_pairs):
        """Creates an instance of SkeletonGraph (undirected graph).

        Args:
            num_of_joints (int): The number of joints in the skeletal structure.
            edges_pairs (list): A list of edges connecting the joints in the skeletal structure.
        """
        self.__num_of_joints = num_of_joints
        self.__bones = edges_pairs
        self.__joints_incidence = [[] for _ in range(num_of_joints)]
        for pair in edges_pairs:
            self.__joints_incidence[pair[0]].append(pair[1])
            self.__joints_incidence[pair[1]].append(pair[0])
        for joint_list in self.__joints_incidence:
            joint_list.sort()

    def get_skeleton_bones(self):
        return self.__bones
    
    def get_joints_incidence(self):
        return self.__joints_incidence
    
    def get_num_of_joints(self):
        return self.__num_of_joints
    
    def motion_graph_edges_pairs(self, seq_length : int):
        """Returns a dictionary of spatial and temporal edges after construction of the
        motion graph based on the skeletal structure and length of the motion. Edges are
        represented as pairs `[i, j]`.

        Args:
            seq_length (int): The length of the motion sequence in number of frames.

        Returns:
            dict: A dictionary of both spatial and temporal edges of the motion graph.
        """
        spatial_edges = []
        temporal_edges = []
        for t in range(0, seq_length):
            for i, j in self.__bones:
                # spatial links per skeleton
                spatial_edges.append([i + (t*self.__num_of_joints) , j + (t*self.__num_of_joints)])
            # Link to same joint for next and previous frames
            if seq_length > 1 and t < seq_length - 1:
                for i in range(self.__num_of_joints):
                    temporal_edges.append([i + (t*self.__num_of_joints),
                                           i + ((t+1)*self.__num_of_joints)])
        return {"spatial": spatial_edges, "temporal": temporal_edges}
    

    def motion_graph_edges_list(self, seq_length : int):
        """Returns a dictionary of spatial and temporal edges after construction of the
        motion graph based on the skeletal structure and length of the motion. Edges are represented
        per vertex with the list of adjacent vertices.

        Args:
            seq_length (int): _description_

        Returns:
            dict:  A dictionary of both spatial and temporal edges of the motion graph.
        """
        spatial_edges = []
        temporal_edges = []
        for t in range(0, seq_length):
            for i in range(self.__num_of_joints):
                # spatial links per skeleton
                spatial_edges.append([j + (t*self.__num_of_joints) for j in self.get_joints_incidence()[i]])
                # Temporal links (previous and/or next frames)
                t_edges = []
                if seq_length > 1:
                    if t == seq_length - 1:
                        t_edges.append((t-1)*self.__num_of_joints + i)
                    elif t == 0:
                        t_edges.append((t+1)*self.__num_of_joints + i)
                    else:
                        t_edges.append((t-1)*self.__num_of_joints + i)
                        t_edges.append((t+1)*self.__num_of_joints + i)
                    temporal_edges.append(t_edges)
        return {"spatial": spatial_edges, "temporal": temporal_edges}


class HierarchicalSkeletonGraph:
    # TODO Complete class to add hierarchy
    def __init__(self, parents):
        self.num_of_joints = len(parents)
        self.bones = []
        self.joints_incidence = [[] for _ in range(self.num_of_joints)]
        for i, parent in enumerate(parents):
            if parent != -1:
                self.joints_incidence[parent].append(i)
                self.joints_incidence[i].append(parent)
                self.bones.append([parent, i])
        for joint_list in self.joints_incidence:
            joint_list.sort()
        self.children = []
        for _ in range(self.num_of_joints):
            self.children.append([])
        for i, child in enumerate(parents):
            self.children[i].append(child)
    
    def motion_graph_edges_pairs(self, seq_length : int):
        spatial_edges = []
        temporal_edges = []
        for t in range(0, seq_length):
            for i, j in self.bones:
                # spatial links per skeleton
                spatial_edges.append([i + (t*self.num_of_joints) , j + (t*self.num_of_joints)])
                # Link to same joint for next and previous frames
                if seq_length > 1:
                    if t < seq_length - 1:
                        temporal_edges.append([t + i, t + i + self.num_of_joints])
        return {"spatial": spatial_edges, "temporal": temporal_edges}

    def motion_graph_edges_list(self, seq_length : int):
        spatial_edges = []
        temporal_edges = []
        for t in range(0, seq_length):
            for i in range(self.num_of_joints):
                # spatial links per skeleton
                spatial_edges.append([j + (t*self.num_of_joints) for j in self.get_joints_incidence()[i]])
                # Temporal links (previous and/or next frames)
                t_edges = []
                if seq_length > 1:
                    if t == seq_length - 1:
                        t_edges.append((t-1)*self.num_of_joints + i)
                    elif t == 0:
                        t_edges.append((t+1)*self.num_of_joints + i)
                    else:
                        t_edges.append((t-1)*self.num_of_joints + i)
                        t_edges.append((t+1)*self.num_of_joints + i)
                    temporal_edges.append(t_edges)
        return {"spatial": spatial_edges, "temporal": temporal_edges}

