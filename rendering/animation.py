#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import numpy as np
from vpython import *

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

scene.width = 1920
scene.height = 1080
scene.autoscale = True
scene.background = color.gray(0.5)
# scene.autoscale = 0
RADIUS = 20.0
SCALE = 1.0

class Plot3DPose:
    def __init__(self, poses, translation=vec(.0, .0, .0), p_color=color.green):
        self.poses = np.reshape(poses * SCALE, [-1, 17, 3])
        self.translation = translation * SCALE
        self.color = p_color
        self.joint_count = 17
        self.bone_count = 16
        self.frames = poses.shape[0]
        self.bones = []
        self.joints = []

        self.make_joints()
        self.make_bones()

    def make_joints(self):
        for i in range(self.joint_count):
            self.joints.append(sphere(
                pos=self.get_position(0, i),
                radius=RADIUS + 5.0,
                color=self.color
            ))

    def make_bones(self):
        for i, j in ES:
            self.bones.append(
                cylinder(
                    pos=self.get_position(0, i),
                    axis=self.get_position(0, j) - self.get_position(0, i),
                    radius=RADIUS,
                    color=self.color
                )
            )

    def update(self, t):
        f = t % self.frames
        # update joints
        for index in range(self.joint_count):
            joint = self.joints[index]
            joint.pos = self.get_position(f, index)
        # update bones
        for index in range(self.bone_count):
            bone = self.bones[index]
            i, j = ES[index]
            bone.pos = self.get_position(f, i)
            bone.axis = self.get_position(f, j) - self.get_position(f, i)

    def get_position(self, t, i):
        return vec(self.poses[t][i][0],
                   -self.poses[t][i][1],
                   self.poses[t][i][2])


class Plot3DPoseVs:
    def __init__(self, poses_1, poses_2):
        self.poses_1 = np.reshape(poses_1, [-1, 17, 3])
        self.poses_2 = np.reshape(poses_2, [-1, 17, 3])
        self.translation = vec(1000, 0, 0)
        self.color_1 = color.green
        self.color_2 = color.yellow
        self.joint_count = 17
        self.bone_count = 16
        self.frames = poses_1.shape[0]
        self.bones_1 = []
        self.bones_2 = []
        self.joints_1 = []
        self.joints_2 = []

        self.make_joints()
        self.make_bones()

    def make_joints(self):
        for i in range(self.joint_count):
            self.joints_1.append(sphere(
                pos=self.get_position(0, i, 1),
                radius=RADIUS + 5.0,
                color=self.color_1
            ))
            self.joints_2.append(sphere(
                pos=self.get_position(0, i, 2)  + self.translation,
                radius=RADIUS + 5.0,
                color=self.color_2
            ))

    def make_bones(self):
        for i, j in ES:
            self.bones_1.append(
                cylinder(
                    pos=self.get_position(0, i, 1),
                    axis=self.get_position(0, j, 1) - self.get_position(0, i, 1),
                    radius=RADIUS,
                    color=self.color_1
                )
            )
            self.bones_2.append(
                cylinder(
                    pos=self.get_position(0, i, 2) + self.translation,
                    axis=self.get_position(0, j, 2) - self.get_position(0, i, 2),
                    radius=RADIUS,
                    color=self.color_2
                )
            )

    def update(self, t):
        f = t % self.frames
        # update joints
        for index in range(self.joint_count):
            joint = self.joints_1[index]
            joint.pos = self.get_position(f, index, 1)
            joint = self.joints_2[index]
            joint.pos = self.get_position(f, index, 2) + self.translation
        
        # update bones
        for index in range(self.bone_count):
            bone_1 = self.bones_1[index]
            i, j = ES[index]
            bone_1.pos = self.get_position(f, i, 1)
            bone_1.axis = self.get_position(f, j, 1) - self.get_position(f, i, 1)

            bone_2 = self.bones_2[index]
            i, j = ES[index]
            bone_2.pos = self.get_position(f, i, 2) + self.translation
            bone_2.axis = self.get_position(f, j, 2) - self.get_position(f, i, 2)

    def get_position(self, t, i, p):
        if p == 1:
            return vec(self.poses_1[t][i][0],
                       -self.poses_1[t][i][1],
                       self.poses_1[t][i][2])
        else:
            return vec(self.poses_2[t][i][0],
                       -self.poses_2[t][i][1],
                       self.poses_2[t][i][2])


class Plot3DPose3Way:
    def __init__(self, poses_1, poses_2, poses_3):
        self.poses_1 = np.reshape(poses_1, [-1, 17, 3])
        self.poses_2 = np.reshape(poses_2, [-1, 17, 3])
        self.poses_3 = np.reshape(poses_3, [-1, 17, 3])
        self.t1 = vec(500, 0, 0)
        self.t2 = vec(1000, 0, 0)
        self.color_1 = color.green
        self.color_2 = color.red
        self.color_3 = color.yellow
        # self.joint_count = 17
        self.bone_count = 16
        self.frames = poses_1.shape[0]
        self.bones_1 = []
        self.bones_2 = []
        self.bones_3 = []

        self.make_bones()

    def make_bones(self):
        for i, j in ES:
            self.bones_1.append(
                cylinder(
                    pos=self.get_position(0, i, 1),
                    axis=self.get_position(0, j, 1) - self.get_position(0, i, 1),
                    radius=RADIUS,
                    color=self.color_1
                )
            )
            self.bones_2.append(
                cylinder(
                    pos=self.get_position(0, i, 2) + self.t1,
                    axis=self.get_position(0, j, 2) - self.get_position(0, i, 2),
                    radius=RADIUS,
                    color=self.color_2
                )
            )
            self.bones_3.append(
                cylinder(
                    pos=self.get_position(0, i, 3) + self.t2,
                    axis=self.get_position(0, j, 3) - self.get_position(0, i, 3),
                    radius=RADIUS,
                    color=self.color_3
                )
            )

    def update(self, t):
        f = t % self.frames
        # update bones
        for index in range(self.bone_count):
            bone_1 = self.bones_1[index]
            i, j = ES[index]
            bone_1.pos = self.get_position(f, i, 1)
            bone_1.axis = self.get_position(f, j, 1) - self.get_position(f, i, 1)

            bone_2 = self.bones_2[index]
            i, j = ES[index]
            bone_2.pos = self.get_position(f, i, 2) + self.t1
            bone_2.axis = self.get_position(f, j, 2) - self.get_position(f, i, 2)

            bone_3 = self.bones_3[index]
            i, j = ES[index]
            bone_3.pos = self.get_position(f, i, 3) + self.t2
            bone_3.axis = self.get_position(f, j, 3) - self.get_position(f, i, 3)

    def get_position(self, t, i, p):
        if p == 1:
            return vec(self.poses_1[t][i][0],
                       -self.poses_1[t][i][1],
                       self.poses_1[t][i][2])
        elif p == 2:
            return vec(self.poses_2[t][i][0],
                       -self.poses_2[t][i][1],
                       self.poses_2[t][i][2])
        elif p == 3:
            return vec(self.poses_3[t][i][0],
                       -self.poses_3[t][i][1],
                       self.poses_3[t][i][2])


class Plot3DPose2vs2:
    def __init__(self, poses_1, poses_2, poses_3):
        self.poses_1 = np.reshape(poses_1, [-1, 17, 3])
        self.poses_2 = np.reshape(poses_2, [-1, 17, 3])
        self.poses_3 = np.reshape(poses_3, [-1, 17, 3])
        self.t = vec(1000, 0, 0)
        self.phase = vec(10, 0, 0)
        self.color_1 = color.green
        self.color_2 = color.red
        self.color_3 = color.yellow
        # self.joint_count = 17
        self.bone_count = 16
        self.frames = poses_1.shape[0]
        self.bones_gt1 = []
        self.bones_est = []
        self.bones_cor = []
        self.bones_gt2 = []

        self.make_bones()

    def make_bones(self):
        for i, j in ES:
            # Estimation
            self.bones_est.append(
                cylinder(
                    pos=self.get_position(0, i, 2) + self.phase,
                    axis=self.get_position(0, j, 2) - self.get_position(0, i, 2),
                    radius=RADIUS,
                    color=self.color_2
                )
            )
            # GT vs Estimation
            self.bones_gt1.append(
                cylinder(
                    pos=self.get_position(0, i, 1),
                    axis=self.get_position(0, j, 1) - self.get_position(0, i, 1),
                    radius=RADIUS,
                    color=self.color_1
                )
            )
            # Correction
            self.bones_cor.append(
                cylinder(
                    pos=self.get_position(0, i, 3) + self.t + self.phase,
                    axis=self.get_position(0, j, 3) - self.get_position(0, i, 3),
                    radius=RADIUS,
                    color=self.color_3
                )
            )
            # GT vs Correction
            self.bones_gt2.append(
                cylinder(
                    pos=self.get_position(0, i, 1) + self.t,
                    axis=self.get_position(0, j, 1) - self.get_position(0, i, 1),
                    radius=RADIUS,
                    color=self.color_1
                )
            )

    def update(self, t):
        f = t % self.frames
        # update bones
        for index in range(self.bone_count):
            # Estimation
            bone_est = self.bones_est[index]
            i, j = ES[index]
            bone_est.pos = self.get_position(f, i, 2) + self.phase
            bone_est.axis = self.get_position(f, j, 2) - self.get_position(f, i, 2)

            # GT vs Estimation
            bone_gt1 = self.bones_gt1[index]
            i, j = ES[index]
            bone_gt1.pos = self.get_position(f, i, 1)
            bone_gt1.axis = self.get_position(f, j, 1) - self.get_position(f, i, 1)

            # Correction
            bone_cor = self.bones_cor[index]
            i, j = ES[index]
            bone_cor.pos = self.get_position(f, i, 3) + self.t + self.phase
            bone_cor.axis = self.get_position(f, j, 3) - self.get_position(f, i, 3)

            # GT vs Estimation
            bone_gt2 = self.bones_gt2[index]
            i, j = ES[index]
            bone_gt2.pos = self.get_position(f, i, 1) + self.t
            bone_gt2.axis = self.get_position(f, j, 1) - self.get_position(f, i, 1)

    def get_position(self, t, i, p):
        if p == 1:
            return vec(self.poses_1[t][i][0],
                       -self.poses_1[t][i][1],
                       self.poses_1[t][i][2])
        elif p == 2:
            return vec(self.poses_2[t][i][0],
                       -self.poses_2[t][i][1],
                       self.poses_2[t][i][2])
        elif p == 3:
            return vec(self.poses_3[t][i][0],
                       -self.poses_3[t][i][1],
                       self.poses_3[t][i][2])

def save_video(video_name=None):
    import cv2 as cv
    import os
    import glob
    folder = "/home/mansour/Downloads"
    images = [img for img in glob.glob(f"{folder}/*.png") if "shots" in img]
    if video_name:
        video_name = "{}/demo.avi".format(folder)
    images.sort()

    # Set frame from the first image
    frame = cv.imread(os.path.join(folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv.imread(os.path.join(folder, image)))

    # Release the video file
    video.release()
    cv.destroyAllWindows()
    print("Video generated successfully!")
    # Delete files
    for filename in set(images):
        os.remove(filename)
    print("Successively delete files!")

def animate_motion(poses):
    plotter = Plot3DPose(poses)
    run = True
    t = 0
    while run:
        rate(10)
        if run:
            plotter.update(t)
            scene.capture(f"shots{t:06d}")
            t += 1
        if t > plotter.frames:
            run = False

def save_animate_motion(poses, filename):
    plotter = Plot3DPose(poses)
    run = True
    t = 0
    while run:
        rate(10)
        if run:
            plotter.update(t)
            scene.capture(f"shots{t:06d}")
            t += 1
        if t > plotter.frames:
            run = False
    if not run:
        save_video(filename)


def animate_motions_vs(poses_1, poses_2):
    plotter = Plot3DPoseVs(poses_1, poses_2)
    run = True
    t = 1
    import time
    time.sleep(5)
    while True:
        rate(30)
        if run:
            plotter.update(t)
            t += 1


def animate_motions_3way(poses_1, poses_2, poses_3):
    plotter = Plot3DPose3Way(poses_1, poses_2, poses_3)
    run = True
    t = 1
    import time
    time.sleep(5)
    while True:
        rate(30)
        if run:
            plotter.update(t)
            t += 1


def animate_motions_2vs2(poses_1, poses_2, poses_3):
    plotter = Plot3DPose2vs2(poses_1, poses_2, poses_3)
    run = True
    t = 1
    while True:
        rate(30)
        if run:
            plotter.update(t)
            t += 1
