#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import os
import tqdm
import cv2
import glob
# from lib.preprocess import h36m_coco_format, revise_kpts
# from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import utility.ops as ops
from model import ops

# 17 joints
SKELETON_PAIRS_17 = [[0, 1], [0, 4], [0, 7], [7, 8],  # hips and spine
                     [1, 2], [2, 3],  # left leg
                     [4, 5], [5, 6],  # right leg
                     [8, 11], [8, 14],  # shoulders
                     [8, 9], [9, 10],  # head
                     [11, 12], [12, 13],  # right arm
                     [14, 15], [15, 16]  # left arm
                     ]

def show_image(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def show_pose_2D(kps, img):
    LR = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(SKELETON_PAIRS_17):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
    re_kpts = revise_kpts(keypoints, scores, valid_frames)
    print('Generating 2D pose successful!')

    output_dir += 'input_2d/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)

    print(keypoints.shape)
    print(scores.shape)
    kpts = ops.normalize_screen_coordinates(keypoints, width, height)
    kpts = np.concatenate([kpts, np.expand_dims(scores, axis=-1)], axis=-1)
    np.savez_compressed(f'{output_dir}data.npz', reconstruction=kpts)
    return keypoints, scores, kpts


def show_pose_3D(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0, 0, 1)
    rcolor=(1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def show3Dpose(vals, ax, color):
    ax.view_init(elev=15., azim=70)

    # lcolor=(0, 0, 1)
    # rcolor=(1, 0, 0)

    I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        # ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2, color = color)
    for i in np.arange( len(I) ):
        x, z, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.scatter(x, y, z, lw=1, color='black')

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.invert_zaxis()
    ax.invert_yaxis()
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def image_to_video(images_folder, output_folder):
    video_path = f"{output_folder}/demo.mp4"
    if os.path.exists(video_path):
        os.remove(video_path)
    import ffmpeg
    (
    ffmpeg
    .input(f"{images_folder}/*.png", pattern_type='glob', framerate=24)
    .output(video_path)
    .overwrite_output()
    .run()
    )


def generate_image(image_2d_dir, image_3d_dir, output_dir):
    image_2d_dir = sorted(glob.glob(os.path.join(image_2d_dir, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(image_3d_dir, '*.png')))
    for i in tqdm.tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        show_image(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        show_image(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'/images/'
        os.makedirs(output_dir_pose, exist_ok=True)
        # plt.axis('off')
        # plt.gcf().set_size_inches(512 / 100, 512 / 100)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%06d'% i)) + '.png', dpi=200, bbox_inches = 'tight')
        plt.close(fig)


def generate_vs_image(image_2d_dir, image_3d_dir, image_cor_3d_dir, output_dir):
    image_2d_dir = sorted(glob.glob(os.path.join(image_2d_dir, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(image_3d_dir, '*.png')))
    image_cor_3d_dir = sorted(glob.glob(os.path.join(image_cor_3d_dir, '*.png')))
    for i in tqdm.tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])
        image_cor_3d = plt.imread(image_cor_3d_dir[i])

        ## show
        font_size = 12
        fig = plt.figure(figsize=(12, 8))
        ax = plt.subplot(131)
        show_image(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(132)
        show_image(ax, image_3d)
        ax.set_title("Estimation (AANet)", fontsize = font_size)

        ax = plt.subplot(133)
        show_image(ax, image_cor_3d)
        ax.set_title("After Correction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'/images/'
        os.makedirs(output_dir_pose, exist_ok=True)
        # plt.axis('off')
        # plt.gcf().set_size_inches(512 / 100, 512 / 100)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%06d'% i)) + '.png', dpi=200, bbox_inches = 'tight')
        plt.close(fig)


def generate_curves(estimation, correction, output_dir):
    length = estimation.shape[0]
    est_acc = ops.get_motion_dsc_per_part(ops.get_acceleration(estimation), None)
    est_vel = ops.get_motion_dsc_per_part(ops.get_velocity(estimation), None)
    est_bone = ops.get_skeleton_dsc_per_part(ops.get_bones_length_variations(estimation), None)
    cor_acc = ops.get_motion_dsc_per_part(ops.get_acceleration(correction), None)
    cor_vel = ops.get_motion_dsc_per_part(ops.get_velocity(correction), None)
    cor_bone = ops.get_skeleton_dsc_per_part(ops.get_bones_length_variations(correction), None)
    
    for i in tqdm.tqdm(range(length)):
        # r = np.arange(start=0, stop=length, step=1)
        ## show
        fig = plt.figure(figsize=(12.8, 7.2))
        # velocity [0:i+1]
        ax = plt.subplot(311)
        if i>=1:
            ax.plot(est_vel[0:i], c="red", linestyle="-")
            ax.plot(cor_vel[0:i], c="blue", linestyle="-")
        ax.set_xlim([0, length])
        ax.set_ylabel("vél. (mm/f)", fontsize=16)

        # acceleration
        ax = plt.subplot(312)
        if i >=2 :
            ax.plot(est_acc[0:i-1], c="red", linestyle="-")
            ax.plot(cor_acc[0:i-1], c="blue", linestyle="-")
        ax.set_xlim([0, length])
        ax.set_ylabel("accél. (mm/f²)", fontsize=16)
        # bone length 
        ax = plt.subplot(313)
        ax.plot(est_bone[0:i+1], c="red", linestyle="-", label="estimation")
        ax.plot(cor_bone[0:i+1], c="blue", linestyle="-", label="correction")
        ax.set_xlim([0, length])
        ax.set_ylabel("longueur os (mm)", fontsize=16)

        # ax.set_xlabel("frames")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', fontsize=18)

        ## save
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir + str(('%06d'% i)) + '.png', dpi=100)
        plt.close(fig)


def generate_vs_image_curves(image_2d_dir, image_3d_dir, image_cor_3d_dir, curves_dir, output_dir):
    image_2d_dir = sorted(glob.glob(os.path.join(image_2d_dir, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(image_3d_dir, '*.png')))
    image_cor_3d_dir = sorted(glob.glob(os.path.join(image_cor_3d_dir, '*.png')))
    curves_dir = sorted(glob.glob(os.path.join(curves_dir, '*.png')))
    for i in tqdm.tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])
        image_cor_3d = plt.imread(image_cor_3d_dir[i])
        curves = plt.imread(curves_dir[i])

        ## show
        font_size = 18
        fig = plt.figure(figsize=(25.6, 14.4))
        ax = plt.subplot(221)
        show_image(ax, image_2d)
        # ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(222)
        show_image(ax, curves)
        ax.set_title("Courbes", fontsize = font_size, y=0.0)

        ax = plt.subplot(223)
        show_image(ax, image_3d)
        ax.set_title("Estimation", fontsize = font_size, y=0.9)

        ax = plt.subplot(224)
        show_image(ax, image_cor_3d)
        ax.set_title("Correction", fontsize = font_size, y=0.9)

        ## save
        output_dir_pose = output_dir +'/images/'
        os.makedirs(output_dir_pose, exist_ok=True)
        # plt.axis('off')
        # plt.gcf().set_size_inches(512 / 100, 512 / 100)
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%06d'% i)) + '.png', dpi=100)#, bbox_inches = 'tight')
        plt.close(fig)
