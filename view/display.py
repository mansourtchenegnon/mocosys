#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from PIL import Image
import imageio.v2 as imageio
import numpy as np
import os
# from mpl_toolkits.mplot3d import Axes3D
import cv2 as cv

# 17 joints
SKELETON_PAIRS_17 = [[0, 1], [0, 4], [0, 7], [7, 8],  # hips and spine
                     [1, 2], [2, 3],  # left leg
                     [4, 5], [5, 6],  # right leg
                     [8, 11], [8, 14],  # shoulders
                     [8, 9], [9, 10],  # head
                     [11, 12], [12, 13],  # right arm
                     [14, 15], [15, 16]  # left arm
                     ]

# 25 joints
SKELETON_PAIRS_25 = [[0, 1], [0, 4], [0, 7], [7, 8],  # hips and spine
                     [1, 2], [2, 3],  # left leg
                     [4, 5], [5, 6],  # right leg
                     [8, 11], [8, 14],  # shoulders
                     [8, 9], [9, 10],  # head
                     [11, 12], [12, 13],  # left arm
                     [14, 15], [15, 16]  # right arm
                     ]

# 31 joints
SKELETON_PAIRS_31 = [[0, 1], [0, 6], [0, 11],  # hips
                     [1, 2], [2, 3], [3, 4], [4, 5],  # left leg
                     [6, 7], [7, 8], [8, 9], [9, 10],  # right leg
                     [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],  # spine
                     [13, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [21, 23],  # left arm
                     [13, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [28, 30]  # right arm
                     ]

LR_BONES = np.array([1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
LR_SK_BONES = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0], dtype=bool)
R_COLOR_HEX = '#0d47a1'
R_COLOR = (136, 0, 97, 255)
L_COLOR_HEX = '#c62828'
L_COLOR = (74, 38, 253, 255)
LR_JOINTS = [R_COLOR, L_COLOR, L_COLOR, L_COLOR, R_COLOR, R_COLOR, R_COLOR, R_COLOR, R_COLOR, R_COLOR, R_COLOR,
             R_COLOR, R_COLOR, R_COLOR, L_COLOR, L_COLOR, L_COLOR]
# LR_JOINTS_HEX = [R_COLOR_HEX, L_COLOR_HEX, L_COLOR_HEX, L_COLOR_HEX, R_COLOR_HEX, R_COLOR_HEX, R_COLOR_HEX,
#                  R_COLOR_HEX, R_COLOR_HEX, R_COLOR_HEX, R_COLOR_HEX, R_COLOR_HEX, R_COLOR_HEX, R_COLOR_HEX,
#                  L_COLOR_HEX, L_COLOR_HEX, L_COLOR_HEX]


def show_verbose(verbose, message):
    """
    Display verbose message

    Parameters
    ----------
    verbose : bool
        Tells if message should be displayed.
    message : str
        Message to display.

    Returns
    -------
    None.

    """
    if verbose:
        print(message)

def generate_video(folder):
    images_folder = "{}/images".format(folder)
    images = [img for img in os.listdir(images_folder) if img.endswith((".png"))]
    video_name = "{}/demo.avi".format(folder)
    images.sort()

    # Set frame from the first image
    frame = cv.imread(os.path.join(images_folder, images[0]))
    height, width, layers = frame.shape

    # Video writer to create .avi file
    video = cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

    # Appending images to video
    for image in images:
        video.write(cv.imread(os.path.join(images_folder, image)))

    # Release the video file
    video.release()
    cv.destroyAllWindows()
    print("Video generated successfully!")

def generate_3d_poses_fixed(frames, out_folder,
                            l_color="#3498db",
                            r_color="#e74c3c",
                            add_labels=True,
                            label_min=None,
                            label_max=None):  # blue, orange
    filenames = []
    index = 0
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    images_folder = "{}/images".format(out_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    for kps in frames:
        vals = np.reshape(kps, (-1, 3))
        vi = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])
        vj = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        lr = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)
        vals[:, [1, 2]] = vals[:, [2, 1]]

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes((0, 0, 1, 1), projection='3d')
        # Make connection matrix
        for i in np.arange(len(vi)):
            x, y, z = [np.array([vals[vi[i], j], vals[vj[i], j]]) for j in range(3)]
            ax.plot(x, z, y, lw=2, c=l_color if lr[i] else r_color)

        if add_labels:
            ax.set_xlabel("x")
            ax.set_ylabel("z")
            ax.set_zlabel("y")

        if label_min is not None and label_max is not None:
            ax.set_xlim3d([label_min, label_max])
            ax.get_xaxis().set_ticklabels(list(range(int(label_min), int(label_max), int((label_max - label_min) / 3))))
            ax.set_ylim3d([label_min, label_max])
            ax.get_yaxis().set_ticklabels(list(range(int(label_min), int(label_max), int((label_max - label_min) / 3))))
            ax.set_zlim3d([label_min, label_max])
            ax.set_zticklabels(list(range(int(label_min), int(label_min), int((label_max - label_min) / 3))))

        # ax.set_aspect('auto')

        white = (1.0, 1.0, 1.0, 0.0)
        ax.w_xaxis.set_pane_color(white)
        ax.w_yaxis.set_pane_color(white)

        ax.w_xaxis.line.set_color(white)
        ax.w_yaxis.line.set_color(white)
        ax.w_zaxis.line.set_color(white)

        filename = "{}/{}.png".format(images_folder, index)
        filenames.append(filename)
        plt.savefig(filename, dpi=100)
        plt.close()
        index += 1

    # Build gif
    ims = []
    fig = plt.figure()
    ax = fig.gca()
    ax.set_axis_off()
    for filename in filenames:
        im = plt.imshow(imageio.imread(filename))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=500)

    writer = animation.PillowWriter(fps=24)
    ani.save("{}/demo.gif".format(out_folder), writer=writer)
    plt.close()

    # show_verbose(verbose, '[INFO] Removing images')
    # # Remove files
    # for filename in set(filenames):
    #     os.remove(filename)


def generate_3d_animation(frames,
                          out_folder,
                          radius=600,
                          verbose=True,
                          frame_on=True,
                          keep_images=False):
    """
    Generates animation gif from sequence of poses.

    Parameters
    ----------
    frames : numpy.ndarray
        Sequence of poses.
    out_folder : string
        Output file name. It must be a GIF. The default is "demo.gif".
    radius : int, optional
        Space around the center.
    verbose : bool, optional
        Tells if logs should be print or not. The default is False.
    frame_on : bool, optional
        Tells if plot frame should be kept. The default is True.
    keep_images : bool, optional
        Tells if frame images should be kept. The default is False.

    Returns
    -------
    None.

    """
    filenames = []
    idx = 0
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    images_folder = "{}/images".format(out_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    for f in range(frames.shape[0]):
        kps = frames[f]
        xs = []
        ys = []
        zs = []
        for i in range(len(kps)):
            xs.append(kps[i][0])
            zs.append(kps[i][1])
            ys.append(kps[i][2])
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes((0, 0, 1, 1), projection='3d')
        # vals = np.reshape(kps, (-1, 3))
        # vals[:, [1, 2]] = vals[:, [2, 1]]
        for idx, pair in enumerate(SKELETON_PAIRS_17):
            # i, j = pair
            # x, y, z = [np.array([vals[i, k], vals[j, k]]) for k in range(3)]
            x = [xs[pair[0]], xs[pair[1]]]
            y = [ys[pair[0]], ys[pair[1]]]
            z = [zs[pair[0]], zs[pair[1]]]
            ax.plot(x, y, z, linewidth=4, c=R_COLOR_HEX if LR_SK_BONES[idx] else L_COLOR_HEX)
        # ax.scatter(xs, ys, zs, c='black', linewidth=8)

        # Space around the subject
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])

        ax.invert_zaxis()
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        if not frame_on:
            # Get rid of the ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.set_zticklabels([])

            # Get rid of the panes (actually, make them white)
            white = (1.0, 1.0, 1.0, 0.0)
            ax.w_xaxis.set_pane_color(white)
            ax.w_yaxis.set_pane_color(white)
            # Keep z pane

            # Get rid of the lines in 3d
            ax.w_xaxis.line.set_color(white)
            ax.w_yaxis.line.set_color(white)
            ax.w_zaxis.line.set_color(white)

        ax.view_init(azim=-90., elev=15.)
        filename = "{}/{:06d}.png".format(images_folder, f)
        filenames.append(filename)
        # print(f"\r{filename}", end='')
        plt.savefig(filename, dpi=200, bbox_inches = 'tight')
        plt.close()
        # index += 1

    show_verbose(verbose, "[INFO] Images saved")

    generate_video(out_folder)

    if not keep_images:
        show_verbose(verbose, '[INFO] Removing images')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        os.rmdir(images_folder)
    show_verbose(verbose, '[INFO] DONE')


def generate_2d_animation(frames, out_folder="test", verbose=True, keep_images=False):
    """
    Generates animation gif from sequence of poses.

    Parameters
    ----------
    frames : numpy.ndarray
        Sequence of poses.
    out_folder : string
        Output folder name. The default is "test".
    verbose : bool, optional
        Print log if True.
    keep_images : bool, optional
        Tells if frame images should be kept. The default is False.

    Returns
    -------
    None.

    """
    filenames = []
    index = 0
    images_folder = "{}/images".format(out_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    for kps in frames:
        skeleton = np.ones((600, 600, 4), dtype=np.uint8) * 255
        xs = []
        ys = []
        kps = np.around(kps * 600)
        # x_c = kps[0]
        # y_c = kps[1]
        for i in range(17):
            xs.append(int(kps[i * 2]))
            ys.append(int(kps[i * 2 + 1]))
            # scale 2x
            # xs.append((kps[i * 2] - x_c) * 2 + x_c)
            # ys.append((kps[i * 2 + 1] - y_c) * 2 + y_c)
        for pair in SKELETON_PAIRS_17:
            cv.line(skeleton, (int(xs[pair[0]]), int(ys[pair[0]])), (int(xs[pair[1]]), int(ys[pair[1]])),
                     LR_JOINTS[pair[0]], 4)
        for k in range(17):
            cv.circle(skeleton, (int(xs[k]), int(ys[k])), 4, (0, 0, 0, 255), thickness=-1, lineType=cv.FILLED)
        filename = "{}/{:06d}.png".format(images_folder, index)
        filenames.append(filename)
        cv.imwrite(filename, skeleton)
        index += 1

    show_verbose(verbose, "[INFO] Images saved")

    generate_video(out_folder)

    show_verbose(verbose, '[INFO] Removing images')
    # Remove files
    if not keep_images:
        show_verbose(verbose, '[INFO] Removing images')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        os.rmdir(images_folder)
    show_verbose(verbose, '[INFO] DONE')


def generate_versus_3d_animation(gt, est, out_folder, radius=600, verbose=True, frame_on=True, keep_images=False):
    """
    Generates animation gif from sequence of poses.

    Parameters
    ----------
    gt : numpy.ndarray
        Sequence of poses (ground truth).
    est : numpy.ndarray
        Sequence of poses (estimation).
    out_folder : string
        Output file name. It must be a GIF. The default is "demo.gif".
    radius : int, optional
        Space around the center.
    verbose : bool, optional
        Tells if logs should be print or not. The default is False.
    frame_on : bool, optional
        Tells if plot frame should be kept. The default is True.
    keep_images : bool, optional
        Tells if plot frame should be kept. The default is True.

    Returns
    -------
    None.

    """
    filenames = []
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    images_folder = "{}/images".format(out_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    for index in range(gt.shape[0]):
        xs_gt, ys_gt, zs_gt = [], [], []
        xs_est, ys_est, zs_est = [], [], []
        for i in range(0, len(gt[0]), 3):
            xs_gt.append(gt[index][i])
            zs_gt.append(gt[index][i + 1])
            ys_gt.append(gt[index][i + 2])
            xs_est.append(est[index][i])
            zs_est.append(est[index][i + 1])
            ys_est.append(est[index][i + 2])
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes((0, 0, 1, 1), projection='3d')
        # draw gt
        for pair in SKELETON_PAIRS_17:
            x = [xs_gt[pair[0]], xs_gt[pair[1]]]
            y = [ys_gt[pair[0]], ys_gt[pair[1]]]
            z = [zs_gt[pair[0]], zs_gt[pair[1]]]
            ax.plot(x, y, z, color='green', linewidth=8, alpha=0.7)
        ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)

        # draw estimation
        for pair in SKELETON_PAIRS_17:
            x = [xs_est[pair[0]], xs_est[pair[1]]]
            y = [ys_est[pair[0]], ys_est[pair[1]]]
            z = [zs_est[pair[0]], zs_est[pair[1]]]
            ax.plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
        ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)

        # Space around the subject
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])

        ax.invert_zaxis()
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        if not frame_on:
            # Get rid of the ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.set_zticklabels([])

            # Get rid of the panes (actually, make them white)
            white = (1.0, 1.0, 1.0, 0.0)
            ax.w_xaxis.set_pane_color(white)
            ax.w_yaxis.set_pane_color(white)
            # Keep z pane

            # Get rid of the lines in 3d
            ax.w_xaxis.line.set_color(white)
            ax.w_yaxis.line.set_color(white)
            ax.w_zaxis.line.set_color(white)

        ax.view_init(azim=-20.)
        filename = "{}/{:06d}.png".format(images_folder, index)
        filenames.append(filename)
        plt.savefig(filename, dpi=100)
        plt.close()
        index += 1

    show_verbose(verbose, "[INFO] Images saved")

    import ffmpeg
    generate_video(out_folder)

    if not keep_images:
        show_verbose(verbose, '[INFO] Removing images')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        os.rmdir(images_folder)
    show_verbose(verbose, '[INFO] DONE')


def generate_3_versus_3d_animation(gt, est, cor, out_folder, radius=600, verbose=True, frame_on=True, keep_images=False):
    """
    Generates animation gif from sequence of poses.

    Parameters
    ----------
    gt : numpy.ndarray
        Sequence of poses (ground truth).
    est : numpy.ndarray
        Sequence of poses (estimation).
    est : numpy.ndarray
        Sequence of poses (correction).
    out_folder : string
        Output file name. It must be a GIF. The default is "demo.gif".
    radius : int, optional
        Space around the center.
    verbose : bool, optional
        Tells if logs should be print or not. The default is False.
    frame_on : bool, optional
        Tells if plot frame should be kept. The default is True.
    keep_images : bool, optional
        Tells if plot frame should be kept. The default is True.

    Returns
    -------
    None.

    """
    filenames = []
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    images_folder = "{}/images".format(out_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    for index in range(gt.shape[0]):
        xs_gt, ys_gt, zs_gt = [], [], []
        xs_est, ys_est, zs_est = [], [], []
        xs_cor, ys_cor, zs_cor = [], [], []
        for i in range(0, len(gt[0]), 3):
            xs_gt.append(gt[index][i])
            zs_gt.append(gt[index][i + 1])
            ys_gt.append(gt[index][i + 2])
            xs_est.append(est[index][i])
            zs_est.append(est[index][i + 1])
            ys_est.append(est[index][i + 2])
            xs_cor.append(cor[index][i])
            zs_cor.append(cor[index][i + 1])
            ys_cor.append(cor[index][i + 2])
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_axes((0, 0, 1, 1), projection='3d')
        # draw gt
        for pair in SKELETON_PAIRS_17:
            x = [xs_gt[pair[0]], xs_gt[pair[1]]]
            y = [ys_gt[pair[0]], ys_gt[pair[1]]]
            z = [zs_gt[pair[0]], zs_gt[pair[1]]]
            ax.plot(x, y, z, color='green', linewidth=8, alpha=0.7)
        ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)

        # draw estimation
        for pair in SKELETON_PAIRS_17:
            x = [xs_est[pair[0]], xs_est[pair[1]]]
            y = [ys_est[pair[0]], ys_est[pair[1]]]
            z = [zs_est[pair[0]], zs_est[pair[1]]]
            ax.plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
        ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)

        # draw correction
        for pair in SKELETON_PAIRS_17:
            x = [xs_cor[pair[0]], xs_cor[pair[1]]]
            y = [ys_cor[pair[0]], ys_cor[pair[1]]]
            z = [zs_cor[pair[0]], zs_cor[pair[1]]]
            ax.plot(x, y, z, color='blue', linewidth=8, alpha=0.7)
        ax.scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=10, alpha=0.8)

        # Space around the subject
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])

        ax.invert_zaxis()
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        if not frame_on:
            # Get rid of the ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.set_zticklabels([])

            # Get rid of the panes (actually, make them white)
            white = (1.0, 1.0, 1.0, 0.0)
            ax.w_xaxis.set_pane_color(white)
            ax.w_yaxis.set_pane_color(white)
            # Keep z pane

            # Get rid of the lines in 3d
            ax.w_xaxis.line.set_color(white)
            ax.w_yaxis.line.set_color(white)
            ax.w_zaxis.line.set_color(white)

        ax.view_init(azim=-20.)
        filename = "{}/{}.png".format(images_folder, index)
        filenames.append(filename)
        plt.savefig(filename, dpi=100)
        plt.close()
        index += 1

    show_verbose(verbose, "[INFO] Images saved")

    # Build gif
    show_verbose(verbose, '[INFO] Creating gif')
    ims = []
    fig = plt.figure()
    ax = fig.gca()
    ax.set_axis_off()
    for filename in filenames:
        im = plt.imshow(imageio.imread(filename))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=500)

    writer = animation.PillowWriter(fps=24)
    ani.save("{}/demo.gif".format(out_folder), writer=writer)
    plt.close()
    show_verbose(verbose, '[INFO] Gif saved')

    if not keep_images:
        show_verbose(verbose, '[INFO] Removing images')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        os.rmdir(images_folder)
    show_verbose(verbose, '[INFO] DONE')


# def generate_3d_vs_plots_animation(gt, est, cor, out_folder, radius=600, verbose=True, frame_on=True, keep_images=False):
#     """
#     Generates animation gif from sequence of poses.

#     Parameters
#     ----------
#     gt : numpy.ndarray
#         Sequence of poses (ground truth).
#     est : numpy.ndarray
#         Sequence of poses (estimation).
#     est : numpy.ndarray
#         Sequence of poses (correction).
#     out_folder : string
#         Output file name. It must be a GIF. The default is "demo.gif".
#     radius : int, optional
#         Space around the center.
#     verbose : bool, optional
#         Tells if logs should be print or not. The default is False.
#     frame_on : bool, optional
#         Tells if plot frame should be kept. The default is True.
#     keep_images : bool, optional
#         Tells if plot frame should be kept. The default is True.

#     Returns
#     -------
#     None.

#     """
#     filenames = []
#     if not os.path.exists(out_folder):
#         os.makedirs(out_folder)
#     images_folder = "{}/images".format(out_folder)
#     if not os.path.exists(images_folder):
#         os.makedirs(images_folder)

#     for index in range(gt.shape[0]):
#         xs_gt, ys_gt, zs_gt = [], [], []
#         xs_est, ys_est, zs_est = [], [], []
#         xs_cor, ys_cor, zs_cor = [], [], []
#         for i in range(0, len(gt[0]), 3):
#             xs_gt.append(gt[index][i])
#             zs_gt.append(gt[index][i + 1])
#             ys_gt.append(gt[index][i + 2])
#             xs_est.append(est[index][i])
#             zs_est.append(est[index][i + 1])
#             ys_est.append(est[index][i + 2])
#             xs_cor.append(cor[index][i])
#             zs_cor.append(cor[index][i + 1])
#             ys_cor.append(cor[index][i + 2])
#         # fig = plt.figure(figsize=(24, 8))
#         # ax = fig.add_axes((0, 0, 1, 1), projection='3d')
#         fig, axs = plt.subplots(1, 3, figsize=(24, 8), projection='3d')
#         # draw gt
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_gt[pair[0]], xs_gt[pair[1]]]
#             y = [ys_gt[pair[0]], ys_gt[pair[1]]]
#             z = [zs_gt[pair[0]], zs_gt[pair[1]]]
#             axs[0].plot(x, y, z, color='green', linewidth=8, alpha=0.7)
#         axs[0].scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)
#         axs[0].title.set_text("GT")

#         # draw estimation
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_est[pair[0]], xs_est[pair[1]]]
#             y = [ys_est[pair[0]], ys_est[pair[1]]]
#             z = [zs_est[pair[0]], zs_est[pair[1]]]
#             axs[1].plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
#         axs[1].scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)
#         axs[1].title.set_text("Estimation")

#         # draw correction
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_cor[pair[0]], xs_cor[pair[1]]]
#             y = [ys_cor[pair[0]], ys_cor[pair[1]]]
#             z = [zs_cor[pair[0]], zs_cor[pair[1]]]
#             axs[2].plot(x, y, z, color='blue', linewidth=8, alpha=0.7)
#         axs[2].scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=10, alpha=0.8)
#         axs[2].title.set_text("Correction")

#         # Space around the subject
#         axs[0].set_xlim([-radius, radius])
#         axs[0].set_ylim([-radius, radius])
#         axs[0].set_zlim([-radius, radius])

#         axs[0].invert_zaxis()
#         axs[0].get_xaxis().set_ticklabels([])
#         axs[0].get_yaxis().set_ticklabels([])
#         axs[0].set_zticklabels([])

#         axs[1].set_xlim([-radius, radius])
#         axs[1].set_ylim([-radius, radius])
#         axs[1].set_zlim([-radius, radius])

#         axs[1].invert_zaxis()
#         axs[1].get_xaxis().set_ticklabels([])
#         axs[1].get_yaxis().set_ticklabels([])
#         axs[1].set_zticklabels([])

#         axs[2].set_xlim([-radius, radius])
#         axs[2].set_ylim([-radius, radius])
#         axs[2].set_zlim([-radius, radius])

#         axs[2].invert_zaxis()
#         axs[2].get_xaxis().set_ticklabels([])
#         axs[2].get_yaxis().set_ticklabels([])
#         axs[2].set_zticklabels([])

#         if not frame_on:
#             white = (1.0, 1.0, 1.0, 0.0)
#             # Get rid of the ticks and tick labels
#             axs[0].set_xticks([])
#             axs[0].set_yticks([])
#             axs[0].set_zticks([])
#             axs[0].get_xaxis().set_ticklabels([])
#             axs[0].get_yaxis().set_ticklabels([])
#             axs[0].set_zticklabels([])
#             axs[1].set_xticks([])
#             axs[1].set_yticks([])
#             axs[1].set_zticks([])
#             axs[1].get_xaxis().set_ticklabels([])
#             axs[1].get_yaxis().set_ticklabels([])
#             axs[1].set_zticklabels([])
#             axs[2].set_xticks([])
#             axs[2].set_yticks([])
#             axs[2].set_zticks([])
#             axs[2].get_xaxis().set_ticklabels([])
#             axs[2].get_yaxis().set_ticklabels([])
#             axs[2].set_zticklabels([])

#             # Get rid of the panes (actually, make them white)
#             axs[0].w_xaxis.set_pane_color(white)
#             axs[0].w_yaxis.set_pane_color(white)
#             axs[1].w_xaxis.set_pane_color(white)
#             axs[1].w_yaxis.set_pane_color(white)
#             axs[2].w_xaxis.set_pane_color(white)
#             axs[2].w_yaxis.set_pane_color(white)
#             # Keep z pane

#             # Get rid of the lines in 3d
#             axs[0].w_xaxis.line.set_color(white)
#             axs[0].w_yaxis.line.set_color(white)
#             axs[0].w_zaxis.line.set_color(white)
#             axs[1].w_xaxis.line.set_color(white)
#             axs[1].w_yaxis.line.set_color(white)
#             axs[1].w_zaxis.line.set_color(white)
#             axs[2].w_xaxis.line.set_color(white)
#             axs[2].w_yaxis.line.set_color(white)
#             axs[2].w_zaxis.line.set_color(white)

#         axs[0].view_init(azim=-20.)
#         axs[1].view_init(azim=-20.)
#         axs[2].view_init(azim=-20.)
#         filename = "{}/{}.png".format(images_folder, index)
#         filenames.append(filename)
#         plt.savefig(filename, dpi=100)
#         plt.close()
#         index += 1

#     show_verbose(verbose, "[INFO] Images saved")

#     # Build gif
#     show_verbose(verbose, '[INFO] Creating gif')
#     ims = []
#     fig = plt.figure()
#     ax = fig.gca()
#     ax.set_axis_off()
#     for filename in filenames:
#         im = plt.imshow(imageio.imread(filename))
#         ims.append([im])

#     ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
#                                     repeat_delay=500)

#     writer = animation.PillowWriter(fps=24)
#     ani.save("{}/demo.gif".format(out_folder), writer=writer)
#     plt.close()
#     show_verbose(verbose, '[INFO] Gif saved')

#     if not keep_images:
#         show_verbose(verbose, '[INFO] Removing images')
#         # Remove files
#         for filename in set(filenames):
#             os.remove(filename)
#         os.rmdir(images_folder)
#     show_verbose(verbose, '[INFO] DONE')

def generate_3d_vs_plots_animation(gt, est, cor, out_folder, radius=600, verbose=True, frame_on=True, keep_images=False):
    """
    Generates animation gif from sequence of poses.

    Parameters
    ----------
    gt : numpy.ndarray
        Sequence of poses (ground truth).
    est : numpy.ndarray
        Sequence of poses (estimation).
    est : numpy.ndarray
        Sequence of poses (correction).
    out_folder : string
        Output file name. It must be a GIF. The default is "demo.gif".
    radius : int, optional
        Space around the center.
    verbose : bool, optional
        Tells if logs should be print or not. The default is False.
    frame_on : bool, optional
        Tells if plot frame should be kept. The default is True.
    keep_images : bool, optional
        Tells if plot frame should be kept. The default is True.

    Returns
    -------
    None.

    """
    filenames = []
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    images_folder = "{}/images".format(out_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    white = (1.0, 1.0, 1.0, 0.0)

    for index in range(gt.shape[0]):
        xs_gt, ys_gt, zs_gt = [], [], []
        xs_est, ys_est, zs_est = [], [], []
        xs_cor, ys_cor, zs_cor = [], [], []
        for i in range(0, len(gt[0]), 3):
            xs_gt.append(gt[index][i])
            zs_gt.append(gt[index][i + 1])
            ys_gt.append(gt[index][i + 2])
            xs_est.append(est[index][i])
            zs_est.append(est[index][i + 1])
            ys_est.append(est[index][i + 2])
            xs_cor.append(cor[index][i])
            zs_cor.append(cor[index][i + 1])
            ys_cor.append(cor[index][i + 2])
        fig = plt.figure(figsize=(24, 8))
        # draw gt
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        for pair in SKELETON_PAIRS_17:
            x = [xs_gt[pair[0]], xs_gt[pair[1]]]
            y = [ys_gt[pair[0]], ys_gt[pair[1]]]
            z = [zs_gt[pair[0]], zs_gt[pair[1]]]
            ax.plot(x, y, z, color='green', linewidth=8, alpha=0.7)
        ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)
        ax.title.set_text("GT")

        # Space around the subject
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])

        ax.invert_zaxis()
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        if not frame_on:
            # Get rid of the ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.set_zticklabels([])

            # Get rid of the panes (actually, make them white)
            ax.w_xaxis.set_pane_color(white)
            ax.w_yaxis.set_pane_color(white)
            # Keep z pane

            # Get rid of the lines in 3d
            ax.w_xaxis.line.set_color(white)
            ax.w_yaxis.line.set_color(white)
            ax.w_zaxis.line.set_color(white)

        # draw estimation
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        for pair in SKELETON_PAIRS_17:
            x = [xs_est[pair[0]], xs_est[pair[1]]]
            y = [ys_est[pair[0]], ys_est[pair[1]]]
            z = [zs_est[pair[0]], zs_est[pair[1]]]
            ax.plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
        ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)
        ax.title.set_text("Estimation")
        # Space around the subject
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])

        ax.invert_zaxis()
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        if not frame_on:
            # Get rid of the ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.set_zticklabels([])

            # Get rid of the panes (actually, make them white)
            ax.w_xaxis.set_pane_color(white)
            ax.w_yaxis.set_pane_color(white)
            # Keep z pane

            # Get rid of the lines in 3d
            ax.w_xaxis.line.set_color(white)
            ax.w_yaxis.line.set_color(white)
            ax.w_zaxis.line.set_color(white)

        # draw correction
        ax = fig.add_subplot(1, 3, 3, projection='3d')
        for pair in SKELETON_PAIRS_17:
            x = [xs_cor[pair[0]], xs_cor[pair[1]]]
            y = [ys_cor[pair[0]], ys_cor[pair[1]]]
            z = [zs_cor[pair[0]], zs_cor[pair[1]]]
            ax.plot(x, y, z, color='blue', linewidth=8, alpha=0.7)
        ax.scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=10, alpha=0.8)
        ax.title.set_text("Correction")
        # Space around the subject
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])

        ax.invert_zaxis()
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

        if not frame_on:
            # Get rid of the ticks and tick labels
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])
            ax.set_zticklabels([])

            # Get rid of the panes (actually, make them white)
            ax.w_xaxis.set_pane_color(white)
            ax.w_yaxis.set_pane_color(white)
            # Keep z pane

            # Get rid of the lines in 3d
            ax.w_xaxis.line.set_color(white)
            ax.w_yaxis.line.set_color(white)
            ax.w_zaxis.line.set_color(white)

        
        filename = "{}/{:06d}.png".format(images_folder, index)
        filenames.append(filename)
        plt.savefig(filename, dpi=100)
        plt.close()
        index += 1

    show_verbose(verbose, "[INFO] Images saved")

    # Build gif
    show_verbose(verbose, '[INFO] Creating gif')
    ims = []
    fig = plt.figure()
    ax = fig.gca()
    ax.set_axis_off()
    for filename in filenames:
        im = plt.imshow(imageio.imread(filename))
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=500)

    writer = animation.PillowWriter(fps=24)
    ani.save("{}/demo.gif".format(out_folder), writer=writer)
    plt.close()
    show_verbose(verbose, '[INFO] Gif saved')

    if not keep_images:
        show_verbose(verbose, '[INFO] Removing images')
        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        os.rmdir(images_folder)
    show_verbose(verbose, '[INFO] DONE')

FRAMESNO = 500

# def animation_with_curves(gt, est, cor, jointno=16, output_folder="./test",
#                           radius=600, verbose=True, frame_on=True, keep_images=False):
#     filenames = []
#     import random
#     start = random.randint(0, gt.shape[0]-(FRAMESNO+10))
#     end = start + FRAMESNO + 1
#     r = np.arange(start=start, stop=end, step=1)
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     images_folder = "{}/images".format(output_folder)
#     if not os.path.exists(images_folder):
#         os.makedirs(images_folder)
#     est_acc = mvt_desc.get_acceleration(est)
#     est_vel = mvt_desc.get_velocity(est)
#     est_bone = mvt_desc.get_bones_length_variations(est)
#     cor_acc = mvt_desc.get_acceleration(cor)
#     cor_vel = mvt_desc.get_velocity(cor) 
#     cor_bone = mvt_desc.get_bones_length_variations(cor)
#     gt_acc = mvt_desc.get_acceleration(gt)
#     gt_vel = mvt_desc.get_velocity(gt) 
#     gt_bone = mvt_desc.get_bones_length_variations(gt)

#     # for f in range(FRAMESNO):
#     for f in r:
#         sf = f+start
#         xs_gt, ys_gt, zs_gt = [], [], []
#         xs_est, ys_est, zs_est = [], [], []
#         xs_cor, ys_cor, zs_cor = [], [], []
#         for i in range(0, len(cor[0]), 3):
#             xs_est.append(est[sf][i])
#             zs_est.append(est[sf][i + 1])
#             ys_est.append(est[sf][i + 2])
#             xs_cor.append(cor[sf][i])
#             zs_cor.append(cor[sf][i + 1])
#             ys_cor.append(cor[sf][i + 2])
#             xs_gt.append(gt[sf][i])
#             zs_gt.append(gt[sf][i + 1])
#             ys_gt.append(gt[sf][i + 2])
#         fig = plt.figure(figsize=(24, 12))

#         # draw positions
#         ax = fig.add_subplot(1, 2, 1, projection='3d')
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_gt[pair[0]], xs_gt[pair[1]]]
#             y = [ys_gt[pair[0]], ys_gt[pair[1]]]
#             z = [zs_gt[pair[0]], zs_gt[pair[1]]]
#             ax.plot(x, y, z, color='green', linewidth=8, alpha=0.7)
#         ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)

#         # draw estimation
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_est[pair[0]], xs_est[pair[1]]]
#             y = [ys_est[pair[0]], ys_est[pair[1]]]
#             z = [zs_est[pair[0]], zs_est[pair[1]]]
#             ax.plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
#         ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)

#         # draw correction
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_cor[pair[0]], xs_cor[pair[1]]]
#             y = [ys_cor[pair[0]], ys_cor[pair[1]]]
#             z = [zs_cor[pair[0]], zs_cor[pair[1]]]
#             ax.plot(x, y, z, color='blue', linewidth=8, alpha=0.7)
#         ax.scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=10, alpha=0.8)

#         # Space around the subject
#         ax.set_xlim([-radius, radius])
#         ax.set_ylim([-radius, radius])
#         ax.set_zlim([-radius, radius])

#         ax.invert_zaxis()
#         ax.get_xaxis().set_ticklabels([])
#         ax.get_yaxis().set_ticklabels([])
#         ax.set_zticklabels([])

#         if not frame_on:
#             # Get rid of the ticks and tick labels
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_zticks([])
#             ax.get_xaxis().set_ticklabels([])
#             ax.get_yaxis().set_ticklabels([])
#             ax.set_zticklabels([])

#             # Get rid of the panes (actually, make them white)
#             white = (1.0, 1.0, 1.0, 0.0)
#             ax.w_xaxis.set_pane_color(white)
#             ax.w_yaxis.set_pane_color(white)
#             # Keep z pane

#             # Get rid of the lines in 3d
#             ax.w_xaxis.line.set_color(white)
#             ax.w_yaxis.line.set_color(white)
#             ax.w_zaxis.line.set_color(white)

#         ax.view_init(azim=-20.)

#         # draw curves
#         # velocity
#         ax = fig.add_subplot(3, 2, 2)
#         ax.plot(x, gt_vel[:, jointno], c="green", linestyle="-")
#         ax.plot(x, est_vel[:, jointno], c="orange", linestyle="-")
#         ax.plot(x, cor_vel[:, jointno], c="blue", linestyle="-")
#         # acceleration
#         ax = fig.add_subplot(3, 2, 4)
#         ax.plot(x, gt_acc[:, jointno], c="green", linestyle="-")
#         ax.plot(x, est_acc[:, jointno], c="orange", linestyle="-")
#         ax.plot(x, cor_acc[:, jointno], c="blue", linestyle="-")
#         # bones
#         ax = fig.add_subplot(3, 2, 6)
#         ax.plot(x, gt_bone[:, -1], c="green", linestyle="-")
#         ax.plot(x, est_bone[:, -1], c="orange", linestyle="-")
#         ax.plot(x, cor_bone[:, -1], c="blue", linestyle="-")

#         filename = "{}/{:06d}.png".format(images_folder, f)
#         filenames.append(filename)
#         plt.savefig(filename, dpi=100)
#         plt.close()
#         # index += 1

#     show_verbose(verbose, "[INFO] Images saved")

#     # Build gif
#     import ffmpeg
#     (
#     ffmpeg
#     .input(f"{images_folder}/*.png", pattern_type='glob', framerate=30)
#     .output(f"{output_folder}/demo.mp4")
#     .run()
#     )

#     if not keep_images:
#         show_verbose(verbose, '[INFO] Removing images')
#         # Remove files
#         for filename in set(filenames):
#             os.remove(filename)
#         os.rmdir(images_folder)
#     show_verbose(verbose, '[INFO] DONE')


# # def animation_with_curves_shared(gt, est, cor, jointno=16, output_folder="./test",
# #                           radius=600, verbose=True, frame_on=True, keep_images=False):
# #     filenames = []
# #     if not os.path.exists(output_folder):
# #         os.makedirs(output_folder)
# #     images_folder = "{}/images".format(output_folder)
# #     if not os.path.exists(images_folder):
# #         os.makedirs(images_folder)
# #     est_acc = mvt_desc.get_acceleration(est)
# #     est_vel = mvt_desc.get_velocity(est)
# #     est_bone = mvt_desc.get_bones_length_variations(est)
# #     cor_acc = mvt_desc.get_acceleration(cor)
# #     cor_vel = mvt_desc.get_velocity(cor) 
# #     cor_bone = mvt_desc.get_bones_length_variations(cor)
# #     gt_acc = mvt_desc.get_acceleration(gt)
# #     gt_vel = mvt_desc.get_velocity(gt) 
# #     gt_bone = mvt_desc.get_bones_length_variations(gt)

# #     for f in range(FRAMESNO):
# #         xs_gt, ys_gt, zs_gt = [], [], []
# #         xs_est, ys_est, zs_est = [], [], []
# #         xs_cor, ys_cor, zs_cor = [], [], []
# #         for i in range(0, len(cor[0]), 3):
# #             xs_est.append(est[f][i])
# #             zs_est.append(est[f][i + 1])
# #             ys_est.append(est[f][i + 2])
# #             xs_cor.append(cor[f][i])
# #             zs_cor.append(cor[f][i + 1])
# #             ys_cor.append(cor[f][i + 2])
# #             xs_gt.append(gt[f][i])
# #             zs_gt.append(gt[f][i + 1])
# #             ys_gt.append(gt[f][i + 2])
# #         fig = plt.figure(figsize=(24, 12))

# #         # draw positions
# #         ax = fig.add_subplot(1, 2, 1, projection='3d')
# #         for pair in SKELETON_PAIRS_17:
# #             x = [xs_gt[pair[0]], xs_gt[pair[1]]]
# #             y = [ys_gt[pair[0]], ys_gt[pair[1]]]
# #             z = [zs_gt[pair[0]], zs_gt[pair[1]]]
# #             ax.plot(x, y, z, color='green', linewidth=8, alpha=0.7)
# #         ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)

# #         # draw estimation
# #         for pair in SKELETON_PAIRS_17:
# #             x = [xs_est[pair[0]], xs_est[pair[1]]]
# #             y = [ys_est[pair[0]], ys_est[pair[1]]]
# #             z = [zs_est[pair[0]], zs_est[pair[1]]]
# #             ax.plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
# #         ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)

# #         # draw correction
# #         for pair in SKELETON_PAIRS_17:
# #             x = [xs_cor[pair[0]], xs_cor[pair[1]]]
# #             y = [ys_cor[pair[0]], ys_cor[pair[1]]]
# #             z = [zs_cor[pair[0]], zs_cor[pair[1]]]
# #             ax.plot(x, y, z, color='blue', linewidth=8, alpha=0.7)
# #         ax.scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=10, alpha=0.8)

# #         # Space around the subject
# #         ax.set_xlim([-radius, radius])
# #         ax.set_ylim([-radius, radius])
# #         ax.set_zlim([-radius, radius])

# #         ax.invert_zaxis()
# #         ax.get_xaxis().set_ticklabels([])
# #         ax.get_yaxis().set_ticklabels([])
# #         ax.set_zticklabels([])

# #         if not frame_on:
# #             # Get rid of the ticks and tick labels
# #             ax.set_xticks([])
# #             ax.set_yticks([])
# #             ax.set_zticks([])
# #             ax.get_xaxis().set_ticklabels([])
# #             ax.get_yaxis().set_ticklabels([])
# #             ax.set_zticklabels([])

# #             # Get rid of the panes (actually, make them white)
# #             white = (1.0, 1.0, 1.0, 0.0)
# #             ax.w_xaxis.set_pane_color(white)
# #             ax.w_yaxis.set_pane_color(white)
# #             # Keep z pane

# #             # Get rid of the lines in 3d
# #             ax.w_xaxis.line.set_color(white)
# #             ax.w_yaxis.line.set_color(white)
# #             ax.w_zaxis.line.set_color(white)

# #         ax.view_init(azim=-20.)

# #         # draw curves
# #         # position
# #         ax = fig.add_subplot(4, 2, 2)
# #         ax.plot(gt[:, jointno], c="green", linestyle="-")
# #         ax.plot(est[:, jointno], c="orange", linestyle="-")
# #         ax.plot(cor[:, jointno], c="blue", linestyle="-")
# #         ax.set_ylabel("position (mm)")
# #         # velocity
# #         ax = fig.add_subplot(4, 2, 4)
# #         ax.plot(gt_vel[:, jointno], c="green", linestyle="-")
# #         ax.plot(est_vel[:, jointno], c="orange", linestyle="-")
# #         ax.plot(cor_vel[:, jointno], c="blue", linestyle="-")
# #         ax.set_ylabel("velocity (mm/f)")
# #         # acceleration
# #         ax = fig.add_subplot(4, 2, 6)
# #         ax.plot(gt_acc[:, jointno], c="green", linestyle="-")
# #         ax.plot(est_acc[:, jointno], c="orange", linestyle="-")
# #         ax.plot(cor_acc[:, jointno], c="blue", linestyle="-")
# #         ax.set_ylabel("acceleration (mm/f²)")
# #         # bones
# #         ax = fig.add_subplot(4, 2, 8)
# #         ax.plot(gt_bone[:, -1], c="green", linestyle="-", label="gt")
# #         ax.plot(est_bone[:, -1], c="orange", linestyle="-", label="estimation")
# #         ax.plot(cor_bone[:, -1], c="blue", linestyle="-", label="correction")
# #         ax.set_ylabel("bone length (mm)")
# #         ax.set_xlabel("frames")
# #         handles, labels = ax.get_legend_handles_labels()
# #         fig.legend(handles, labels, loc='upper center')

# #         filename = "{}/{:06d}.png".format(images_folder, f)
# #         filenames.append(filename)
# #         plt.savefig(filename, dpi=100)
# #         plt.close()

# #     show_verbose(verbose, "[INFO] Images saved")

# #     import ffmpeg
# #     (
# #     ffmpeg
# #     .input(f"{images_folder}/*.png", pattern_type='glob', framerate=30)
# #     .output(f"{output_folder}/demo.mp4")
# #     .run()
# #     )
# #     if not keep_images:
# #         show_verbose(verbose, '[INFO] Removing images')
# #         # Remove files
# #         for filename in set(filenames):
# #             os.remove(filename)
# #         os.rmdir(images_folder)
# #     show_verbose(verbose, '[INFO] DONE')

# def animation_with_curves_shared(gt, est, cor, part="rightarm", output_folder="./test",
#                           radius=600, verbose=True, frame_on=True, keep_images=False):
#     filenames = []
#     import random
#     start = random.randint(0, gt.shape[0]-(FRAMESNO+10))
#     end = start + FRAMESNO
#     r = np.arange(start=start, stop=end, step=1)
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     images_folder = "{}/images".format(output_folder)
#     if not os.path.exists(images_folder):
#         os.makedirs(images_folder)
#     est_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(est), part)
#     est_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(est), part)
#     est_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(est), part)
#     cor_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(cor), part)
#     cor_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(cor), part)
#     cor_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(cor), part)
#     gt_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(gt), part)
#     gt_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(gt), part)
#     gt_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(gt), part)

#     for f in range(FRAMESNO):
#         sf = f+start
#         xs_gt, ys_gt, zs_gt = [], [], []
#         xs_est, ys_est, zs_est = [], [], []
#         xs_cor, ys_cor, zs_cor = [], [], []
#         for i in range(0, len(cor[0]), 3):
#             xs_est.append(est[sf][i])
#             zs_est.append(est[sf][i + 1])
#             ys_est.append(est[sf][i + 2])
#             xs_cor.append(cor[sf][i])
#             zs_cor.append(cor[sf][i + 1])
#             ys_cor.append(cor[sf][i + 2])
#             xs_gt.append(gt[sf][i])
#             zs_gt.append(gt[sf][i + 1])
#             ys_gt.append(gt[sf][i + 2])
#         fig = plt.figure(figsize=(24, 12))

#         # draw positions
#         ax = fig.add_subplot(1, 2, 1, projection='3d')
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_gt[pair[0]], xs_gt[pair[1]]]
#             y = [ys_gt[pair[0]], ys_gt[pair[1]]]
#             z = [zs_gt[pair[0]], zs_gt[pair[1]]]
#             ax.plot(x, y, z, color='green', linewidth=8, alpha=0.7)
#         ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)

#         # draw estimation
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_est[pair[0]], xs_est[pair[1]]]
#             y = [ys_est[pair[0]], ys_est[pair[1]]]
#             z = [zs_est[pair[0]], zs_est[pair[1]]]
#             ax.plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
#         ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)

#         # draw correction
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_cor[pair[0]], xs_cor[pair[1]]]
#             y = [ys_cor[pair[0]], ys_cor[pair[1]]]
#             z = [zs_cor[pair[0]], zs_cor[pair[1]]]
#             ax.plot(x, y, z, color='blue', linewidth=8, alpha=0.7)
#         ax.scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=10, alpha=0.8)

#         # Space around the subject
#         ax.set_xlim([-radius, radius])
#         ax.set_ylim([-radius, radius])
#         ax.set_zlim([-radius, radius])

#         ax.invert_zaxis()
#         ax.get_xaxis().set_ticklabels([])
#         ax.get_yaxis().set_ticklabels([])
#         ax.set_zticklabels([])

#         if not frame_on:
#             # Get rid of the ticks and tick labels
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_zticks([])
#             ax.get_xaxis().set_ticklabels([])
#             ax.get_yaxis().set_ticklabels([])
#             ax.set_zticklabels([])

#             # Get rid of the panes (actually, make them white)
#             white = (1.0, 1.0, 1.0, 0.0)
#             ax.w_xaxis.set_pane_color(white)
#             ax.w_yaxis.set_pane_color(white)
#             # Keep z pane

#             # Get rid of the lines in 3d
#             ax.w_xaxis.line.set_color(white)
#             ax.w_yaxis.line.set_color(white)
#             ax.w_zaxis.line.set_color(white)

#         ax.view_init(azim=-20.)

#         # draw curves
#         # velocity
#         ax = fig.add_subplot(3, 2, 2)
#         ax.plot(r, gt_vel[start:end], c="green", linestyle="-")
#         ax.plot(r, est_vel[start:end], c="orange", linestyle="-")
#         ax.plot(r, cor_vel[start:end], c="blue", linestyle="-")
#         ax.set_ylabel("velocity (mm/f)")
#         # acceleration
#         ax = fig.add_subplot(3, 2, 4)
#         ax.plot(r, gt_acc[start:end], c="green", linestyle="-")
#         ax.plot(r, est_acc[start:end], c="orange", linestyle="-")
#         ax.plot(r, cor_acc[start:end], c="blue", linestyle="-")
#         ax.set_ylabel("acceleration (mm/f²)")
#         # bones
#         ax = fig.add_subplot(3, 2, 6)
#         ax.plot(r, gt_bone[start:end], c="green", linestyle="-", label="ground truth")
#         ax.plot(r, est_bone[start:end], c="orange", linestyle="-", label="estimation")
#         ax.plot(r, cor_bone[start:end], c="blue", linestyle="-", label="correction")
#         ax.set_ylabel("bone length (mm)")
#         ax.set_xlabel("frames")
#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(handles, labels, loc='upper center', fontsize='x-large')

#         filename = "{}/{:06d}.png".format(images_folder, f)
#         filenames.append(filename)
#         plt.savefig(filename, dpi=100)
#         plt.close()

#     show_verbose(verbose, "[INFO] Images saved")

#     import ffmpeg
#     (
#     ffmpeg
#     .input(f"{images_folder}/*.png", pattern_type='glob', framerate=24)
#     .output(f"{output_folder}/demo.mp4")
#     .run()
#     )
#     if not keep_images:
#         show_verbose(verbose, '[INFO] Removing images')
#         # Remove files
#         for filename in set(filenames):
#             os.remove(filename)
#         os.rmdir(images_folder)
#     show_verbose(verbose, '[INFO] DONE')

# def new_animation_with_curves_shared(gt, est, cor, part=None, output_folder="./test",
#                           radius=600, verbose=True, frame_on=True, keep_images=False):
#     filenames = []
#     import random
#     start = random.randint(0, gt.shape[0]-(FRAMESNO+10))
#     end = start + FRAMESNO
#     r = np.arange(start=start, stop=end, step=1)
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     images_folder = "{}/images".format(output_folder)
#     if not os.path.exists(images_folder):
#         os.makedirs(images_folder)
#     est_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(est), part)
#     est_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(est), part)
#     est_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(est), part)
#     cor_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(cor), part)
#     cor_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(cor), part)
#     cor_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(cor), part)
#     gt_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(gt), part)
#     gt_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(gt), part)
#     gt_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(gt), part)

#     for f in range(FRAMESNO):
#         sf = f+start
#         xs_gt, ys_gt, zs_gt = [], [], []
#         xs_est, ys_est, zs_est = [], [], []
#         xs_cor, ys_cor, zs_cor = [], [], []
#         for i in range(0, len(cor[0]), 3):
#             xs_est.append(est[sf][i])
#             zs_est.append(est[sf][i + 1])
#             ys_est.append(est[sf][i + 2])
#             xs_cor.append(cor[sf][i])
#             zs_cor.append(cor[sf][i + 1])
#             ys_cor.append(cor[sf][i + 2])
#             xs_gt.append(gt[sf][i])
#             zs_gt.append(gt[sf][i + 1])
#             ys_gt.append(gt[sf][i + 2])
#         fig = plt.figure(figsize=(24, 12))

#         # draw positions
#         ax = fig.add_subplot(1, 2, 1, projection='3d')
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_gt[pair[0]], xs_gt[pair[1]]]
#             y = [ys_gt[pair[0]], ys_gt[pair[1]]]
#             z = [zs_gt[pair[0]], zs_gt[pair[1]]]
#             ax.plot(x, y, z, color='green', linewidth=8, alpha=0.7)
#         ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=10, alpha=0.8)

#         # draw estimation
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_est[pair[0]], xs_est[pair[1]]]
#             y = [ys_est[pair[0]], ys_est[pair[1]]]
#             z = [zs_est[pair[0]], zs_est[pair[1]]]
#             ax.plot(x, y, z, color='orange', linewidth=8, alpha=0.7)
#         ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=10, alpha=0.8)

#         # draw correction
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_cor[pair[0]], xs_cor[pair[1]]]
#             y = [ys_cor[pair[0]], ys_cor[pair[1]]]
#             z = [zs_cor[pair[0]], zs_cor[pair[1]]]
#             ax.plot(x, y, z, color='blue', linewidth=8, alpha=0.7)
#         ax.scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=10, alpha=0.8)

#         # Space around the subject
#         ax.set_xlim([-radius, radius])
#         ax.set_ylim([-radius, radius])
#         ax.set_zlim([-radius, radius])

#         ax.invert_zaxis()
#         ax.get_xaxis().set_ticklabels([])
#         ax.get_yaxis().set_ticklabels([])
#         ax.set_zticklabels([])

#         if not frame_on:
#             # Get rid of the ticks and tick labels
#             ax.set_xticks([])
#             ax.set_yticks([])
#             ax.set_zticks([])
#             ax.get_xaxis().set_ticklabels([])
#             ax.get_yaxis().set_ticklabels([])
#             ax.set_zticklabels([])

#             # Get rid of the panes (actually, make them white)
#             white = (1.0, 1.0, 1.0, 0.0)
#             ax.w_xaxis.set_pane_color(white)
#             ax.w_yaxis.set_pane_color(white)
#             # Keep z pane

#             # Get rid of the lines in 3d
#             ax.w_xaxis.line.set_color(white)
#             ax.w_yaxis.line.set_color(white)
#             ax.w_zaxis.line.set_color(white)

#         ax.view_init(azim=-20.)

#         # draw curves
#         # velocity
#         ax = fig.add_subplot(3, 2, 2)
#         ax.plot(r, gt_vel[start:end], c="green", linestyle="-")
#         ax.plot(r, est_vel[start:end], c="orange", linestyle="-")
#         ax.plot(r, cor_vel[start:end], c="blue", linestyle="-")
#         ax.set_ylabel("velocity (mm/f)")
#         # acceleration
#         ax = fig.add_subplot(3, 2, 4)
#         ax.plot(r, gt_acc[start:end], c="green", linestyle="-")
#         ax.plot(r, est_acc[start:end], c="orange", linestyle="-")
#         ax.plot(r, cor_acc[start:end], c="blue", linestyle="-")
#         ax.set_ylabel("acceleration (mm/f²)")
#         # bones
#         ax = fig.add_subplot(3, 2, 6)
#         ax.plot(r, gt_bone[start:end], c="green", linestyle="-", label="ground truth")
#         ax.plot(r, est_bone[start:end], c="orange", linestyle="-", label="estimation")
#         ax.plot(r, cor_bone[start:end], c="blue", linestyle="-", label="correction")
#         ax.set_ylabel("average bone length (mm)")
#         ax.set_xlabel("frames")
#         handles, labels = ax.get_legend_handles_labels()
#         fig.legend(handles, labels, loc='upper center', fontsize='x-large')

#         filename = "{}/{:06d}.png".format(images_folder, f)
#         filenames.append(filename)
#         plt.savefig(filename, dpi=100)
#         plt.close()

#     show_verbose(verbose, "[INFO] Images saved")

#     import ffmpeg
#     (
#     ffmpeg
#     .input(f"{images_folder}/*.png", pattern_type='glob', framerate=24)
#     .output(f"{output_folder}/demo.mp4")
#     .run()
#     )
#     if not keep_images:
#         show_verbose(verbose, '[INFO] Removing images')
#         # Remove files
#         for filename in set(filenames):
#             os.remove(filename)
#         os.rmdir(images_folder)
#     show_verbose(verbose, '[INFO] DONE')


# def render_images(gt, est, cor, part=None, output_folder="./test",
#                   radius=600, verbose=True, frame_on=True, keep_images=False):
#     import random
#     start = random.randint(0, gt.shape[0]-(FRAMESNO+10))
#     end = start + FRAMESNO
#     os.makedirs(output_folder, exist_ok=True)
#     r = np.arange(start=start, stop=end, step=1)
#     est_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(est), part)
#     est_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(est), part)
#     est_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(est), part)
#     cor_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(cor), part)
#     cor_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(cor), part)
#     cor_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(cor), part)
#     gt_acc = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_acceleration(gt), part)
#     gt_vel = mvt_desc.get_motion_dsc_per_part(mvt_desc.get_velocity(gt), part)
#     gt_bone = mvt_desc.get_skeleton_dsc_per_part(mvt_desc.get_bones_length_variations(gt), part)
#     sampling_export = np.random.choice(r, size=3, replace=False)
#     sampling_export = np.sort(sampling_export)
#     # print(sampling_export)
#     fig = plt.figure(figsize=(24, 12))
#     idx = 1
#     for f in sampling_export:
#         xs_gt, ys_gt, zs_gt = [], [], []
#         xs_est, ys_est, zs_est = [], [], []
#         xs_cor, ys_cor, zs_cor = [], [], []
#         for i in range(0, len(cor[0]), 3):
#             xs_est.append(est[f][i])
#             zs_est.append(est[f][i + 1])
#             ys_est.append(est[f][i + 2])
#             xs_cor.append(cor[f][i])
#             zs_cor.append(cor[f][i + 1])
#             ys_cor.append(cor[f][i + 2])
#             xs_gt.append(gt[f][i])
#             zs_gt.append(gt[f][i + 1])
#             ys_gt.append(gt[f][i + 2])

#         # draw gt
#         ax = fig.add_subplot(2, 3, idx, projection='3d')
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_gt[pair[0]], xs_gt[pair[1]]]
#             y = [ys_gt[pair[0]], ys_gt[pair[1]]]
#             z = [zs_gt[pair[0]], zs_gt[pair[1]]]
#             ax.plot(x, y, z, color='green', linewidth=2, alpha=0.7)
#         ax.scatter(xs_gt, ys_gt, zs_gt, c='green', linewidth=1, alpha=0.8)

#         # draw estimation
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_est[pair[0]], xs_est[pair[1]]]
#             y = [ys_est[pair[0]], ys_est[pair[1]]]
#             z = [zs_est[pair[0]], zs_est[pair[1]]]
#             ax.plot(x, y, z, color='orange', linewidth=2, alpha=0.7)
#         ax.scatter(xs_est, ys_est, zs_est, c='orange', linewidth=1, alpha=0.8)

#         # draw correction
#         for pair in SKELETON_PAIRS_17:
#             x = [xs_cor[pair[0]], xs_cor[pair[1]]]
#             y = [ys_cor[pair[0]], ys_cor[pair[1]]]
#             z = [zs_cor[pair[0]], zs_cor[pair[1]]]
#             ax.plot(x, y, z, color='blue', linewidth=2, alpha=0.7)
#         ax.scatter(xs_cor, ys_cor, zs_cor, c='blue', linewidth=1, alpha=0.8)
#         idx += 1

#         # Space around the subject
#         ax.set_xlim([-radius, radius])
#         ax.set_ylim([-radius, radius])
#         ax.set_zlim([-radius, radius])

#         ax.invert_zaxis()
#         ax.get_xaxis().set_ticklabels([])
#         ax.get_yaxis().set_ticklabels([])
#         ax.set_zticklabels([])
#         ax.view_init(azim=-20.)
#         ax.set_title(f"frame {f}")

#     # draw curves
#     # velocity
#     ax = fig.add_subplot(2, 3, 4)
#     ax.plot(r, gt_vel[start:end], c="green", linestyle="-")
#     ax.plot(r, est_vel[start:end], c="orange", linestyle="-")
#     ax.plot(r, cor_vel[start:end], c="blue", linestyle="-")
#     ax.set_ylabel("velocity (mm/f)")
#     ax.set_xlabel("frames")
#     # acceleration
#     ax = fig.add_subplot(2, 3, 5)
#     ax.plot(r, gt_acc[start:end], c="green", linestyle="-")
#     ax.plot(r, est_acc[start:end], c="orange", linestyle="-")
#     ax.plot(r, cor_acc[start:end], c="blue", linestyle="-")
#     ax.set_ylabel("acceleration (mm/f²)")
#     ax.set_xlabel("frames")
#     # bones
#     ax = fig.add_subplot(2, 3, 6)
#     ax.plot(r, gt_bone[start:end], c="green", linestyle="-", label="ground truth")
#     ax.plot(r, est_bone[start:end], c="orange", linestyle="-", label="estimation")
#     ax.plot(r, cor_bone[start:end], c="blue", linestyle="-", label="correction")
#     ax.set_ylabel("average bone length (mm)")
#     ax.set_xlabel("frames")
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', fontsize='x-large')
#     plt.savefig(f"{output_folder}/material.png", dpi=200, format='png', bbox_inches = 'tight', pad_inches=0)