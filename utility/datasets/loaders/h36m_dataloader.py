#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 04.2025
"""
import tensorflow as tf
import numpy as np

from utility import ops
from utility.datasets.h36m_dataset import Human36mDatasetGenerator

# To change to the folder where your .npz files are
LOCATION = '/home/mansour/Workspace/data/human36m'
KEYPOINT_TYPE = {
    'cpn': 'cpn_ft_h36m_dbb',
    'detectron': 'detectron_ft_h36m',
    'gt': 'gt'
}

class Human36mSotaDataset:
    def __init__(self, training_set=True, estimation="aanet", location=LOCATION,
                 chunk_size=0, keypoints="gt", fused=False):
        """ Initialises human36m dataset for motion correction.

        Parameters
        ----------
        training_set : bool, optional
            Dataset for training or evaluation. The default is True.
        estimation: str, optional
            The state-of-the-art estimator. Can be 'aanet', 'mnet'
            or 'poseformer'. The default is 'aanet'.
        location: str, optional
            Location folder of the data.
        chunk_size: int, optional
            Size of the clips in case of clipped motion sequence. The default
            is None.
        keypoints: str, optional
            Type of 2D inputs keypoints. "gt" or "cpn".
        fused: bool, optional
            Tells whether to fuse the features and the joints of not

        """
        self.sequence_index = None
        self._training_set = training_set
        self._chunk_size = chunk_size
        self._fused = fused
        if training_set:
            filename = f'{location}/{estimation}_{keypoints}_train.npz'
        else:
            filename = f'{location}/{estimation}_{keypoints}_test.npz'
        contents = np.load(filename, allow_pickle=True)['data'].item()

        self._estimations = np.concatenate(contents["estimation"], axis=0)
        self._targets = np.concatenate(contents["gt"], axis=0)
        self._video_names = contents["names"]
        self._inputs = np.concatenate(contents["inputs"], axis=0)
        self._total_frames = self._inputs.shape[0]
        # input between [0, 1]
        # self._inputs = (self._inputs + 1.) / 2.
        # root related inputs
        # self.inputs = self.inputs - np.tile(self.inputs[:, :2], [1, int(self.inputs.shape[-1] / 2)])
        bones = get_h36m_17_bones_length(self._targets)
        bones, self._bones_mean, self._bones_std = ops.normalize_data(bones)
        self._inputs2d_mean, self._inputs2d_std = np.ones((1,)), np.ones((1,))
        if self._fused:
            self._inputs = np.reshape(self._inputs, (self._total_frames, -1))
            self._estimations = np.reshape(self._estimations, (self._total_frames, -1))
            self._targets = np.reshape(self._targets, (self._total_frames, -1))
        self._cut_names = []

        self._frame_numbers = list(contents["frame_numbers"])

        self.__set_sequences__()

    def __getitem__(self, index):
        items_index = self.sequence_index[index]
        inp = self._inputs[items_index]
        est = self._estimations[items_index]
        gt = self._targets[items_index]
        if self._chunk_size != 0:
            return inp, est, gt, self._cut_names[index]
        else:
            return inp, est, gt, self._video_names[index]

    def __generator__(self):
        i = 0
        while i < len(self):
            yield self.__getitem__(i)
            i += 1

    def __len__(self):
        return len(self.sequence_index)
    
    def get_frame_count(self):
        return self._total_frames

    def __set_sequences__(self):
        def slice_set(cut_size, frame_numbers):
            sequence_index = []
            start_index = 0
            if cut_size != 0:
                for k, frames in enumerate(frame_numbers):
                    if frames > cut_size:
                        clips_number = int(frames // self._chunk_size)
                        for i in range(clips_number):
                            start = int(start_index + i * cut_size)
                            end = int(start + cut_size)
                            sequence_index.append(list(range(start, end)))
                            self._cut_names.append(self._video_names[k])
                    start_index += frames
            else:
                for frames in frame_numbers:
                    sequence_index.append(list(range(start_index, start_index + frames)))
                    start_index += frames

            return sequence_index

        self.sequence_index = slice_set(self._chunk_size, self._frame_numbers)

    def get_parameters(self):
        return self._inputs2d_mean, self._inputs2d_std, self._bones_mean, self._bones_std

    def dataset(self):
        if self._fused:
            ds = tf.data.Dataset.from_generator(self.__generator__,
                                                output_signature=(
                                                    tf.TensorSpec(shape=(None, 34), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(None, 51), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(None, 51), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(), dtype=tf.string))
                                                )
        else:
            ds = tf.data.Dataset.from_generator(self.__generator__,
                                                output_signature=(
                                                    tf.TensorSpec(shape=(None, 17, 2), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(None, 17, 3), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(None, 17, 3), dtype=tf.float32),
                                                    tf.TensorSpec(shape=(), dtype=tf.string))
                                                )
        return ds, self.get_parameters(), self.__len__()
    

class Human36mSotaDatasetLoader:
    def __init__(self, training_set=True, estimation="aanet", location=LOCATION, batch_size=1, chunk_size=0, keypoints="gt", fused=False, seed=97):
        datas = Human36mSotaDataset(training_set, estimation, location, chunk_size, keypoints, fused)
        self._total_frames = datas.get_frame_count()
        self._batch_size = batch_size
        self._seed = seed
        self._dataset, self.parameters, self._size = datas.dataset()
        # self._dataset = self._dataset.batch(batch_size)
        self._parameters = [tf.convert_to_tensor(item, dtype=tf.float32) for item in self.parameters]

    def get_dataset(self):
        return self._dataset.batch(self._batch_size)
    
    def get_train_validation_datasets(self, split=0.8):
        train_size = int(len(self) * split)
        train_set = self._dataset.take(train_size)
        validation_set = self._dataset.skip(train_size)
        return train_set.shuffle(1000, seed=self._seed).batch(self._batch_size), validation_set.shuffle(1000, seed=self._seed).batch(self._batch_size)
    
    def get_parameters(self):
        return self._parameters
    
    def __len__(self):
        return self._size
    
    def total_frames(self):
        return self._total_frames
    

class Human36mDatasetLoader:
    class TFDataset:
        def __init__(self, camera_params, poses_3d, poses_2d, codenames, chunk_size=0, fused=False):
            self._inputs2d = []
            self._targets3d = []
            self._camera_params = camera_params
            self._names = []
            self._frames_count = []
            self._chunk_size = chunk_size
            self._fused = fused
            for i in range(len(poses_3d)):
                self._frames_count.append(poses_3d[i].shape[0])
            self._inputs2d = np.concatenate(poses_2d, axis=0)
            self._targets3d = np.concatenate(poses_3d, axis=0)  # [global root + root related joints]
            self._root3d = self._targets3d[:, [0], :]  # get global root trajectory
            self._targets3d[:, 0, :] = 0  # remove global root trajectory
            self._names = codenames
            if self._fused:
                self._inputs2d = np.reshape(self._inputs2d.shape[0], (self._inputs2d.shape[0], -1))
                self._targets3d = np.reshape(self._targets3d, (self._targets3d.shape[0], -1))
            self._cut_names = []
            self.sequence_index = None
            self.__set_sequences__()
        
        def __set_sequences__(self):
            def slice_set(cut_size, frame_numbers):
                sequence_index = []
                start_index = 0
                if cut_size != 0:
                    for k, frames in enumerate(frame_numbers):
                        if frames > cut_size:
                            clips_number = int(frames // self._chunk_size)
                            for i in range(clips_number):
                                start = int(start_index + i * cut_size)
                                end = int(start + cut_size)
                                sequence_index.append(list(range(start, end)))
                                self._cut_names.append(self._names[k])
                        start_index += frames
                else:
                    for frames in frame_numbers:
                        sequence_index.append(list(range(start_index, start_index + frames)))
                        start_index += frames

                return sequence_index

            self.sequence_index = slice_set(self._chunk_size, self._frames_count)
                
        def __len__(self):
            return len(self.sequence_index)
        
        def get_frame_count(self):
            return self._inputs2d.shape[0]
    
        def __getitem__(self, idx):
            index = self.sequence_index[idx]
            if self._chunk_size != 0:
                return self._inputs2d[index], self._targets3d[index], self._cut_names[idx]
            else:
                return self._inputs2d[index], self._bones[index], self._names[idx] 
        
        def __generator__(self):
            i = 0
            while i < len(self):
                yield self.__getitem__(i)
                i += 1

        def generate_dataset_with_seperated_joints_features(self):
            ds = tf.data.Dataset.from_generator(
                self.__generator__,
                output_signature=(
                    tf.TensorSpec(shape=(None, 17, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 17, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.string))
                )
            return ds
        
        def generate_dataset_with_fused_joints_features(self):
            ds = tf.data.Dataset.from_generator(
                self.__generator__,
                output_signature=(
                    tf.TensorSpec(shape=(None, 34), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 51), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.string))
                )
            return ds
    
    def __init__(self, training_set=True, batch_size=1, chunk_size=0, keypoints="gt", fused=False):
        keypoints_path = f'./data/h36m/data_2d_h36m_{KEYPOINT_TYPE[keypoints]}.npz'
        data_generator = Human36mDatasetGenerator(keypoints_path=keypoints_path)
        if training_set:
            subjects = ["S1","S5","S6","S7","S8"]
        else:
            subjects = ["S9", "S11"]
        camera_params, poses_3d, poses_2d, codenames = data_generator.generate(subjects)
        self._batch_size = batch_size
        self._tf_dataset = Human36mDatasetLoader.TFDataset(camera_params, poses_3d, poses_2d, codenames, chunk_size, fused)
        self._dataset = self._tf_dataset.generate_dataset()
        # self._dataset = self._dataset.batch(batch_size)
    
    def get_dataset(self):
        return self._dataset.batch(self._batch_size)
    
    def get_parameters(self):
        return self._parameters
    
    def size(self):
        return len(self._tf_dataset)
    
    def total_frames(self):
        return self._tf_dataset.get_frame_count() 


class Human36mBoneDatasetLoader:
    class TFBoneDataset:
        def __init__(self, camera_params, poses_3d, poses_2d, codenames, chunk_size=0, fused=True):
            self._inputs2d = []
            self._targets3d = []
            self._camera_params = camera_params
            self._names = []
            self._frames_count = []
            self._chunk_size = chunk_size
            self._fused = fused
            for i in range(len(poses_3d)):
                self._frames_count.append(poses_3d[i].shape[0])
            self._inputs2d = np.concatenate(poses_2d, axis=0)
            # self._inputs2d = (self._inputs2d + 1.) / 2.
            targets3d = np.concatenate(poses_3d, axis=0)  # except root, all other joints are root related
            # root3d = targets3d[:, [0], :]  # root contains the trajectory
            targets3d[:, 0, :] = 0  # remove root trajectory
            self._bones = get_h36m_17_bones_length(targets3d)
            if self._fused:
                self._inputs2d = np.reshape(self._inputs2d, (self._inputs2d.shape[0], -1))
            self._bones, self._bones_mean, self._bones_std = ops.normalize_data(self._bones)
            self._inputs2d_mean, self._inputs2d_std = np.ones((1,)), np.ones((1,))
            # self._inputs2d, self._inputs2d_mean, self._inputs2d_std = tools.normalize_data(self._inputs2d)
            self._names = codenames
            self._cut_names = []
            self.sequence_index = None
            self.__set_sequences__()
        
        def __set_sequences__(self):
            def slice_set(cut_size, frame_numbers):
                sequence_index = []
                start_index = 0
                if cut_size != 0:
                    for k, frames in enumerate(frame_numbers):
                        if frames > cut_size:
                            clips_number = int(frames // self._chunk_size)
                            for i in range(clips_number):
                                start = int(start_index + i * cut_size)
                                end = int(start + cut_size)
                                sequence_index.append(list(range(start, end)))
                                self._cut_names.append(self._names[k])
                        start_index += frames
                else:
                    for frames in frame_numbers:
                        sequence_index.append(list(range(start_index, start_index + frames)))
                        start_index += frames

                return sequence_index

            self.sequence_index = slice_set(self._chunk_size, self._frames_count)
                
        def __len__(self):
            return len(self.sequence_index)
        
        def get_frame_count(self):
            return self._inputs2d.shape[0]
    
        def __getitem__(self, idx):
            index = self.sequence_index[idx]
            if self._chunk_size != 0:
                return self._inputs2d[index], self._bones[index], self._cut_names[idx]
            else:
                return self._inputs2d[index], self._bones[index], self._names[idx] 
        
        def __generator__(self):
            i = 0
            while i < len(self):
                yield self.__getitem__(i)
                i += 1

        def generate_dataset(self):
            if self._fused:
                ds = tf.data.Dataset.from_generator(
                    self.__generator__,
                    output_signature=(
                        tf.TensorSpec(shape=(None, 34), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                        # tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.string))
                    )
            else:
                ds = tf.data.Dataset.from_generator(
                    self.__generator__,
                    output_signature=(
                        tf.TensorSpec(shape=(None, 17, 2), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                        # tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                        tf.TensorSpec(shape=(), dtype=tf.string))
                    )
            return ds
        
        def get_bones_mean_std(self):
            return self._bones_mean, self._bones_std
        
        def get_inputs_mean_std(self):
            return self._inputs2d_mean, self._inputs2d_std
        
        def get_normalisation_parameters(self):
            return self._inputs2d_mean, self._inputs2d_std, self._bones_mean, self._bones_std

    def __init__(self, training_set=True, batch_size=1, chunk_size=0, keypoints="gt", fused=True, seed=97):
        keypoints_path = f'./data/human36m/data_2d_h36m_{KEYPOINT_TYPE[keypoints]}.npz'
        data_generator = Human36mDatasetGenerator(keypoints_path=keypoints_path)
        if training_set:
            subjects = ["S1","S5","S6","S7","S8"]
        else:
            subjects = ["S9", "S11"]
        camera_params, poses_3d, poses_2d, codenames = data_generator.generate(subjects)
        self._tf_dataset = Human36mBoneDatasetLoader.TFBoneDataset(camera_params, poses_3d, poses_2d, codenames, chunk_size, fused)
        self._dataset = self._tf_dataset.generate_dataset()
        self._batch_size = batch_size
        self._seed = seed
        self._parameters = self._tf_dataset.get_normalisation_parameters()
        # self._dataset = self._dataset.batch(batch_size)
    
    def get_tf_dataset(self):
        return self._tf_dataset
    
    def get_dataset(self):
        return self._dataset.batch(self._batch_size)
    
    def get_train_validation_datasets(self, split=0.8):
        train_size = int(self.size() * split)
        train_set = self._dataset.take(train_size)
        validation_set = self._dataset.skip(train_size)
        return train_set.shuffle(1000, seed=self._seed).batch(self._batch_size), validation_set.shuffle(1000, seed=self._seed).batch(self._batch_size)
    
    def get_parameters(self):
        return self._parameters
    
    def get_bones_mean_std(self):
        return self._tf_dataset.get_bones_mean_std()
    
    def size(self):
        return len(self._tf_dataset)
    
    def total_frames(self):
        return self._tf_dataset.get_frame_count() 


def get_h36m_17_bones_length(positions):
    def distance(start, end):
        return np.linalg.norm(start - end, axis=-1)

    hips = (
        distance(positions[..., 0:1, :], positions[..., 1:2, :])
        + distance(positions[..., 0:1, :], positions[..., 4:5, :])
    ) / 2
    femur = (
        distance(positions[..., 1:2, :], positions[..., 2:3, :])
        + distance(positions[..., 4:5, :], positions[..., 5:6, :])
    ) / 2  # 
    tibia = (
        distance(positions[..., 2:3, :], positions[..., 3:4, :])
        + distance(positions[..., 5:6, :], positions[..., 6:7, :])
    ) / 2
    spine_back = distance(
        positions[..., 0:1, :], positions[..., 7:8, :]
    )
    spine_top = distance(
        positions[..., 7:8, :], positions[..., 8:9, :]
    )
    neck = distance(
        positions[..., 8:9, :], positions[..., 9:10, :]
    )
    head = distance(
        positions[..., 9:10, :], positions[..., 10:11, :]
    )
    clavicle = (
        distance(positions[..., 8:9, :], positions[..., 11:12, :])
        + distance(positions[..., 8:9, :], positions[..., 14:15, :])
    ) / 2
    humerus = (
        distance(positions[..., 14:15, :], positions[..., 15:16, :])
        + distance(positions[..., 11:12, :], positions[..., 12:13, :])
    ) / 2
    radius = (
        distance(positions[..., 15:16, :], positions[..., 16:17, :])
        + distance(positions[..., 12:13, :], positions[..., 13:14, :])
    ) / 2
    return np.concatenate((hips, femur, tibia, spine_back, spine_top, neck, head, clavicle, humerus, radius),
                         axis=-1)


def get_bones(position_3d):
    def distance(position1, position2):
        return np.sqrt(np.sum(np.square(position1 - position2), axis=-1))

    length = np.zeros((*position_3d.shape[:-1], 10))
    length[..., 0] = (
        distance(position_3d[..., 3 * 0: 3 * 0 + 3], position_3d[..., 3 * 1: 3 * 1 + 3])
        + distance(position_3d[..., 3 * 0: 3 * 0 + 3], position_3d[..., 3 * 4: 3 * 4 + 3])
    ) / 2
    length[..., 1] = (
        distance(position_3d[..., 3 * 1: 3 * 1 + 3], position_3d[..., 3 * 2: 3 * 2 + 3])
        + distance(position_3d[..., 3 * 4: 3 * 4 + 3], position_3d[..., 3 * 5: 3 * 5 + 3])
    ) / 2
    length[..., 2] = (
        distance(position_3d[..., 3 * 2: 3 * 2 + 3], position_3d[..., 3 * 3: 3 * 3 + 3])
        + distance(position_3d[..., 3 * 5: 3 * 5 + 3], position_3d[..., 3 * 6: 3 * 6 + 3])
    ) / 2
    length[..., 3] = distance(
        position_3d[..., 3 * 0: 3 * 0 + 3], position_3d[..., 3 * 7: 3 * 7 + 3]
    )
    length[..., 4] = distance(
        position_3d[..., 3 * 7: 3 * 7 + 3], position_3d[..., 3 * 8: 3 * 8 + 3]
    )
    length[..., 5] = distance(
        position_3d[..., 3 * 8: 3 * 8 + 3], position_3d[..., 3 * 9: 3 * 9 + 3]
    )
    length[..., 6] = distance(
        position_3d[..., 3 * 9: 3 * 9 + 3], position_3d[..., 3 * 10: 3 * 10 + 3]
    )
    length[..., 7] = (
        distance(position_3d[..., 3 * 8: 3 * 8 + 3], position_3d[..., 3 * 11: 3 * 11 + 3])
        + distance(position_3d[..., 3 * 8: 3 * 8 + 3], position_3d[..., 3 * 14: 3 * 14 + 3])
    ) / 2
    length[..., 8] = (
        distance(position_3d[..., 3 * 14: 3 * 14 + 3], position_3d[..., 3 * 15: 3 * 15 + 3])
        + distance(position_3d[..., 3 * 11: 3 * 11 + 3], position_3d[..., 3 * 12: 3 * 12 + 3])
    ) / 2
    length[..., 9] = (
        distance(position_3d[..., 3 * 15: 3 * 15 + 3], position_3d[..., 3 * 16: 3 * 16 + 3])
        + distance(position_3d[..., 3 * 12: 3 * 12 + 3], position_3d[..., 3 * 13: 3 * 13 + 3])
    ) / 2
    return length