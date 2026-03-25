#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 14.04.2025
"""
import tensorflow as tf
import numpy as np

from utility.datasets.h36m_dataset import Human36mDataset, Human36mDatasetGenerator

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
        self._inputs = (self._inputs + 1.) / 2.
        # root related inputs
        # self.inputs = self.inputs - np.tile(self.inputs[:, :2], [1, int(self.inputs.shape[-1] / 2)])
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
        return [0.0], [0.0]

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
    def __init__(self, training_set=True, estimation="aanet", location=LOCATION, batch_size=1, chunk_size=0, keypoints="gt", fused=False):
        datas = Human36mSotaDataset(training_set, estimation, location, chunk_size, keypoints, fused)
        self._total_frames = datas.get_frame_count()
        self._dataset, self.parameters, self._size = datas.dataset()
        self._dataset = self._dataset.batch(batch_size)
        self._parameters = [tf.convert_to_tensor(item, dtype=tf.float32) for item in self.parameters]

    def get_dataset(self):
        return self._dataset
    
    def get_parameters(self):
        return self._parameters
    
    def size(self):
        return self._size
    
    def total_frames(self):
        return self._total_frames
    

class Human36mDatasetLoader:
    class TFDataset:
        def __init__(self, camera_params, poses_3d, poses_2d, codenames, chunk_size=0):
            self._inputs2d = []
            self._targets3d = []
            self._camera_params = camera_params
            self._names = []
            self._frames_count = []
            self._chunk_size = chunk_size
            for i in range(len(poses_3d)):
                self._frames_count.append(poses_3d[i].shape[0])
            self._inputs2d = np.concatenate(poses_2d, axis=0)
            self._targets3d = np.concatenate(poses_3d, axis=0)
            self._root3d = self._targets3d[:, [0], :]
            self._targets3d[:, 0, :] = 0
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
    
    def __init__(self, training_set=True, batch_size=1, chunk_size=0, keypoints="gt"):
        keypoints_path = f'./data/h36m/data_2d_h36m_{KEYPOINT_TYPE[keypoints]}.npz'
        data_generator = Human36mDatasetGenerator(keypoints_path=keypoints_path)
        if training_set:
            subjects = ["S1","S5","S6","S7","S8"]
        else:
            subjects = ["S9", "S11"]
        camera_params, poses_3d, poses_2d, codenames = data_generator.generate(subjects)
        self._tf_dataset = Human36mDatasetLoader.TFDataset(camera_params, poses_3d, poses_2d, codenames, chunk_size)
        self._dataset = self._tf_dataset.generate_dataset()
        self._dataset = self._dataset.batch(batch_size)
    
    def get_dataset(self):
        return self._dataset
    
    def get_parameters(self):
        return self._parameters
    
    def size(self):
        return len(self._tf_dataset)
    
    def total_frames(self):
        return self._tf_dataset.get_frame_count() 



class Human36mBoneDatasetLoader:
    class TFBoneDataset:
        def __init__(self, camera_params, poses_3d, poses_2d, codenames, chunk_size=0):
            self._inputs2d = []
            self._targets3d = []
            self._camera_params = camera_params
            self._names = []
            self._frames_count = []
            self._chunk_size = chunk_size
            for i in range(len(poses_3d)):
                self._frames_count.append(poses_3d[i].shape[0])
            self._inputs2d = np.concatenate(poses_2d, axis=0)
            targets3d = np.concatenate(poses_3d, axis=0)
            self._bones = h36m_17_get_bones(targets3d)
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
            ds = tf.data.Dataset.from_generator(
                self.__generator__,
                output_signature=(
                    tf.TensorSpec(shape=(None, 17, 2), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.string))
                )
            return ds

    def __init__(self, training_set=True, batch_size=1, chunk_size=0, keypoints="gt"):
        keypoints_path = f'./data/h36m/data_2d_h36m_{KEYPOINT_TYPE[keypoints]}.npz'
        data_generator = Human36mDatasetGenerator(keypoints_path=keypoints_path)
        if training_set:
            subjects = ["S1","S5","S6","S7","S8"]
        else:
            subjects = ["S9", "S11"]
        camera_params, poses_3d, poses_2d, codenames = data_generator.generate(subjects)
        self._tf_dataset = Human36mBoneDatasetLoader.TFBoneDataset(camera_params, poses_3d, poses_2d, codenames, chunk_size)
        self._dataset = self._tf_dataset.generate_dataset()
        self._dataset = self._dataset.batch(batch_size)
    
    def get_dataset(self):
        return self._dataset
    
    def get_parameters(self):
        return self._parameters
    
    def size(self):
        return len(self._tf_dataset)
    
    def total_frames(self):
        return self._tf_dataset.get_frame_count() 


def h36m_17_get_bones(positions):
    def distance(start, end):
        return np.sqrt(np.sum(np.square(start - end), axis=-1))

    length = np.zeros((positions.shape[0], 10))
    length[:, 0] = (
        distance(positions[:, 0, :], positions[:, 1, :])
        + distance(positions[:, 0, :], positions[:, 4, :])
    ) / 2
    length[:, 1] = (
        distance(positions[:, 1, :], positions[:, 2, :])
        + distance(positions[:, 4, :], positions[:, 5, :])
    ) / 2
    length[:, 2] = (
        distance(positions[:, 2, :], positions[:, 3, :])
        + distance(positions[:, 5, :], positions[:, 6, :])
    ) / 2
    length[:, 3] = distance(
        positions[:, 0, :], positions[:, 7, :]
    )
    length[:, 4] = distance(
        positions[:, 7, :], positions[:, 8, :]
    )
    length[:, 5] = distance(
        positions[:, 8, :], positions[:, 9, :]
    )
    length[:, 6] = distance(
        positions[:, 9, :], positions[:, 10, :]
    )
    length[:, 7] = (
        distance(positions[:, 8, :], positions[:, 11, :])
        + distance(positions[:, 8, :], positions[:, 14, :])
    ) / 2
    length[:, 8] = (
        distance(positions[:, 14, :], positions[:, 15, :])
        + distance(positions[:, 11, :], positions[:, 12, :])
    ) / 2
    length[:, 9] = (
        distance(positions[:, 15, :], positions[:, 16, :])
        + distance(positions[:, 12, :], positions[:, 13, :])
    ) / 2
    return length


def get_bones(position_3d):
    def distance(position1, position2):
        return np.sqrt(np.sum(np.square(position1 - position2), axis=-1))

    length = np.zeros((position_3d.shape[0], 10))
    length[:, 0] = (
        distance(position_3d[:, 3 * 0: 3 * 0 + 3], position_3d[:, 3 * 1: 3 * 1 + 3])
        + distance(position_3d[:, 3 * 0: 3 * 0 + 3], position_3d[:, 3 * 4: 3 * 4 + 3])
    ) / 2
    length[:, 1] = (
        distance(position_3d[:, 3 * 1: 3 * 1 + 3], position_3d[:, 3 * 2: 3 * 2 + 3])
        + distance(position_3d[:, 3 * 4: 3 * 4 + 3], position_3d[:, 3 * 5: 3 * 5 + 3])
    ) / 2
    length[:, 2] = (
        distance(position_3d[:, 3 * 2: 3 * 2 + 3], position_3d[:, 3 * 3: 3 * 3 + 3])
        + distance(position_3d[:, 3 * 5: 3 * 5 + 3], position_3d[:, 3 * 6: 3 * 6 + 3])
    ) / 2
    length[:, 3] = distance(
        position_3d[:, 3 * 0: 3 * 0 + 3], position_3d[:, 3 * 7: 3 * 7 + 3]
    )
    length[:, 4] = distance(
        position_3d[:, 3 * 7: 3 * 7 + 3], position_3d[:, 3 * 8: 3 * 8 + 3]
    )
    length[:, 5] = distance(
        position_3d[:, 3 * 8: 3 * 8 + 3], position_3d[:, 3 * 9: 3 * 9 + 3]
    )
    length[:, 6] = distance(
        position_3d[:, 3 * 9: 3 * 9 + 3], position_3d[:, 3 * 10: 3 * 10 + 3]
    )
    length[:, 7] = (
        distance(position_3d[:, 3 * 8: 3 * 8 + 3], position_3d[:, 3 * 11: 3 * 11 + 3])
        + distance(position_3d[:, 3 * 8: 3 * 8 + 3], position_3d[:, 3 * 14: 3 * 14 + 3])
    ) / 2
    length[:, 8] = (
        distance(position_3d[:, 3 * 14: 3 * 14 + 3], position_3d[:, 3 * 15: 3 * 15 + 3])
        + distance(position_3d[:, 3 * 11: 3 * 11 + 3], position_3d[:, 3 * 12: 3 * 12 + 3])
    ) / 2
    length[:, 9] = (
        distance(position_3d[:, 3 * 15: 3 * 15 + 3], position_3d[:, 3 * 16: 3 * 16 + 3])
        + distance(position_3d[:, 3 * 12: 3 * 12 + 3], position_3d[:, 3 * 13: 3 * 13 + 3])
    ) / 2
    return length