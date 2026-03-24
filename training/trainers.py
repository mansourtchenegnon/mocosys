#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 15.04.2025
"""

import math
import pickle
import sys
import time
from model import losses as losses
from utility.datasets.loaders.h36m_dataloader import Human36mBoneDatasetLoader, Human36mSotaDatasetLoader
import utility.logger as logs
import utility.tools as tools
import tensorflow as tf

from model import solvers
from model.models import MotionFineTuningModel, SkeletonModel

class Trainer:
    def __init__(self, config, model, model_type="default"):
        """
        Initialises instance of BaseTrainer.

        Parameters
        ----------
        config : types.SimpleNamespace
            Configuration parameters.
        model: tf.keras.Model
            The nn object.
        model_type: str
            The type of the model, for naming purpose.

        Returns
        -------
        None.

        """
        self.config = config
        self.checkpoint_dir = "./checkpoints/%s/%s" % (model_type, config.running.version)
        self.log_dir = "./checkpoints/logs/%s/%s" % (model_type, config.running.version)
        self.model = model
        tools.make_dir(self.checkpoint_dir)
        tools.make_dir(self.log_dir)
        tools.make_dir(f"{self.checkpoint_dir}/best")
        self.logger = logs.Logger("%s/training.log" % self.log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(f"{self.log_dir}/train")
        self.test_summary_writer = tf.summary.create_file_writer(f"{self.log_dir}/test")


    def save_checkpoint(self, epoch: int = None):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'config': self.model.params
        }
        if epoch:
            folder = f'{self.checkpoint_dir}/ckpt-{epoch:04d}'
            tools.make_dir(folder)
            self.model.save_weights(f"{folder}/weights.keras")
            with open(f"{folder}/state.pkl", 'wb') as fp:
                pickle.dump(state, fp)
        else:
            self.model.save_weights(f"{self.checkpoint_dir}/best/weights.keras")
            with open(f"{self.checkpoint_dir}/best/state.pkl", 'wb') as fp:
                pickle.dump(state, fp)


class MFTModelTrainer(Trainer):
    def __init__(self,
                config,
                model : MotionFineTuningModel,
                train_dataloader : Human36mSotaDatasetLoader,
                test_dataloader : Human36mSotaDatasetLoader
        ):
        super().__init__(config, model, "mftmodel")
        self._trainset = train_dataloader
        self._testset = test_dataloader
        self.BATCH_SIZE = config.running.batch_size
        self.EPOCHS = config.running.epochs
        self.mse = tf.keras.losses.MeanSquaredError()

        # Loss weights
        self.a = tf.constant(1.)  # position
        self.b = tf.constant(1.)  # delta
        self.c = tf.constant(0.)  # bone
        self.d = tf.constant(0.)  # velocity

        self.delta_converter = solvers.DeltaConverter(
            3,
            self.model.skeleton,
            self.model.window)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # Metrics to track losses
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.train_position_loss = tf.keras.metrics.Mean(name="train_position_loss")
        self.test_position_loss = tf.keras.metrics.Mean(name="test_position_loss")
        self.train_bone_loss = tf.keras.metrics.Mean(name="train_bone_loss")
        self.test_bone_loss = tf.keras.metrics.Mean(name="test_bone_loss")
        self.train_velocity_loss = tf.keras.metrics.Mean(name="train_velocity_loss")
        self.test_velocity_loss = tf.keras.metrics.Mean(name="test_velocity_loss")
        self.train_delta_loss = tf.keras.metrics.Mean(name="train_delta_loss")
        self.test_delta_loss = tf.keras.metrics.Mean(name="test_delta_loss")
        
        self.save_freq = 10
        self.monitor_best = math.inf
        self.epoch_log = "\r\r{}/{} [{:03d}%] loss={:.6f}, e_pos={:.6f}, " \
                        "e_delta={:.6f}, e_bone={:.6f}, e_vel={:.6f}"
        self.epoch_summary = (
            "Epoch {:03d}/{:03d}:\n"
            "----------------------------------------------------------\n"
            "Loss: {:.3f}, Test Loss: {:.3f}\n"
            "Position Error: {:.3f}, Test Position Error: {:.3f}\n"
            "Delta Error: {:.6f}, Test Delta Error: {:.6f}\n"
            "Bone Length Error: {:.3f}, Test Bone Length Error: {:.3f}\n"
            "Velocity Error: {:.3f}, Test Velocity Error: {:.3f}\n"
        )

    def __reset_trackers__(self, training=True):
        if training:
            self.train_loss.reset_state()
            self.train_position_loss.reset_state()
            self.train_delta_loss.reset_state()
            self.train_bone_loss.reset_state()
            self.train_velocity_loss.reset_state()
        else:
            self.test_loss.reset_state()
            self.test_position_loss.reset_state()
            self.test_delta_loss.reset_state()
            self.test_bone_loss.reset_state()
            self.test_velocity_loss.reset_state()

    def compute_losses(self, targets, predictions):
        deltas, poses = predictions
        position_loss = losses.mean_distance_loss(targets, poses)
        delta_loss = tf.keras.losses.mse(
            self.delta_converter(targets),
            deltas
        )
        bone_loss = losses.mean_bone_length_loss(targets, poses)
        velocity_loss = losses.mean_velocity_loss(targets, poses)
        loss = tf.constant(0.0, dtype=float) + self.a * position_loss \
        + self.b * delta_loss + self.c * bone_loss + self.d * velocity_loss
        return loss, position_loss, delta_loss, bone_loss, velocity_loss

    def train_epoch(self, epoch):
        def train_step(data):
            _, estimation, gt, _ = data
            with tf.GradientTape() as tape:
                predictions = self.model(estimation, training=True)
                loss, loss_position, loss_delta, loss_bone, loss_vel = self.compute_losses(gt, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss.update_state(loss)
            self.train_position_loss.update_state(loss_position)
            self.train_delta_loss.update_state(loss_delta)
            self.train_bone_loss.update_state(loss_bone)
            self.train_velocity_loss.update_state(loss_vel)
            return estimation.shape[0] * estimation.shape[1]
        
        counter = 0
        self.__reset_trackers__(training=True)
        logs.print_info("Epoch {}/{} - Train".format(epoch+1, self.EPOCHS))
        for sample in self._trainset.get_dataset():
            counter += train_step(sample)
            percent = counter / self._trainset.total_frames()
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._trainset.total_frames(),
                    int(percent * 100),
                    self.train_loss.result().numpy(),
                    self.train_position_loss.result().numpy(),
                    self.train_delta_loss.result().numpy(),
                    self.train_bone_loss.result().numpy(),
                    self.train_velocity_loss.result().numpy()
                )
            )
            sys.stdout.flush()
        print()
        print("---"*5)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('position loss', self.train_position_loss.result(), step=epoch)
            tf.summary.scalar('delta loss', self.train_delta_loss.result(), step=epoch)
            tf.summary.scalar('bone loss', self.train_bone_loss.result(), step=epoch)
            tf.summary.scalar('velocity loss', self.train_velocity_loss.result(), step=epoch)

    def test_epoch(self, epoch):
        def test_step(data):
            _, estimation, gt, _ = data
            predictions = self.model(estimation, training=False)
            loss, loss_position, loss_delta, loss_bone, loss_vel = self.compute_losses(gt, predictions)
            self.test_loss.update_state(loss)
            self.test_position_loss.update_state(loss_position)
            self.test_delta_loss.update_state(loss_delta)
            self.test_bone_loss.update_state(loss_bone)
            self.test_velocity_loss.update_state(loss_vel)
            return estimation.shape[0] * estimation.shape[1]
        counter = 0
        self.__reset_trackers__(training=False)
        logs.print_info("Epoch {}/{} - Test".format(epoch+1, self.EPOCHS))
        for sample in self._testset.get_dataset():
            counter += test_step(sample)
            percent = counter / self._testset.total_frames()
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._testset.total_frames(),
                    int(percent * 100),
                    self.test_loss.result().numpy(),
                    self.test_position_loss.result().numpy(),
                    self.test_delta_loss.result().numpy(),
                    self.test_bone_loss.result().numpy(),
                    self.test_velocity_loss.result().numpy()
                )
            )
            sys.stdout.flush()
        print()
        print("---"*5)
        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
            tf.summary.scalar('position loss', self.test_position_loss.result(), step=epoch)
            tf.summary.scalar('delta loss', self.test_delta_loss.result(), step=epoch)
            tf.summary.scalar('bone loss', self.test_bone_loss.result(), step=epoch)
            tf.summary.scalar('velocity loss', self.test_velocity_loss.result(), step=epoch)

    def train(self):
        t = time.time()
        for epoch in range(self.EPOCHS):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            best = False
            if self.monitor_best > self.test_loss.result().numpy():
                best = True
                self.monitor_best = self.test_loss.result().numpy()
            if best and epoch > 2:
                self.save_checkpoint()
                logs.print_info("\nSaving new best checkpoints at epoch {:03d}".format(epoch))
            print()
            msg = self.epoch_summary.format(
                epoch, self.EPOCHS, self.train_loss.result(), self.test_loss.result(),
                self.train_position_loss.result(), self.test_position_loss.result(),
                self.train_delta_loss.result(), self.test_delta_loss.result(),
                self.train_bone_loss.result(), self.test_bone_loss.result(),
                self.train_velocity_loss.result(), self.test_velocity_loss.result()
            )
            logs.print_info(msg)
            print("===" * 10)
        t = tools.secondsToStr(time.time() - t)
        logs.print_info(f"Training completed ! Duration: {t}")


class SkeletonModelTrainer(Trainer):
    # TODO: complete this class
    def __init__(self,
                config,
                model : SkeletonModel,
                train_dataloader : Human36mBoneDatasetLoader,
                test_dataloader : Human36mBoneDatasetLoader
        ):
        super().__init__(config, model, "mftmodel")
        self._trainset = train_dataloader
        self._testset = test_dataloader
        self.BATCH_SIZE = config.running.batch_size
        self.EPOCHS = config.running.epochs
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # Metrics to track losses
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.train_bone_loss = tf.keras.metrics.Mean(name="train_bone_loss")
        self.test_bone_loss = tf.keras.metrics.Mean(name="test_bone_loss")
        
        self.save_freq = 10
        self.monitor_best = math.inf
        self.epoch_log = "\r\r{}/{} [{:03d}%] loss={:.6f}, e_bone={:.6f}"
        self.epoch_summary = (
            "Epoch {:03d}/{:03d}:\n"
            "----------------------------------------------------------\n"
            "Loss: {:.3f}, Test Loss: {:.3f}\n"
            "Bone Length Error: {:.3f}, Test Bone Length Error: {:.3f}\n"
        )

    def __reset_trackers__(self, training=True):
        if training:
            self.train_loss.reset_state()
            self.train_bone_loss.reset_state()
        else:
            self.test_loss.reset_state()
            self.test_bone_loss.reset_state()

    def compute_losses(self, targets, predictions):
        targets = tf.reduce_max(targets, axis=1, keepdims=True)
        loss = self.mse(
            targets, predictions
        )
        err_bone = self.mae(
            targets, predictions
        ) # / 1000
        return loss, err_bone

    def train_epoch(self, epoch):
        def train_step(data):
            poses2d, bones_gt, _ = data
            with tf.GradientTape() as tape:
                predictions = self.model(poses2d, training=True)
                loss, loss_bone = self.compute_losses(bones_gt, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss.update_state(loss)
            self.train_bone_loss.update_state(loss_bone)
            return poses2d.shape[0] * poses2d.shape[1]
        
        counter = 0
        self.__reset_trackers__(training=True)
        logs.print_info("Epoch {}/{} - Train".format(epoch+1, self.EPOCHS))
        for sample in self._trainset.get_dataset():
            counter += train_step(sample)
            percent = counter / self._trainset.total_frames()
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._trainset.total_frames(),
                    int(percent * 100),
                    self.train_loss.result().numpy(),
                    self.train_bone_loss.result().numpy(),
                )
            )
            sys.stdout.flush()
        print()
        print("---"*5)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('bone loss', self.train_bone_loss.result(), step=epoch)

    def test_epoch(self, epoch):
        def test_step(data):
            poses2d, bones_gt, _ = data
            predictions = self.model(poses2d, training=False)
            loss, loss_bone = self.compute_losses(bones_gt, predictions)
            self.test_loss.update_state(loss)
            self.test_bone_loss.update_state(loss_bone)
            return poses2d.shape[0] * poses2d.shape[1]
        counter = 0
        self.__reset_trackers__(training=False)
        logs.print_info("Epoch {}/{} - Test".format(epoch+1, self.EPOCHS))
        for sample in self._testset.get_dataset():
            counter += test_step(sample)
            percent = counter / self._testset.total_frames()
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._testset.total_frames(),
                    int(percent * 100),
                    self.test_loss.result().numpy(),
                    self.test_bone_loss.result().numpy(),
                )
            )
            sys.stdout.flush()
        print()
        print("---"*5)
        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
            tf.summary.scalar('bone loss', self.test_bone_loss.result(), step=epoch)

    def train(self):
        t = time.time()
        for epoch in range(self.EPOCHS):
            self.train_epoch(epoch)
            self.test_epoch(epoch)
            best = False
            if self.monitor_best > self.test_loss.result().numpy():
                best = True
                self.monitor_best = self.test_loss.result().numpy()
            if best and epoch > 2:
                self.save_checkpoint()
                logs.print_info("\nSaving new best checkpoints at epoch {:03d}".format(epoch))
            print()
            msg = self.epoch_summary.format(
                epoch, self.EPOCHS, self.train_loss.result(), self.test_loss.result(),
                self.train_bone_loss.result(), self.test_bone_loss.result(),
            )
            logs.print_info(msg)
            print("===" * 10)
        t = tools.secondsToStr(time.time() - t)
        logs.print_info(f"Training completed ! Duration: {t}")