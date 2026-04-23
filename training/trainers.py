#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 2025.04
"""

import math
import yaml
import sys
import time

from model import losses, metrics
from utility.datasets.loaders.h36m_dataloader import Human36mBoneDatasetLoader, Human36mSotaDatasetLoader
import utility.logger as logs
import utility.tools as tools
import tensorflow as tf
from keras import ops as kops

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
        self.checkpoint_dir = "./checkpoints/%s/%s" % (model_type, config["running"]["version"])
        self.log_dir = "./checkpoints/logs/%s/%s" % (model_type, config["running"]["version"])
        self.model = model
        self.normalization_parameters = []
        tools.make_dir(self.checkpoint_dir)
        tools.make_dir(self.log_dir)
        self.logger = logs.Logger("%s/training.log" % self.log_dir)
        self.train_summary_writer = tf.summary.create_file_writer(f"{self.log_dir}")
        self.history = {}


    def save_checkpoint(self, epoch : int):
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'config': self.config,
            'normalization': self.normalization_parameters
        }
        self.model.save(f"{self.checkpoint_dir}/best.keras")
        self.model.save_weights(f"{self.checkpoint_dir}/best.weights.h5")
        with open(f"{self.checkpoint_dir}/state.yaml", 'w') as fp:
            yaml.dump(state, fp, default_flow_style=False)


class MFTModelTrainer(Trainer):
    def __init__(self,
                config,
                model : MotionFineTuningModel,
                train_dataloader : Human36mSotaDatasetLoader,
        ):
        super().__init__(config, model, "mftmodel")
        self._trainset, self._testset = train_dataloader.get_train_validation_datasets()
        self._trainset_size = 0
        self._testset_size = 0
        self.BATCH_SIZE = config["running"]["batch_size"]
        self.EPOCHS = config["running"]["epochs"]
        self.mse = tf.keras.losses.MeanSquaredError()
        self.distance_loss = losses.DistanceLoss()

        # Loss weights
        self.alpha = 1.  # position
        self.beta = 1.  # delta

        self.delta_converter = solvers.DeltaConverter(self.model.skeleton, 3, self.model.window)

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # Metrics to track
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.train_position_error = tf.keras.metrics.Mean(name="train_position_error")
        self.test_position_error = tf.keras.metrics.Mean(name="test_position_error")
        self.train_bone_error = tf.keras.metrics.Mean(name="train_bone_error")
        self.test_bone_error = tf.keras.metrics.Mean(name="test_bone_error")
        self.train_velocity_error = tf.keras.metrics.Mean(name="train_velocity_error")
        self.test_velocity_error = tf.keras.metrics.Mean(name="test_velocity_error")
        self.train_acceleration_error = tf.keras.metrics.Mean(name="train_acceleration_error")
        self.test_acceleration_error = tf.keras.metrics.Mean(name="test_acceleration_error")
        
        self.save_freq = 10
        self.monitor_best = math.inf
        self.epoch_log = "\r\r{}/{} [{:03d}%] loss={:.6f}, e_pos={:.3f}, e_vel={:.3f}, e_acc={:.3f}, e_bone={:.3f}"
        self.epoch_summary = (
            "Epoch {:03d}/{:03d}:\n"
            "----------------------------------------------------------\n"
            "Loss: {:.3f}, Test Loss: {:.3f}\n"
            "Position Error: {:.3f}, Test Position Error: {:.3f}\n"
            "Velocity Error: {:.3f}, Test Velocity Error: {:.3f}\n"
            "Acceleration Error: {:.3f}, Test Acceleration Error: {:.3f}\n"
            "Bone Length Error: {:.3f}, Test Bone Length Error: {:.3f}\n"
        )

    def __reset_trackers__(self, training=True):
        if training:
            self.train_loss.reset_state()
            self.train_position_error.reset_state()
            self.train_bone_error.reset_state()
            self.train_velocity_error.reset_state()
            self.train_acceleration_error.reset_state()
        else:
            self.test_loss.reset_state()
            self.test_position_error.reset_state()
            self.test_bone_error.reset_state()
            self.test_velocity_error.reset_state()
            self.test_acceleration_error.reset_state()

    def compute_losses(self, targets, outputs):
        deltas, poses = outputs
        delta_loss = tf.keras.losses.mse(
            self.delta_converter(targets, keepdims=True, format=True),
            deltas
        )
        position_loss = self.distance_loss(targets, poses)
        loss = self.beta * delta_loss + self.alpha * position_loss
        return loss

    def train_epoch(self, epoch):
        def train_step(data):
            _, estimation, gt, _ = data
            with tf.GradientTape() as tape:
                deltas, poses = self.model(estimation, training=True)
                loss = self.compute_losses(gt, [deltas, poses])

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss.update_state(loss)
            self.train_position_error.update_state(metrics.mean_position_error(gt, poses))
            self.train_velocity_error.update_state(metrics.mean_velocity_error(gt, poses))
            self.train_acceleration_error.update_state(metrics.mean_acceleration_error(gt, poses))
            self.train_bone_error.update_state(metrics.mean_bone_length_error(gt, poses))
            return estimation.shape[0] * estimation.shape[1]
        
        counter = 0
        self.__reset_trackers__(training=True)
        logs.print_info("Epoch {}/{} - Train".format(epoch, self.EPOCHS))
        for sample in self._trainset:
            counter += train_step(sample)
            if self._trainset_size == 0:
                percent = 0
            else:
                percent = counter / self._trainset_size
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._trainset_size,
                    int(percent * 100),
                    self.train_loss.result().numpy(),
                    self.train_position_error.result().numpy() * 1000,
                    self.train_velocity_error.result().numpy() * 1000,
                    self.train_acceleration_error.result().numpy() * 1000,
                    self.train_bone_error.result().numpy() * 1000
                )
            )
            sys.stdout.flush()
        print()
        print("---"*5)

        if self._trainset_size == 0:
           self._trainset_size = counter

        with self.train_summary_writer.as_default():
            tf.summary.scalar('train loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('train position error', self.train_position_error.result(), step=epoch)
            tf.summary.scalar('train acceleration error', self.train_acceleration_error.result(), step=epoch)
            tf.summary.scalar('train bone error', self.train_bone_error.result(), step=epoch)
            tf.summary.scalar('train velocity error', self.train_velocity_error.result(), step=epoch)

    def test_epoch(self, epoch):
        def test_step(data):
            _, estimation, gt, _ = data
            deltas, poses = self.model(estimation, training=False)
            loss = self.compute_losses(gt, [deltas, poses])
            self.test_loss.update_state(loss)
            self.test_position_error.update_state(metrics.mean_position_error(gt, poses))
            self.test_acceleration_error.update_state(metrics.mean_acceleration_error(gt, poses))
            self.test_bone_error.update_state(metrics.mean_bone_length_error(gt, poses))
            self.test_velocity_error.update_state(metrics.mean_velocity_error(gt, poses))
            return estimation.shape[0] * estimation.shape[1]
        counter = 0
        self.__reset_trackers__(training=False)
        logs.print_info("Epoch {}/{} - Test".format(epoch, self.EPOCHS))
        for sample in self._testset:
            counter += test_step(sample)
            if self._testset_size == 0:
                percent = 0
            else:
                percent = counter / self._testset_size
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._testset_size,
                    int(percent * 100),
                    self.test_loss.result().numpy(),
                    self.test_position_error.result().numpy() * 1000,
                    self.test_velocity_error.result().numpy() * 1000,
                    self.test_acceleration_error.result().numpy() * 1000,
                    self.test_bone_error.result().numpy() * 1000
                )
            )
            sys.stdout.flush()

        if self._testset_size == 0:
            self._testset_size = counter

        print()
        print("---"*5)
        with self.train_summary_writer.as_default():
            tf.summary.scalar('test loss', self.test_loss.result(), step=epoch)
            tf.summary.scalar('test position loss', self.test_position_error.result() * 1000, step=epoch)
            tf.summary.scalar('test acceleration loss', self.test_acceleration_error.result() * 1000, step=epoch)
            tf.summary.scalar('test bone loss', self.test_bone_error.result() * 1000, step=epoch)
            tf.summary.scalar('test velocity loss', self.test_velocity_error.result() * 1000, step=epoch)

    def train(self):
        # Initialise necessary variables
        train_loss_results = []
        test_loss_results = []
        train_position_error_results = []
        test_position_error_results = []
        train_bone_error_results = []
        test_bone_error_results = []
        train_acceleration_error_results = []
        test_acceleration_error_results = []
        train_velocity_error_results = []
        test_velocity_error_results = []
        t = time.time()
        for epoch in range(1, self.EPOCHS+1):
            self.train_epoch(epoch)
            self.test_epoch(epoch)

            train_loss_results.append(self.train_loss.result().numpy())
            test_loss_results.append(self.test_loss.result().numpy())
            train_position_error_results.append(self.train_position_error.result().numpy() * 1000)
            test_position_error_results.append(self.test_position_error.result().numpy() * 1000)
            train_acceleration_error_results.append(self.train_acceleration_error.result().numpy() * 1000)
            test_acceleration_error_results.append(self.test_acceleration_error.result().numpy() * 1000)
            train_bone_error_results.append(self.train_bone_error.result().numpy() * 1000)
            test_bone_error_results.append(self.test_bone_error.result().numpy() * 1000)
            train_velocity_error_results.append(self.train_velocity_error.result().numpy() * 1000)
            test_velocity_error_results.append(self.test_velocity_error.result().numpy() * 1000)

            best = False
            if self.monitor_best > self.test_loss.result().numpy():
                best = True
                self.monitor_best = self.test_loss.result().numpy()
            if best: # and epoch > 2:
                self.save_checkpoint(epoch)
                logs.print_info("\nSaving new best checkpoints at epoch {:03d}".format(epoch))
            print()
            msg = self.epoch_summary.format(
                epoch, self.EPOCHS, self.train_loss.result(), self.test_loss.result(),
                self.train_position_error.result() * 1000, self.test_position_error.result() * 1000,
                self.train_velocity_error.result() * 1000, self.test_velocity_error.result() * 1000,
                self.train_acceleration_error.result() * 1000, self.test_acceleration_error.result() * 1000,
                self.train_bone_error.result() * 1000, self.test_bone_error.result() * 1000
            )
            logs.print_info(msg)
            self.logger.info(msg)
            self.logger.printing("\nSaving new best checkpoints at epoch {:03d}".format(epoch))
            self.logger.printing("==========" * 3)
            print("===" * 10)
        t = tools.secondsToStr(time.time() - t)
        # History dictionary
        self.history = {
            "train loss": train_loss_results,
            "test loss": test_loss_results,
            "train position error": train_position_error_results,
            "test position error": test_position_error_results,
            "train bone error": train_bone_error_results,
            "test bone error": test_bone_error_results,
            "train acceleration error": train_acceleration_error_results,
            "test acceleration error": test_acceleration_error_results,
            "train velocity error": train_velocity_error_results,
            "test velocity error": test_velocity_error_results,
        }
        self.logger.info(f"Training completed! Duration: {t}")
        self.logger.commit_logs()
        logs.print_info(f"Training completed ! Duration: {t}")


class SkeletonModelTrainer(Trainer):
    def __init__(self,
                config,
                model : SkeletonModel,
                train_dataloader : Human36mBoneDatasetLoader
        ):
        super().__init__(config, model, "skelmodel")
        self._trainset, self._testset = train_dataloader.get_train_validation_datasets()
        self.normalization_parameters = train_dataloader.get_parameters()
        model.set_normalization_parameters(self.normalization_parameters)
        self._trainset_size = 0
        self._testset_size = 0
        self.BATCH_SIZE = config["running"]["batch_size"]
        self.EPOCHS = config["running"]["epochs"]
        self.mse = tf.keras.losses.MeanSquaredError()
        self.mae = tf.keras.losses.MeanAbsoluteError()

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # Metrics to track losses
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.train_bone_error = tf.keras.metrics.Mean(name="train_bone_error")
        self.test_bone_error = tf.keras.metrics.Mean(name="test_bone_error")
        
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
            self.train_bone_error.reset_state()
        else:
            self.test_loss.reset_state()
            self.test_bone_error.reset_state()

    def compute_losses(self, targets, predictions):
        targets = kops.max(targets, axis=1, keepdims=True)
        loss = self.mse(
            targets, predictions
        )
        bone_error = self.mae(
            tools.denormalise(targets, self.normalization_parameters[2], self.normalization_parameters[3]),
            tools.denormalise(predictions, self.normalization_parameters[2], self.normalization_parameters[3])
        ) # / 1000
        return loss, bone_error

    def train_epoch(self, epoch):
        def train_step(data):
            poses2d, bones_gt, _ = data
            with tf.GradientTape() as tape:
                predictions = self.model(poses2d, training=True)
                loss, loss_bone = self.compute_losses(bones_gt, predictions)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.train_loss.update_state(loss)
            self.train_bone_error.update_state(loss_bone)
            return poses2d.shape[0] * poses2d.shape[1]
        
        counter = 0
        self.__reset_trackers__(training=True)
        logs.print_info("Epoch {}/{} - Train".format(epoch+1, self.EPOCHS))
        for sample in self._trainset:
            counter += train_step(sample)
            if self._trainset_size == 0:
                percent = 0
            else:
                percent = counter / self._trainset_size
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._trainset_size,
                    int(percent * 100),
                    self.train_loss.result().numpy(),
                    self.train_bone_error.result().numpy() * 1000,
                )
            )
            sys.stdout.flush()
        print()
        print("---"*5)

        if self._trainset_size == 0:
            self._trainset_size = counter

        with self.train_summary_writer.as_default():
            tf.summary.scalar('train loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('train bone error', self.train_bone_error.result(), step=epoch)

    def test_epoch(self, epoch):
        def test_step(data):
            poses2d, bones_gt, _ = data
            predictions = self.model(poses2d, training=False)
            loss, loss_bone = self.compute_losses(bones_gt, predictions)
            self.test_loss.update_state(loss)
            self.test_bone_error.update_state(loss_bone)
            return poses2d.shape[0] * poses2d.shape[1]
        counter = 0
        self.__reset_trackers__(training=False)
        logs.print_info("Epoch {}/{} - Test".format(epoch+1, self.EPOCHS))
        for sample in self._testset:
            counter += test_step(sample)
            if self._testset_size == 0:
                percent = 0
            else:
                percent = counter / self._testset_size
            sys.stdout.write(
                self.epoch_log.format(
                    counter,
                    self._testset_size,
                    int(percent * 100),
                    self.test_loss.result().numpy(),
                    self.test_bone_error.result().numpy() * 1000,
                )
            )
            sys.stdout.flush()
        print()
        print("---"*5)

        if self._testset_size == 0:
            self._testset_size = counter

        with self.train_summary_writer.as_default():
            tf.summary.scalar('test loss', self.test_loss.result(), step=epoch)
            tf.summary.scalar('test bone error', self.test_bone_error.result() * 1000, step=epoch)

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
                self.save_checkpoint(epoch=epoch)
                logs.print_info("\nSaving new best checkpoints at epoch {:03d}".format(epoch))
            print()
            msg = self.epoch_summary.format(
                epoch, self.EPOCHS, self.train_loss.result(), self.test_loss.result(),
                self.train_bone_error.result() * 1000, self.test_bone_error.result() * 1000,
            )
            logs.print_info(msg)
            print("===" * 10)
        t = tools.secondsToStr(time.time() - t)
        logs.print_info(f"Training completed ! Duration: {t}")