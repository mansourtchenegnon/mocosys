#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mansour Tchenegnon
@version: 25.04.2025
"""

import tensorflow as tf
import argparse
from utility.datasets.h36m_dataset import Human36mDataset

# %% Functions and variables
def _feature_of_bytes(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def _data_example(**kwargs):
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible data type.
    sample_features = {}
    for key in kwargs:
        sample_features[key] = kwargs[key]
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=sample_features))
    return example_proto

default_features_desc = {
    'input_2d': tf.io.FixedLenFeature([], tf.string),
    'target_3d': tf.io.FixedLenFeature([], tf.string),
    'name': tf.io.FixedLenFeature([], tf.string)
}

def _parse_data(example_proto):
    return tf.io.parse_single_example(example_proto, default_features_desc)

def save_default_tf_record(dataset, record_file):
    with tf.io.TFRecordWriter(record_file) as writer:
        for input_2d, target_3d, name in dataset:
            tf_example = _data_example(input_2d, target_3d, name)
            writer.write(tf_example.SerializeToString())

def save_sota_tf_record(dataset, record_file):
    with tf.io.TFRecordWriter(record_file) as writer:
        for input_2d, target_3d, name in dataset:
            tf_example = _data_example(input_2d, target_3d, name)
            writer.write(tf_example.SerializeToString())

# Parse args
parser = argparse.ArgumentParser(
        description="Argument parser."
    )

parser.add_argument(
    "--sota",
    action="store_true",
    help='Tells whether it is dataset with SOTA input or default dataset.',
)

parser.add_argument("--batch_size", default=128, type=int, help="The batch size")

parser.add_argument(
    "--set",
    type=str,
    help='Set of the dataset. "Train" or "Test"')

args = parser.parse_args()

if args.set == "Train":
    dataloader = Human36mDataset(True)
else:
    dataloader = Human36mDataset()
