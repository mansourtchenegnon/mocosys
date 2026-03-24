# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import tensorflow as tf
import numpy as np
import hashlib

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a tensorflow function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) is np.ndarray:
            args[i] = tf.convert_to_tensor(arg)
            if unsqueeze:
                args[i] = tf.expand_dims(args[i], 0)
        
    result = func(*args)
    
    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) is tf.Tensor:
                if unsqueeze:
                    res = tf.squeeze(res, 0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) is tf.Tensor:
        if unsqueeze:
            result = tf.squeeze(result, 0)
        return result.numpy()
    else:
        return result
    
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value
