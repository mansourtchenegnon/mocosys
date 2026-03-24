# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a numpy array of shape (*, 4) for q and a numpy array of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a numpy of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    qvec = q[..., 1:]  # vector part (x, y, z)
    uv = numpy.cross(qvec, v)
    uuv = numpy.cross(qvec, uv)

    return v + 2.0 * (q[..., :1] * uv + uuv)
    
    
def qinverse(q, inplace=False):
    # We assume the quaternion to be normalized
    if inplace:
        q[..., 1:] *= -1
        return q
    else:
        w = q[..., :1]
        xyz = q[..., 1:]
        return numpy.concatenate((w, -xyz), axis=len(q.shape)-1)