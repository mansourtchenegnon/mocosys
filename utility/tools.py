import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import json
from types import SimpleNamespace as Namespace
from keras import ops as kops
import yaml


np.seterr(divide='ignore', invalid='ignore')

ROTATION_NUMBERS = {'q': 4, '6d': 6, 'eular': 3}


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w] 


def make_dataset(dir_list, phase, data_split=3, sort=False, sort_index=1):
    images = []
    for dataroot in dir_list:
        _images = []
        image_filter = []

        assert os.path.isdir(dataroot), '%s is not a valid directory' % dataroot
        for root, _, fnames in sorted(os.walk(dataroot)):
            for fname in fnames:
                if phase in fname:
                    path = os.path.join(root, fname)
                    _images.append(path)
        if sort:
            _images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[sort_index]))
        if data_split is not None:
            for i in range(int(len(_images)/data_split - 1)):
                image_filter.append(_images[data_split*i])
            images += image_filter
            return images
        else:
            return _images


def mkdir(folder):
    if os.path.exists(folder):
        return 1
    else:
        os.makedirs(folder)


def normalize_data(orig_data):
    data_mean = np.mean(orig_data, axis=0, dtype='float32')
    data_std = np.std(orig_data, axis=0, dtype='float32')
    normalized_data = np.divide((orig_data - data_mean), data_std)
    normalized_data[normalized_data != normalized_data] = 0
    return normalized_data, data_mean, data_std


def umnormalize_data(normalized_data, data_mean, data_std):
    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # Dimensionality
    ndims = len(normalized_data.shape)
    shape = [1 for i in range(ndims-1)] + [D]
    repeats = [normalized_data.shape[i] for i in range(ndims-1)] + [1]

    stdMat = data_std.reshape(shape)
    stdMat = np.tile(stdMat, repeats)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.tile(meanMat, repeats)
    orig_data = np.multiply(normalized_data, stdMat) + meanMat
    return orig_data

def normalise(data):
    # TODO : complete
    data_mean = kops.mean(data, axis=0)
    data_std = kops.std(data, axis=0)

def denormalise(normalised_data, data_mean, data_std):
    D = data_mean.shape[0]  # Dimensionality
    ndims = len(normalised_data.shape)
    shape = [1 for i in range(ndims-1)] + [D]
    repeats = [normalised_data.shape[i] for i in range(ndims-1)] + [1]

    std_mat = kops.reshape(data_std, shape)
    std_mat = kops.tile(std_mat, repeats)
    mean_mat = kops.reshape(data_mean, shape)
    mean_mat = kops.tile(mean_mat, repeats)
    original_data = kops.multiply(normalised_data, std_mat) + mean_mat
    return original_data

def interp_pose(pose_array, confidences, k=2):
    frame_number, joint_number = confidences.shape[0], confidences.shape[1]
    pose_completed = np.zeros_like(pose_array)
    confi_completed = np.zeros_like(confidences)
    for joint_index in range(joint_number):
        x_loc = pose_array[:, joint_index*2]
        y_loc = pose_array[:, joint_index*2 + 1]
        joint_confi = confidences[:, joint_index]
        select_indics = (joint_confi >= 0.4)
        x_spl = InterpolatedUnivariateSpline(np.arange(0, frame_number)[select_indics], x_loc[select_indics], k=2)
        y_spl = InterpolatedUnivariateSpline(np.arange(0, frame_number)[select_indics], y_loc[select_indics], k=2)
        c_spl = InterpolatedUnivariateSpline(np.arange(0, frame_number)[select_indics], joint_confi[select_indics], k=2)
        pose_completed[:, joint_index*2] = x_spl(np.arange(0, frame_number))
        pose_completed[:, joint_index*2 + 1] = y_spl(np.arange(0, frame_number))
        confi_completed[:, joint_index] = c_spl(np.arange(0, frame_number))
    return pose_completed, confi_completed


def secondsToStr(t):
    dd = t // (3600*24)
    r = t - dd * (3600*24)
    hh = r // 3600
    r -= hh * 3600
    mm = r // 60
    r -= mm * 60
    ss = r
    return "%02d-%02d:%02d:%02d" % (dd, hh, mm, ss)


def dict_to_namespace(data):
    if type(data) is list:
        return list(map(dict_to_namespace, data))
    elif type(data) is dict:
        sns = Namespace()
        for key, value in data.items():
            setattr(sns, key, dict_to_namespace(value))
        return sns
    else:
        return data


def save_config_json(conf, config_file):
    # dict_conf = vars(conf)
    dict_conf = json.loads(json.dumps(conf, default=lambda s: vars(s)))
    with open(config_file, "w") as cf:
        json.dump(dict_conf, cf)


def load_config_json(config_file):
    config = None
    with open(config_file, "r") as cf:
        config = json.load(cf, object_hook=lambda d: Namespace(**d))
    return config

def save_config_yaml(conf, config_file):
    dict_conf = json.loads(json.dumps(conf, default=lambda s: vars(s)))
    with open(config_file, "w") as fd:
        yaml.dump(dict_conf, fd)


def load_config_yaml(config_file):
    config = None
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config
