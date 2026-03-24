#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import random

LENGTH = 200


# HFONT = {'fontname':'Helvetica'}
# def draw_error_curves(errors, error_type, filename, path):
#     if not os.path.exists(path):
#         os.makedirs(path)
#     x = np.arange(start=1, stop=len(errors) + 1)
#     fig = plt.figure()
#     plt.plot(x, errors)
#     plt.xlabel("time", HFONT)
#     plt.ylabel("errors", HFONT)
#     plt.title("%s" % error_type, HFONT)
#     plt.savefig(f"{path}/{filename}.png", dpi=100)
#     plt.close(fig)


def draw_error_curves(errors, error_type, filename, path):
    if not os.path.exists(path):
        os.makedirs(path)
    begin = random.randint(0, len(errors) - LENGTH)
    x = np.arange(start=1, stop=LENGTH + 1)
    fig = plt.figure()
    plt.plot(x, errors[begin:begin + LENGTH])
    plt.xlabel("time")
    plt.ylabel("errors")
    plt.title("%s" % error_type)
    plt.savefig(f"{path}/{filename}.png", dpi=100)
    plt.close(fig)


def draw_error_comparison_curves(errors, error_type, filename, path, size):
    if not os.path.exists(path):
        os.makedirs(path)
    fig = plt.figure()
    begin = random.randint(0, size - LENGTH)
    x = np.arange(start=1, stop=LENGTH + 1)
    for key in errors.keys():
        plt.plot(x, errors[key][begin:begin + LENGTH], label=key, linestyle="dashed")
    plt.xlabel("time")
    plt.ylabel("errors")
    plt.title("%s" % error_type)
    plt.legend()
    plt.savefig(f"{path}/{filename}.png", dpi=100)
    plt.close(fig)
