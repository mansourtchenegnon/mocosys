#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: mansour
"""

import json
import logging
import datetime


class Logger:
    """
    Training process logger
    """

    def __init__(self, filename):
        self.logs = ""
        self.filename = filename
        # self.logger = logging.getLogger(self.__class__.__name__)
        # self.logger.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%d %b %Y %H:%M:%S')

        # file_handler = logging.FileHandler(filename, mode='w')
        # file_handler.setLevel(logging.INFO)
        # file_handler.setFormatter(formatter)

        # stream_handler = logging.StreamHandler()
        # stream_handler.setLevel(logging.INFO)
        # stream_handler.setFormatter(formatter)

        # self.logger.addHandler(file_handler)
        # self.logger.addHandler(stream_handler)
        logging.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def info(self, strs):
        info = "{} : [INFO] {}\n".format(datetime.datetime.now().strftime('%d %b %Y %H:%M:%S'), strs)
        self.logs += info

    def warning(self, strs):
        warning = "{} : [WARNING] {}\n".format(datetime.datetime.now().strftime('%d %b %Y %H:%M:%S'), strs)
        self.logs += warning

    def printing(self, strs):
        message = "{}\n".format(strs)
        self.logs += message

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)

    def commit_logs(self):
        with open(self.filename, "w") as trainlogs:
            trainlogs.writelines(self.logs)


def print_message(verbosity, message):
    """
    Displays message if allowed.

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
    if verbosity:
        print(message)


def print_info(message, verbosity=True):
    """
    Prints informative messages

    Parameters
    ----------
    message : str
        The message to print.
    verbosity : bool, optional
        Tells if message should be displayed. The default is True.

    Returns
    -------
    None.

    """
    info = "{} : [INFO] {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    print_message(verbosity, info)


def print_warning(message, verbosity=True):
    """
    Prints warning messages

    Parameters
    ----------
    message : str
        The message to print.
    verbosity : bool, optional
        Tells if message should be displayed. The default is True.

    Returns
    -------
    None.

    """
    warning = "{} : [WARNING] {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    print_message(verbosity, warning)


def print_error(message, verbosity=True):
    """
    Prints warning messages

    Parameters
    ----------
    message : str
        The message to print.
    verbosity : bool, optional
        Tells if message should be displayed. The default is True.

    Returns
    -------
    None.

    """
    error = "{} : [ERROR] {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), message)
    print_message(verbosity, error)
