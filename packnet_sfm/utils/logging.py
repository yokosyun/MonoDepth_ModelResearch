# Copyright 2020 Toyota Research Institute.  All rights reserved.

import os
from termcolor import colored
from functools import partial

from packnet_sfm.utils.horovod import on_rank_0


def prepare_dataset_prefix(config, dataset_idx):
    """
    Concatenates dataset path and split for metrics logging

    Parameters
    ----------
    config : CfgNode
        Dataset configuration
    dataset_idx : int
        Dataset index for multiple datasets

    Returns
    -------
    prefix : str
        Dataset prefix for metrics logging
    """
    # Path is always available
    prefix = "{}".format(os.path.splitext(config.path[dataset_idx].split("/")[-1])[0])
    # If split is available and does not contain { character
    if config.split[dataset_idx] != "" and "{" not in config.split[dataset_idx]:
        prefix += "-{}".format(
            os.path.splitext(os.path.basename(config.split[dataset_idx]))[0]
        )
    # If depth type is available
    if config.depth_type[dataset_idx] != "":
        prefix += "-{}".format(config.depth_type[dataset_idx])
    # If we are using specific cameras
    if len(config.cameras[dataset_idx]) == 1:  # only allows single cameras
        prefix += "-{}".format(config.cameras[dataset_idx][0])
    # Return full prefix
    return prefix


def s3_url(config):
    """
    Generate the s3 url where the models will be saved

    Parameters
    ----------
    config : CfgNode
        Model configuration

    Returns
    -------
    url : str
        String containing the URL pointing to the s3 bucket
    """
    return "https://s3.console.aws.amazon.com/s3/buckets/{}/{}".format(
        config.checkpoint.s3_path[5:], config.name
    )


class AvgMeter:
    """Average meter for logging"""

    def __init__(self, n_max=100):
        """
        Initializes a AvgMeter object.

        Parameters
        ----------
        n_max : int
            Number of steps to average over
        """
        self.n_max = n_max
        self.values = []

    def __call__(self, value):
        """Appends new value and returns average"""
        self.values.append(value)
        if len(self.values) > self.n_max:
            self.values.pop(0)
        return self.get()

    def get(self):
        """Get current average"""
        return sum(self.values) / len(self.values)

    def reset(self):
        """Reset meter"""
        self.values.clear()

    def get_and_reset(self):
        """Gets current average and resets"""
        average = self.get()
        self.reset()
        return average
