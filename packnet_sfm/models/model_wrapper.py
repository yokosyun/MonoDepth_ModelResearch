# Copyright 2020 Toyota Research Institute.  All rights reserved.

from collections import OrderedDict
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

from packnet_sfm.datasets.transforms import get_transforms
from packnet_sfm.utils.depth import (
    inv2depth,
    post_process_inv_depth,
    compute_depth_metrics,
)

from packnet_sfm.utils.image import flip_lr
from packnet_sfm.utils.load import (
    load_class,
    load_class_args_create,
    load_network,
    filter_args,
)
from packnet_sfm.utils.reduce import (
    reduce_dict,
    create_dict,
    average_loss_and_metrics,
)

from torchvision.utils import save_image


def set_random_seed(seed):
    if seed >= 0:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def setup_depth_net(config, **kwargs):
    """
    Create a depth network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    prepared : bool
        True if the network has been prepared before
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    depth_net : nn.Module
        Create depth network
    """
    depth_net = load_class_args_create(
        config.name,
        paths=[
            "packnet_sfm.networks.depth",
        ],
        args={**config, **kwargs},
    )
    if config.checkpoint_path is not "":
        depth_net = load_network(
            depth_net, config.checkpoint_path, ["depth_net", "disp_network"]
        )
    return depth_net


def setup_pose_net(config, **kwargs):
    """
    Create a pose network

    Parameters
    ----------
    config : CfgNode
        Network configuration
    kwargs : dict
        Extra parameters for the network

    Returns
    -------
    pose_net : nn.Module
        Created pose network
    """
    pose_net = load_class_args_create(
        config.name,
        paths=[
            "packnet_sfm.networks.pose",
        ],
        args={**config, **kwargs},
    )
    if config.checkpoint_path is not "":
        pose_net = load_network(
            pose_net, config.checkpoint_path, ["pose_net", "pose_network"]
        )
    return pose_net


def setup_model(config, **kwargs):
    """
    Create a model

    Parameters
    ----------
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    prepared : bool
        True if the model has been prepared before
    kwargs : dict
        Extra parameters for the model

    Returns
    -------
    model : nn.Module
        Created model
    """
    model = load_class(
        config.name,
        paths=[
            "packnet_sfm.models",
        ],
    )(**{**config.loss, **kwargs})
    # Add depth network if required
    if model.network_requirements["depth_net"]:
        model.add_depth_net(setup_depth_net(config.depth_net))
    # Add pose network if required
    if model.network_requirements["pose_net"]:
        model.add_pose_net(setup_pose_net(config.pose_net))
    # If a checkpoint is provided, load pretrained model
    if config.checkpoint_path is not "":
        model = load_network(model, config.checkpoint_path, "model")
    # Return model
    return model


def setup_dataset(config, mode, requirements, **kwargs):
    """
    Create a dataset class

    Parameters
    ----------
    config : CfgNode
        Configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataset
    requirements : dict (string -> bool)
        Different requirements for dataset loading (gt_depth, gt_pose, etc)
    kwargs : dict
        Extra parameters for dataset creation

    Returns
    -------
    dataset : Dataset
        Dataset class for that mode
    """
    # If no dataset is given, return None
    if len(config.path) == 0:
        return None

    # Global shared dataset arguments
    dataset_args = {
        "back_context": config.back_context,
        "forward_context": config.forward_context,
        "data_transform": get_transforms(mode, **kwargs),
    }

    # Loop over all datasets
    datasets = []
    for i in range(len(config.split)):
        path_split = os.path.join(config.path[i], config.split[i])

        # Individual shared dataset arguments
        dataset_args_i = {
            "depth_type": config.depth_type[i] if requirements["gt_depth"] else None,
            "with_pose": requirements["gt_pose"],
        }

        # KITTI dataset
        if config.dataset[i] == "KITTI":
            from packnet_sfm.datasets.kitti_dataset import KITTIDataset

            dataset = KITTIDataset(
                config.path[i],
                path_split,
                **dataset_args,
                **dataset_args_i,
            )
        else:
            ValueError("Unknown dataset %d" % config.dataset[i])

        # Repeat if needed
        if "repeat" in config and config.repeat[i] > 1:
            dataset = ConcatDataset([dataset for _ in range(config.repeat[i])])
        datasets.append(dataset)

    # If training, concatenate all datasets into a single one
    if mode == "train":
        datasets = [ConcatDataset(datasets)]

    return datasets


def setup_dataloader(datasets, config, mode):
    """
    Create a dataloader class

    Parameters
    ----------
    datasets : list of Dataset
        List of datasets from which to create dataloaders
    config : CfgNode
        Model configuration (cf. configs/default_config.py)
    mode : str {'train', 'validation', 'test'}
        Mode from which we want the dataloader

    Returns
    -------
    dataloaders : list of Dataloader
        List of created dataloaders for each input dataset
    """
    return [
        (
            DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=config.num_workers,
            )
        )
        for dataset in datasets
    ]
