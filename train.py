# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
import os
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from datasets.transforms import get_transforms

# from model import PackNet01
from models import PackNet01 as PackNet01
from models import PoseNet as PoseNet
from geometry.pose import Pose
from losses.multiview_photometric_loss import MultiViewPhotometricLoss

parser = argparse.ArgumentParser(description="Mono Depth")
parser.add_argument("--model", default="da", help="select model")
parser.add_argument("--loadmodel", default=None, help="load model")
parser.add_argument("--maxdisp", type=int, default=192, help="maxium disparity")
parser.add_argument(
    "--datapath",
    default="/media/yoko/SSD-PGU3/workspace/datasets/KITTI/monodepth/KITTI_tiny/",
    help="datapath",
)

parser.add_argument("--epochs", type=int, default=300, help="number of epochs to train")
parser.add_argument("--batch_size", type=int, default=1, help="number of batch size")
parser.add_argument("--savemodel", default="./result", help="save model")
parser.add_argument(
    "--use_cuda", action="store_true", default=True, help="enables CUDA training"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

writer_train = SummaryWriter(log_dir="./logs/train")
writer_test = SummaryWriter(log_dir="./logs/test")


def deterministic_training(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_datasampler(dataset, mode):
    """Distributed data sampler"""
    return torch.utils.data.distributed.DistributedSampler(
        dataset,
        shuffle=(mode == "train"),
        num_replicas=1,
        rank=0,
    )


def worker_init_fn(worker_id):
    """Function to initialize workers"""
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)


def stack_batch(batch):
    """
    Stack multi-camera batches (B,N,C,H,W becomes BN,C,H,W)

    Parameters
    ----------
    batch : dict
        Batch

    Returns
    -------
    batch : dict
        Stacked batch
    """
    # If there is multi-camera information
    if len(batch["rgb"].shape) == 5:
        assert (
            batch["rgb"].shape[0] == 1
        ), "Only batch size 1 is supported for multi-cameras"
        # Loop over all keys
        for key in batch.keys():
            # If list, stack every item
            if is_list(batch[key]):
                if is_tensor(batch[key][0]) or is_numpy(batch[key][0]):
                    batch[key] = [sample[0] for sample in batch[key]]
            # Else, stack single item
            else:
                batch[key] = batch[key][0]
    return batch

    # def compute_inv_depths(image, model):
    #     """Computes inverse depth maps from single images"""
    #     # Randomly flip and estimate inverse depth maps
    #     flip_lr = random.random() < self.flip_lr_prob if self.training else False
    #     inv_depths = make_list(flip_model(self.depth_net, image, flip_lr))
    #     # If upsampling depth maps
    #     if self.upsample_depth_maps:
    #         inv_depths = interpolate_scales(inv_depths, mode="nearest", align_corners=None)
    #     # Return inverse depth maps
    #     return inv_depths


def compute_poses(image, contexts, model_pose):
    """Compute poses from image and a sequence of context images"""
    pose_vec = model_pose(image, contexts)
    return [Pose.from_vec(pose_vec[:, i], "euler") for i in range(pose_vec.shape[1])]


def training_step(batch, model_depth, model_pose, args):
    """Processes a training batch."""

    model_depth.train()
    model_pose.train()
    if args.cuda:
        batch["rgb"] = batch["rgb"].cuda()
        batch["rgb_original"] = batch["rgb_original"].cuda()
        batch["rgb_context_original"][0] = batch["rgb_context_original"][0].cuda()
        batch["rgb_context_original"][1] = batch["rgb_context_original"][1].cuda()
        # batch["intrinsics"] = batch["intrinsics"].cuda()

    batch = stack_batch(batch)
    inv_depths = model_depth(batch["rgb"])
    save_image(batch["rgb"], "result/train/input.png")

    # Generate pose predictions if available
    pose = None
    if "rgb_context" in batch and model_pose is not None:
        pose = compute_poses(batch["rgb"], batch["rgb_context"], model_pose)

    self_sup_output = self_supervised_loss(
        batch["rgb_original"],
        batch["rgb_context_original"],
        inv_depths,
        pose,
        batch["intrinsics"],
        return_logs=False,
        progress=0.0,
    )
    print(self_sup_output["loss"])
    if True:
        print(inv_depths[0].shape)
        save_image(
            inv_depths[0] / torch.max(inv_depths[0]), "result/train/inv_depths.png"
        )
    return self_sup_output["loss"]


def self_supervised_loss(
    image,
    ref_images,
    inv_depths,
    poses,
    intrinsics,
    return_logs=False,
    progress=0.0,
):
    """
    Calculates the self-supervised photometric loss.

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Original image
    ref_images : list of torch.Tensor [B,3,H,W]
        Reference images from context
    inv_depths : torch.Tensor [B,1,H,W]
        Predicted inverse depth maps from the original image
    poses : list of Pose
        List containing predicted poses between original and context images
    intrinsics : torch.Tensor [B,3,3]
        Camera intrinsics
    return_logs : bool
        True if logs are stored
    progress :
        Training progress percentage

    Returns
    -------
    output : dict
        Dictionary containing a "loss" scalar a "metrics" dictionary
    """
    _photometric_loss = MultiViewPhotometricLoss()
    return _photometric_loss(
        image,
        ref_images,
        inv_depths,
        intrinsics,
        intrinsics,
        poses,
        return_logs=return_logs,
        progress=progress,
    )


if __name__ == "__main__":
    # args = parse_args()
    args = parser.parse_args()
    args.cuda = args.use_cuda and torch.cuda.is_available()

    # seed
    deterministic_training(args.seed)

    # KITTI dataset
    from datasets.kitti_dataset import KITTIDataset

    args.back_context = 1
    args.forward_context = 1
    args.depth_type = ""
    dataset_args = {
        "back_context": args.back_context,
        "forward_context": args.forward_context,
        "data_transform": get_transforms(
            "train", image_shape=(192, 640), jittering=(0.2, 0.2, 0.2, 0.05)
        ),
    }
    path_split = os.path.join(args.datapath, "kitti_tiny.txt")
    requirements = {
        "gt_depth": False,  # No ground-truth depth required
        "gt_pose": False,  # No ground-truth pose required
    }

    dataset_args_i = {
        "depth_type": args.depth_type[i] if requirements["gt_depth"] else None,
        "with_pose": requirements["gt_pose"],
    }

    dataset = KITTIDataset(
        args.datapath,
        path_split,
        **dataset_args,
        **dataset_args_i,
    )

    TrainImgLoader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        worker_init_fn=worker_init_fn,
        sampler=get_datasampler(dataset, "train"),
    )

    # model

    # model = PackNet01()
    model_depth = PackNet01.PackNet01(dropout=None, version="1A")
    model_pose = PoseNet.PoseNet(nb_ref_imgs=2, rotation_mode="euler")

    # optimizer
    optimizer = optim.Adam(
        model_depth.parameters(),
        lr=0.0002,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
    )
    # optimizer_pose = optim.Adam(
    #     model_pose.parameters(),
    #     lr=0.0002,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=0,
    # )
    if args.cuda:
        model_depth = nn.DataParallel(model_depth)
        model_depth.cuda()
        model_pose = nn.DataParallel(model_pose)
        model_pose.cuda()

    iteration = 0
    for epoch in range(0, args.epochs):
        ## training ##
        total_train_loss = 0
        for batch_idx, (batch) in enumerate(TrainImgLoader):
            optimizer.zero_grad()
            loss = training_step(batch, model_depth, model_pose, args)
            loss.backward()
            optimizer.step()
            # output["loss"] = output["loss"].detach()
            # outputs.append(output)

            # Return loss and metrics
            # return {
            #     "loss": self_sup_output["loss"],
            #     **merge_outputs(output, self_sup_output),
            # }
