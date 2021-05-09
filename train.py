# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse

# from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.models.model_checkpoint import ModelCheckpoint

# from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.utils.config import parse_train_file

# from packnet_sfm.utils.load import set_debug
from packnet_sfm.utils.load import filter_args_create
from packnet_sfm.utils.horovod import hvd_init, rank


import torch

# from tqdm import tqdm
from packnet_sfm.utils.logging import prepare_dataset_prefix


def sample_to_cuda(data, dtype=None):
    if isinstance(data, str):
        return data
    elif isinstance(data, dict):
        return {key: sample_to_cuda(data[key], dtype) for key in data.keys()}
    elif isinstance(data, list):
        return [sample_to_cuda(val, dtype) for val in data]
    else:
        # only convert floats (e.g., to half), otherwise preserve (e.g, ints)
        dtype = dtype if torch.is_floating_point(data) else None
        return data.to("cuda", dtype=dtype)


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description="PackNet-SfM training script")
    parser.add_argument("file", type=str, help="Input file (.ckpt or .yaml)")
    args = parser.parse_args()
    assert args.file.endswith(
        (".ckpt", ".yaml")
    ), "You need to provide a .ckpt of .yaml file"
    return args


def training_step(model, batch, *args):
    """Processes a training batch."""
    batch = stack_batch(batch)
    output = model(batch, progress=0)
    return {"loss": output["loss"], "metrics": output["metrics"]}


if __name__ == "__main__":
    args = parse_args()

    # # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(args.file)

    from packnet_sfm.utils.load import (
        load_class,
        load_class_args_create,
        load_network,
        filter_args,
    )
    from packnet_sfm.models.model_wrapper import (
        setup_model,
        setup_dataset,
        set_random_seed,
        setup_dataloader,
    )
    import torch.optim as optim
    from torchvision.utils import save_image

    # from packnet_sfm.trainers.base_trainer import sample_to_cuda
    from packnet_sfm.models.model_utils import stack_batch

    set_random_seed(config.arch.seed)

    model = setup_model(config.model)

    augmentation = config.datasets.augmentation
    # Setup train dataset (requirements are given by the model itself)
    train_dataset = setup_dataset(
        config.datasets.train,
        "train",
        model.train_requirements,
        **augmentation,
    )

    train_dataloader = setup_dataloader(train_dataset, config.datasets.train, "train")[
        0
    ]

    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))

    model.cuda()

    outputs = []

    for epoch in range(0, 100):
        ## training ##
        total_train_loss = 0
        for batch_idx, batch_data in enumerate(train_dataloader):
            optimizer.zero_grad()
            # Send samples to GPU and take a training step
            batch = sample_to_cuda(batch_data)
            output = training_step(model, batch, batch_idx)
            print(output)
            output["loss"].backward()
            optimizer.step()
            # Append output to list of outputs
            output["loss"] = output["loss"].detach()
            outputs.append(output)

        # save_image("result/train/image.png", batch_data["rgb"])
