import os

import sys
from pathlib import Path

import json

import ray
from ray import tune

# from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.logger import JsonLogger, CSVLogger
from ray.tune.integration.wandb import WandbLogger

import torch
from torchvision import datasets, transforms

from model_definitions.def_NN_experiment import NN_tune_trainable


PATH_ROOT = Path(".")


def main():
    # ray init to limit memory and storage
    cpus = 24
    gpus = 4

    cpu_per_trial = 6
    gpu_fraction = ((gpus * 100) // (cpus / cpu_per_trial)) / 100
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_fraction}

    # experiment name
    project = "CIFAR10 Resnet18 kaiming-uniform"
    experiment_name = "tune_zoo_cifar10_resnet18_kaiming_uniform"

    # set module parameters
    config = {}
    config["model::type"] = "Resnet18"
    config["model::channels_in"] = 3
    config["model::o_dim"] = 10
    config["model::nlin"] = "relu"
    config["model::dropout"] = 0.0
    config["model::init_type"] = "kaiming_uniform"
    config["model::use_bias"] = False
    config["optim::optimizer"] = "sgd"
    config["optim::lr"] = 0.05
    config["optim::wd"] = 5e-4
    config["optim::momentum"] = 0.9
    config["optim::scheduler"] = "OneCycleLR"
    config["training::loss"] = "nll"
    # config["training::loss"] = "crossentropy"
    config["training::dataloader"] = "normal"
    config["testloader::workers"] = 3

    # set seeds for reproducibility
    seeds = list(range(1, 1001))
    config["seed"] = tune.grid_search(seeds)

    # set training parameters
    net_dir = PATH_ROOT.joinpath("zoos/CIFAR10/resnet19/kaiming_uniform")
    try:
        net_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    print(f"Zoo directory: {net_dir.absolute()}")

    # config["training::batchsize"] = tune.grid_search([8, 4, 2])
    config["training::batchsize"] = 256
    config["training::epochs_train"] = 50
    config["training::start_epoch"] = 1
    config["training::output_epoch"] = 1
    config["training::val_epochs"] = 1
    config["training::idx_out"] = 500
    config["training::checkpoint_dir"] = None

    config["cuda"] = True if gpus > 0 and torch.cuda.is_available() else False

    cifar_path = PATH_ROOT.joinpath("data/CIFAR10")
    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
            ),
        ]
    )

    val_and_trainset = datasets.CIFAR10(
        root="./data/CIFAR10",
        train=True,
        transform=train_transforms,
        download=True,
    )

    testset = datasets.CIFAR10(
        root="./data/CIFAR10",
        train=False,
        transform=test_transforms,
        download=True,
    )

    dataset_seed = 42
    trainset, valset = torch.utils.data.random_split(
        val_and_trainset,
        [40000, 10000],
        generator=torch.Generator().manual_seed(dataset_seed),
    )

    # save dataset and seed in data directory
    dataset = {
        "trainset": trainset,
        "valset": valset,
        "testset": testset,
        "dataset_seed": dataset_seed,
    }
    torch.save(dataset, cifar_path.joinpath("dataset_28px.pt"))

    # save datasets in zoo directory
    net_dir.joinpath(experiment_name).mkdir(exist_ok=True)
    config["dataset::dump"] = net_dir.joinpath(experiment_name, "dataset.pt").absolute()
    torch.save(dataset, config["dataset::dump"])

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
    )

    # save config as json file
    with open((net_dir.joinpath(experiment_name, "config.json")), "w") as f:
        json.dump(config, f, default=str)

    # check if readme.md exists
    readme_file = net_dir.joinpath(experiment_name, "readme.md")
    if readme_file.is_file():
        pass
    # if not, make empty readme
    else:
        with open(readme_file, "w") as f:
            pass

    assert ray.is_initialized() == True

    # run tune trainable experiment
    analysis = tune.run(
        NN_tune_trainable,
        name=experiment_name,
        stop={
            "training_iteration": config["training::epochs_train"],
        },
        checkpoint_score_attr="test_acc",
        checkpoint_freq=config["training::output_epoch"],
        config=config,
        local_dir=net_dir,
        reuse_actors=False,
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        resources_per_trial=resources_per_trial,
        verbose=3,
    )

    ray.shutdown()
    assert ray.is_initialized() == False


if __name__ == "__main__":
    main()
