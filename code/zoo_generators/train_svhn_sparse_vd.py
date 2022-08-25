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

from model_definitions.def_NN_experiment_distillation import NN_tune_trainable_from_checkpoint


PATH_ROOT = Path(".")


def main():
    # ray init to limit memory and storage
    cpus = 20
    gpus=0
    cpus_per_trial = 1
    resources_per_trial = {"cpu": cpus_per_trial, "gpu":0}

    # experiment name
    project = "cnn_small_svhn_ard"
    experiment_name = "cnn_small_svhn_ard"

    source_zoo_path = Path('./zoos/SVHN/tune_zoo_svhn_uniform/')
    source_path_list = [pdx for pdx in source_zoo_path.iterdir() if pdx.is_dir()]
    config_path = source_path_list[0].joinpath('params.json')
    config = json.load(config_path.open('r'))
    config["model::type"] = "CNN_ARD"

    # 
    config["training::init_checkpoint_path"] = tune.grid_search(source_path_list)
    config["training::sample_epoch"] = 50
    # set module parameters

    # set training parameters
    net_dir = PATH_ROOT.joinpath("zoos/SVHN/ARD")
    try:
        net_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    print(f"Zoo directory: {net_dir.absolute()}")

    # config["training::batchsize"] = tune.grid_search([8, 4, 2])
    config["training::epochs_train"] = 25

    config["cuda"] = True if gpus > 0 and torch.cuda.is_available() else False


    dataset = torch.load(source_zoo_path.joinpath('dataset.pt'))

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

    # generate empty readme.md file   ?? maybe create and copy template
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
        NN_tune_trainable_from_checkpoint,
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
