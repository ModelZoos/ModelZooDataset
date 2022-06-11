import os
from torchvision import datasets, transforms
import torch
from ray.tune.integration.wandb import WandbLogger
from ray.tune.logger import JsonLogger, CSVLogger
from ray import tune
import ray
import json
from pathlib import Path
import sys
sys.path.append('./..')
from model_definitions.def_NN_experiment import NN_tune_trainable


PATH_ROOT = Path(".")


def main():
    # ray init to limit memory and storage
    cpus = 8
    gpus = 0

    cpu_per_trial = 1
    gpu_fraction = ((gpus * 100) // (cpus / cpu_per_trial)) / 100
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_fraction}

    # experiment name
    project = "Corr: STL Random Large"
    experiment_name = "tune_zoo_stl_large_hyperparameter_10_random_seeds"

    # set module parameters
    config = {}
    config["model::type"] = "CNN3"
    config["model::channels_in"] = 3
    config["model::o_dim"] = 10
    config["model::nlin"] = tune.grid_search(
        ["tanh", "relu", "sigmoid", "gelu"])
    config["model::dropout"] = tune.grid_search([0, 0.5])
    config["model::init_type"] = tune.grid_search(
        ["uniform", "normal", "kaiming_uniform", "kaiming_normal"])
    config["model::use_bias"] = False
    config["optim::optimizer"] = tune.grid_search(["adam", "sgd"])
    config["optim::lr"] = tune.grid_search([1e-3, 1e-4])
    config["optim::wd"] = tune.grid_search([1e-2, 1e-3])
    config["optim::momentum"] = 0.9

    # set seeds for reproducibility
    repetitions = 10
    seeds = []
    for _ in range(repetitions):
        seeds.append(tune.randint(0, 1000000))
    config["seed"] = tune.grid_search(seeds)

    # set training parameters
    net_dir = PATH_ROOT.joinpath("zoos/STL-10/large")
    try:
        net_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass
    print(f"Zoo directory: {net_dir.absolute()}")

    # config["training::batchsize"] = tune.grid_search([8, 4, 2])
    config["training::batchsize"] = 10
    config["training::epochs_train"] = 50
    config["training::start_epoch"] = 1
    config["training::output_epoch"] = 1
    config["training::val_epochs"] = 1
    config["training::idx_out"] = 500
    config["training::checkpoint_dir"] = None

    config["cuda"] = True if gpus > 0 and torch.cuda.is_available() else False

    stl_path = PATH_ROOT.joinpath("data/STL")
    print(f"Data directory: {stl_path.absolute()}")
    try:
        # load existing dataset
        dataset = torch.load(str(stl_path.joinpath("dataset_32px.pt")))
        print("using existing dataset")
    except FileNotFoundError:
        # if file not found, generate and save dataset
        # seed for reproducibility
        dataset_seed = 42

        # load raw dataset
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        val_and_trainset_raw = datasets.STL10(
            stl_path, split="train", download=True, transform=transform)
        testset_raw = datasets.STL10(
            stl_path, split="test", download=True, transform=transform)
        trainset_raw, valset_raw = torch.utils.data.random_split(
            val_and_trainset_raw, [4200, 800], generator=torch.Generator().manual_seed(dataset_seed))

        # temp dataloaders
        trainloader_raw = torch.utils.data.DataLoader(
            dataset=trainset_raw, batch_size=len(trainset_raw), shuffle=True
        )
        valloader_raw = torch.utils.data.DataLoader(
            dataset=valset_raw, batch_size=len(valset_raw), shuffle=True
        )
        testloader_raw = torch.utils.data.DataLoader(
            dataset=testset_raw, batch_size=len(testset_raw), shuffle=True
        )
        # one forward pass
        assert trainloader_raw.__len__() == 1, "temp trainloader has more than one batch"
        for train_data, train_labels in trainloader_raw:
            pass
        assert valloader_raw.__len__() == 1, "temp valloader has more than one batch"
        for val_data, val_labels in valloader_raw:
            pass
        assert testloader_raw.__len__() == 1, "temp testloader has more than one batch"
        for test_data, test_labels in testloader_raw:
            pass

        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        valset = torch.utils.data.TensorDataset(val_data, val_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # save dataset and seed in data directory
        dataset = {
            "trainset": trainset,
            "valset": valset,
            "testset": testset,
            "dataset_seed": dataset_seed
        }
        torch.save(dataset, stl_path.joinpath("dataset_32px.pt"))

    # save datasets in zoo directory
    net_dir.joinpath(experiment_name).mkdir(exist_ok=True)
    config["dataset::dump"] = net_dir.joinpath(
        experiment_name, "dataset.pt").absolute()
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
        NN_tune_trainable,
        name=experiment_name,
        stop={"training_iteration": config["training::epochs_train"], },
        checkpoint_score_attr="test_acc",
        checkpoint_freq=config["training::output_epoch"],
        config=config,
        local_dir=net_dir,
        #loggers=DEFAULT_LOGGERS + (WandbLogger,),
        loggers=(JsonLogger,) + (CSVLogger,),  # + (WandbLogger,),
        queue_trials=False,
        reuse_actors=False,
        # resume="ERRORED_ONLY",  # resumes from previous run. if run should be done all over, set resume=False
        # resume="LOCAL",  # resumes from previous run. if run should be done all over, set resume=False
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        resources_per_trial=resources_per_trial,
        verbose=1
    )

    ray.shutdown()
    assert ray.is_initialized() == False


if __name__ == "__main__":
    main()
