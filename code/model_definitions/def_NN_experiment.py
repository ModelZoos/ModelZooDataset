from ray.tune import Trainable

import torch
import sys

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

from .def_FastTensorDataLoader import FastTensorDataLoader

from .def_net import NNmodule

from checkpoints_to_datasets.dataset_auxiliaries import (
    vectorize_checkpoint,
    vector_to_checkpoint,
)

import json

"""
define Tune Trainable
##############################################################################
"""


class NN_tune_trainable(Trainable):
    def setup(self, config):
        self.config = config
        self.seed = config["seed"]
        self.cuda = config["cuda"]

        if config.get("dataset::dump", None) is not None:
            # print(f"loading dataset from {config['dataset::dump']}")
            # load dataset from file
            print(f"loading data from {config['dataset::dump']}")
            dataset = torch.load(config["dataset::dump"])
            self.trainset = dataset["trainset"]
            self.testset = dataset["testset"]
            self.valset = dataset.get("valset", None)
        else:

            data_path = config["training::data_path"]
            fname = f"{data_path}/train_data.pt"
            train_data = torch.load(fname)
            train_data = torch.stack(train_data)
            fname = f"{data_path}/train_labels.pt"
            train_labels = torch.load(fname)
            train_labels = torch.tensor(train_labels)
            # test
            fname = f"{data_path}/test_data.pt"
            test_data = torch.load(fname)
            test_data = torch.stack(test_data)
            fname = f"{data_path}/test_labels.pt"
            test_labels = torch.load(fname)
            test_labels = torch.tensor(test_labels)
            #
            # Flatten images for MLP
            if config["model::type"] == "MLP":
                train_data = train_data.flatten(start_dim=1)
                test_data = test_data.flatten(start_dim=1)
            # send data to device
            if config["cuda"]:
                train_data, train_labels = train_data.cuda(), train_labels.cuda()
                test_data, test_labels = test_data.cuda(), test_labels.cuda()
            else:
                print(
                    "### WARNING ### : using tensor dataloader without cuda. probably slow"
                )
            # create new tensor datasets
            self.trainset = torch.utils.data.TensorDataset(train_data, train_labels)
            self.testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # instanciate Tensordatasets
        self.dl_type = config.get("training::dataloader", "tensor")
        if self.dl_type == "tensor":
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=config["training::batchsize"],
                shuffle=True,
                num_workers=self.config.get("testloader::workers", 2),
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )
        else:
            self.trainloader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
                num_workers=self.config.get("testloader::workers", 2),
            )
            self.testloader = torch.utils.data.DataLoader(
                dataset=self.testset, batch_size=self.config["training::batchsize"], shuffle=False
            )
            if self.valset is not None:
                self.valloader = torch.utils.data.DataLoader(
                    dataset=self.valset, batch_size=self.config["training::batchsize"], shuffle=False
                )
        # TODO: allow different dataloaders for CIFAR

        config["scheduler::steps_per_epoch"] = len(self.trainloader)

        # init model
        self.NN = NNmodule(
            config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
        )

        # run first test epoch and log results
        self._iteration = -1

    def step(self):
        # here, all manual writers are disabled. tune takes care of that
        # run one training epoch
        if self._iteration < 0:
            print("test first validation mode")
            loss_train, acc_train = self.NN.test_epoch(self.trainloader, 0)

        else:
            loss_train, acc_train = self.NN.train_epoch(self.trainloader, 0, idx_out=10)
        # run one test epoch
        loss_test, acc_test = self.NN.test_epoch(self.testloader, 0)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "test_loss": loss_test,
            "test_acc": acc_test,
        }
        if self.valset is not None:
            loss_val, acc_val = self.NN.test_epoch(self.valloader, 0)
            result_dict["val_loss"] = loss_val
            result_dict["val_acc"] = acc_val

        return result_dict

    def save_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        torch.save(self.NN.model.state_dict(), path)
        # save optimizer
        path = Path(tmp_checkpoint_dir).joinpath("optimizer")
        torch.save(self.NN.optimizer.state_dict(), path)

        # tune apparently expects to return the directory
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        checkpoint = torch.load(path)
        self.NN.model.load_state_dict(checkpoint)
        # load optimizer
        try:
            path = Path(tmp_checkpoint_dir).joinpath("optimizer")
            opt_dict = torch.load(path)
            self.NN.optimizer.load_state_dict(opt_dict)
        except:
            print(f"Could not load optimizer state_dict. (not found at path {path})")

    def reset_config(self, new_config):
        success = False
        try:
            print(
                "### warning: reuse actors / reset_config only if the dataset remains exactly the same. \n ### only dataloader and model are reconfiugred"
            )
            self.config = new_config
            self.seed = self.config["seed"]
            self.cuda = self.config["cuda"]

            # init model
            self.NN = NNmodule(
                config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
            )

            # instanciate Tensordatasets
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )

            # drop inital checkpoint
            self.save()

            # run first test epoch and log results
            self._iteration = -1

            # if we got to this point:
            success = True

        except Exception as e:
            print(e)

        return success
