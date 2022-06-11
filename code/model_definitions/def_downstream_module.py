import tqdm
from .def_net import NNmodule
import numpy as np
from .def_FastTensorDataLoader import FastTensorDataLoader
import torch
import torch.nn as nn
import sys

# for classification


class DownstreamTaskLearner:
    """
    This class implements a module to evalute learned representations on downstream tasks.
    It expects sample-wise embeddings and target values.
    By the variable type it it choses linear classification or regresion.
    """

    def __init__(self):
        """
        nothing going on here, yet.
        """
        return

    ##### main func to run ds tasks ##################################################################
    def eval_dstasks(
        self,
        model,
        trainset,
        testset,
        valset=None,
        batch_size=100,
        polar_coordinates=False,
    ):
        self.polar_coordinates = polar_coordinates
        # initialize return dictionary
        performance = {}
        # figure out device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        # prepare embeddings
        print(f"Prepare embeddings")
        w_train = trainset.__get_weights__()
        z_train = self.map_embeddings(
            weights=w_train, model=model, batch_size=batch_size
        )
        w_test = testset.__get_weights__()
        z_test = self.map_embeddings(
            weights=w_test, model=model, batch_size=batch_size)
        if valset is not None:
            w_val = valset.__get_weights__()
            z_val = self.map_embeddings(
                weights=w_val,
                model=model,
                batch_size=batch_size,
            )
        else:
            z_val = None
        # iterate over properties
        print(f"Compute downstream task performance")
        for key in tqdm.tqdm(trainset.properties.keys()):
            # figure out task
            # print(f"identify task")
            task_dx = self.identify_task(trainset.properties[key])
            # if task: regression:
            if task_dx == "regression":
                # print(f"start solving regression problem")
                r2_train, r2_test, r2_val = self.compute_closed_form_solution(
                    z_train=z_train,
                    prop_train=trainset.properties[key],
                    z_test=z_test,
                    prop_test=testset.properties[key],
                    z_val=z_val if z_val is not None else None,
                    prop_val=valset.properties[key] if valset is not None else None,
                )
                performance[f"{key}_train"] = r2_train
                performance[f"{key}_test"] = r2_test
                if r2_val is not None:
                    performance[f"{key}_val"] = r2_val

            # elif task classification
            elif task_dx == "classification":
                # print(f"start solving classification problem")
                # use classification_single function to run class task
                acc_train, acc_test, acc_val = self.classify_single(
                    z_train=z_train,
                    prop_train=trainset.properties[key],
                    z_test=z_test,
                    prop_test=testset.properties[key],
                    z_val=z_val if z_val is not None else None,
                    prop_val=valset.properties[key] if valset is not None else None,
                )
                performance[f"{key}_train"] = acc_train
                performance[f"{key}_test"] = acc_test
                if acc_val is not None:
                    performance[f"{key}_val"] = acc_val
            else:
                # we don't really know what to do with the data, so we don't do anything
                pass

        return performance

    ##### helper: map weights to embeddings ##################################################################
    def map_embeddings(
        self,
        weights: torch.Tensor,
        model: nn.Module,
        batch_size: int = 100,
    ) -> torch.Tensor:
        """
        computes the
        """
        # init return value
        z = []
        # prepare datasets
        if not isinstance(weights, torch.Tensor):
            weights = torch.Tensor(weights)
        weightset = torch.utils.data.TensorDataset(weights)
        weightloader = FastTensorDataLoader(weightset, batch_size=batch_size)
        # iterate over batches
        model.to(self.device)
        # put model in eval mode
        model.eval()
        with torch.no_grad():
            # for wdx in tqdm.tqdm(weightloader):
            for wdx in weightloader:
                # batches are tuples...
                wdx = wdx[0]
                # end batch to self.device
                wdx = wdx.to(self.device)
                zdx = model.forward_encoder(wdx)
                z.append(zdx)
            # cat all batches in one tensor
        z = torch.cat(z, dim=0)
        # send z back to cpu, just in case
        z.to("cpu")
        # print(z.shape)
        if self.polar_coordinates:
            z = self.map_cartesian_to_polar(z)
        return z

    ##### helper: identify task for property ##################################################################
    def identify_task(self, property: list) -> str:
        """
        identifies the task for a given property: if continuous -> regression, categorical -> classifcation
        """
        # take first entry as sample from property list
        sample = property[0]
        # figure out task
        if isinstance(sample, float) or isinstance(sample, int):
            # target is number -> inferred task is regression
            task = "regression"
        elif isinstance(sample, str):
            task = "classification"
        else:
            task = "unidentified"

        return task

    ##### helper: run classification for one property ##################################################################
    def classify_single(
        self,
        z_train: torch.Tensor,
        prop_train: list,
        z_test: torch.Tensor,
        prop_test: list,
        z_val: torch.Tensor = None,
        prop_val: list = None,
    ):
        # get list of classes and class labels
        # cast to float and push to cpu. back to gpu will be done at dataloader time
        z_train = z_train.float().to(torch.device("cpu"))
        z_test = z_test.float().to(torch.device("cpu"))
        if z_val is not None:
            z_val = z_val.float().to(torch.device("cpu"))
        # train set classes
        classes = list(np.unique(prop_train))
        no_classes = len(classes)
        # assert all classes are in trainset
        classes_test = list(np.unique(prop_test))
        assert set(classes_test).issubset(
            set(classes)
        ), "test set contains classes which are not in train set"
        if prop_val is not None:
            classes_val = list(np.unique(prop_val))
            assert set(classes_val).issubset(
                set(classes)
            ), "val set contains classes which are not in train set"

        # one hot encoding
        labels_train = torch.tensor(
            [float(classes.index(vdx)) for idx, vdx in enumerate(prop_train)]
        ).long()
        labels_test = torch.tensor(
            [float(classes.index(vdx)) for idx, vdx in enumerate(prop_test)]
        ).long()
        if prop_val is not None:
            labels_val = torch.tensor(
                [float(classes.index(vdx)) for idx, vdx in enumerate(prop_val)]
            ).long()

        # dataset
        # train
        trainset = torch.utils.data.TensorDataset(z_train, labels_train)
        trainloader = FastTensorDataLoader(
            trainset, batch_size=10, shuffle=True)
        # test
        testset = torch.utils.data.TensorDataset(z_test, labels_test)
        testloader = FastTensorDataLoader(testset, batch_size=10, shuffle=True)
        # val
        if prop_val is not None:
            valset = torch.utils.data.TensorDataset(z_val, labels_val)
            valloader = FastTensorDataLoader(
                valset, batch_size=10, shuffle=True)

        # create model
        # configure model
        config = {}
        config["model::type"] = "MLP"
        config["model::h_dim"] = []  # no hidden layers
        config["model::i_dim"] = z_train.shape[1]
        config["model::o_dim"] = no_classes
        config["model::init_type"] = "kaiming_normal"
        config["model::nlin"] = "relu"
        config["model::dropout"] = 0
        config["model::use_bias"] = True
        config["optim::optimizer"] = "adam"
        config["optim::lr"] = 1e-4
        config["optim::wd"] = 1e-6
        config["training::task"] = "classification"
        config["training::batchsize"] = 4500
        config["training::start_epoch"] = 1
        config["training::epochs_train"] = 50
        config["training::val_epochs"] = 10
        config["training::output_epoch"] = 10
        config["training::idx_out"] = 1500
        config["training::checkpoint_dir"] = None
        config["training::tensorboard_dir"] = None
        config["seed"] = 42
        config["training::trainloader"] = trainloader
        config["training::testloader"] = testloader

        # instanciate model
        cuda = True if torch.cuda.is_available() else False
        MLP_regrs = NNmodule(config, cuda=cuda, verbosity=0)
        # start training loop
        MLP_regrs.train_loop(config)
        # get perforamnce
        _, acc_train = MLP_regrs.test(trainloader, epoch=-1)
        _, acc_test = MLP_regrs.test(testloader, epoch=-1)
        if prop_val is not None:
            _, acc_val = MLP_regrs.test(valloader, epoch=-1)
        else:
            acc_val = None

        return acc_train, acc_test, acc_val

    def compute_closed_form_solution(
        self,
        z_train: torch.Tensor,
        prop_train: list,
        z_test: torch.Tensor,
        prop_test: list,
        z_val: torch.Tensor = None,
        prop_val: list = None,
        verbosity=0,
        return_reg=False,
    ):
        # prepare data
        # cast to tensor
        prop_train = torch.Tensor(prop_train)
        prop_test = torch.Tensor(prop_test)
        if prop_val is not None:
            prop_val = torch.Tensor(prop_val)
        # find nan values
        # train
        idx_no_nan_train = [
            idx for idx, pdx in enumerate(prop_train.isnan()) if not pdx == True
        ]
        X_train = z_train[idx_no_nan_train]
        y_train = prop_train[idx_no_nan_train]
        if len(y_train.shape) == 2:
            y_train = y_train.squeeze()
        # test
        idx_no_nan_test = [
            idx for idx, pdx in enumerate(prop_test.isnan()) if not pdx == True
        ]
        X_test = z_test[idx_no_nan_test]
        y_test = prop_test[idx_no_nan_test]
        if len(y_test.shape) == 2:
            y_test = y_test.squeeze()
        if prop_val is not None:
            idx_no_nan_val = [
                idx for idx, pdx in enumerate(prop_val.isnan()) if not pdx == True
            ]
            X_val = z_val[idx_no_nan_val]
            y_val = prop_val[idx_no_nan_val]
            if len(y_val.shape) == 2:
                y_val = y_val.squeeze()
        else:
            X_val = None
            y_val = None

        # baseline regularization very small
        reg_list = [
            1e3,
            3e2,
            1e2,
            3e1,
            1e1,
            3e0,
            1e0,
            3e-1,
            1e-1,
            3e-2,
            1e-2,
            3e-3,
            1e-3,
            3e-4,
            1e-4,
            3e-5,
            1e-5,
        ]
        r2_list_train = []
        r2_list_test = []
        r2_list_val = []

        # for reg in tqdm.tqdm(reg_list):
        for reg in reg_list:
            r2_train_curr, r2_test_curr, r2_val_curr = self.solve_linear_system(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X_val=X_val if X_val is not None else None,
                y_val=y_val if y_val is not None else None,
                regularization=reg,
            )
            r2_list_train.append(r2_train_curr)
            r2_list_test.append(r2_test_curr)
            r2_list_val.append(r2_val_curr)
        if prop_val is not None:
            # choose best validation performance
            r2_val_best = max(r2_list_val)
            index_best = r2_list_val.index(r2_val_best)
        else:
            # fallback to best train performance
            r2_train_best = max(r2_list_train)
            index_best = r2_list_train.index(r2_train_best)
        # retrieve best r2s
        r2_train_best = r2_list_train[index_best]
        r2_test_best = r2_list_test[index_best]
        if prop_val is not None:
            r2_val_best = r2_list_val[index_best]
        else:
            r2_val_best = None
        reg_best = reg_list[index_best]
        if return_reg:
            return reg_best, r2_train_best, r2_test_best, r2_val_best
        if verbosity > 0:
            print(f"best regularization : {reg_best}")
        return r2_train_best, r2_test_best, r2_val_best

    def solve_linear_system(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        X_val: torch.Tensor = None,
        y_val: torch.Tensor = None,
        regularization: float = 0.3,
        mode: str = "iterative",
        return_predictions=False,
    ):
        """
        solves the linear system of the type:
        X b = y
        inversion: solved by b = (X^T X)^(-1) X^T y
        iterative: use torch.linalg.solve((X^T X)b = X^T y)
        """
        # send to device
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        # append row of ones -> bias / offset
        X_train = torch.cat(
            [X_train, torch.ones(X_train.shape[0]).unsqueeze(
                dim=1).to(self.device)],
            dim=1,
        )
        X_test = torch.cat(
            [X_test, torch.ones(X_test.shape[0]).unsqueeze(
                dim=1).to(self.device)],
            dim=1,
        )
        if X_val is not None:
            X_val = torch.cat(
                [X_val, torch.ones(X_val.shape[0]).unsqueeze(
                    dim=1).to(self.device)],
                dim=1,
            )
        # cast tensors to double
        X_train = X_train.double()
        y_train = y_train.double()
        X_test = X_test.double()
        y_test = y_test.double()
        if X_val is not None:
            X_val = X_val.double()
            y_val = y_val.double()

        if mode == "inversion":
            # transpose
            X_t_train = torch.einsum("ij->ji", X_train)
            # compute X^T X
            X_t_X_train = torch.einsum("ij,jk->ik", X_t_train, X_train)
            # attempt to compute inverse solution
            while True:
                thiko_train = regularization * torch.eye(X_t_X_train.shape[0])
                try:
                    X_t_X_inv_train = torch.inverse(X_t_X_train + thiko_train)
                except:
                    # catch if inversion is unstable...
                    regularization *= 10.0 / 3.0
                    continue
                break

            # compute parameters
            X_t_y_train = torch.einsum("ij,j->i", X_t_train, y_train)
            b = torch.einsum("ij,j->i", X_t_X_inv_train, X_t_y_train)
        else:  # mode=='iterative'
            # torch.linalg.solve expects square matrices
            # transpose
            X_t_train = torch.einsum("ij->ji", X_train)
            # compute X^T X
            X_t_X_train = torch.einsum("ij,jk->ik", X_t_train, X_train)
            # compute X^t y
            X_t_y_train = torch.einsum("ij,j->i", X_t_train, y_train)
            # add regularization
            thiko_train = regularization * torch.eye(X_t_X_train.shape[0])
            X_t_X_train += thiko_train.to(self.device)
            # solve system
            b = torch.linalg.solve(X_t_X_train, X_t_y_train)

        # compute predictions
        y_train_pred = torch.einsum("ij,j->i", X_train, b)
        y_test_pred = torch.einsum("ij,j->i", X_test, b)
        if X_val is not None:
            y_val_pred = torch.einsum("ij,j->i", X_val, b)
        # compute r2
        r2_train = self.compute_r2(y=y_train_pred, t=y_train)
        r2_test = self.compute_r2(y=y_test_pred, t=y_test)
        if X_val is not None:
            r2_val = self.compute_r2(y=y_val_pred, t=y_val)
        else:
            y_val_pred = None
            r2_val = None
        if return_predictions:
            return y_train_pred, y_test_pred, y_val_pred, r2_train, r2_test, r2_val
        else:
            return r2_train, r2_test, r2_val

    def compute_r2(self, y: torch.Tensor, t: torch.Tensor) -> float:
        # compute error
        e = t - y
        e_mean = torch.einsum("i,i->", e, e) / e.shape[0]
        # compute mean of targets
        t_mean = torch.zeros(t.shape).to(self.device).add(t.mean(dim=0))
        # compute error with mean of targets
        e_t_mean = t - t_mean
        e_var = torch.einsum("i,i->", e_t_mean, e_t_mean) / e_t_mean.shape[0]
        # r2
        r2 = 1 - e_mean / e_var
        return r2.item()

    def eval_ood_dstask(
        self,
        model,
        trainset,
        testset_dict,
        valset=None,
        batch_size=1000,
        tasks=["regression", "classification"],
        force_reg=False,
    ):
        # initialize return dictionary
        performance = {}
        # figure out device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available(
            ) else torch.device("cpu")
        )
        # prepare embeddings
        print(f"Prepare embeddings")
        w_train = trainset.__get_weights__()
        z_train = self.map_embeddings(
            weights=w_train, model=model, batch_size=batch_size
        )
        # init new dict for test / ood data
        test_dict = {}
        for datasetkey in testset_dict.keys():
            # add weights and embeddings
            w_test = testset_dict[datasetkey].__get_weights__()
            z_test = self.map_embeddings(
                weights=w_test, model=model, batch_size=batch_size
            )
            test_dict[datasetkey] = {
                # "w_test": w_test, # weights are unnecessary and large...
                "z_test": z_test,
            }
            # copy property data over
            for propkey in testset_dict[datasetkey].properties.keys():
                test_dict[datasetkey][propkey] = testset_dict[datasetkey].properties[
                    propkey
                ]

        if valset is not None:
            w_val = valset.__get_weights__()
            z_val = self.map_embeddings(
                weights=w_val,
                model=model,
                batch_size=batch_size,
            )
        else:
            z_val = None
        # iterate over properties
        print(f"Compute downstream task performance")
        for propkey in tqdm.tqdm(trainset.properties.keys()):
            # figure out task
            # print(f"identify task")
            task_dx = self.identify_task(trainset.properties[propkey])
            # tests if task should be performed
            if task_dx not in tasks:
                continue
            # if task: regression:
            if task_dx == "regression":
                test_dict = self.compute_closed_form_solution_ood(
                    z_train=z_train,
                    prop_train=trainset.properties[propkey],
                    testset_dict=test_dict,
                    test_prop_key=propkey,
                    z_val=z_val if z_val is not None else None,
                    prop_val=valset.properties[propkey] if valset is not None else None,
                    force_reg=force_reg,
                )

            # elif task classification
            elif task_dx == "classification":
                # print(f"start solving classification problem")
                # use classification_single function to run class task
                test_dict = self.classify_single_ood(
                    z_train=z_train,
                    prop_train=trainset.properties[propkey],
                    testset_dict=test_dict,
                    test_prop_key=propkey,
                    z_val=z_val if z_val is not None else None,
                    prop_val=valset.properties[propkey] if valset is not None else None,
                )
            else:
                # we don't really know what to do with the data, so we don't do anything
                pass

        return test_dict

    def compute_closed_form_solution_ood(
        self,
        z_train: torch.Tensor,
        prop_train: list,
        testset_dict: dict,
        test_prop_key: str,
        z_val: torch.Tensor = None,
        prop_val: list = None,
        verbosity=0,
        force_reg=False,
    ):
        # prepare data
        # cast to tensor
        prop_train = torch.Tensor(prop_train)
        if prop_val is not None:
            prop_val = torch.Tensor(prop_val)
        # find nan values
        # train
        idx_no_nan_train = [
            idx for idx, pdx in enumerate(prop_train.isnan()) if not pdx == True
        ]
        X_train = z_train[idx_no_nan_train]
        y_train = prop_train[idx_no_nan_train]
        if len(y_train.shape) == 2:
            y_train = y_train.squeeze()
        if prop_val is not None:
            idx_no_nan_val = [
                idx for idx, pdx in enumerate(prop_val.isnan()) if not pdx == True
            ]
            X_val = z_val[idx_no_nan_val]
            y_val = prop_val[idx_no_nan_val]
            if len(y_val.shape) == 2:
                y_val = y_val.squeeze()
        else:
            X_val = None
            y_val = None

        if not force_reg:
            # baseline regularization very small
            reg_list = [
                1e3,
                3e2,
                1e2,
                3e1,
                1e1,
                3e0,
                1e0,
                3e-1,
                1e-1,
                3e-2,
                1e-2,
                3e-3,
                1e-3,
                3e-4,
                1e-4,
                3e-5,
                1e-5,
            ]
            r2_list_train = []
            # r2_list_test = []
            r2_list_val = []
            # for reg in tqdm.tqdm(reg_list):
            for reg in reg_list:
                r2_train_curr, _, r2_val_curr = self.solve_linear_system(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_val,
                    y_test=y_val,
                    X_val=X_val if X_val is not None else None,
                    y_val=y_val if y_val is not None else None,
                    regularization=reg,
                )
                r2_list_train.append(r2_train_curr)
                # r2_list_test.append(r2_test_curr)
                r2_list_val.append(r2_val_curr)
            if prop_val is not None:
                # choose best validation performance
                r2_val_best = max(r2_list_val)
                index_best = r2_list_val.index(r2_val_best)
            else:
                # fallback to best train performance
                r2_train_best = max(r2_list_train)
                index_best = r2_list_train.index(r2_train_best)
            reg_best = reg_list[index_best]
        else:
            reg_best = 1e-5
        ####
        # with best regularization, iterate over test sets, compute predictions and calculate r2,kendal's tau
        for key in testset_dict.keys():
            # get pair of embeddings,targets
            z_test_curr = testset_dict[key]["z_test"]
            prop_curr = torch.Tensor(testset_dict[key][test_prop_key])
            idx_no_nan_test = [
                idx for idx, pdx in enumerate(prop_curr.isnan()) if not pdx == True
            ]
            X_test = z_test_curr[idx_no_nan_test]
            y_test = prop_curr[idx_no_nan_test]
            if len(y_test.shape) == 2:
                y_test = y_test.squeeze()
            # solve linear system with train,val data ID, curr data as test data
            (
                _,  # y_train_pred,
                y_test_pred,
                _,  # y_val_pred,
                _,  # r2_train,
                r2_test,
                _,  # r2_val,
            ) = self.solve_linear_system(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                X_val=X_val if X_val is not None else None,
                y_val=y_val if y_val is not None else None,
                regularization=reg_best,
                return_predictions=True,
            )
            # compute kendall's tau
            from scipy.stats import kendalltau

            tau_test = kendalltau(y_test.cpu().numpy(),
                                  y_test_pred.cpu().numpy())
            testset_dict[key][f"{test_prop_key}_prediction"] = y_test_pred
            testset_dict[key][f"{test_prop_key}_rsq"] = r2_test
            testset_dict[key][f"{test_prop_key}_ktau"] = tau_test

        return testset_dict

    ##### helper: run classification for one property ##################################################################
    def classify_single_ood(
        self,
        z_train: torch.Tensor,
        prop_train: list,
        testset_dict: dict,
        test_prop_key: str,
        z_val: torch.Tensor = None,
        prop_val: list = None,
    ):
        # get list of classes and class labels
        # cast to float and push to cpu. back to gpu will be done at dataloader time
        z_train = z_train.float().to(torch.device("cpu"))
        if z_val is not None:
            z_val = z_val.float().to(torch.device("cpu"))
        # train set classes
        classes = list(np.unique(prop_train))
        no_classes = len(classes)
        # assert all classes are in trainset
        if prop_val is not None:
            classes_val = list(np.unique(prop_val))
            assert set(classes_val).issubset(
                set(classes)
            ), "val set contains classes which are not in train set"

        # one hot encoding
        labels_train = torch.tensor(
            [float(classes.index(vdx)) for idx, vdx in enumerate(prop_train)]
        ).long()
        if prop_val is not None:
            labels_val = torch.tensor(
                [float(classes.index(vdx)) for idx, vdx in enumerate(prop_val)]
            ).long()

        # dataset
        # train
        trainset = torch.utils.data.TensorDataset(z_train, labels_train)
        trainloader = FastTensorDataLoader(
            trainset, batch_size=10, shuffle=True)
        # val
        if prop_val is not None:
            valset = torch.utils.data.TensorDataset(z_val, labels_val)
            valloader = FastTensorDataLoader(
                valset, batch_size=10, shuffle=True)

        # create model
        # configure model
        config = {}
        config["model::type"] = "MLP"
        config["model::h_dim"] = []  # no hidden layers
        config["model::i_dim"] = z_train.shape[1]
        config["model::o_dim"] = no_classes
        config["model::init_type"] = "kaiming_normal"
        config["model::nlin"] = "relu"
        config["model::dropout"] = 0
        config["model::use_bias"] = True
        config["optim::optimizer"] = "adam"
        config["optim::lr"] = 1e-4
        config["optim::wd"] = 1e-6
        config["training::task"] = "classification"
        config["training::batchsize"] = 4500
        config["training::start_epoch"] = 1
        config["training::epochs_train"] = 50
        config["training::val_epochs"] = 10
        config["training::output_epoch"] = 10
        config["training::idx_out"] = 1500
        config["training::checkpoint_dir"] = None
        config["training::tensorboard_dir"] = None
        config["seed"] = 42
        config["training::trainloader"] = trainloader
        config["training::testloader"] = valloader

        # instanciate model
        cuda = True if torch.cuda.is_available() else False
        MLP_regrs = NNmodule(config, cuda=cuda, verbosity=0)
        # start training loop
        MLP_regrs.train_loop(config)
        # get perforamnce
        _, acc_train = MLP_regrs.test(trainloader, epoch=-1)
        if prop_val is not None:
            _, acc_val = MLP_regrs.test(valloader, epoch=-1)
        else:
            acc_val = None
        # iterate over testsets
        for key in testset_dict.keys():
            # test
            z_test = testset_dict[key]["z_test"]
            z_test = z_test.float().to(torch.device("cpu"))
            prop_test = testset_dict[key][test_prop_key]
            classes_test = list(np.unique(prop_test))
            assert set(classes_test).issubset(
                set(classes)
            ), "test set contains classes which are not in train set"
            labels_test = torch.tensor(
                [float(classes.index(vdx))
                 for idx, vdx in enumerate(prop_test)]
            ).long()
            testset = torch.utils.data.TensorDataset(z_test, labels_test)
            testloader = FastTensorDataLoader(
                testset, batch_size=10, shuffle=True)
            _, acc_test = MLP_regrs.test(testloader, epoch=-1)
            testset_dict[key][f"{test_prop_key}_accuracy"] = acc_test

        return testset_dict

    ######################################################################################################################################################
    ### transformation cartesian <-> polar ###############################################################################################################
    ######################################################################################################################################################
    # https://stats.stackexchange.com/questions/331253/draw-n-dimensional-uniform-sample-from-a-unit-n-1-sphere-defined-by-n-1-dime
    # TODO: this transformation is invertible, but not geometrically correct. fix it if there's time
    def map_cartesian_to_polar(self, z):
        """
        translates bxd sized tensor interpreted as cartesian coordinates to b x (d+1) polar coordinates.
        The first entry per sample is the radius, the other are angles from the individual axis.
        Technically, the polar coordinates are over-defined, but that way all the signs are recoverable.
        """
        from einops import repeat

        # compute radius
        # r = sqrt(sum_i(z_i**2))
        r_norm = torch.linalg.norm(z, ord=2, dim=1)
        # apply trigonometry:
        # z_j = r cos(theta_j) <=> theta_j = arccos(z_j/r)
        r_norm_rep = repeat(r_norm, "b -> b d", d=z.shape[1])
        r = torch.acos(z / r_norm_rep)
        # cat the radius and angles
        r = torch.cat((r_norm.unsqueeze(dim=1), r), dim=1)
        return r

    def map_polar_to_cartesian(self, r):
        """
        translates polar coordinates of size b x (d+1) to cartesian coordinates of size b x d
        As before, polar coordinates are overdefined (e.g. one angle too many).
        First column is radius, rest are angles per dimension
        """
        # apply trigonometry:
        # z_j = r cos(theta_j) <=> theta_j = arccos(z_j/r)
        r_norm = repeat(r[:, 0], "b -> b d", d=r.shape[1] - 1)
        z = r_norm * torch.cos(r[:, 1:])
        return z
