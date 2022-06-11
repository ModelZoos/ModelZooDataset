from pathlib import Path

import torch

from torch.utils.data import Dataset


from .dataset_auxiliaries import (
    test_checkpoint_for_nan,
    test_checkpoint_with_threshold,
    vectorize_checkpoint,
)

import random
import copy
import json
import tqdm


import ray
from .progress_bar import ProgressBar


class ModelDatasetBase(Dataset):
    """
    This dataset class loads checkpoints from path, stores them in the memory
    The checkpoints are stored in self.data_in
    The weights of the models can be accessed as tensor via self.__get_weights__()
    The model properties, such as accuracy, generalization gap, or hyperparameters are in self.properties
    The __getitem__() function currently returns checkpoints / weights only. Adapt this to your needs by returning additional values. 
    """

    # class arguments

    # init
    def __init__(
        self,
        root,
        # layer index and type of the layers to laod
        # for the small model zoos:
        layer_lst=[
            (0, "conv2d"),
            (3, "conv2d"),
            (6, "conv2d"),
            (9, "fc"),
            (11, "fc"),
        ],
        # for the large model zoos:
        #     [
        #     (0, "conv2d"),
        #     (4, "conv2d"),
        #     (8, "conv2d"),
        #     (13, "fc"),
        #     (16, "fc"),
        # ]
        # epochs of the models to load
        epoch_lst=10,
        # wether to store models as checkpoints or weights ("checkpoints","vector")
        mode="checkpoint",
        # "reconstruction" (w->w), "sequence_prediction" (x^i -> x^i+1),
        task="reconstruction",
        # Load only weights or also bias parameters.
        use_bias=True,
        # determines which dataset split to use ["train","val","test"]
        train_val_test="train",
        # Determines the dataset split ration. List of 2 or 3 floats, which have to add to 1.
        ds_split=[0.7, 0.3],
        # Set number of maximum samples to load. Set to None to load all samples.
        max_samples=None,
        # Set weight threshold to exclude samples with absolute weight values above the threshold.
        weight_threshold=float("inf"),
        # Function to filter samlpes. Function receives sample path as argument and returns True if model needs to be filtered out
        filter_function=None,
        # Keys to load from model config / results.
        property_keys=None,
        # Threads for multiprocessing
        num_threads=4,
        # Set whether to shuffle the paths
        shuffle_path=True,
        # Set verbosity
        verbosity=0,
    ):
        self.layer_lst = layer_lst
        self.epoch_lst = epoch_lst
        self.mode = mode
        self.task = task
        self.use_bias = use_bias
        self.verbosity = verbosity
        self.weight_threshold = weight_threshold
        self.property_keys = copy.deepcopy(property_keys)
        self.train_val_test = train_val_test
        self.ds_split = ds_split
        # self.filter_function = filter_function

        ### prepare directories and path list ################################################################

        # check if root is list. if not, make root a list
        if not isinstance(root, list):
            root = [root]

        # make path an absolute pathlib Path
        for rdx in root:
            if isinstance(rdx, torch._six.string_classes):
                rdx = Path(rdx)
        self.root = root

        # get list of folders in directory
        self.path_list = []
        for rdx in self.root:
            pth_lst_tmp = [f for f in rdx.iterdir() if f.is_dir()]
            self.path_list.extend(pth_lst_tmp)

        # shuffle self.path_list
        if shuffle_path:
            random.seed(42)
            random.shuffle(self.path_list)

        ### Split Train and Test set ###########################################################################
        if max_samples is not None:
            self.path_list = self.path_list[:max_samples]

        ### Split Train and Test set ###########################################################################
        assert sum(self.ds_split) == 1.0, "dataset splits do not equal to 1"
        # two splits
        if len(self.ds_split) == 2:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[:idx1]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[idx1:]
            else:
                raise NotImplementedError(
                    "validation split requested, but only two splits provided."
                )
        # three splits
        elif len(self.ds_split) == 3:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[:idx1]
            elif self.train_val_test == "val":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(self.path_list))
                self.path_list = self.path_list[idx1:idx2]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(self.path_list))
                self.path_list = self.path_list[idx2:]
        else:
            print(f"dataset splits are unintelligble. Load 100% of dataset")
            pass

        ### initialize data over epochs #####################
        if not isinstance(epoch_lst, list):
            epoch_lst = [epoch_lst]
        # get iterator for epochs
        if self.task == "sequence_prediction":
            # -1 because we need pairs of != epochs
            eldx_lst = range(len(epoch_lst) - 1)
        else:
            eldx_lst = range(len(epoch_lst))

        ### prepare data lists ###############
        data_in = []
        data_out = []
        labels_in = []
        labels_out = []
        paths = []
        epochs = []

        ## init multiprocessing environment ############
        ray.init(num_cpus=num_threads)

        ### gather data #############################################################################################
        print(f"loading checkpoints from {self.root}")
        pb = ProgressBar(total=len(self.path_list * len(eldx_lst)))
        pb_actor = pb.actor
        for idx, path in enumerate(self.path_list):
            # if self.verbosity > 2:
            # printProgressBar(iteration=idx + 1, total=len(self.path_list))
            # iterate over all epochs in list
            for eldx in eldx_lst:
                edx = epoch_lst[eldx]

                # # call function in parallel
                # din, lin, dout, lout = self.load_checkpoints.remote(
                #     idx=idx, edx=edx, eldx=eldx, path=path
                # )
                # call function in parallel
                (
                    din,
                    lin,
                    dout,
                    lout,
                    path_dx,
                    epoch_dx,
                ) = load_checkpoints_remote.remote(
                    idx=idx,
                    edx=edx,
                    eldx=eldx,
                    path=path,
                    layer_lst=self.layer_lst,
                    epoch_lst=self.epoch_lst,
                    use_bias=self.use_bias,
                    task=self.task,
                    weight_threshold=self.weight_threshold,
                    filter_function=filter_function,
                    verbosity=self.verbosity,
                    pba=pb_actor,
                )

                data_in.append(din)
                labels_in.append(lin)
                data_out.append(dout)
                labels_out.append(lout)
                paths.append(path_dx)
                epochs.append(epoch_dx)

        pb.print_until_done()

        # collect actual data
        data_in = ray.get(data_in)
        data_out = ray.get(data_out)
        labels_in = ray.get(labels_in)
        labels_out = ray.get(labels_out)
        paths = ray.get(paths)
        epochs = ray.get(epochs)

        ray.shutdown()

        # remove None values
        data_in = [ddx for ddx in data_in if ddx]
        data_out = [ddx for ddx in data_out if ddx]
        labels_in = [ddx for ddx in labels_in if ddx]
        labels_out = [ddx for ddx in labels_out if ddx]
        epochs = [edx for edx, pdx in zip(epochs, paths) if pdx]
        paths = [pdx for pdx in paths if pdx]

        self.data_in = copy.deepcopy(data_in)
        self.data_out = copy.deepcopy(data_out)
        self.labels_in = copy.deepcopy(labels_in)
        self.labels_out = copy.deepcopy(labels_out)
        self.paths = copy.deepcopy(paths)
        self.epochs = copy.deepcopy(epochs)

        if self.verbosity > 2:
            print(
                f"Data loaded. found {len(self.data_in)} usable samples out of potential {len(self.path_list * len(eldx_lst))} samples."
            )

        if self.property_keys is not None:
            if self.verbosity > 2:
                print(f"Load properties for samples from paths.")

            # get propertys from path
            result_keys = self.property_keys.get("result_keys", [])
            config_keys = self.property_keys.get("config_keys", [])
            # figure out offset
            try:
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=1,
                )
            except AssertionError as e:
                print(e)
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=0,
                )
            if self.verbosity > 2:
                print(f"Properties loaded.")
        else:
            self.properties = None

        if self.mode == "vector":
            print('Vectorize weights')
            self.data_in = self.__get_weights__()

    ## getitem ####################################################################################################################################################################
    def __getitem__(self, index):
        # adapt this to fit your need
        return self.data_in[index]
        ## len ####################################################################################################################################################################

    def __len__(self):
        return len(self.data_in)

    ### get_weights ##################################################################################################################################################################
    def __get_weights__(self):
        if self.mode == "checkpoint":
            self.vectorize_data()
        else:
            self.weights_in = torch.stack(self.data_in, dim=0)
        return self.weights_in

    ## read properties from path ##############################################################################################################################################
    def read_properties(self, results_key_list, config_key_list, idx_offset=1):
        # copy results_key_list to prevent kickback of delete to upstream function
        results_key_list = [key for key in results_key_list]
        # init dict
        properties = {}
        for key in results_key_list:
            properties[key] = []
        for key in config_key_list:
            properties[key] = []
        # remove ggap from results_key_list -> cannot be read, has to be computed.
        read_ggap = False
        if "ggap" in results_key_list:
            results_key_list.remove("ggap")
            read_ggap = True

        if self.verbosity > 2:
            print(f"### load data for {properties.keys()}")

        # iterate over samples
        for iidx, (ppdx, eedx) in tqdm.tqdm(enumerate(zip(self.paths, self.epochs))):
            res_tmp = read_properties_from_path(
                ppdx, eedx, idx_offset=idx_offset)
            for key in results_key_list:
                properties[key].append(res_tmp[key])
            for key in config_key_list:
                properties[key].append(res_tmp["config"][key])
            # compute ggap
            if read_ggap:
                gap = res_tmp["train_acc"] - res_tmp["test_acc"]
                properties["ggap"].append(gap)
            # assert epoch == training_iteration -> match correct data
            if iidx == 0:
                train_it = int(res_tmp["training_iteration"])
                assert (
                    int(eedx) == train_it
                ), f"training iteration {train_it} and epoch {eedx} don't match."

            if self.verbosity > 2 and iidx == 123:
                print(f"check existance of keys")
                for key in properties.keys():
                    print(
                        f"key: {key} - len {len(properties[key])} - last entry: {properties[key][-1]}"
                    )
        self.properties = properties

    def vectorize_data(self):
        # save base checkpoint
        self.checkpoint_base = self.data_in[0]
        # iterate over length of dataset
        self.weights_in = []
        for idx in tqdm.tqdm(range(self.__len__())):
            checkpoint_in = copy.deepcopy(self.data_in[idx])
            ddx_in = vectorize_checkpoint(
                checkpoint_in, self.layer_lst, self.use_bias)
            self.weights_in.append(ddx_in)

        self.weights_in = torch.stack(self.weights_in, dim=0)


# helper function for property reading
def read_properties_from_path(path, idx, idx_offset):
    """
    reads path/result.json
    returns the dict for training_iteration=idx
    idx_offset=0 if checkpoint_0 was written, else idx_offset=1
    """
    # read json
    try:
        fname = Path(path).joinpath("result.json")
        results = []
        for line in fname.open():
            results.append(json.loads(line))
        # trial_id = results[0]["trial_id"]
    except Exception as e:
        print(f"error loading {fname}")
        print(e)
    # pick results
    jdx = idx - idx_offset
    return results[jdx]


############## load_checkpoint_remote ########################################################
@ray.remote(num_returns=6)
def load_checkpoints_remote(
    idx,
    edx,
    eldx,
    path,
    layer_lst,
    epoch_lst,
    use_bias,
    task,
    weight_threshold,
    filter_function,
    verbosity,
    pba,
):
    ## get full path to files ################################################################
    chkpth = path.joinpath(f"checkpoint_{edx}", "checkpoints")
    ## load checkpoint #######################################################################
    chkpoint = {}
    try:
        # try with conventional naming scheme
        try:
            # load chkpoint to cpu memory
            chkpoint = torch.load(
                str(chkpth), map_location=torch.device("cpu"))
        except FileNotFoundError as e:
            if verbosity > 5:
                print(f"{e}")
                print(f"try again with different formatting")

            # use other formatting
            chkpth = path.joinpath(f"checkpoint_{edx:06d}", "checkpoints")
            # load chkpoint to cpu memory
            chkpoint = torch.load(
                str(chkpth), map_location=torch.device("cpu"))
    except Exception as e:
        if verbosity > 5:
            print(f"error while loading {chkpth}")
            print(f"{e}")
        # instead of appending empty stuff, jump to next
        pba.update.remote(1)
        return None, None, None, None, None, None
    ## create label ##########################################################################
    label = f"{path}#_#epoch_{edx}#_#layer_{layer_lst}"

    ### check for NAN values #################################################################
    nan_flag = test_checkpoint_for_nan(
        copy.deepcopy(chkpoint), layer_lst, use_bias)
    if nan_flag == True:
        if verbosity > 5:
            # jump to next sample
            raise ValueError(f"found nan values in checkpoint {label}")
        pba.update.remote(1)
        return None, None, None, None, None, None
    # apply filter function
    if filter_function is not None:
        filter_flag = filter_function(path)
        if filter_flag == True:  # model needs to be filtered
            pba.update.remote(1)
            return None, None, None, None, None, None

    # apply threhold
    thresh_flag = test_checkpoint_with_threshold(
        copy.deepcopy(chkpoint), layer_lst, use_bias, weight_threshold
    )
    if thresh_flag == True:
        if verbosity > 5:
            # jump to next sample
            raise ValueError(
                f"found values above threshold in checkpoint {label}")
        pba.update.remote(1)
        return None, None, None, None, None, None
    else:  # use data
        din = copy.deepcopy(chkpoint)
        lin = copy.deepcopy(label)

    ### task: reconstruction: copy input to output ############################################
    if task == "reconstruction":
        dout = copy.deepcopy(din)
        lout = copy.deepcopy(lin)

    ### task: sequence prediction: load next checkpoint #######################################
    elif task == "sequence_prediction":
        ## get full path to files ################################################################
        # get next
        chkpth = path.joinpath(
            f"checkpoint_{epoch_lst[eldx+1]}", "checkpoints")
        ## load checkpoint #######################################################################
        chkpoint = {}
        # try with conventional naming scheme
        try:
            # load chkpoint to cpu memory
            chkpoint = torch.load(
                str(chkpth), map_location=torch.device("cpu"))
        except FileNotFoundError as e:
            if verbosity > 5:
                print(f"{e}")
                print(f"try again with different formatting")

            # use other formatting
            chkpth = path.joinpath(
                f"checkpoint_{epoch_lst[eldx+1]:06d}", "checkpoints")
            # load chkpoint to cpu memory
            chkpoint = torch.load(
                str(chkpth), map_location=torch.device("cpu"))
        except Exception as e:
            if verbosity > 5:
                print(f"error loading {chkpth}")
                print(f"{e}")
            # instead of appending empty stuff, jump to next
            pba.update.remote(1)
            return None, None, None, None, None, None

        ## create label ##########################################################################
        label = f"{path} epoch {epoch_lst[eldx+1]} layer {layer_lst}"

        ## get weights and check for NAN #########################################################
        nan_flag = test_checkpoint_for_nan(
            copy.deepcopy(chkpoint), layer_lst, use_bias)
        if nan_flag == True:
            if verbosity > 5:
                # jump to next sample
                raise ValueError(f"found nan values in checkpoint {label}")
            pba.update.remote(1)
            return None, None, None, None, None, None
        if filter_function is not None:
            filter_flag = filter_function(path)
            if filter_flag == True:  # model needs to be filtered
                pba.update.remote(1)
                return None, None, None, None, None, None
        # apply threhold
        thresh_flag = test_checkpoint_with_threshold(
            copy.deepcopy(chkpoint), layer_lst, use_bias, weight_threshold
        )
        if thresh_flag == True:
            if verbosity > 5:
                # jump to next sample
                raise ValueError(
                    f"found values above threshold in checkpoint {label}")
            pba.update.remote(1)
            return None, None, None, None, None, None
        else:
            # use data
            dout = copy.deepcopy(chkpoint)
            lout = copy.deepcopy(label)

    # return
    pba.update.remote(1)
    return din, lin, dout, lout, path, edx
