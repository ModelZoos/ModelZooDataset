# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import timeit

from functools import reduce
import operator

"""
Sparsity implemented via Variational Dropout
https://arxiv.org/abs/1701.05369
Code taken and adapted from https://github.com/HolyBayes/pytorch_ard (provided under MIT license)
"""

def get_ard_reg(module):
    """
    :param module: model to evaluate ard regularization for
    :param reg: auxilary cumulative variable for recursion
    :return: total regularization for module
    """
    if isinstance(module, LinearARD) or isinstance(module, Conv2dARD):
        return module.get_reg()
    elif hasattr(module, 'children'):
        return sum([get_ard_reg(submodule) for submodule in module.children()])
    return 0


def _get_dropped_params_cnt(module):
    if hasattr(module, 'get_dropped_params_cnt'):
        return module.get_dropped_params_cnt()
    elif hasattr(module, 'children'):
        return sum([_get_dropped_params_cnt(submodule) for submodule in module.children()])
    return 0


def _get_params_cnt(module):
    if any([isinstance(module, l) for l in [LinearARD, Conv2dARD]]):
        return reduce(operator.mul, module.weight.shape, 1)
    elif hasattr(module, 'children'):
        return sum(
            [_get_params_cnt(submodule) for submodule in module.children()])
    return sum(p.numel() for p in module.parameters())


def get_dropped_params_ratio(model):
    return _get_dropped_params_cnt(model) * 1.0 / _get_params_cnt(model)

""" LOSS FUNCTION"""

class ELBOLoss(nn.Module):
    def __init__(self, net, loss_fn):
        super(ELBOLoss, self).__init__()
        self.loss_fn = loss_fn
        self.net = net

    def forward(self, input, target, loss_weight=1., kl_weight=1.):
        assert not target.requires_grad
        # Estimate ELBO
        return loss_weight * self.loss_fn(input, target)  \
            + kl_weight * get_ard_reg(self.net)


""" FULLY CONNECTED LAYER"""
class LinearARD(nn.Module):
    """
    Dense layer implementation with weights ARD-prior (arxiv:1701.05369)
    """

    def __init__(self, in_features, out_features, bias=True, thresh=3, ard_init=-10):
        super(LinearARD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.thresh = thresh
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.ard_init = ard_init
        self.log_sigma2 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigma2_bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def forward(self, input):
        if self.training:
            W_mu = F.linear(input, self.weight)
            std_w = torch.exp(self.log_alpha).permute(1,0)
            W_std = torch.sqrt((input.pow(2)).matmul(std_w*(self.weight.permute(1,0)**2)) + 1e-15)

            epsilon = W_std.new(W_std.shape).normal_()
            output = W_mu + W_std * epsilon
            if self.bias is not None: 
              output += self.bias
        else:
            W = self.weights_clipped
            output = F.linear(input, W) + self.bias
        return output

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 0.02)
        if self.bias is not None:
            self.bias.data.zero_()
        self.log_sigma2.data.fill_(self.ard_init)

    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        mdkl = k1 * torch.sigmoid(k2 + k3 * self.log_alpha) - \
            0.5 * torch.log1p(torch.exp(-self.log_alpha)) + C
        return -torch.sum(mdkl)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (with log alpha greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -10, 10)


""" CONVOLUTIONAL LAYER """

class Conv2dARD(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, ard_init=-10, thresh=3,bias=True):
        # bias = False  # Goes to nan if bias = True
        super(Conv2dARD, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)
        # self.bias = None
        self.thresh = thresh
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ard_init = ard_init
        self.log_sigma2 = nn.Parameter(ard_init * torch.ones_like(self.weight))
        self.log_sigma2_bias = nn.Parameter(ard_init * torch.ones_like(self.bias))
        # self.log_sigma2 = Parameter(2 * torch.log(torch.abs(self.weight) + eps).clone().detach()+ard_init*torch.ones_like(self.weight))
        
    def forward(self, input):
        """
        Forward with all regularized connections and random activations (Beyesian mode). Typically used for train
        """
        if self.training == False:
            weights_clipped = self.weights_clipped
            # bias_clipped = self.bias_clipped()
            bias_clipped = self.bias_clipped()
            return F.conv2d(input, weights_clipped,
                            bias_clipped, self.stride,
                            self.padding, self.dilation, self.groups)
            # return F.conv2d(input, self.weights_clipped,
            #                 self.bias, self.stride,
            #                 self.padding, self.dilation, self.groups)
        W = self.weight
        b = self.bias
        conved_mu = F.conv2d(input, W, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        log_alpha = self.log_alpha
        log_alpha_bias = self.log_alpha_bias()
        conved_si = torch.sqrt(1e-15 + F.conv2d(input * input,
                                                torch.exp(log_alpha) * W *
                                                W, torch.exp(log_alpha_bias)*b*b, self.stride,
                                                self.padding, self.dilation, self.groups))
        
        # conved_si = torch.sqrt(1e-15 + F.conv2d(input * input,
        #                                         torch.exp(log_alpha) * W *
        #                                         W, self.bias, self.stride,
        #                                         self.padding, self.dilation, self.groups))
        conved = conved_mu + \
            conved_si * \
            torch.normal(torch.zeros_like(conved_mu),
                         torch.ones_like(conved_mu))
        return conved

    @property
    def weights_clipped(self):
        clip_mask = self.get_clip_mask()
        return torch.where(clip_mask, torch.zeros_like(self.weight), self.weight)
    
    def get_clip_mask(self):
        log_alpha = self.log_alpha
        return torch.ge(log_alpha, self.thresh)
    
    # @property
    def bias_clipped(self):
        clip_mask = self.get_clip_mask_bias()
        return torch.where(clip_mask, torch.zeros_like(self.bias), self.bias)

    def get_clip_mask_bias(self):
        log_alpha_bias = self.log_alpha_bias()
        return torch.ge(log_alpha_bias, self.thresh)

    def get_reg(self, **kwargs):
        """
        Get weights regularization (KL(q(w)||p(w)) approximation)
        """
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.log_alpha
        mdkl = k1 * torch.sigmoid(k2 + k3 * log_alpha) - \
            0.5 * torch.log1p(torch.exp(-log_alpha)) + C
        # add bias
        log_alpha_bias = self.log_alpha_bias()
        mdkl_b = k1 * torch.sigmoid(k2 + k3 * log_alpha_bias) - \
            0.5 * torch.log1p(torch.exp(-log_alpha_bias)) + C
        return -torch.sum(mdkl) - torch.sum(mdkl_b)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels, self.out_channels, self.bias is not None
        )

    def get_dropped_params_cnt(self):
        """
        Get number of dropped weights (greater than "thresh" parameter)
        :returns (number of dropped weights, number of all weight)
        """
        return self.get_clip_mask().sum().cpu().numpy()

    @property
    def log_alpha(self):
        log_alpha = self.log_sigma2 - 2 * \
            torch.log(torch.abs(self.weight) + 1e-15)
        return torch.clamp(log_alpha, -8, 8)
    
    # @property
    def log_alpha_bias(self):
        log_alpha_bias = self.log_sigma2_bias - 2 * \
            torch.log(torch.abs(self.bias) + 1e-15)
        return torch.clamp(log_alpha_bias, -8, 8)


###############################################################################
# define net
# ##############################################################################
def compute_outdim(i_dim, stride, kernel, padding, dilation):
    o_dim = (i_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
    return o_dim


class CNN_ARD(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(Conv2dARD(in_channels=channels_in,out_channels=8, kernel_size=5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(Conv2dARD(in_channels=8, out_channels=6, kernel_size=5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(Conv2dARD(in_channels=6, out_channels=4, kernel_size=2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(LinearARD(3 * 3 * 4, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(LinearARD(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == LinearARD or type(m) == Conv2dARD:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                try:
                    # set bias to some small non-zero value
                    m.bias.data.fill_(0.01)
                except Exception as e:
                    print(e)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class CNN2_ARD(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(Conv2dARD(in_channels=channels_in, out_channels=6, kernel_size=5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(Conv2dARD(in_channels=6, out_channels=9, kernel_size=5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(Conv2dARD(in_channels=9, out_channels=6, kernel_size=2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(LinearARD(3 * 3 * 6, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(LinearARD(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == LinearARD or type(m) == Conv2dARD:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return x, activations


class CNN3_ARD(nn.Module):
    def __init__(
        self,
        channels_in,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 32x32 image size
        ## chn_in * 32 * 32
        ## compose layer 0
        self.module_list.append(Conv2dARD(in_channels = channels_in, out_channels = 16, kernel_size = 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 16 * 15 * 15
        ## compose layer 1
        self.module_list.append(Conv2dARD(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 32 * 7 * 7 // 32 * 6 * 6
        ## compose layer 2
        self.module_list.append(Conv2dARD(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(LinearARD(15 * 2 * 2, 20))
        self.module_list.append(self.get_nonlin(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(LinearARD(20, 10))

        ### initialize weights with se methods
        self.initialize_weights(init_type)

    def initialize_weights(self, init_type):
        # print("initialze model")
        for m in self.module_list:
            if type(m) == LinearARD or type(m) == Conv2dARD:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(m.weight)
                if init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(m.weight)
                if init_type == "uniform":
                    torch.nn.init.uniform_(m.weight)
                if init_type == "normal":
                    torch.nn.init.normal_(m.weight)
                if init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(m.weight)
                if init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(m.weight)
                # set bias to some small non-zero value
                m.bias.data.fill_(0.01)

    def get_nonlin(self, nlin):
        # apply nonlinearity
        if nlin == "leakyrelu":
            return nn.LeakyReLU()
        if nlin == "relu":
            return nn.ReLU()
        if nlin == "tanh":
            return nn.Tanh()
        if nlin == "sigmoid":
            return nn.Sigmoid()
        if nlin == "silu":
            return nn.SiLU()
        if nlin == "gelu":
            return nn.GELU()

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations




###############################################################################
# define FNNmodule
# ##############################################################################
class NNmoduleARD(nn.Module):
    def __init__(self, config, cuda=False, seed=42, verbosity=0):
        super(NNmoduleARD, self).__init__()

        # set verbosity
        self.verbosity = verbosity

        self.config = config

        if cuda and torch.cuda.is_available():
            self.cuda = True
            if self.verbosity > 0:
                print("cuda availabe:: send model to GPU")
        else:
            self.cuda = False
            if self.verbosity > 0:
                print("cuda unavailable:: train model on cpu")

        # setting seeds for reproducibility
        # https://pytorch.org/docs/stable/notes/randomness.html
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # construct model
        if config["model::type"] == "MLP":
            raise NotImplementedError

        elif config["model::type"] == "CNN_ARD":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN_ARD(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN2_ARD":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN2_ARD(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "CNN3_ARD":
            # calling MLP constructor
            if self.verbosity > 0:
                print("=> creating model CNN")
            model = CNN3_ARD(
                channels_in=config["model::channels_in"],
                nlin=config["model::nlin"],
                dropout=config["model::dropout"],
                init_type=config["model::init_type"],
            )
        elif config["model::type"] == "ResCNN":
            raise NotImplementedError
        elif config["model::type"] == "CNN_more_layers":
            raise NotImplementedError
        elif config["model::type"] == "CNN_residual":
            raise NotImplementedError
        elif config["model::type"] == "CNN_more_layers_residual":
            raise NotImplementedError
        elif config["model::type"] == "Resnet18":
            raise NotImplementedError
        else:
            raise NotImplementedError("error: model type unkown")

        if self.cuda:
            model = model.cuda()

        self.model = model

        # define loss function (criterion) and optimizer
        # set loss
        self.criterion = ELBOLoss(
            net=self.model, 
            loss_fn=F.cross_entropy
            )
        self.n_epochs = config["training::epochs_train"]
        if self.cuda:
            self.criterion.cuda()

        # set opimizer
        self.set_optimizer(config)

        self.scheduler = None

        self.best_epoch = None
        self.loss_best = None

    # module forward function
    def forward(self, x):
        # compute model prediction
        y = self.model(x)
        return y

    # set optimizer function - maybe we'll only use one of them anyways..
    def set_optimizer(self, config):
        if config["optim::optimizer"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=config["optim::lr"],
                momentum=config["optim::momentum"],
                weight_decay=config["optim::wd"],
                nesterov=config.get("optim::nesterov", False),
            )
        if config["optim::optimizer"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
            )
        if config["optim::optimizer"] == "rms_prop":
            self.optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=config["optim::lr"],
                weight_decay=config["optim::wd"],
                momentum=config["optim::momentum"],
            )

    def get_kl_weight(self,epoch):
        return min(1,1e-2*epoch/self.n_epochs)
        

    def load_state_dict_from_nonARD(self,checkpoint):
        if self.config["model::type"] == "CNN_ARD" or self.config["model::type"] == "CNN_ARD3":
            assert self.model.module_list[0].weight.shape == checkpoint['module_list.0.weight'].shape
            self.model.module_list[0].weight.data = checkpoint['module_list.0.weight']
            assert self.model.module_list[0].bias.data.shape == checkpoint['module_list.0.bias'].shape
            self.model.module_list[0].bias.data = checkpoint['module_list.0.bias']
            assert self.model.module_list[3].weight.data.shape == checkpoint['module_list.3.weight'].shape
            self.model.module_list[3].weight.data = checkpoint['module_list.3.weight']
            assert self.model.module_list[3].bias.data.shape == checkpoint['module_list.3.bias'].shape
            self.model.module_list[3].bias.data = checkpoint['module_list.3.bias']
            assert self.model.module_list[6].weight.data.shape == checkpoint['module_list.6.weight'].shape
            self.model.module_list[6].weight.data = checkpoint['module_list.6.weight']
            assert self.model.module_list[6].bias.data.shape == checkpoint['module_list.6.bias'].shape
            self.model.module_list[6].bias.data = checkpoint['module_list.6.bias']
            assert self.model.module_list[9].weight.data.shape == checkpoint['module_list.9.weight'].shape
            self.model.module_list[9].weight.data = checkpoint['module_list.9.weight']
            assert self.model.module_list[9].bias.data.shape == checkpoint['module_list.9.bias'].shape
            self.model.module_list[9].bias.data = checkpoint['module_list.9.bias']
            assert self.model.module_list[11].weight.data.shape == checkpoint['module_list.11.weight'].shape
            self.model.module_list[11].weight.data = checkpoint['module_list.11.weight']
            assert self.model.module_list[11].bias.data.shape == checkpoint['module_list.11.bias'].shape
            self.model.module_list[11].bias.data = checkpoint['module_list.11.bias']
            print(f'Checkpoint loaded')
        else:
            raise NotImplementedError

    def get_dropped_params_ratio(self):
        sparsity_ratio = get_dropped_params_ratio(self.model)
        return sparsity_ratio

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            # print(fname)
            perf_dict["state_dict"] = self.model.state_dict()
            torch.save(perf_dict, fname)
        return None

    # one training step / batch
    def train_step(self, input, target):
        # zero grads before training steps
        self.optimizer.zero_grad()
        # compute pde residual
        output = self.forward(input)
        # compute loss
        loss = self.criterion(output, target, loss_weight = 1.0, kl_weight = self.kl_weight)
        # prop loss backwards to
        loss.backward()
        # update parameters
        self.optimizer.step()
        # scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        # compute correct
        correct = 0
        _, predicted = torch.max(output.data, 1)
        correct = (predicted == target).sum().item()
        return loss.item(), correct

    # one training epoch
    def train_epoch(self, trainloader, epoch, idx_out=10):
        if self.verbosity > 2:
            print(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        if self.verbosity > 2:
            printProgressBar(
                0,
                len(trainloader),
                prefix="Batch Progress:",
                suffix="Complete",
                length=50,
            )
        # init accumulated loss, accuracy
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        # set k1 weight
        self.kl_weight = self.get_kl_weight(epoch)
        #
        if self.verbosity > 4:
            start = timeit.default_timer()

        # enter loop over batches
        for idx, data in enumerate(trainloader):
            input, target = data
            # send to cuda
            if self.cuda:
                input, target = input.cuda(), target.cuda()

            # take one training step
            if self.verbosity > 2:
                printProgressBar(
                    idx + 1,
                    len(trainloader),
                    prefix="Batch Progress:",
                    suffix="Complete",
                    length=50,
                )
            loss, correct = self.train_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
            # logging
            if idx > 0 and idx % idx_out == 0:
                loss_running = loss_acc / n_data
                accuracy = correct_acc / n_data
                if self.verbosity > 1:
                    print(
                        f"epoch {epoch} -batch {idx}/{len(trainloader)} --- running ::: loss: {loss_running}; accuracy: {accuracy} "
                    )

        if self.verbosity > 4:
            end = timeit.default_timer()
            print(f"training time for epoch {epoch}: {end-start} seconds")

        self.model.eval()
        # compute epoch running losses
        loss_running = loss_acc / n_data
        accuracy = correct_acc / n_data
        return loss_running, accuracy

    # test batch
    def test_step(self, input, target):
        with torch.no_grad():
            # forward pass: prediction
            output = self.forward(input)
            # compute loss
            loss = self.criterion(output, target, loss_weight = 1.0, kl_weight = self.kl_weight)
            correct = 0
            # compute correct
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == target).sum().item()
            return loss.item(), correct

    # test epoch
    def test_epoch(self, testloader, epoch):
        if self.verbosity > 1:
            print(f"validate at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # initilize counters
        loss_acc = 0
        correct_acc = 0
        n_data = 0
        # set k1 weight
        self.kl_weight = self.get_kl_weight(epoch)
        for idx, data in enumerate(testloader):
            input, target = data
            # send to cuda
            if self.cuda:
                input, target = input.cuda(), target.cuda()
            # take one training step
            loss, correct = self.test_step(input, target)
            # scale loss with batchsize
            loss_acc += loss * len(target)
            correct_acc += correct
            n_data += len(target)
        # logging
        # compute epoch running losses
        loss_running = loss_acc / n_data
        accuracy = correct_acc / n_data
        if self.verbosity > 1:
            print(f"test ::: loss: {loss_running}; accuracy: {accuracy}")

        return loss_running, accuracy

    # training loop over all epochs
    def train_loop(self, config, tune=False):
        if self.verbosity > 0:
            print("##### enter training loop ####")

        # unpack training_config
        batchsize = config["training::batchsize"]
        epochs_train = config["training::epochs_train"]
        start_epoch = config["training::start_epoch"]
        output_epoch = config["training::output_epoch"]
        val_epochs = config["training::val_epochs"]
        idx_out = config["training::idx_out"]
        checkpoint_dir = config["training::checkpoint_dir"]

        trainloader = config["training::trainloader"]
        testloader = config["training::testloader"]
        # dataloader

        if self.task == "regression":
            self.compute_mean_loss(testloader)

        perf_dict = {
            "train_loss": 1e15,
            "train_accuracy": 0.0,
            "test_loss": 1e15,
            "test_accuracy": 0.0,
        }
        self.save_model(epoch=0, perf_dict=perf_dict, path=checkpoint_dir)
        self.best_epoch = 0
        self.loss_best = 1e15

        # initialize the epochs list
        epoch_iter = range(start_epoch, start_epoch + epochs_train)
        # enter training loop
        for epoch in epoch_iter:

            # enter training loop over all batches
            loss, accuracy = self.train_epoch(trainloader, epoch, idx_out=idx_out)

            if epoch % val_epochs == 0:
                loss_test, accuracy_test = self.test_epoch(testloader, epoch)

                if loss_test < self.loss_best:
                    self.best_epoch = epoch
                    self.loss_best = loss_test
                    perf_dict["epoch"] = epoch
                    perf_dict["train_loss"] = loss
                    perf_dict["train_accuracy"] = accuracy
                    perf_dict["test_loss"] = loss_test
                    perf_dict["test_accuracy"] = accuracy_test
                    self.save_model(
                        epoch="best", perf_dict=perf_dict, path=checkpoint_dir
                    )
                if self.verbosity > 1:
                    print(f"best loss: {self.loss_best} at epoch {self.best_epoch}")

            if epoch % output_epoch == 0:
                perf_dict["train_loss"] = loss
                perf_dict["train_accuracy"] = accuracy
                perf_dict["test_loss"] = loss_test
                perf_dict["test_accuracy"] = accuracy_test
                self.save_model(epoch=epoch, perf_dict=perf_dict, path=checkpoint_dir)

        if self.verbosity > 0:
            print(f"best loss: {self.loss_best} at epoch {self.best_epoch}")
        return self.loss_best


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
    printEnd="\r",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
