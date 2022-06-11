import torch
import torch.nn as nn
import numpy as np

### just an identity model, maps input on itself
class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        self.modules = torch.nn.Linear(1, 1)

    def forward(self, x):
        return x

    def forward_encoder(self, x):
        return x


class LayerQuintiles(nn.Module):
    def __init__(self, index_dict, use_bias=True):
        super(LayerQuintiles, self).__init__()
        self.index_dict = index_dict
        self.modules = torch.nn.Linear(1, 1)
        self.use_bias = use_bias

    def forward(self, x):
        z = self.compute_layer_quintiles(x)
        return z

    def forward_encoder(self, x):
        z = self.compute_layer_quintiles(x)
        return z

    def compute_layer_quintiles(self, weights):
        # weights need to be on the cpu for numpy
        weights = weights.to(torch.device("cpu"))
        quantiles = [1, 25, 50, 75, 99]
        features = []
        for idx, layer in enumerate(self.index_dict["layer"]):
            # print(f"compute features for layer {layer}")
            # print(f"### layer {layer} ###")
            # get slices
            kernel_start = self.index_dict["idx_start"][idx]
            kernel_end = (
                self.index_dict["idx_start"][idx]
                + self.index_dict["kernel_no"][idx]
                * self.index_dict["kernel_size"][idx]
                * self.index_dict["channels_in"][idx]
            )
            index_kernel = list(range(kernel_start, kernel_end))
            if self.use_bias:
                bias_start = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                )
                bias_end = (
                    self.index_dict["idx_start"][idx]
                    + self.index_dict["kernel_no"][idx]
                    * self.index_dict["kernel_size"][idx]
                    * self.index_dict["channels_in"][idx]
                    + self.index_dict["kernel_no"][idx]
                )
                index_bias = list(range(bias_start, bias_end))
            # compute kernel stat values
            features_ldx_weights = np.percentile(
                a=weights[:, index_kernel].detach().numpy(), q=quantiles, axis=1
            )
            features_ldx_weights = torch.tensor(features_ldx_weights)
            features_ldx_weights = torch.einsum("ij->ji", features_ldx_weights)
            # print(features_ldx_weights.shape)
            mean_ldx_weights = torch.tensor(
                np.mean(a=weights[:, index_kernel].detach().numpy(), axis=1)
            ).unsqueeze(dim=1)
            var_ldx_weights = torch.tensor(
                np.var(a=weights[:, index_kernel].detach().numpy(), axis=1)
            ).unsqueeze(dim=1)
            features.extend([mean_ldx_weights, var_ldx_weights, features_ldx_weights])

            # compute bias stat values
            if self.use_bias:
                features_ldx_bias = np.percentile(
                    a=weights[:, index_bias].detach().numpy(), q=quantiles, axis=1
                )
                features_ldx_bias = torch.tensor(features_ldx_bias)
                features_ldx_bias = torch.einsum("ij->ji", features_ldx_bias)
                # print(features_ldx_bias.shape)
                mean_ldx_bias = torch.tensor(
                    np.mean(a=weights[:, index_bias].detach().numpy(), axis=1)
                ).unsqueeze(dim=1)
                var_ldx_bias = torch.tensor(
                    np.var(a=weights[:, index_bias].detach().numpy(), axis=1)
                ).unsqueeze(dim=1)
                features.extend([mean_ldx_bias, var_ldx_bias, features_ldx_bias])

        # put together
        z = torch.cat(features, dim=1)
        return z


########################################################################
### Kernel PCA
########################################################################
# idx_full_train = list(range(dataset["din_train"].shape[0]))
# idx_full_test = list(range(dataset["din_test"].shape[0]))
# keep = 3
# idx_subset_train = [idx for idx in idx_full_train if idx % keep == 0]
# idx_subset_test = [idx for idx in idx_full_test if idx % keep == 0]
# lat_dims = [50, 33]
# kernel_list = ["linear", "poly", "rbf", "sigmoid", "cosine"]


class KPCAModel(nn.Module):
    def __init__(self, weights_fit, lat_dim, kernel):
        super(KPCAModel, self).__init__()
        from sklearn.decomposition import KernelPCA

        self.transformer = KernelPCA(
            n_components=lat_dim, kernel=kernel, fit_inverse_transform=False
        )
        self.transformer.fit(weights_fit.cpu().detach().numpy())

    def forward(self, x):
        x = self.transformer.transform(x.cpu().detach().numpy())
        return torch.tensor(x)

    def forward_encoder(self, x):
        x = self.forward(x)
        return x


########################################################################
### Umap
########################################################################


class UMAPModel(nn.Module):
    def __init__(self, weights_fit, lat_dim, metric="euclidean"):
        super(UMAPModel, self).__init__()
        import umap

        n_neighbors = 25  #  smaller values -> focus on local dependencies / larger values -> better global resolution
        min_dist = 0.001  # how close can points be mapped together in reduced space
        # cosine: perfect
        # correlation: also perfect
        self.transformer = umap_transformer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=lat_dim,
            metric=metric,
            transform_seed=42,
        )

        self.transformer.fit(weights_fit.cpu().detach().numpy())

    def forward(self, x):
        x = self.transformer.transform(x.cpu().detach().numpy())
        return torch.tensor(x)

    def forward_encoder(self, x):
        x = self.forward(x)
        return x

