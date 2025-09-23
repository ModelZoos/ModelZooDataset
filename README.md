This repository complements the Paper "Model Zoos: A Dataset of Diverse Populations of Neural Network Models". The paper can be found [here](https://arxiv.org/abs/2209.14764). The repository contains code to recreate, adapt or extend the zoos, links to the zoos, code to load the zoos and reproduce the benchmarks. The vision for the dataset is to extend model zoos over time allowing the community to investigate different aspects of model populations. 

![alt text](assets/model_zoo_overview.png)

# Abstract
In the last years, neural networks have evolved from laboratory environments to the state-of-the-art for many real-world problems. Our hypothesis is that neural network models (i.e., their weights and biases) evolve on unique, smooth trajectories in weight space during training. Following, a population of such neural network models (refereed to as “model zoo”) would form topological structures in weight space. We think that the geometry, curvature and smoothness of these structures contain information about the state of training and can be reveal latent properties of individual models. With such zoos, one could investigate novel approaches for (i) model analysis, (ii) discover unknown learning dynamics, (iii) learn rich representations of such populations, or (iv) exploit the model zoos for generative modelling of neural network weights and biases. Unfortunately, the lack of standardized model zoos and available benchmarks significantly increases the friction for further research about populations of neural networks. With this work, we publish a novel dataset of model zoos containing systematically generated and diverse populations of neural network models for further research. In total the proposed model zoo dataset is based on six image datasets, consist of 27 model zoos with varying hyperparameter combinations are generated and includes 50’360 unique neural network models resulting in over 2’585’360 collected model states. Additionally, to the model zoo data we provide an in-depth analysis of the zoos and provide benchmarks for multiple downstream tasks as mentioned before. Further, each model zoo will be accompanied with a sparsified counterpart.

# Model Zoos
The model zoo datasets are hosted on Zenodo. Zenodo guarantees at least 20 years of availability, provides searchable metadata and dataset DOIs. We provide the raw checkpoints as well as performance metrics for every model at every epoch,like the accuray visualized in the figure below, to assess performance and diversity of the zoo. 
![alt text](assets/boxplots_mix.png)

The zoos of each image dataset are uploaded in individual zenodo repositories, to make them extendable. We provide DOI links below.  
In the Zenodo repositories, we provide `zip` files with the raw model zoos, as well as `.pt` (pytorch) files. 
The `.pt` files contain preprocessed datasets wrapped in our custom dataset class (`code/checkpoints_to_datasets/dataset_base.py`).
The `index_dict.json` contains information on where weights in the vectorized form belong in the original model.  


| Image Dataset | DOI Link to Zoo  |  
| ----------- | ----------- |
| MNIST CNN-s (raw and prepocessed)| https://doi.org/10.5281/zenodo.6632086 |  
| Fashion-MNIST CNN-s (raw and prepocessed)| https://doi.org/10.5281/zenodo.6632104 |   
| SVHN CNN-s (raw and prepocessed)| https://doi.org/10.5281/zenodo.6632120 |  
| USPS CNN-s (raw and prepocessed)| https://doi.org/10.5281/zenodo.6633626 |  
| Cifar10 CNN-s and CNN-l (raw and prepocessed)| https://doi.org/10.5281/zenodo.6620868 |  
| STL10 CNN-s and CNN-l (raw) | https://doi.org/10.5281/zenodo.6631783 |  
| STL10 CNN-s and CNN-l (preprocessed) | https://doi.org/10.5281/zenodo.6634138 | 
| CIFAR10 ResNet-18 (raw, squeezed) | https://doi.org/10.5281/zenodo.6974028 |
| CIFAR10 ResNet-18 (raw, full) | https://ai.unisg.ch/files/tar_files/tune_zoo_cifar10_resnet18_kaiming_uniform.tar |
| CIFAR100 ResNet-18 (raw, squeezed) | https://doi.org/10.5281/zenodo.6977381 |
| CIFAR100 ResNet-18 (raw, full) | https://ai.unisg.ch/files/tar_files/tune_zoo_cifar100_resnet18_kaiming_uniform.tar |
| Tiny-Imagenet ResNet-18 (raw, squeezed) | https://doi.org/10.5281/zenodo.7023277 |
| EuroSAT CNN-s (raw) | https://doi.org/10.5281/zenodo.8141666 |

# Sparsified Model Zoo Twins
Model sparsification and distillation are an important topic to efficiently operate neural networks in production. To study sparisifaction at a population level, we introduce *sparsified model zoo twins*.  
Re-using the populations of full models (from above), the sparsified populations contain *sparsified twins* of each of the full models. 
Starting of the last epoch of the full zoo, sparsification with [Variational Dropout](https://arxiv.org/abs/1701.05369) generates a sparsification trajectory for each model, along which we track the performance, degree of sparsity and the sparsified checkpoint.  Sparsified model zoos add several potential use-cases. The zoos can be used to study the sparsification performance on a population level, study emerging patterns of populations of sparse models, or the relation of full models and their sparse counterparts.   
The Figure below shows the performance of the first sparsified zoo (MNIST Sparsified CNN-s) is already added to the collection. After 25 sparsification epochs, the models achieve on average 80% sparsity at only 2% accuracy loss.
![MNIST Sparsified Twin Zoo](assets/MNIST_Sparse_Twin.png)

The link to the full datasets containing the sparsification trajectories and performance metrics are listed below. The code to sparsify models is contained in `def_net_distillation.py` and def_NN_experiment_distillation.py`, the zoo generators are uploaded in the corresponding directory. 
| Image Dataset | DOI Link to Zoo  |  
| ----------- | ----------- |
| MNIST Sparsified CNN-s (raw)| https://doi.org/10.5281/zenodo.7023335 |  
| SVHN Sparsified CNN-s (raw)| https://doi.org/10.5281/zenodo.7027566 |  
| EuroSAT CNN-s (raw) | https://doi.org/10.5281/zenodo.8141666 |



# Accessibility
We provide a custom pytorch dataset class to load and preprocess the raw model zoos. Code related to the dataset class is in the module `code/checkpoints_to_datasets/`. The class definition can be found under `dataset_base.py`.
The class takes care of loading the model checkpoints and their properties, if necessary vectorizing their weights, and sorting out models with faulty data.
We further provide pre-computed dataset files, with train, test and validation datasets.
To further simplify access to the datasets, a jupyter notebook `code/load_dataset.ipynb` contains examples of loading preprocessed or raw datasets and explores their properties.   
A conda `environmnent.yml` to create a working `conda` environment can be found in `code/`. With `conda` installed, run `conda env create -f code/environment.yml` to create the environment. 

# Zoo Generation
The scripts to generate the model zoos can be found under `code/zoo_generators/`. 
With these scripts, the zoos can be rectreated, adapted or extended. 
The class definition for the models in the zoo is contained in `code/model_definitions/def_net.py`. 
We use [ray tune](#https://docs.ray.io/en/latest/tune/index.html) in version 1.8.0 to train populations. The `tune.trainable` class wrapper around the model is in `code/model_definitions/def_NN_experiment.py`.

# Benchmark Results
A notebook with code to replicate the benchmark numbers from the paper can be found in `code/benchmark_results.ipynb`. The benchmark classes are located in `code/model_definitions/def_downstream_module.py` and `code/model_definitions/def_baseline_models.py`.  

# License
The model zoos are licensed under the Creative Commons Attribution 4.0 International license (CC-BY 4.0).


Get in touch!
====
If there are any questions, reach out to us! We'll respond to issues here, or via email `konstantin.schuerholt@unisg.ch`. 

Citation
====
If you want to cite this work, please use 
```
@inproceedings{schurholtModelZoosDataset2022,
  title = {Model Zoos: A Dataset of Diverse Populations of Neural Network Models},
  booktitle = {Thirty-Sixth Conference on Neural Information Processing Systems (NeurIPS) Track on Datasets and Benchmarks},
  author = {Sch{\"u}rholt, Konstantin and Taskiran, Diyar and Knyazev, Boris and Gir{\'o}-i-Nieto, Xavier and Borth, Damian},
  year = {2022},
  month = sep,
}
```
