# SDSC4016 Fundamentals of Machine Learning II

SDSC4016 Project

## Description

Deep learning based Recommender systems Using Amazon Reviews

This project involves using a large dataset containing customer reviews and product information from Amazon to construct a recommender system. The aim is to leverage deep learning techniques to predict consumer preferences based on their consumption habits and product metadata. 

In this project, a hybrid recommendation model is implemented, combining a neural matrix factorization model with content-based models. This hybrid model integrates user-item interactions and item attributes, following the approach proposed by Zhang et al. (2017) for joint representation learning. The neural matrix factorization model's architecture is based on the framework proposed by Chakrabarti and Das (2019).

## Data Source

[Amazon review data (2018)](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/)

## Getting Started

### Dependencies

- Python
  - Python 3.10+
  - Jupyter
  - tensorflow

### Install mini-conda and mamba

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
conda install mamba -n base -c conda-forge
```

### Set up conda environment

```bash
mamba create -n tf
mamba activate tf
```

### Installing dependencies

```bash
# conda or mamba
mamba install -c conda-forge tensorflow-gpu
mamba install -c nvidia cuda-nvcc
mamba install -c conda-forge Jupyter ipykernel pylint autopep8
mamba install -c conda-forge pandas numpy scikit-learn tqdm wordcloud nltk
```

#### Run the code

1. You can run it by ```script.sh```

2. Or you can run it by the following command:

    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    source ~/miniconda3/etc/profile.d/mamba.sh
    mamba activate tf
    export TF_CPP_MIN_LOG_LEVEL=1
    python src/script.py
    ```

#### Fetching image only

1. You can run it by ```fetcher.sh```

2. Or you can run it by the following command:

    ```bash
    source ~/miniconda3/etc/profile.d/conda.sh
    source ~/miniconda3/etc/profile.d/mamba.sh
    mamba activate tf
    export TF_CPP_MIN_LOG_LEVEL=1
    python src/fetcher.py
    ```

### Image location

- Path: ```./image/```

### Model Structure

![image](./model_structure.png)
