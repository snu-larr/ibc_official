# IBC

### Accepted to the Fortieth International Conference on Machine Learning (ICML 2023)

This is an official implementation for **IBC** from our paper: [**Demonstration-free Autonomous Reinforcement Learning via Implicit and Bidirectional Curriculum**](https://arxiv.org/abs/2305.09943) by [Jigang Kim*](https://jigang.kim/), [Daesol Cho*](https://dscho1234.github.io)  (*Equally contributed), and [H. Jin Kim](https://scholar.google.co.kr/citations?user=TLQUwIMAAAAJ)

The instructions below were tested on Ubuntu 20.04, but should work on other Linux distros as well.

## Installation

### 1. Install Conda package manager
Conda package manager is required for installing python dependencies. Follow the link below to install conda:

https://docs.conda.io/projects/conda/en/latest/user-guide/install/

### 2. Create a Conda environment
```
conda env create -f conda_env.yml
conda activate ibc
```

### 3. Manually install other dependencies 
```
# Install a version of pytorch appropriate for your machine. For example,
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

# Install metaworld for sawyer env.
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
```

## Running experiments
Set the path parameters *default_save_path_prefix* and *workspace_path* for your machine by following the instructions in [config/paths/template.yaml](https://github.com/snu-larr/ibc_official/blob/main/config/paths/template.yaml).

Below are the commands for running IBC for the six environments:
```
python train.py env=tabletop_manipulation
```
```
python train.py env=sawyer_door
```
```
python train.py env=fetch_pickandplace_ergodic
```
```
python train.py env=fetch_push_ergodic
```
```
python train.py env=fetch_reach_ergodic
```
```
python train.py env=point_umaze
```

## Acknowledgements

This repository contains modified open-source code from the official implementation of [HGG](https://github.com/Stilwell-Git/Hindsight-Goal-Generation). It also contains open-source implementations of various RL environments such as [earl_benchmark](https://github.com/architsharma97/earl_benchmark.git), [mujoco-maze](https://github.com/kngwyu/mujoco-maze), and [metaworld](https://github.com/rlworkgroup/metaworld).

## BibTeX

```bibtex
@article{kim2023demonstration,
  title={Demonstration-free Autonomous Reinforcement Learning via Implicit and Bidirectional Curriculum},
  author={Kim, Jigang and Cho, Daesol and Kim, H Jin},
  journal={arXiv preprint arXiv:2305.09943},
  year={2023}
}
```
