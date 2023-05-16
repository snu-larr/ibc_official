# IBC

This is a implementation **IBC** from our paper: **Deomnstration-free Autonomous Reinforcement Learning via Implicit and Bidirectional Curriculum (ICML 2023)**

By [Jigang Kim*](https://jigang.kim/), [Daesol Cho*](https://dscho1234.github.io)  (*Equally contributed), and H. Jin Kim

## Conda Install
```
conda env create -f conda_env.yml

conda activate ibc

```

## Install other dependencies 
```
install torch appropriate to your machine.

(ex: conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia)

install metaworld for sawyer env.

pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld

```



## Instructions
Firstly, you should write some paths following the instructions in config/paths/template.yaml.

Then, run by following code for each environment:

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


Our code sourced and modified from official implementation of [HGG](https://github.com/Stilwell-Git/Hindsight-Goal-Generation) Algorithm. Also, we utilize [mujoco-maze](https://github.com/kngwyu/mujoco-maze) and [metaworld](https://github.com/rlworkgroup/metaworld) to validate our proposed method.