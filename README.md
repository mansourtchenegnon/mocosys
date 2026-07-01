# MoCoSys : Human Motion Correction based on Deep Learning Coupled with 3D+t Laplacian Motion Representation

This repository is the official implementation for *"MoCoSys : Human Motion Correction based on Deep Learning Coupled with 3D+t Laplacian Motion Representation"*.

## News

## Introduction

**Abstract.** In this paper, we present a 3D motion correction system whose objective is to produce motion for computer animation purposes from 3D human pose estimation. Currently, most 3D pose estimation approaches fail to reconstruct motion that meets the specific requirements for 3D avatar animation. In particular, they often do not ensure the stability of the underlying skeleton or the temporal fluidity and coherence of the motion. We propose an additional process to address these limitations. Our system uses deep learning techniques in combination with Laplacian motion modeling, along with algorithms designed to enhance the temporal characteristics of the motion while preserving the integrity of the skeletal structure throughout the sequence. The approach is based on two deep neural networks. The first network utilizes a 3D+t graph representation of motion, combined with a discrete Laplacian operator, to improve the spatio-temporal deformation of the skeleton over time. The second network estimates fixed bone lengths within the skeletal structure, enabling the correction process to maintain skeletal consistency throughout the motion, while minimizing bone lengths errors in the reconstructed motion. Experiments conducted on the outputs of state-of-the-art neural architectures demonstrate that our system significantly enhances both the spatial and temporal characteristics of the reconstructed motion. This improvement ensures that the corrected movements are suitable for use in data-driven animation applications.

## Environment

The code is developed and tested under the following environment.

- Python >=3.10
- Tensorflow >= 2.16.1
- CUDA 11.3

```pip install -r requirements.txt```

## Usage

### Dataset preparation

Please refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset as follows:

```
code_root/
└── data/
	├── data_2d_h36m_gt.npz
	├── data_2d_h36m_cpn_ft_h36m_dbb.npz
	└── data_3d_h36m.npz
```
### Training
#### Motion Fine-Tuning Model
Before the training, you need to generate estimated 3D poses from a state-of-the-art solution and store them in the `npz` format in folder `data`. We provide you the compress data file used for training containing the estimated poses we used in our experiments. The content is describes [here](./documentation/data.md). 

#### Skeleton Model
```
python run_training.py -a skelmodel
```

### Evaluation
#### Motion Fine-Tuning Model

#### Skeleton Model
```
python run_evaluation.py 
```