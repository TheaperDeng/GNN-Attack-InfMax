# **GNN-Attack-InfMax**

This repo provides the official implementations for the experiments described in the following paper:

[**Adversarial Attack on Graph Neural Networks as An Influence Maximization Problem**](https://arxiv.org/abs/2106.10785)

Jiaqi Ma\*, Junwei Deng\*, and Qiaozhu Mei. ArXiv 2021.

(\*: equal constribution)

## Requirements
- dgl 0.4.2
- torch 1.4.0    
- networkx 2.3  
- numpy 1.16.4 

## Note for the Experiment Setup

This repo is built on top of the [code](https://github.com/Mark12Ding/GNN-Practical-Attack) for a NeurIPS 2020 publication: [*Towards More Practical Adversarial Attacks on Graph Neural Networks*](https://arxiv.org/abs/2006.05057). 

A major change in the experiment setup between this work and the NeurIPS 2020 publication is the way we construct the node attribute perturbation, as noted in Section 5.2 of our paper. The perturbation construction implemented in this repo is more strictly black-box.

## Run the code

Example command to run the code: `python main.py --seed 0 --model GAT --dataset cora --threshold 0.1 --norm_length 10`. 

## Cite
```
@article{ma2021adversarial,
  title={Adversarial Attack on Graph Neural Networks as An Influence Maximization Problem},
  author={Ma, Jiaqi and Deng, Junwei and Mei, Qiaozhu},
  journal={arXiv preprint arXiv:2106.10785},
  year={2021}
}
```

