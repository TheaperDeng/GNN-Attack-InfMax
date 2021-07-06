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

