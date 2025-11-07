# Planner3D

Implementation of the paper _Planner3D: LLM-enhanced graph prior meets 3D indoor scene explicit regularization_. 


### [Project page](https://sites.google.com/view/planner3d)
  
## Usage

### Preparation

#### Data

Download the [3D-FRONT](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-scene-dataset) dataset from their official site. Preprocess the dataset following [ATISS](https://github.com/nv-tlabs/ATISS#data-preprocessing). Download [3D-FUTURE-SDF](https://github.com/ymxlzgy/commonscenes) and [SG-FRONT](https://github.com/ymxlzgy/commonscenes/blob/main/SG-FRONT.md).

#### Weights

Put the [weight](https://drive.google.com/file/d/19TC_F6BVZluVJQ0C_JVr81-XpZtkbvu_/view?usp=sharing) under ./scripts/checkpoint

### Train

```
cd scripts
python train_3dfront.py --exp .../all --room_type livingroom --dataset /path/to/FRONT --residual True --network_type v2_full --with_SDF True --with_CLIP True --batchSize 4 --workers 4 --loadmodel False --nepoch 10000 --large False

```

### Evaluation

```
cd scripts
python eval_3dfront.py --exp .../all --epoch 180 --visualize False --evaluate_diversity False --num_samples 5 --gen_shape False --no_stool True
```

##  Citation

Please cite our work if you find this code is useful in your research.
```
@inproceedings{Wei_2025_Planner3d,
author    = {Yao Wei and Martin Renqiang Min and George Vosselman and Li Erran Li and Michael Ying Yang},
title     = {Planner3D: LLM-enhanced Graph Prior Meets 3D Indoor Scene Explicit Regularization},
booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
publisher = {IEEE},
year      = {2025}
}
```

## Acknowledgements
We build our code based on the following codebases, many thanks to the contributors.

[Graph-to-3D](https://github.com/he-dhamo/graphto3d) [Dhamo et al., ICCV'21]
[SDFusion](https://github.com/yccyenchicheng/SDFusion) [Cheng et al., CVPR'23]
[CommonScenes](https://github.com/ymxlzgy/commonscenes) [Zhai et al., NeurIPS'23]
