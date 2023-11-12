# Jointly Optimized Global-Local Visual Localization of UAVs

The current project page provides [pytorch](http://pytorch.org/) code that implements the following paper:

**Title:**      "Jointly Optimized Global-Local Visual Localization of UAVs"

**Authors:**     Haoling Li, Jiuniu Wang, Zhiwei Wei, [Wenjia Xu*](https://wenjia.ruantang.top/)

**Paper:**  [https://arxiv.org/pdf/2310.08082.pdf](https://arxiv.org/pdf/2310.08082.pdf)

**Abstract:**
Navigation and localization of UAVs present a challenge when global navigation satellite systems (GNSS) are disrupted and unreliable. Traditional techniques, such as simultaneous localization and mapping (SLAM) and visual odometry (VO), exhibit certain limitations in furnishing absolute coordinates and mitigating error accumulation. Existing visual localization methods achieve autonomous visual localization without error accumulation by matching with ortho satellite images. However, doing so cannot guarantee real-time performance due to the complex matching process. To address these challenges, we propose a novel Global-Local Visual Localization (GLVL) network. Our GLVL network is a two-stage visual localization approach, combining a large-scale retrieval module that finds similar regions with the UAV flight scene, and a fine-grained matching module that localizes the precise UAV coordinate, enabling real-time and precise localization. The training process is jointly optimized in an end-to-end manner to further enhance the model capability. Experiments on six UAV flight scenes encompassing both texture-rich and texture-sparse regions demonstrate the ability of our model to achieve the realtime precise localization requirements of UAVs. Particularly, our method achieves a localization error of only 2.39 meters in 0.48 seconds in a village scene with sparse texture features.

## Requirements

Python 3.8

PyTorch = 1.8.1

All experiments are performed with two Nvidia RTX4090 GPU.


## Prerequisites

-  Dataset
 Please download all datasets via this [link](https://cloud.tsinghua.edu.cn/d/cba28e0a39db4de2ba7b/). And change the path where your data is stored in the parser_joint.py and .yaml files. 

## Train
```python
python train.py
```

Tips:

- The model parameter configuration file can be modified via the parser_joint.py and .yaml files (mainly regarding SuperPoint). 
- If you want to further finetune a new dataset or continuous pre-training on an existing weight model, you can modify the path to best_model_state_dict in the train.py file to accomplish this process. 
- If you want to modify the training ratio between the retrieval module and the SuperPoint module, you can manually modify Line185 of the code in the train.py file. In fact, you can implement fine-grained module fusion in a more elegant way. 
- I apologize for not being able to provide pre-training weights for the model.

## Test

```python
python eval_joint.py
```
The test case code for all the  datasets claimed in the paper is currently provided in the superpoint.py file, and you can modify it yourself at Line48 in the eval_joint.py file.


```
@article{li2023jointly,
  title={Jointly Optimized Global-Local Visual Localization of UAVs},
  author={Li, Haoling and Wang, Jiuniu and Wei, Zhiwei and Xu, Wenjia},
  journal={arXiv preprint arXiv:2310.08082},
  year={2023}
}
```

The code is under construction. If you have problems, feel free to reach me at [li-hl23@mails.tsinghua.edu.cn](mailto:li-hl23@mails.tsinghua.edu.cn)

## Acknowledgment

We thank the following repos providing helpful components/functions in our work.

-  [pytorch-superpoint](https://github.com/eric-yyjau/pytorch-superpoint) 
-  [deep-visual-geo-localization-benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) 
-  [gps-denied-uav-localization](https://github.com/hmgoforth/gps-denied-uav-localization) 
