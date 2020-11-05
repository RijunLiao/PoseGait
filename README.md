# PoseGait
PoseGait is a model-based gait recognition method with body pose and human prior knowledge. Our model PoseGait exploits human 3D pose estimated from images by Convolutional Neural Network as the input feature for gait recognition. The 3D pose, defined by the 3D coordinates of joints of the human body, is invariant to view changes and other external factors of variation.

Link to paper:
- [A model-based gait recognition method with body pose and human
prior knowledge](http://r.web.umkc.edu/rlyfv/papers/poseGait.pdf)
Please contact with me (rijun.liao@gmail.com) if you have any questions. Thank you!

## Prerequisites

- Pytorch
- Python
- GPU

## Dataset & Preparation
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the 2D human pose keypoints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- Step 3: Extract the 3D human pose keypoints by using [3DHumanPose](https://github.com/flyawaychase/3DHumanPose)

## Train
```bash
python pose_train.py
```

## Extract Gait Feature
```bash
python pose_test.py
```

## Citation
Please cite these papers in your publications if it helps your research:
```
@article{liao2020model,
  title={A model-based gait recognition method with body pose and human prior knowledge},
  author={Liao, Rijun and Yu, Shiqi and An, Weizhi and Huang, Yongzhen},
  journal={Pattern Recognition},
  volume={98},
  pages={107069},
  year={2020},
  publisher={Elsevier}
}
@inproceedings{an2018improving,
  title={Improving gait recognition with 3D pose estimation},
  author={An, Weizhi and Liao, Rijun and Yu, Shiqi and Huang, Yongzhen and Yuen, Pong C},
  booktitle={Chinese Conference on Biometric Recognition},
  pages={137--147},
  year={2018},
  organization={Springer}
}
@inproceedings{liao2017pose,
  title={Pose-based temporal-spatial network (PTSN) for gait recognition with carrying and clothing variations},
  author={Liao, Rijun and Cao, Chunshui and Garcia, Edel B and Yu, Shiqi and Huang, Yongzhen},
  booktitle={Chinese Conference on Biometric Recognition},
  pages={474--483},
  year={2017},
  organization={Springer}
}
```
