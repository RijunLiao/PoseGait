# PoseGait
PoseGait is a Pose-based temporal-spatial network to handle with the carrying and clothing variations for gait recognition.
Link to paper:
- [Pose-based temporal-spatial network (PTSN) for gait recognition with carrying and clothing variations](http://r.web.umkc.edu/rlyfv/papers/2017_ccbr.pdf)

## Prerequisites

- Caffe
- Python
- GPU

## Dataset & Preparation
- Step 1: Download [CASIA-B Dataset](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
- Step 2: Extract the human pose keypoints by using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

## Train
```bash
sh run_lstm_RGB.sh
```

## Citation
Please cite these papers in your publications if it helps your research:
```
@inproceedings{liao2017pose,
  title={Pose-based temporal-spatial network (PTSN) for gait recognition with carrying and clothing variations},
  author={Liao, Rijun and Cao, Chunshui and Garcia, Edel B and Yu, Shiqi and Huang, Yongzhen},
  booktitle={Chinese Conference on Biometric Recognition},
  pages={474--483},
  year={2017},
  organization={Springer}
}
```

## Acknowledgments
This code was based on the source of [LRCN_activity_recognition](https://github.com/intel/caffe/tree/master/examples/LRCN_activity_recognition)
