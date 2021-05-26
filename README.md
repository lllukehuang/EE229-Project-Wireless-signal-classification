# **EE229-Project-Wireless-signal-classification**

#### Introduction

This project shows the complete process from signal cutting to classification. Deep learning method(faster rcnn, yolov4) also supported.

#### File structure (traditional method)

- stft_with_color.m : generate the spectrogram
- recombination.py: adjusting the spectrogram
- process.py: run segmentation and get the signal pieces 
- wash.m: zero-padding
- hog_kmeans.m: K-Means clustering

#### Wireless Signal Auto-Classification Dataset(WSAD)

- Number of data labels:

  | Class   | Wi-Fi | BLE  | Others |
  | ------- | ----- | ---- | ------ |
  | Numbers | 4018  | 1531 | 1715   |

- Scale: 2k images

- Some models' performance on this dataset:

  | Model       | mAP    | Inference Time |
  | ----------- | ------ | -------------- |
  | Faster RCNN | 82.64% | 57.7ms/img     |
  | YOLOv4      | 80.28% | 5.6ms/img      |
  

You can download the whole dataset at https://jbox.sjtu.edu.cn/l/oFjsbK.

#### Note

The main code of deep learning method references the [Detectron2](https://github.com/facebookresearch/detectron2/tree/master/detectron2) platform and [Darknet](https://github.com/AlexeyAB/darknet).