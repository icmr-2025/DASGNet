<div align="center">
<h1>Direction-aware Attention and Semantic Guidance Network for Salient Object Detection in Optical Remote Sensing Images </h1>
</div>

## 📰 News
This project provides the code and results for DASGNet.

## ⭐ Abstract
Salient object detection in optical remote sensing images (ORSI-SOD) aims to automatically identify the most visually prominent objects or regions in remote sensing images. However, due to the diverse orientations and varying scales of salient objects, as well as cluttered backgrounds, it remains a challenging task. To tackle these issues, we propose a direction-aware attention and semantic guidance network (DASGNet), a novel framework designed to enhance sensitivity to both orientation and multi-scale information while improving the depiction of boundaries in complex scenes. DASGNet integrates two key modules: a multi-scale direction-aware attention module (MDAM) and a semantic-guided edge reconstruction module (SERM). MDAM combines the attention mechanism with orientation information, effectively suppressing redundant information while capturing multi-scale orientation features. SERM employs 3D convolution to construct a stereoscopic receptive field, facilitating the integration of high-level semantic information across scales to guide the reconstruction of low-level texture information and thereby achieving precise edge delineation, particularly in complex scenes. Extensive experiments on three benchmark datasets demonstrate that DASGNet outperforms 14 state-of-the-art methods, achieving significant improvements in both accuracy and precision.

## 🌏 Network Architecture
   <div align=center>
   <img src="https://github.com/ICMR-2025/DASGNet/blob/main/images/DASGNet.png">
   </div>
Overview of DASGNet. The overall architecture is shown in the upper left. It consists of two key modules: a multi-scale direction-aware attention module (MDAM) and a semantic-guided edge reconstruction module (SERM). MDAM is composed of a direction-aware spatial attention block (DSAB), a direction-aware channel self-attention block (DCSAB), and a multi-scale convolution block (MCB). SERM is composed of a multi-dimensional semantic integration block (MSIB) and a dynamic edge semantic fusion block (DESFB).


   <div align=center>
   <img src="https://github.com/ICMR-2025/DASGNet/blob/main/images/SERM.png">
   </div>
Illustrations of the proposed SERM.
   
## 🖥️ Requirements
   python 3.8 + pytorch 1.9.0
   
## 🚀 Training
   1. Modify paths of datasets, then run train.py.

Note: Our main model is under './model/DASGNet.py'
      We provided an augmented dataset for training and the original dataset for testing.

## 🛸 Testing
   1. Modify paths of pre-trained models and datasets.

   2. Run test.py.

## 🖼️ Quantitative comparison
   <div align=center>
   <img src="https://github.com/ICMR-2025/DASGNet/blob/main/images/table.png">
   </div>
   
## 🌃 Visualization
   <div align=center>
   <img src="https://github.com/ICMR-2025/DASGNet/blob/main/images/Visualization.png">
   </div>
