# SimulAI: Complete Deep Computer-Vision Methodology for Investigating Hydrodynamic Instabilities
This repository contains the official source code used to produce the results reported in the following papers:
[ourpaper]
All models, images and data can be found in this [URL](https://drive.google.com/drive/folders/1OlS5ZuTunQlkYFN0bHJczLQoNC_Gqcgr).
If you use this code, please cite one of those papers (the first one when you work with hierarchy-based semantic embeddings, the second one when you use the cosine loss for classification).
The remainder of this ReadME will contain explanation on the work, database, source codes. Whilst each folder will contain how to run the specific model.



<details><summary><strong>Table of Contents</strong></summary>

1. [Rayleigh-Taylor Instability and Significance](#1-rayleigh-taylor-instability-and-significance)
2. [RayleAI - RTI Database](#2-rayleai-rti-database)
3. [LIRE](#3-lire)
4. [QATM](#4-qatm)
5. [InfoGAN](#5-infogan)
6. [pReg](#5-pReg)
7. [PredRNN](#5-predrnn)

</details>


## 1. Rayleigh-Taylor Instability and Significance

In fluid dynamics, one of the most important research fields is hydrodynamic instabilities and their evolution in different flow regimes. The investigation of said instabilities is concerned with the highly non-linear dynamics. Currently, three main methods are used for the understanding of such phenomenon - namely analytical models, experiments and simulations - and all of them are primarily investigated and correlated using human expertise. In this work we claim and demonstrate that a major portion of this research effort could and should be analysed using recent breakthrough advancements in the field of Computer Vision with Deep Learning (CVDL or Deep Computer-Vision). Specifically, we target and evaluate specific state-of-the-art techniques - such as Image Retrieval, Template Matching, Parameters Regression and Spatiotemporal Prediction - for the quantitative and qualitative benefits they provide. In order to do so we focus in this research on one of the most representative instabilities, the Rayleigh-Taylor one, simulate its behaviour and create an open-sourced state-of-the-art annotated database RayleAI. Finally, we use adjusted experimental results and novel physical loss methodologies to validate the correspondence of the predicted results to actual physical reality to prove the models correctness.
The techniques which were developed and proved in this work can be served as essential tools for physicists in the field of hydrodynamics for investigating a variety of physical systems, and also could be used via Transfer Learning to other instabilities research. A part of the techniques can be easily applied on already exist simulation results.

<p align="center" style="text-align:center">
<img src="https://user-images.githubusercontent.com/27349725/78000356-d2cb6b00-733c-11ea-831d-0a9b5342673a.jpg" alt=Rayleigh-Taylor Instability>
</p>

<p align="center" style="text-align:center">
Rayleigh-Taylor Instability.
</p>

## 2. RayleAI Database
The first model is the state-of-the-art database - RayleAI can be found and downloaded executing the following command:

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=156_GlmdF3jKgBaToc8eYYUTf9_bw7jlj' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=156_GlmdF3jKgBaToc8eYYUTf9_bw7jlj" -O RayleAI.tar.gz && rm -rf /tmp/cookies.txt
```

or simply download in this [URL](https://drive.google.com/drive/folders/1YGPY17bej0OzM3yyP4JZgR0xJW8KmAa-)
The database contains thresholded images from a simulation of a simple single-mode RTI perturbation with a resolution of 64x128 cells, 2.7cm in x axis and 5.4cm in y axis, while each fluid follows the equation of state of an ideal gas. The simulation input consists of three free parameters: Atwood number, gravity and the amplitude of the perturbation. The database contains of 101,250 images produced by 1350 different simulations (75 frames each) with unique pair of the free parameters. The format of the repository is built upon directories, each represents a simulation execution with the directory name indicating the parameters of the execution.
<p align="center" style="text-align:center">
| Parameter | From | To  | Stride |           
|---------- | ---- | --- | ------ |
| Atwood    | 0.02 | 0.5 | 0.02   | 
| Gravity   | 600  | 800 | 25     |
| Amplitude | 0.1  | 0.5 | 0.1    |
| X         | 2.7  | 2.7 | 0      |
| Y         | 5.4  | 5.4 | 0      |
</p>

## 3. LIRE

LIRE is a library that provides a way to retrieve images from databases based on color and texture characteristics among other classic features. LIRE creates a Lucene index of image features using both local and global methods. For the evaluation of the similarity of two images, one can calculate their distance in the space they were indexed to. Many state-of-the-art methods for extracting features can be used, such as Gabor Texture Features, Tamura Features, or FCTH. For our purposes, we found that the Tamura Features method is better than the other methods that LIRE provides as it indexes RayleAI images in a more dispersed fashion. The Tamura feature vector of an image is an 18 double values descriptor that represents texture features in the image that correspond to human visual perception.

<p align="center" style="text-align:center">
<img src="https://user-images.githubusercontent.com/27349725/78253782-93984800-74fd-11ea-80ef-ba850b4b62dd.png" height="300px">
</p>

<p align="center" style="text-align:center">
LIRE results with a new method evaluation - "Physical loss" (Smaller y-value is better).
</p>

Instructions on how installation requirments, execution and more can be found in this [folder](https://github.com/scientific-computing-nrcn/SimulAI/tree/master/LIRE_Model) inside the git repository


## 4. QATM

Quality-Aware Template Matching (QATM) method is a standalone template matching algorithm and a trainable layer with trainable parameters that can be used in a Deep Neural Network. QATM is inspired by assessing the matching quality of the source and target templates. It defines the <img src="https://render.githubusercontent.com/render/math?math=QATM(t,s)"> - measure as the product of likelihoods that a patch <img src="https://render.githubusercontent.com/render/math?math=s"> in <img src="https://render.githubusercontent.com/render/math?math=S"> is matched in <img src="https://render.githubusercontent.com/render/math?math=T"> and a patch <img src="https://render.githubusercontent.com/render/math?math=t"> in <img src="https://render.githubusercontent.com/render/math?math=T"> is matched in <img src="https://render.githubusercontent.com/render/math?math=S"> . Once <img src="https://render.githubusercontent.com/render/math?math=QATM(t, s)"> is computed, we can compute the template matching map for the template image <img src="https://render.githubusercontent.com/render/math?math=T"> and the target search image <img src="https://render.githubusercontent.com/render/math?math=S">. Eventually, we can find the best-matched region <img src="https://render.githubusercontent.com/render/math?math={R^*}"> which maximizes the overall matching quality. Therefore, the technique is of great need when templates are complicated and targets are noisy. Thus most suitable for RTI images from simulations and experiments. 

<p align="center" style="text-align:center">
<img src="https://user-images.githubusercontent.com/27349725/78253281-cee64700-74fc-11ea-9ae3-261c04316b3f.png" height="300px">
</p>

<p align="center" style="text-align:center">
PCA and k-means clustering methodology made on QATM results.
</p>

Instructions on how installation requirments, execution and more can be found in this [folder](https://github.com/scientific-computing-nrcn/SimulAI/tree/master/QATM_Model) inside the git repository

## 5. InfoGAN

Generative Advreserial Networks (GANs) is a framework capable to learn  network <img src="https://render.githubusercontent.com/render/math?math=G">, that transforms noise variable z from some noise distribution into a generated sample <img src="https://render.githubusercontent.com/render/math?math=G(z)">, while training the generator is optimized against a discriminator network <img src="https://render.githubusercontent.com/render/math?math=D">, which targets to distinguish between real samples with generated ones. The fruitful competition of both <img src="https://render.githubusercontent.com/render/math?math=G"> and <img src="https://render.githubusercontent.com/render/math?math=D">, in the form of MinMax game, allows <img src="https://render.githubusercontent.com/render/math?math=G"> to generate samples such that <img src="https://render.githubusercontent.com/render/math?math=D"> will have difficulty with distinguishing real samples between them. The ability to generate indistinguishable new data in an unsupervised manner is one example of a machine learning approach that is able to understand an underlying deep, abstract and generative representation of the data. Information Maximizing Generative Adversarial Network (InfoGAN) utilizes latent code variables <img src="https://render.githubusercontent.com/render/math?math=C_i">, which are added to the noise variable. These noise variables are randomly generated, although from a user-specified domain.

<p align="center" style="text-align:center">
<img src="https://user-images.githubusercontent.com/27349725/78253637-621f7c80-74fd-11ea-938d-625842220a8a.png" height="300px">
</p>

<p align="center" style="text-align:center">
InfoGAN results with a new method evaluation - "Physical loss" (Smaller y-value is better).
</p>

Instructions on how installation requirments, execution and more can be found in this [folder](https://github.com/scientific-computing-nrcn/SimulAI/tree/master/InfoGAN_Model) inside the git repository

## 6. pReg
Many Deep Learning techniques obtain state-of-the-art results for regression tasks, in a wide range of CV applications: Pose Estimation, Facial Landmark Detection, Age Estimation, Image Registration and Image Orientation. Most of the deep learning architectures used for regression tasks on images are Convolutional Neural Networks (ConvNets), which are usually composed of blocks of Convolutional layers followed by a Pooling layer, and finally Fully-Connected layers. The dimension of the output layer depends on the task, and its activation function is usually linear or sigmoid. ConvNets can be used for retrieving the parameters of an experiment image, via regression.

<p align="center" style="text-align:center">
<img src="https://user-images.githubusercontent.com/27349725/78252321-559a2480-74fb-11ea-8e65-870412691355.png" alt=On the left the experiment input image and on the right the simulation output image with its parameters height="400px">
</p>

<p align="center" style="text-align:center">
On the left the experiment input image and on the right the simulation output image with its parameters
</p>

Instructions on how installation requirments, execution and more can be found in this [folder](https://github.com/scientific-computing-nrcn/SimulAI/tree/master/pReg_Model) inside the git repository

## 7. PredRNN
PredRNN is a state-of-the-art Recurrent Neural Network for predictive learning using LSTMs. PredRNN memorizes both spatial appearances and temporal variations in a unified memory pool. Unlike standard LSTMs, and in addition to the standard memory transition within them, memory in PredRNN can travel through the whole network in a zigzag direction, therefore from the top unit of some time step to the bottom unit of the other. Thus, PredRNN is able to preserve the temporal as well as the spatial memory for long-term motions. In this work, we use PredRNN for predicting future time steps of simulations as well as experiments, based on the given sequence of time steps.

<img src="https://user-images.githubusercontent.com/27349725/78253397-02c16c80-74fd-11ea-9c7c-c565553ce631.png">

PredRNN prediction on a simulation and an experiment

Instructions on how installation requirments, execution and more can be found in this [folder](https://github.com/scientific-computing-nrcn/SimulAI/tree/master/predrnn_Model) inside the git repository

