# SimulAI: Complete Deep Computer-Vision Methodology for Investigating Hydrodynamic Instabilities
This repository contains the official source code used to produce the results reported in the following papers:
[ourpaper]
All models, images and data can be found in this URL: https://drive.google.com/drive/folders/1OlS5ZuTunQlkYFN0bHJczLQoNC_Gqcgr.
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

![Rayleigh-Taylor Instability](https://user-images.githubusercontent.com/27349725/78000356-d2cb6b00-733c-11ea-831d-0a9b5342673a.jpg)


## 2. RayleAI Database
The first model is the state-of-the-art database - RayleAI can be found and downloaded here https://drive.google.com/drive/folders/1OlS5ZuTunQlkYFN0bHJczLQoNC_Gqcgr. The database contains thresholded images from a simulation of a simple single-mode RTI perturbation with a resolution of 64x128 cells, 2.7cm in x axis and 5.4cm in y axis, while each fluid follows the equation of state of an ideal gas. The simulation input consists of three free parameters: Atwood number, gravity and the amplitude of the perturbation. The database contains of 101,250 images produced by 1350 different simulations (75 frames each) with unique pair of the free parameters. The format of the repository is built upon directories, each represents a simulation execution with the directory name indicating the parameters of the execution.
| Parameter | From | To  | Stride |           
|---------- | ---- | --- | ------ |
| Atwood    | 0.02 | 0.5 | 0.02   | 
| Gravity   | 600  | 800 | 25     |
| Amplitude | 0.1  | 0.5 | 0.1    |
| X         | 2.7  | 2.7 | 0      |
| Y         | 5.4  | 5.4 | 0      |


## 3. LIRE

LIRE is a library that provides a way to retrieve images from databases based on color and texture characteristics among other classic features. LIRE creates a \textit{Lucene} index of image features using both local and global methods. For the evaluation of the similarity of two images, one can calculate their distance in the space they were indexed to. Many state-of-the-art methods for extracting features can be used, such as Gabor Texture Features \cite{zhang2000content}, Tamura Features \cite{tamurafeatures}, or FCTH \cite{chatzichristofis2008fcth}. For our purposes, we found that the Tamura Features method is better than the other methods that LIRE provides as it indexes \textit{RayleAI} images in a more dispersed fashion. The Tamura feature vector of an image is an 18 double values descriptor that represents texture features in the image that correspond to human visual perception.

## 4. QATM
One variation of the Template Matching problem is defined as follows: Given an exemplar image <img src="https://render.githubusercontent.com/render/math?math=T">, find the most similar region of interest in a target image <img src="https://render.githubusercontent.com/render/math?math=S"> \cite{brunelli2009template}. Classic template matching methods often use Sum-of-Squared Differences (SSD) or Normalized Cross-Correlation (NCC) to asses the similarity score between a template and an underlying image. These approaches work well when the transformation between the template and the target search image is simple. However, with non-rigid transformations, which are common in real-life, they start to fail. Quality-Aware Template Matching (QATM) \cite{qatm} method is a standalone template matching algorithm and a trainable layer with trainable parameters that can be used in a Deep Neural Network. QATM is inspired by assessing the matching quality of the source and target templates. It defines the $QATM(t,s)$-measure as the product of likelihoods that a patch $s$ in $S$ is matched in $T$ and a patch $t$ in $T$ is matched in $S$. Once $QATM(t, s)$ is computed, we can compute the template matching map for the template image $T$ and the target search image $S$. Eventually, we can find the best-matched region ${R^*}$ which maximizes the overall matching quality. Therefore, the technique is of great need when templates are complicated and targets are noisy. Thus most suitable for RTI images from simulations and experiments.  

## 5. InfoGAN

Generative Advreserial Networks (GANs) is a framework capable to learn  network $G$, that transforms noise variable z from some noise distribution into a generated sample G(z), while training the generator is optimized against a discriminator network D, which targets to distinguish between real samples with generated ones. The fruitful competition of both G and D, in the form of MinMax game, allows $G$ to generate samples such that D will have difficulty with distinguishing real samples between them. The ability to generate indistinguishable new data in an unsupervised manner is one example of a machine learning approach that is able to understand an underlying deep, abstract and generative representation of the data. Information Maximizing Generative Adversarial Network (InfoGAN) utilizes latent code variables Ci, which are added to the noise variable. These noise variables are randomly generated, although from a user-specified domain.
The latent variables impose an Information Theory Regularization term to the optimization problem, which forces G to preserve the information stored in ci through the generation process. This allows learning interpretative and meaningful representations of the data, with a negligible computation cost, on top of a GAN. The high-abstract-level representation can be extracted from the discriminator (e.g. the last layer before the classification) into a features vector. We use these features in order to measure the similarity between some input image to any other image, by applying some distance function (e.g. L2 norm) on the features of the input to the features of the other image. This methodology provides the ability to order images similarity to a given input image .

### 6. pReg
Many Deep Learning techniques obtain state-of-the-art results for regression tasks, in a wide range of CV applications: Pose Estimation, Facial Landmark Detection, Age Estimation, Image Registration and Image Orientation. Most of the deep learning architectures used for regression tasks on images are Convolutional Neural Networks (ConvNets), which are usually composed of blocks of Convolutional layers followed by a Pooling layer, and finally Fully-Connected layers. The dimension of the output layer depends on the task, and its activation function is usually linear or sigmoid. 

ConvNets can be used for retrieving the parameters of an experiment image, via regression. Our model consists of 3 Convolutional layers with 64 filters, with a kernel size 5 times 5, and with L2 regularization, each followed by a Max-Pooling layer, a Dropout of $0.1$ rate and finally Batch Normalization. Then, there are two Fully-Connected layers of 250 and 200 features, which are separated again by a Batch Normalization layer. Finally, the Output layer of our network has 2 features (as will described next), and is activated by sigmoid to prevent the exploding gradients problem. Since the most significant parameters for describing each image are Amplitude and Time - which \textit{pReg} is trained to predict - we used only a subset of \textit{RayleAI} for the training set, namely images with the following parameters: Atwood of [0.08, 0.5] (with a stride of 0.02), gravity of 625, 700, 750, 800, amplitude of 0.1, 0.5 (with a stride of 0.1 and T in [0.1, 0.6] (with a stride of 0.01). We fixed a small amount of values for Gravity and for Amplitude, so the network will not try and learn the variance that these parameters impose while expanding our database with as minimal noise as possible. We chose the value ranges of Atwood and Time to expose the model to images with both small and big perturbations, such that the amount of the latter ones will not be negligible. Our reduced training set consists of $\sim 16K$ images, and our validation set consists of $\sim 4K$ images. Nonetheless, for increasing generalization and for decreasing model overfitting, we employed data augmentation. Since there is high significance for the perspective from which each image is taken, the methods of data augmentation should be carefully chosen: Rotation, shifting and flipping methods may generate images such that the labels of the original parameters do not fit for them. Therefore, we augment our training set with only zooming in/out (zoom range=0.1) via TensorFlow preprocessing.

[1]: https://arxiv.org/pdf/1809.09924
[2]: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
[3]: https://www.cs.toronto.edu/~kriz/cifar.html
[4]: http://image-net.org/challenges/LSVRC/2012/
[5]: http://dl.allaboutbirds.org/nabirds
[6]: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
[7]: http://hera.inf-cv.uni-jena.de:6680/pdf/Barz18:GoodTraining
[8]: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
[9]: https://arxiv.org/pdf/1605.07146.pdf
[10]: https://ieeexplore.ieee.org/abstract/document/8100151
[11]: https://arxiv.org/pdf/1409.1556.pdf
[12]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
[13]: https://github.com/broadinstitute/keras-resnet
[14]: https://github.com/broadinstitute/keras-resnet/pull/47
[15]: https://arxiv.org/pdf/1707.07012.pdf
[16]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/cifar_unitsphere-embed+cls_plain11.model.h5
[17]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/cifar_unitsphere-embed+cls_resnet-110-fc.model.h5
[18]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/cifar_unitsphere-embed+cls_pyramidnet-272-200.model.h5
[19]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/nab_unitsphere-embed+cls_rn50.model.h5
[20]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/nab_unitsphere-embed+cls_rn50_finetuned.model.h5
[21]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/imagenet_unitsphere-embed+cls_rn50.model.h5
[22]: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
[23]: https://arxiv.org/pdf/1901.09054
[24]: CosineLoss.md
[25]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/cub_unitsphere-embed+cls_deep-hierarchy_rn50.model.h5
[26]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/cub_unitsphere-embed+cls_deep-hierarchy_rn50_finetuned.model.h5
[27]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/nab-large_unitsphere-embed+cls_rn50.model.h5
[28]: https://github.com/cvjena/semantic-embeddings/releases/download/v1.1.0/nab-large_unitsphere-embed+cls_rn50_finetuned.model.h5
[29]: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
[30]: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
[31]: https://github.com/visipedia/inat_comp/tree/2018
[32]: https://www.kaggle.com/c/inaturalist-2019-fgvc6/
