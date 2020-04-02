# SimulAI: Complete Deep Computer-Vision Methodology for Investigating Hydrodynamic Instabilities
This repository contains the official source code used to produce the results reported in the following papers:
[ourpaper]
All models, images and data can be found in this [URL](https://drive.google.com/drive/folders/1OlS5ZuTunQlkYFN0bHJczLQoNC_Gqcgr)
 .
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

<img src="https://user-images.githubusercontent.com/27349725/78000356-d2cb6b00-733c-11ea-831d-0a9b5342673a.jpg" alt=Rayleigh-Taylor Instability>


## 2. RayleAI Database
The first model is the state-of-the-art database - RayleAI can be found and downloaded [here] (https://drive.google.com/drive/folders/1OlS5ZuTunQlkYFN0bHJczLQoNC_Gqcgr). The database contains thresholded images from a simulation of a simple single-mode RTI perturbation with a resolution of 64x128 cells, 2.7cm in x axis and 5.4cm in y axis, while each fluid follows the equation of state of an ideal gas. The simulation input consists of three free parameters: Atwood number, gravity and the amplitude of the perturbation. The database contains of 101,250 images produced by 1350 different simulations (75 frames each) with unique pair of the free parameters. The format of the repository is built upon directories, each represents a simulation execution with the directory name indicating the parameters of the execution.
| Parameter | From | To  | Stride |           
|---------- | ---- | --- | ------ |
| Atwood    | 0.02 | 0.5 | 0.02   | 
| Gravity   | 600  | 800 | 25     |
| Amplitude | 0.1  | 0.5 | 0.1    |
| X         | 2.7  | 2.7 | 0      |
| Y         | 5.4  | 5.4 | 0      |


## 3. LIRE

LIRE is a library that provides a way to retrieve images from databases based on color and texture characteristics among other classic features. LIRE creates a Lucene index of image features using both local and global methods. For the evaluation of the similarity of two images, one can calculate their distance in the space they were indexed to. Many state-of-the-art methods for extracting features can be used, such as Gabor Texture Features, Tamura Features, or FCTH. For our purposes, we found that the Tamura Features method is better than the other methods that LIRE provides as it indexes RayleAI images in a more dispersed fashion. The Tamura feature vector of an image is an 18 double values descriptor that represents texture features in the image that correspond to human visual perception.

## 4. QATM

Quality-Aware Template Matching (QATM) method is a standalone template matching algorithm and a trainable layer with trainable parameters that can be used in a Deep Neural Network. QATM is inspired by assessing the matching quality of the source and target templates. It defines the <img src="https://render.githubusercontent.com/render/math?math=QATM(t,s)"> - measure as the product of likelihoods that a patch <img src="https://render.githubusercontent.com/render/math?math=s"> in <img src="https://render.githubusercontent.com/render/math?math=S"> is matched in <img src="https://render.githubusercontent.com/render/math?math=T"> and a patch <img src="https://render.githubusercontent.com/render/math?math=t"> in <img src="https://render.githubusercontent.com/render/math?math=T"> is matched in <img src="https://render.githubusercontent.com/render/math?math=S"> . Once <img src="https://render.githubusercontent.com/render/math?math=QATM(t, s)"> is computed, we can compute the template matching map for the template image <img src="https://render.githubusercontent.com/render/math?math=T"> and the target search image <img src="https://render.githubusercontent.com/render/math?math=S">. Eventually, we can find the best-matched region <img src="https://render.githubusercontent.com/render/math?math={R^*}"> which maximizes the overall matching quality. Therefore, the technique is of great need when templates are complicated and targets are noisy. Thus most suitable for RTI images from simulations and experiments.  

## 5. InfoGAN

Generative Advreserial Networks (GANs) is a framework capable to learn  network <img src="https://render.githubusercontent.com/render/math?math=G">, that transforms noise variable z from some noise distribution into a generated sample <img src="https://render.githubusercontent.com/render/math?math=G(z)">, while training the generator is optimized against a discriminator network <img src="https://render.githubusercontent.com/render/math?math=D">, which targets to distinguish between real samples with generated ones. The fruitful competition of both <img src="https://render.githubusercontent.com/render/math?math=G"> and <img src="https://render.githubusercontent.com/render/math?math=D">, in the form of MinMax game, allows <img src="https://render.githubusercontent.com/render/math?math=G"> to generate samples such that <img src="https://render.githubusercontent.com/render/math?math=D"> will have difficulty with distinguishing real samples between them. The ability to generate indistinguishable new data in an unsupervised manner is one example of a machine learning approach that is able to understand an underlying deep, abstract and generative representation of the data. Information Maximizing Generative Adversarial Network (InfoGAN) utilizes latent code variables <img src="https://render.githubusercontent.com/render/math?math=C_i">, which are added to the noise variable. These noise variables are randomly generated, although from a user-specified domain.

### 6. pReg
Many Deep Learning techniques obtain state-of-the-art results for regression tasks, in a wide range of CV applications: Pose Estimation, Facial Landmark Detection, Age Estimation, Image Registration and Image Orientation. Most of the deep learning architectures used for regression tasks on images are Convolutional Neural Networks (ConvNets), which are usually composed of blocks of Convolutional layers followed by a Pooling layer, and finally Fully-Connected layers. The dimension of the output layer depends on the task, and its activation function is usually linear or sigmoid. ConvNets can be used for retrieving the parameters of an experiment image, via regression.
<img src="https://user-images.githubusercontent.com/27349725/78252321-559a2480-74fb-11ea-8e65-870412691355.png" alt=On the left the experiment input image and on the right the simulation output image with its parameters  width="400px" height="400px">

## 4. Pre-trained models

### 4.1. Download links

|  Dataset  |              Model              | Input Size | mAHP@250 | Balanced Accuracy |
|-----------|---------------------------------|:----------:|---------:|------------------:|
| CIFAR-100 | [Plain-11][16]                  |    32x32   |   82.05% |            74.10% |
| CIFAR-100 | [ResNet-110-wfc][17]            |    32x32   |   83.29% |            76.60% |
| CIFAR-100 | [PyramidNet-272-200][18]        |    32x32   |   86.38% |            80.49% |
| NABirds   | [ResNet-50 (from scratch)][19]  |   224x224  |   73.99% |            59.46% |
| NABirds   | [ResNet-50 (fine-tuned)][20]    |   224x224  |   81.46% |            69.49% |
| NABirds   | [ResNet-50 (from scratch)][27]  |   448x448  |   82.33% |            70.43% |
| NABirds   | [ResNet-50 (fine-tuned)][28]    |   448x448  |   88.11% |            76.79% |
| CUB       | [ResNet-50 (from scratch)][25]  |   448x448  |   83.33% |            70.14% |
| CUB       | [ResNet-50 (fine-tuned)][26]    |   448x448  |   92.24% |            80.23% |
| ILSVRC    | [ResNet-50][21] *               |   224x224  |   83.15% |            70.42% |

<p style="font-size: 0.8em">
* This is an updated model with slightly better performance than reported in the paper (~1 percent point).
The original model can be obtained <a href="https://github.com/cvjena/semantic-embeddings/releases/download/v1.0.0/imagenet_unitsphere-embed+cls_rn50.model.h5">here</a>.
</p>

### 4.2. Pre-processing

The pre-trained models provided above assume input images to be given in RGB color format and standardized by subtracting a dataset-specific channel-wise mean and dividing by a dataset-specific standard deviation.
The means and standard deviations for each dataset are provided in the following table.

|            Dataset          |                     Mean                     |            Standard Deviation            |
|-----------------------------|----------------------------------------------|------------------------------------------|
| CIFAR-100                   | `[129.30386353, 124.06987, 112.43356323]`    | `[68.17019653, 65.39176178, 70.4180603]` |
| NABirds (from scratch)      | `[125.30513277, 129.66606421, 118.45121113]` | `[57.0045467, 56.70059436, 68.44430446]` |
| CUB (from scratch)          | `[123.82988033, 127.35116805, 110.25606303]` | `[59.2230949, 58.0736071, 67.80251684]`  |
| ILSVRC & fine-tuned models  | `[122.65435242, 116.6545058, 103.99789959]`  | `[71.40583196, 69.56888997, 73.0440314]` |

### 4.3. Troubleshooting

Sometimes, loading of the pre-trained models fails with the error message "unknown opcode".
In the case of this or other issues, you can still create the architecture yourself and load the pre-trained weights from the model files provided above.
For CIFAR-100 and the `resnet-110-wfc` architecture, for example, this can be done as follows:

```python
import keras
import utils
from learn_image_embeddings import cls_model

model = utils.build_network(100, 'resnet-110-wfc')
model = keras.models.Model(
    model.inputs,
    keras.layers.Lambda(utils.l2norm, name = 'l2norm')(model.output)
)
model = cls_model(model, 100)

model.load_weights('cifar_unitsphere-embed+cls_resnet-110-wfc.model.h5')
```


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
