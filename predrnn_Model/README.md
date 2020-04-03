
# PredRNN
Based on:
` https://github.com/thuml/predrnn-pytorch `

PredRNN [[paper](https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms)], a recurrent network with *twisted and zigzag space-time memory cells* for video data. Given a sequence of previous frames, our model generates future frames for multiple timestamps.

## Installation

This project is using the following python packages:
```
Python=3.7
OpenCV==3.4
PyTorch==1.3
```

You will also need a dataset, for example the RayleAI database which can be downloaded [here](https://drive.google.com/drive/folders/1YGPY17bej0OzM3yyP4JZgR0xJW8KmAa-).

## Training

To train PredRNN on all images run: 
`sh scripts/predrnn_train.sh`

To get the predict images run:
`sh script/predrnn_predict.sh `

