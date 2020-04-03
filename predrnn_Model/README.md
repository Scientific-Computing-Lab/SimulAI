
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
or alternativley:
`
python3 -u run.py \
    --device cuda\
    --is_training 1 \
    --dataset_name rayleai \
    --train_data_paths /home/reemh/Simulation_Resize_refactor/train/ \
    --valid_data_paths /home/reemh/Simulation_Resize_refactor/valid/ \
    --save_dir checkpoints/mnist_predrnn \
    --gen_frm_dir results/mnist_predrnn \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 64 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 65 \
    --num_hidden 64,64,64,64 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 1 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0003 \
    --batch_size 8 \
    --max_iterations 80000 \
    --display_interval 50 \
    --test_interval 5000 \
    --snapshot_interval 50
`

To get the predict images run:
`sh script/predrnn_predict.sh `

