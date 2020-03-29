# Files
1. extract_file.py is used for extracting data with specific parameters from the whole database
2. valid_train.py is used for separating the data to train and validation sets
3. data_prepare.py used for inner data preparation
3. params_regression.py used for training the network
4. prediction.py used for predicting from the network
5. simulai_graph_regression.py used for the physical loss visualization graphs

# Training

To train the network on all the images: 
`python3 params_regression.py --model_name="<model name without file extension>" --data="data_input_path/" --train --params=<regressed parameters, we used "at, time">`

To calculate similarity for randomly picked images:
`python3 params_regression.py --model_name="<model name without file extension>" --data="data_input_path/" --similarity --params=<regressed parameters, we used "at, time">`

For parameters prediction:
`python3 params_regression.py --model_name="<model name without file extension>" --data="data_input_path/" --predict --params=<regressed parameters, we used "at, time">`


