# Utils

Utility files used to produce the results in the [paper]

## Files

1. convert_file_name.py - Splits the RayleAI dataset in to two seperate folders containing train and valid by folder. Note it only takes the 50 first time frames.
2. simulai_graph.py - General script that produces the graphs shown in the paper. In addition it contains the logic of the Physical Loss methodology.
3. simulai_graph_infogan.py - same as simulai_graph.py but dedicated for InfoGAN results.
4. simulai_graph_lire.py - same as simulai_graph.py but dedicated for LIRE results.
5. simulai_graph_regression.py - same as simulai_graph.py but dedicated for pReg results.
6. valid_train.py - Splits the RayleAI dataset in to two seperate folders containing train and valid by file.

## Execute
1. convert_file_name.py:
`python convert_file_name.py -f WHOLE_DATASET_PATH -out_dir OUTPUT_DIRECTORY`

2. simulai_graph.py:
`python simulai_graph.py -f JSON_FORMAT_PATH`

3. simulai_graph_infogan.py:
`python simulai_graph_infogan.py -f JSON_FORMAT_PATH`

4. simulai_graph_lire.py:
`python simulai_graph_lire.py -f JSON_FORMAT_PATH`

5. simulai_graph_regression.py:
`python simulai_graph_regression.py -f JSON_FORMAT_PATH`

6. valid_train.py:
`python convert_file_name.py -f WHOLE_DATASET_PATH -out_dir OUTPUT_DIRECTORY`
