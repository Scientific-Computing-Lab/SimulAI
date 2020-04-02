# [**QATM**: Quality-Aware Template Matching for Deep Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Cheng_QATM_Quality-Aware_Template_Matching_for_Deep_Learning_CVPR_2019_paper.pdf)


***
We used the QATM DNN layer (CVPR2019). For method details, please refer to 

```
 @InProceedings{Cheng_2019_CVPR,
    author = {Cheng, Jiaxin and Wu, Yue and AbdAlmageed, Wael and Natarajan, Premkumar},
    title = {QATM: Quality-Aware Template Matching for Deep Learning},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2019}
    }
```
See git repository `https://github.com/kamata1729/QATM_pytorch.git`

# Overview
## What is QATM?
QATM is an algorithmic DNN layer that implements template matching idea with learnable parameters.

## How does QATM work?
QATM learns the similarity scores reflecting the (soft-)repeatness of a pattern. Emprically speaking, matching a background patch in QATM will produce a much lower score than matching a foreground patch. 


# Dependencies
- Dependencies in our experiment, not necessary to be exactly same version but later version is preferred
- keras=2.2.4
- tensorflow=1.9.0
- opencv=4.0.0 (opencv>=3.1.0 should also works)

# The use of QATM in SimulAI:
We use QATM to find templates of experiment images in the RayleAI Database. We used the pre-trained model (as you can see in models.py) to conduct template matching queries. 
In order to run template matching queries, run the following jupyter notebook:
[`run_all_templates_experiment.ipynb`](./run_all_templates_experiment.ipynb)

# Other files:

The files: __init__.py, _config.yml, models.py, utils.py are necessary f
for running the jupyter notebook.

Files in QATM_Analysis directory:
qatm.py - used to arrange the results of the jupyter running into json files.
qatm_json_to_graph.py - takes the jsons, and creates graphs of qatm score with clustering results
qatm_clustering.py - responsible for clustering (PCA, k-means)
