# LIRE - Lucene Image Retrieval
LIRE (Lucene Image Retrieval) is an open source library for content based image retrieval, which means you can search for images that look similar. Besides providing multiple common and state of the art retrieval mechanisms LIRE allows for easy use on multiple platforms. LIRE is actively used for research, teaching and commercial applications. Due to its modular nature it can be used on process level (e.g. index images and search) as well as on image feature level. Developers and researchers can easily extend and modify LIRE to adapt it to their needs.

Most [recent documentation is found here at the github repo] (https://github.com/dermotte/LIRE/blob/master/src/main/docs/developer-docs/docs/index.md).

# LIRE in SimulAI:
We use LIRE to conduct image similarity queries for experiment/simulation images over the RayleAI Database.
We compared this tool with InfoGAN, a deep learning tool for image similarity. 

# Files and Directories: 

1) CBIR.java - this java code is responsible for indexing the entire database into a feature space and conduct the image similarity queries. The results are wrriten into a json file.

2) net/semanticmetadata/lire - this lib (by LIRE) is necessary to run CBIR.java. 