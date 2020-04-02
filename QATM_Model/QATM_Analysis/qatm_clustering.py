from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  #only needed for 3D plots
from sklearn.cluster import KMeans
import os, sys
import json
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import xlrd
np.set_printoptions(threshold=sys.maxsize)

class QATM_Clustering:

    def __init__(self, dataset, real_path):
        self.dataset = dataset
        self.real_path = real_path
    ############################################################################################################
    def clustering(self):
        self.dataset = np.asarray(self.dataset, dtype=np.float32)
        self.dataset = self.dataset / 255.0

        df = pd.DataFrame(self.dataset.reshape(len(self.dataset), -1))
        X_std = StandardScaler().fit_transform(df)
        pca = PCA(n_components=260)
        principalComponents = pca.fit_transform(X_std)
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='black')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.show()
        PCA_components = pd.DataFrame(principalComponents)
        model = KMeans(n_clusters=4)
        model.fit(PCA_components.iloc[:, :3])
        print("done clustering process")
        return model.labels_
