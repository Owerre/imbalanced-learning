# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data pre-processing
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

# Helps with importing functions from different directory
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

# import custom class
from helper import log_transfxn as cf 

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class TransformationPipeline:
    """
    A class for transformation pipeline
    """

    def __init__(self):
        """
        Define parameters
        """

    def num_pipeline(self, train_feat_mtx, test_feat_mtx):
        """
        Transformation pipeline of data with only numerical variables

        Parameters
        ___________
        train_feat_mtx: Training feature matrix
        test_feat_mtx: Test feature matrix

        Returns
        __________
        Transformation pipeline and transformed data in array
        """
        # Create pipeline
        # num_pipeline = Pipeline([ ('log', cf.LogTransformer()),
        #                         ('std_scaler', StandardScaler()),
        #                         ])

        # Instantiate the class
        scaler = StandardScaler() 

        # Fit transform the training set
        X_train_scaled = scaler.fit_transform(train_feat_mtx)
        
        # Only transform the test set
        X_test_scaled = scaler.transform(test_feat_mtx)
        return scaler, X_train_scaled, X_test_scaled

    def complete_pipeline(self, train_feat_mtx, test_feat_mtx):
        """
        Transformation pipeline of data with both numerical and categorical 
        variables. This transformation returns a pandas dataframe

        Parameters
        ___________
        train_feat_mtx: Training feature matrix
        test_feat_mtx: Test feature matrix

        Returns
        __________
        Transformed data in Pandas DataFrame
        """
        # List of categorical attributes
        cat_attribs = list(train_feat_mtx.select_dtypes('O'))

        # List of numerical attributes
        num_attribs = list(train_feat_mtx.select_dtypes('number'))

        # Binarize the categorical attributes
        cat_attribs = [([cat], LabelBinarizer()) for cat in cat_attribs]

        # Power transform and Standardize the numerical attributes
        num_attribs = [([num], StandardScaler()) for num in num_attribs]

        # Build a dataframe mapper pipeline
        mapper = DataFrameMapper(cat_attribs + num_attribs, df_out = True)

        # Fit transform the training set
        X_train_scaled = mapper.fit_transform(train_feat_mtx)

        # Only transform the test set
        X_test_scaled = mapper.transform(test_feat_mtx)
        return X_train_scaled, X_test_scaled
  
    def complete_pipeline_2(self, train_feat_mtx, test_feat_mtx):
        """
        Transformation pipeline of data with both numerical and categorical 
        variables. This transformation returns a dense array

        Parameters
        ___________
        train_feat_mtx: Training feature matrix
        test_feat_mtx: Test feature matrix

        Returns
        __________
        Transformed data in array
        """
        # List of categorical attributes
        cat_attribs = list(train_feat_mtx.select_dtypes('O'))

        # List of numerical attributes
        num_attribs = list(train_feat_mtx.select_dtypes('number'))

        # Call numerical transformation pipepline
        num_pipeline_ = self.num_pipeline(train_feat_mtx.select_dtypes('number'),
                                            test_feat_mtx.select_dtypes('number'))[0]

        full_pipeline = ColumnTransformer([("num", num_pipeline_, num_attribs),
                                            ("cat", OneHotEncoder(), cat_attribs),
                                            ])

        # Fit transform the training set
        X_train_scaled = full_pipeline.fit_transform(train_feat_mtx)

        # Only transform the test set
        X_test_scaled = full_pipeline.transform(test_feat_mtx)
        return X_train_scaled, X_test_scaled

    def pca_plot_labeled(self, data_, labels, palette = None):
        """
        Dimensionality reduction of labeled data using PCA 

        Parameters
        __________
        data: scaled data
        labels: labels of the data
        palette: color list

        Returns
        __________
        Matplotlib plot of two component PCA
        """
        #PCA
        pca = PCA(n_components = 2)
        X_pca = pca.fit_transform(data_)

        # put in dataframe
        X_reduced_pca = pd.DataFrame(data = X_pca)
        X_reduced_pca.columns = ['PC1', 'PC2']
        X_reduced_pca['class'] = labels.reset_index(drop = True)

        # plot results
        plt.rcParams.update({'font.size': 15})
        plt.subplots(figsize = (12,8))
        sns.scatterplot(x = 'PC1', y = 'PC2', data = X_reduced_pca,
        hue = 'class', palette = palette)

        # axis labels
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.title("Dimensionality reduction")
        plt.legend(loc = 'best')
        # plt.savefig('../image/pca.png')
        plt.show()