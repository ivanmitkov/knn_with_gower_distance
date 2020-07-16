# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:54:25 2020

@author: ivan.mitkov
"""

# Make Predictions with k-nearest neighbors on the Iris Flowers Dataset
import gower
from csv import reader
from math import sqrt
from sklearn import datasets
from collections import Counter
import pandas as pd
import numpy as np

class knn():

    """
    The class calculates the similarity distance for mixed type of data (continious and nominal)
    and finds the lines with most similar observations.
    """
    def __init__(self,
                 train_df,
                 test_df
                 ):
        self.train_df = train_df
        self.test_df = test_df
        
    def ignore_class(self,
                     name_of_class_column):
        self.class_index = self.train_df[[name_of_class_column]]
        #print(self.class_index )
        self.class_ignored_df = self.train_df.drop(columns = [name_of_class_column])
        
    def find_nearest_neighbors(self,
                               number_of_neighbors
                               ):
        predictions_df = pd.DataFrame()
        for index, row in self.test_df.iterrows():
            row = pd.DataFrame([row], columns=list(self.test_df))
            row.columns = list(self.test_df)
            nearest_index = list(gower.gower_topn(pd.DataFrame(row), self.class_ignored_df, n = number_of_neighbors)['index'])
            output = list(self.class_index.iloc[nearest_index, 0])
            result = pd.DataFrame(dict((x, output.count(x)/number_of_neighbors) for x in set(output)), index = [index])
            predictions_df = predictions_df.append(result)
            print('for row: ', index, 'the most similar lines are ', nearest_index)
        return predictions_df
        
    if __name__ == "__main__":
        iris = datasets.load_iris()
        iris_df = pd.DataFrame(iris.data)
        iris_df.columns = iris['feature_names']
        iris_df['target'] = iris.target
        train_ds = iris_df.sample(100).reset_index(drop = True)
        X_test = iris_df.sample(3).reset_index(drop = True)
        y_test = X_test['target']
        X_test = X_test.drop(columns = ['target'])
        class_test = knn(train_ds, X_test)
        class_test.ignore_class('target')
        class_test.find_nearest_neighbors(3)