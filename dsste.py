import pandas as pd
import numpy as np
import math
import random
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours



def class_attribute(data, threshold):
    X, y = data.iloc[:,:-1], data.iloc[:,-1:]
    vcs = y.value_counts(normalize=True)
    print(vcs)
    
    minority_cls, majority_cls = [], []
    
    for idx, vc in zip(range(len(vcs)), vcs):
        if vc <= threshold:
            minority_cls.append(list(vcs.index[idx])[0])
        else:
            majority_cls.append(list(vcs.index[idx])[0])
    print(minority_cls, majority_cls)
    return minority_cls, majority_cls
    

def difficult_data(data):
    X, y = data.iloc[:,:-1], data.iloc[:,-1:]
    print('Original dataset shape %s' % Counter(y))
    data_df = pd.concat([X, y], axis=1)

    enn = EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=N, kind_sel='all', n_jobs=-1)
    X_res, y_res = enn.fit_resample(X, y)

    X_res = pd.DataFrame(X_res)
    y_res = pd.DataFrame(y_res)

    X_res = X_res.reset_index()
    y_res = y_res.reset_index()

    easy_df = pd.concat([X_res, y_res], axis=1)
    easy_df = easy_df.drop(['index'], axis=1)
    print('easy samples %s' % Counter(easy_df.iloc[:,-1:]))

    difficult_df = easy_df.append(data_df)
    difficult_df = easy_df.append(data_df)
    difficult_df = difficult_df.drop_duplicates(keep=False)
    print('difficult samples %s' % Counter(difficult_df.iloc[:,-1:]))
    return difficult_df

def compress_data(data):
    X, y = data.iloc[:,:-1], data.iloc[:,-1:]

    cc = ClusterCentroids(sampling_strategy='auto', random_state=None, estimator=None, voting='auto', n_jobs='deprecated')

    X_res, y_res = cc.fit_resample(X, y)

    X_res = pd.DataFrame(X_res)
    y_res = pd.DataFrame(y_res)

    X_res = X_res.reset_index()
    y_res = y_res.reset_index()

    compress_df = pd.concat([X_res, y_res], axis=1)
    compress_df = compress_df.drop(['index'], axis=1)

    print('Cluster Centroids samples %s' % Counter(compress_df.iloc[:,-1:]))
    return compress_df

def synthetic_data(data, value):    
    multiple = value / data.shape[0]
    weight_list = [] 
    n = N  

    # weight list
    for x in range(n, (int(multiple/2))+n):
        weight1 = 1 + (1/x)
        weight2 = 1 - (1/x)
        weight_list.append(weight1)
        weight_list.append(weight2)
        weight_list.sort()
    
    # synthete samples
    synthetic_data = pd.DataFrame()
    for weight in weight_list:
        '''
        随机选中 [0.25, 0.75] 的特征进行缩放
        '''
        data_sample = data.iloc[:,:-1].sample(n=random.randint(int(len(data.columns)*0.25), int(len(data.columns)*0.75)), random_state=None, axis=1)*(weight)
        data[data_sample.columns] = data_sample
        synthetic_data = synthetic_data.append(data, ignore_index=True)

    print('synthete samples %s' % Counter(synthetic_data.iloc[:,-1:]))
    return synthetic_data

def aug_data(origin_data, synthetic_data):
    return pd.concat([origin_data, synthetic_data], axis=0, ignore_index=True)


if __name__=="__main__":
    
    N=10

    # step 0): make dataset
    X, y = make_classification(n_classes=2, class_sep=2, 
            weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
            n_features=40, n_clusters_per_class=1, n_samples=1000, random_state=10)

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    data = pd.concat([X, y], axis=1)

    # step 1): get difficult samples
    difficult_data = get_difficult_samples(data)

    # step 2): get difficult samples
    difficult_data = get_difficult_samples(data)

    # step 3): compress difficult samples
    compress_data = compress_data(difficult_data, 10)

    # step 4): synthetic difficult samples
    synthetic_data = synthetic_data(difficult_data, 10)

    # step 5): synthetic difficult samples
    aug_data = aug_data(data, synthetic_data)