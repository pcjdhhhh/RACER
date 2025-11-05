# -*- coding: utf-8 -*-
import os
import warnings
import ClusterEnsembles as ce
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
import sklearn
from sklearn.metrics import pairwise_kernels
from sklearn import metrics
from sklearn.cluster import KMeans,SpectralClustering,AgglomerativeClustering
from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
#import matplotlib.pyplot as plt
from numba import njit, prange, vectorize
import pymetis
import random
import time
import umap

warnings.filterwarnings("ignore")




random.seed(2025)  
np.random.seed(2025)

datasets_df = pd.read_csv('https://www.cs.ucr.edu/~eamonn/time_series_data_2018/DataSummary.csv')
datasets128 = datasets_df['Name'].values


#在development_datasets
development_datasets = ['Beef', 'BirdChicken', 'Car', 'CricketX', 'CricketY', 'CricketZ', 'DistalPhalanxTW',
                        'ECG200', 'ECG5000', 'FiftyWords', 'Fish', 'FordA', 'FordB', 'Haptics', 'Herring',
                        'InsectWingbeatSound', 'ItalyPowerDemand', 'LargeKitchenAppliances', 'Lightning2',
                        'Lightning7', 'Meat', 'MedicalImages', 'MiddlePhalanxOutlineAgeGroup', 'OSULeaf',
                        'OliveOil', 'Phoneme', 'Plane', 'ProximalPhalanxOutlineCorrect', 'ProximalPhalanxTW',
                        'ScreenType', 'ShapeletSim', 'Strawberry', 'SwedishLeaf', 'SyntheticControl',
                        'ToeSegmentation1', 'Trace', 'UWaveGestureLibraryY', 'Wafer', 'WordSynonyms',
                        'Worms', 'Yoga']

varying_length_datasets = ['PLAID', 'AllGestureWiimoteX', 'AllGestureWiimoteY',
                   'AllGestureWiimoteZ', 'GestureMidAirD1', 'GestureMidAirD2',
                    'GestureMidAirD3', 'GesturePebbleZ1', 'GesturePebbleZ2',
                    'PickupGestureWiimoteZ', 'ShakeGestureWiimoteZ']

#difference between datasets with setdiff1d
validation_datasets = np.setdiff1d(datasets128, development_datasets)

#remove varying length datasets from validation datasets 把长度不一致的也删除了
validation_datasets = np.setdiff1d(validation_datasets, varying_length_datasets)

#Total num of valid validation datasets and label:
#print('num of valid validation datasets: ', len(validation_datasets))

#Total num of valid validation datasets and developement datasets:
#print('num of valid develompent datasets: ', len(development_datasets))

#all valid datasets (datasets128 except varying length datasets)
all_valid_datasets = np.setdiff1d(datasets128, varying_length_datasets)
#print('num of all valid datasets: ', len(all_valid_datasets))


#Datasetes with missing values
datasets_with_nans = ['DodgerLoopDay', 'DodgerLoopGame', 'DodgerLoopWeekend', 'MelbournePedestrian']

def fix_MelbournePedestrian(X, y):
    """
    Filters out elements from X and y where the data in X has length different than 24.

    Parameters:
    - X: List of 2D numpy arrays, where each 2D array has shape (1, N).
    - y: Numpy array of labels, where the ith label corresponds to the ith element in X.

    Returns:
    - filtered_data_np: 2D numpy array containing only the data with length 24.
    - filtered_labels_np: Numpy array containing the corresponding labels.
    """

    # Initialize empty lists for filtered data and labels
    filtered_data = []
    filtered_labels = []

    # Loop through the data and labels and filter out elements where the data length is not 24
    for i, arr in enumerate(X):
        if arr.shape[1] == 24:
            filtered_data.append(arr)
            filtered_labels.append(y[i])

    # Convert filtered data to a 2D NumPy array
    filtered_data_np = np.concatenate(filtered_data, axis=0)

    # Convert filtered labels to a NumPy array
    filtered_labels_np = np.array(filtered_labels)

    #if the number of unique values in y_clean is not equal to the number of unique values in y
    #then there are missing values in the labels. Print it out
    if len(np.unique(filtered_labels)) != len(np.unique(y)):
        print("There are missing values in the labels")

    return filtered_data_np, filtered_labels_np

def clean_nans(X, y):
    """
    Removes time series with missing values and their corresponding labels.

    Parameters:
    - X: List of 2D numpy arrays, where each 2D array has shape (1, N).
    - y: Numpy array of labels, where the ith label corresponds to the ith element in X.

    Returns:
    - X_clean: 2D numpy array containing only the data with length 24.
    - y_clean: Numpy array containing the corresponding labels.
    """

    # Find which time series contain missing values
    missing_values_indices = np.any(np.isnan(X), axis=1)

    # Remove time series with missing values and their corresponding labels
    X_clean = X[~missing_values_indices]
    y_clean = y[~missing_values_indices]

    #if the number of unique values in y_clean is not equal to the number of unique values in y
    #then there are missing values in the labels. Print it out
    if len(np.unique(y_clean)) != len(np.unique(y)):
        print("There are missing values in the labels")

    return X_clean, y_clean

def load_classification_with_fixes(dataset):
    """
    Loads a classification dataset from the UCR archive and applies fixes to it if required
    """

    if(dataset == 'MelbournePedestrian'):
        X, y = load_classification(dataset)
        X, y = fix_MelbournePedestrian(X, y)
        X, y = clean_nans(X, y)
    elif(dataset in datasets_with_nans):
        X, y = load_classification(dataset)
        X = np.squeeze(X)
        X, y = clean_nans(X, y)
    else:
        X, y = load_classification(dataset)
        X = np.squeeze(X)

    return X.astype(np.float32),y










#MiniROCKET 
@njit("float32[:](float32[:,:],int32[:],int32[:],float32[:])", fastmath = True, parallel = False, cache = True)
def _fit_biases(X, dilations, num_features_per_dilation, quantiles):

    num_examples, input_length = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    ###MODIFICATION
    indices = np.array((
       1, 3, 6, 1, 2, 7, 1, 2, 3, 0, 2, 3, 1, 4, 5, 0, 1, 3, 3, 5, 6, 0,
       1, 2, 2, 5, 8, 1, 3, 7, 0, 1, 8, 4, 6, 7, 0, 1, 4, 3, 4, 6, 0, 4,
       5, 2, 6, 7, 5, 6, 7, 0, 1, 6, 4, 5, 7, 4, 7, 8, 1, 6, 8, 0, 2, 6,
       5, 6, 8, 2, 5, 7, 0, 1, 7, 0, 7, 8, 0, 3, 5, 0, 3, 7, 2, 3, 8, 2,
       3, 4, 1, 4, 6, 3, 4, 5, 0, 3, 8, 4, 5, 8, 0, 4, 6, 1, 4, 8, 6, 7,
       8, 4, 6, 8, 0, 3, 4, 1, 3, 4, 1, 5, 7, 1, 4, 7, 1, 2, 8, 0, 6, 7,
       1, 6, 7, 1, 3, 5, 0, 1, 5, 0, 4, 8, 4, 5, 6, 0, 2, 5, 3, 5, 7, 0,
       2, 4, 2, 6, 8, 2, 3, 7, 2, 5, 6, 2, 4, 8, 0, 2, 7, 3, 6, 8, 2, 3,
       6, 3, 7, 8, 0, 5, 8, 1, 2, 6, 2, 3, 5, 1, 5, 8, 3, 6, 7, 3, 4, 7,
       0, 4, 7, 3, 5, 8, 2, 4, 5, 1, 2, 5, 2, 7, 8, 2, 4, 6, 0, 5, 6, 3,
       4, 8, 0, 6, 8, 2, 4, 7, 0, 2, 8, 0, 3, 6, 5, 7, 8, 1, 5, 6, 1, 2,
       4, 0, 5, 7, 1, 3, 8, 1, 7, 8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            _X = X[np.random.randint(num_examples)]   #这里有随机数

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

    return biases

def _fit_dilations(input_length, num_features, max_dilations_per_kernel):

    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((input_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, num_features = 10_000, max_dilations_per_kernel = 32):

    _, input_length = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(input_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    ###MODIFICATION
    quantiles = np.random.permutation(quantiles) #这里有随机

    biases = _fit_biases(X, dilations, num_features_per_dilation, quantiles)


    return dilations, num_features_per_dilation, biases

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
@vectorize("float32(float32,float32)", nopython = True, cache = True)
def _PPV(a, b):
    #print(a)
    if a > b:
        return 1
    else:
        return 0

@njit("float32[:,:](float32[:,:],Tuple((int32[:],int32[:],float32[:])))", fastmath = True, parallel = True, cache = True)
def transform(X, parameters):

    num_examples, input_length = X.shape

    dilations, num_features_per_dilation, biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    ###MODIFICATION
    indices = np.array((
       1, 3, 6, 1, 2, 7, 1, 2, 3, 0, 2, 3, 1, 4, 5, 0, 1, 3, 3, 5, 6, 0,
       1, 2, 2, 5, 8, 1, 3, 7, 0, 1, 8, 4, 6, 7, 0, 1, 4, 3, 4, 6, 0, 4,
       5, 2, 6, 7, 5, 6, 7, 0, 1, 6, 4, 5, 7, 4, 7, 8, 1, 6, 8, 0, 2, 6,
       5, 6, 8, 2, 5, 7, 0, 1, 7, 0, 7, 8, 0, 3, 5, 0, 3, 7, 2, 3, 8, 2,
       3, 4, 1, 4, 6, 3, 4, 5, 0, 3, 8, 4, 5, 8, 0, 4, 6, 1, 4, 8, 6, 7,
       8, 4, 6, 8, 0, 3, 4, 1, 3, 4, 1, 5, 7, 1, 4, 7, 1, 2, 8, 0, 6, 7,
       1, 6, 7, 1, 3, 5, 0, 1, 5, 0, 4, 8, 4, 5, 6, 0, 2, 5, 3, 5, 7, 0,
       2, 4, 2, 6, 8, 2, 3, 7, 2, 5, 6, 2, 4, 8, 0, 2, 7, 3, 6, 8, 2, 3,
       6, 3, 7, 8, 0, 5, 8, 1, 2, 6, 2, 3, 5, 1, 5, 8, 3, 6, 7, 3, 4, 7,
       0, 4, 7, 3, 5, 8, 2, 4, 5, 1, 2, 5, 2, 7, 8, 2, 4, 6, 0, 5, 6, 3,
       4, 8, 0, 6, 8, 2, 4, 7, 0, 2, 8, 0, 3, 6, 5, 7, 8, 1, 5, 6, 1, 2,
       4, 0, 5, 7, 1, 3, 8, 1, 7, 8
    ), dtype = np.int32).reshape(84, 3)


    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        _X = X[example_index]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        for dilation_index in range(num_dilations):

            _padding0 = dilation_index % 2

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros(input_length, dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                C_alpha[-end:] = C_alpha[-end:] + A[:end]
                C_gamma[gamma_index, -end:] = G[:end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                C_alpha[:-start] = C_alpha[:-start] + A[start:]
                C_gamma[gamma_index, :-start] = G[start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                _padding1 = (_padding0 + kernel_index) % 2

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]

                if _padding1 == 0:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()
                else:
                    for feature_count in range(num_features_this_dilation):
                        features[example_index, feature_index_start + feature_count] = _PPV(C[padding:-padding], biases[feature_index_start + feature_count]).mean()

                feature_index_start = feature_index_end

    return features

#@title R-Clustering evaluated across UCR dataset

# Define the number of features or kernels (500 by default based on previous development experiments)
num_features = 500

# Initialize a list to store the results
results_list = []

DR_method = {'pca':PCA,'umap':umap}
Classification_method = {'KMeans':KMeans,'AgglomerativeClustering':AgglomerativeClustering,'GaussianMixture':GaussianMixture}


num_of_datasets = len(validation_datasets)
repeat_times = 20  #

res_dict = {'datasets':validation_datasets,
            'pca+KMeans':np.zeros([repeat_times,num_of_datasets]),
            'pca+AgglomerativeClustering':np.zeros([repeat_times,num_of_datasets]),
            'pca+GaussianMixture':np.zeros([repeat_times,num_of_datasets]),
            'umap+KMeans':np.zeros([repeat_times,num_of_datasets]),
            'umap+AgglomerativeClustering':np.zeros([repeat_times,num_of_datasets]),
            'umap+GaussianMixture':np.zeros([repeat_times,num_of_datasets]),
            'ensemble':np.zeros([repeat_times,num_of_datasets])}

for _time in range(repeat_times):
    print(_time)
    for i,dataset in enumerate(validation_datasets):
        print('--------------------' + str(i)+ '----------------------------------')
        print(dataset)
        cluster_list = []
        clustering_results = dict()  #save results
        
        
        X, Y = load_classification_with_fixes(dataset)
    
       
        parameters = fit(X=X, num_features=num_features)
        transformed_data = transform(X=X, parameters=parameters)
    
        
        sc = StandardScaler()
        X_std = sc.fit_transform(transformed_data)
        num_clusters = len(np.unique(Y))
        #six base clusterers
        for DR_name,DR_way in DR_method.items():
            if DR_name == 'pca':
                pca = DR_way().fit(X_std)
                optimal_dimensions = np.argmax(pca.explained_variance_ratio_ < 0.01)
                pca_optimal = DR_way(n_components=optimal_dimensions)
                transformed_data_pca = pca_optimal.fit_transform(X_std)
                for C_name,C_way in Classification_method.items():
                    str_info = DR_name + ' ' + C_name + ': '
                    if C_name=='KMeans':
                        labels_pred = C_way(n_clusters=num_clusters, n_init=10).fit_predict(transformed_data_pca)
                    elif C_name=='AgglomerativeClustering':
                        labels_pred = C_way(n_clusters=num_clusters).fit_predict(transformed_data_pca)
                    else:
                        labels_pred = C_way(n_components=num_clusters).fit_predict(transformed_data_pca)
                    clustering_results[DR_name + '+' + C_name] = labels_pred
                    cluster_list.append(labels_pred)
                    score = metrics.adjusted_rand_score(labels_true=Y, labels_pred=labels_pred)
                    res_dict[DR_name + '+' + C_name][_time,i] = score
                    print(str_info + " ARI: ", score)
                    print("\n")
            elif DR_name == 'umap':
                reducer = umap.UMAP(n_components=10, random_state=42)
                X_umap = reducer.fit_transform(X_std)
                for C_name,C_way in Classification_method.items():
                    str_info = DR_name + ' ' + C_name + ': '
                    if C_name=='KMeans':
                        labels_pred = C_way(n_clusters=num_clusters, n_init=10).fit_predict(X_umap)
                    elif C_name=='AgglomerativeClustering':
                        labels_pred = C_way(n_clusters=num_clusters).fit_predict(X_umap)
                    else:
                        labels_pred = C_way(n_components=num_clusters).fit_predict(X_umap)
                    clustering_results[DR_name + '+' + C_name] = labels_pred
                    cluster_list.append(labels_pred)
                    score = metrics.adjusted_rand_score(labels_true=Y, labels_pred=labels_pred)
                    res_dict[DR_name + '+' + C_name][_time,i] = score
                    print(str_info + " ARI: ", score)
                    print("\n")
            else:
                pass
        cluster_list = np.asarray(cluster_list)
        cluster_matrix = np.stack(cluster_list, axis=0)
        
        clusters_assign = ce.cluster_ensembles(cluster_matrix,nclass=num_clusters,solver='nmf')  #ensembled
        score = metrics.adjusted_rand_score(labels_true=Y, labels_pred=clusters_assign)
        res_dict['ensemble'][_time,i] = score
        print('ensemble' + " ARI: ", score)
        print("\n")


#to_csv
for _key in res_dict.keys():
    if _key != 'datasets':
        res_dict[_key] = np.mean(res_dict[_key],axis=0)   #average
df = pd.DataFrame(res_dict)
df.to_csv('output_time.csv', index=False)


    
