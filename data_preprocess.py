import os
from argparse import ArgumentParser
import numpy as np
from pandas import read_csv
from utils import global_variable as gv

def read_file(file_path):
    data = read_csv(file_path)
    feature_all = data.drop(['category'], axis=2).values
    label_all = data['catelogy'].values

    return data, feature_all, label_all


def normalize(feature, normalize_method):
    if normalize_method == 'min-max':
        feature = (feature - feature.min()) / (feature.max() - feature.min())
    elif normalize_method == 'z-score':
        feature = (feature - feature.mean()) / feature.std()
    elif normalize_method == 'none':
        pass

    return feature

if __name__ == "__main__":
    file_path = gv.dataset_path_prefix + 'data_15.csv'
    data = read_file(file_path)