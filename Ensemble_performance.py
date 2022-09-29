import time
import random
import importlib
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier

from data_preprocess import read_file, normalize
from hyper_parameter import HyperParameterOptimizer
from utils import global_variable as gv



def perform(n_raw_features, n_folds, hyper_algorithm, ensemble_name):
    # read data
    data_name = 'data_' + str(n_raw_features) + '.csv'
    file_path = gv.dataset_path_prefix + data_name
    data, feature_all, label_all = read_file(file_path)
    normalized_feature_all = normalize(feature_all, 'none')
    print('===== data ready =====')
    
    
    # build ensemble
    if ensemble_name == 'Stacking':
        ensemble = StackingClassifier(
            estimators=[('RF', RandomForestClassifier()),
                        ('GBDT', GradientBoostingClassifier()),
                        ('SVM', SVC()),
                        ('LR', LogisticRegression()),
                        ('KNN', KNeighborsClassifier())],
            final_estimator=LogisticRegression(),
            cv=n_folds
        )
    print('===== ensemble ready =====')

    # train-test validation
    X_train, X_test, y_train, y_test = \
        train_test_split(normalized_feature_all, label_all, test_size=0.3)
    begin = time.time()
    ensemble.fit(X_train, y_train)
    end = time.time()
    train_time = end - begin
    y_predict = ensemble.predict(X_test)
    y_predict_train = ensemble.predict(X_train)

    scores = {}
    score_names = [
        'test_accuracy','test_precision_macro', 'test_recall_macro', 'test_f1_macro',
        'best_parameters', 'best_score', 'search_time', 'train_time'
    ]

    for score_name in score_names:
        if score_name == 'test_accuracy':
            score = metrics.accuracy_score(y_test, y_predict)
        elif score_name == 'test_precision_macro':
            score = metrics.precision_score(y_test, y_predict, average='macro')
        elif score_name == 'test_recall_macro':
            score = metrics.recall_score(y_test, y_predict, average='macro')
        elif score_name == 'test_f1_macro':
            score = metrics.f1_score(y_test, y_predict, average='macro')
        
        elif score_name == 'best_parameters':
            score = best_parameters
        elif score_name == 'best_score':
            score = best_score
        elif score_name == 'search_time':
            score = search_time
        elif score_name == 'train_time':
            score = train_time

        scores[score_name] = score

    print('===== train-test validation finished, here are results: =====')
    for score_name in scores.keys():
        print('%s: %s' % (score_name, scores[score_name]))
    print('train time: %.4f' % (train_time))

    print('===== more results of %s =====' % (ensemble_name))
    print('train classification report:')
    print(metrics.classification_report(y_train, y_predict_train, digits=4))
    print('test classification report:')
    print(metrics.classification_report(y_test, y_predict, digits=4))

    key = '%s_%s_%s_%s' % (n_raw_features, n_folds, hyper_algorithm, ensemble_name)
    value = scores

    return key, value
