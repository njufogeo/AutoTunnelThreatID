from itertools import product
import warnings
warnings.filterwarnings('ignore')

import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from skopt import BayesSearchCV

import pyswarms as ps

class HyperParameterOptimizer():
    def __init__(self, model_name, n_folds, hyper_algorithm, max_trial=50):
        self.model_name = model_name
        self.n_folds = n_folds
        self.hyper_algorithm = hyper_algorithm
        self.max_trial = max_trial
        self.hyper_parameter_space = {}
        # sample model (SVM, LR, KNN)
        if model_name == 'SVM':
            self.hyper_parameter_space = {
                'C': np.linspace(0.1, 50, 30),
                'gamma': np.linspace(0.1, 10, 30)
            }
        elif model_name == 'LR':
            self.hyper_parameter_space = {
                'C': np.linspace(0.1, 50, 30),
            }
        elif model_name == 'KNN':
            self.hyper_parameter_space = {
                'n_neighbors': np.linspace(1, 10, 10).astype(int),
            }

        self.hyper_parameter_space_keys = list(self.hyper_parameter_space.keys())
        self.hyper_parameter_space_values = list(self.hyper_parameter_space.values())
    
    def optimize(self, X, y):
        if self.hyper_algorithm == 'random':
            return self.optimize_random(X, y)
        elif self.hyper_algorithm == 'grid':
            return self.optimize_grid(X, y)
        elif self.hyper_algorithm == 'bayesian':
            if self.model_name == 'SVM': model = SVC()
            elif self.model_name == 'LR': model = LogisticRegression()
            elif self.model_name == 'KNN': model = KNeighborsClassifier()

            opt = BayesSearchCV(model,
                                self.hyper_parameter_space,
                                cv=self.n_folds,
                                n_jobs=-1,
                                n_iter=self.max_trial)
            opt.fit(X, y)
            return opt.best_params_, opt.best_score_

        elif self.hyper_algorithm == 'pso':
            def objective_function(params):
                bs = params.shape[0]
                results = []
                for i in range(bs):
                    param = params[i]
                    assert(len(param) == len(self.hyper_parameter_space_keys))

                    parameters = {}
                    for j in range(len(self.hyper_parameter_space_keys)):
                        param_type = type(self.hyper_parameter_space_values[j][0])
                        parameters[self.hyper_parameter_space_keys[j]] = param[j].astype(param_type)

                    results.append(self.get_score(X, y, parameters))
                return -1.0 * np.array(results)

            min_bound, max_bound = [], []
            for key, value in self.hyper_parameter_space.items():
                min_bound.append(np.min(value))
                max_bound.append(np.max(value))
            min_bound = np.array(min_bound)
            max_bound = np.array(max_bound)

            opt = ps.single.GlobalBestPSO(n_particles=10,
                                          dimensions=len(self.hyper_parameter_space),
                                          options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
                                          bounds=(min_bound, max_bound))

            cost, pos = opt.optimize(objective_function, iters=self.max_trial)

            best_parameters = {}
            for j in range(len(self.hyper_parameter_space_keys)):
                param_type = type(self.hyper_parameter_space_values[j][0])
                best_parameters[self.hyper_parameter_space_keys[j]] = pos[j].astype(param_type)
            best_score = -1.0 * cost

            return best_parameters, cost
    
    def optimize_random(self, X, y):
        print('===== in optimize_random =====')
        best_score = 0
        best_parameters = {}
        for i in range(self.max_trial):
            parameters = {}
            for j in range(len(self.hyper_parameter_space_keys)):
                parameters[self.hyper_parameter_space_keys[j]] = np.random.choice(self.hyper_parameter_space_values[j])
            score = self.get_score(X, y, parameters)
            if score > best_score:
                best_score = score
                best_parameters = parameters
            # print('===== after trial %d: best score = %f =====' % (i, best_score))
        return best_parameters, best_score
    
    def optimize_grid(self, X, y):
        print('===== in optimize_grid =====')
        best_score = 0
        best_parameters = {}

        parameter_table_list = [value for key, value in self.hyper_parameter_space.items()]
        cnt = 0
        for parameter_table in product(*parameter_table_list):
            parameters = {}
            for i in range(len(parameter_table)):
                parameters[self.hyper_parameter_space_keys[i]] = parameter_table[i]
            score = self.get_score(X, y, parameters)
            if score > best_score:
                best_score = score
                best_parameters = parameters
            cnt += 1
            # print('===== after trial %d: best score = %f =====' % (cnt, best_score))
        return best_parameters, best_score