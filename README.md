# AutoTunnelThreatID
Reproducability code for "Listening to underground infrastructure: Machine learning-based distributed acoustic sensing enables automated identification of diverse tunnel threats" by Tai-Yin Zhang, Cheng-Cheng Zhang, and Bin Shi (under review).

## Pre-requirements
* Python 3.9
* numpy 1.2
* pandas 1.3
* scikit-learn


## Setup
Precomputed features for the data used in our study are stored on **Zenodo** at:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7116488.svg)](https://doi.org/10.5281/zenodo.7116488)

* We provide calculated features **after** dimensionality reduction for all the data in `data_15.csv`.
* We provide calculated features **before** dimensionality reduction for all the data used in our manuscript in `data_24.csv`.

Details of raw data and features are stored at:

* We provide the details of raw data in `Details of raw data.xlsx`, including tunnel information, acquisition parameters, event distributions, etc.
* We provide the details of features in `Details of features.xlsx`.


## File format
Input file is a csv file, stores the features for each vibration signal:
* data_15.csv	
* data_24.csv


## Usage
The Python scripts to reproduce analyses from our paper are:
 
* `data_preprocess.py` is designed for data preprocessing, which can classify samples and normalize features.

* `Hyper_parameter.py` is used to compare four optimization algorithms, namely random search, grid search, bayesian optimization, and particle swarm optimization. The optimization algorithm selects SVM, KNN, and LR as examples. 

* `Ensemble_performance.py` is designed to build the stack framework, including data formats, cross-validation, optimization algorithms, and ensemble models. In addition, many codes are adjustable, and researchers can modify them according to their needs.

## Help
If you have any questions or require assistance, please contact the first author of the paper at zhangtaiyin@smail.nju.edu.cn.
