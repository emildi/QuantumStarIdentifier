#!/usr/bin/env python3
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uuid

from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

from qiskit import BasicAer
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.providers.aer.backends import StatevectorSimulator

seed = 42
algorithm_globals.random_seed = seed

print(f'Start run at {datetime.now()}')

datasets_dir = './datasets/'

training_data = '100_stars_dataset_clean_SVM.csv'
testing_data = '100_stars_validation_dataset_SVM.csv'

print(f'Training data file: {training_data}')
print(f'Testing data file: {testing_data}')

### DATA PREP SECTION
usecols = ["bin0","bin1","bin2","bin3","bin4","bin5", "bin6","bin7","bin8","bin9","bin10","bin11","bin12","bin13","bin14","bin15", "bin16","bin17","bin18","bin19","bin20","bin21","bin22","bin23","bin24","label"]
df = pd.read_csv(datasets_dir+training_data, usecols=usecols)
df_val = pd.read_csv(datasets_dir+testing_data, usecols=usecols)

x = df[["bin0","bin1","bin2","bin3","bin4","bin5", "bin6","bin7","bin8","bin9","bin10","bin11","bin12","bin13","bin14","bin15", "bin16","bin17","bin18","bin19","bin20","bin21","bin22","bin23","bin24"]]
x = x.to_numpy().astype(np.float32)
print(f'Training data shape: {x.shape}')

y = df[["label"]]
y = y.to_numpy().astype(np.float32)
print(f'Training data label shape: {y.shape}')

x_val = df_val[["bin0","bin1","bin2","bin3","bin4","bin5", "bin6","bin7","bin8","bin9","bin10","bin11","bin12","bin13","bin14","bin15", "bin16","bin17","bin18","bin19","bin20","bin21","bin22","bin23","bin24"]]
x_val = x_val.to_numpy().astype(np.float32)
print(f'Testing data shape: {x_val.shape}')

y_val = df_val[["label"]]
y_val = y_val.to_numpy().astype(np.float32)
print(f'Testing data label shape: {y_val.shape}')

# FUNCTIONS

def precompute_quantum_kernel(adhoc_dimension, feature_map='ZFeatureMap', reps=1, entanglement=None):

    # CREATE FEAUTURE MAP
    if feature_map == 'ZFeatureMap':
        adhoc_feature_map = ZFeatureMap(feature_dimension=adhoc_dimension, reps=reps)
    elif feature_map == 'ZZFeatureMap':
        adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=reps, entanglement=entanglement)
    else:
        print(f'Unknown feature map {feature_map}')
        return None, None
    quantum_simulator = StatevectorSimulator(device='GPU', precision='single')

    adhoc_backend = QuantumInstance(
        quantum_simulator, seed_simulator=seed, seed_transpiler=seed
    )

    adhoc_kernel = QuantumKernel(feature_map=adhoc_feature_map, quantum_instance=adhoc_backend)

    ### PRECOMPUTE QUANTUM KERNEL
    start_prec_k = datetime.now()
    print(f'Start precomputing kernel at {start_prec_k}')

    adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=x)

    end_prec_k = datetime.now()
    print(f'End precomputing kernel at {end_prec_k}')
    print(f'Duration: {end_prec_k - start_prec_k}')

    return adhoc_matrix_train, adhoc_kernel

def fit_quantum_kernel(adhoc_matrix_train, y):
    ### QSVM FIT
    start_qsvm_fit = datetime.now()
    print(f'Start QSVM fit at {start_qsvm_fit}')

    adhoc_svc = SVC(kernel="precomputed")
    adhoc_svc.fit(adhoc_matrix_train, y.ravel())

    end_qsvm_fit = datetime.now()
    print(f'End QSVM fit at {end_qsvm_fit}')
    print(f'Duration: {end_qsvm_fit - start_qsvm_fit}')

    return adhoc_svc

def train_score_calc(adhoc_svc, adhoc_matrix_train, y):
### TRAIN SCORE
    start_qsvm_train_score = datetime.now()
    print(f'Start train score at {start_qsvm_train_score}')

    adhoc_score_train = adhoc_svc.score(adhoc_matrix_train, y.ravel())
    print(f"Precomputed kernel classification train score: {adhoc_score_train}")

    end_qsvm_train_score = datetime.now()
    print(f'End train score at {end_qsvm_train_score}')
    print(f'Duration: {end_qsvm_train_score - start_qsvm_train_score}')

    return adhoc_score_train

def test_score_calc(adhoc_svc, adhoc_kernel, x_val, y_val):
    ### TEST SCORE
    start_qsvm_test_score = datetime.now()
    print(f'Start test score at {start_qsvm_test_score}')

    test_scores = []
    test_iterator = range(0, x_val.shape[0], x.shape[0])
    test_iterations = len(test_iterator)
    iteration = 0
    for idx in range(0, x_val.shape[0], x.shape[0]):
        #print(x_val[idx:idx+x.shape[0],:].shape)
        iteration += 1
        start_time = datetime.now()
        print(f'Iteration {iteration}/{test_iterations} starts at {start_time}')
        adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=x_val[idx:idx+x.shape[0],:], y_vec=x)
        adhoc_score = adhoc_svc.score(adhoc_matrix_test, y_val[idx:idx+x.shape[0],:].ravel())
        end_time = datetime.now()
        print(f'Iteration {iteration}/{test_iterations} ends at {end_time} - Duration {end_time - start_time}')
        test_scores.append(adhoc_score)

    test_score = sum(test_scores) / len(test_scores)

    print(f"Precomputed kernel classification test score: {test_score}")

    end_qsvm_test_score = datetime.now()
    print(f'End test score at {end_qsvm_test_score}')
    print(f'Duration: {end_qsvm_test_score - start_qsvm_test_score}')

    return test_score

### TESTING WITH DIFFERENT FEATUREMAP OPTIONS
adhoc_dimension = x.shape[1] # corresponds to number of qubits needed
feature_maps_list = ['ZFeatureMap', 'ZZFeatureMap']
reps_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
entanglements_list = ['full', 'linear', 'circular', 'sca']

for feature_map in feature_maps_list:
    for reps in reps_list:
        if feature_map == 'ZFeatureMap':
            print(60*'-')
            print(f'START ({datetime.now()}): {feature_map}, {reps}, NONE')
            print(60*'-')

            adhoc_matrix_train, adhoc_kernel = precompute_quantum_kernel(adhoc_dimension=adhoc_dimension, feature_map=feature_map, reps=reps)
            adhoc_svc = fit_quantum_kernel(adhoc_matrix_train=adhoc_matrix_train, y=y)
            adhoc_score_train = train_score_calc(adhoc_svc=adhoc_svc, adhoc_matrix_train=adhoc_matrix_train, y=y)
            test_score = test_score_calc(adhoc_svc=adhoc_svc, adhoc_kernel=adhoc_kernel, x_val=x_val, y_val=y_val)

            print(60*'-')
            print(f'RESULT: {feature_map}, {reps}, NONE, {adhoc_score_train}, {test_score}')
            print(60*'-')
            kernel_path = 'saved_kernels/' + f'{feature_map}-{reps}-NONE-{str(uuid.uuid4())}'
            print(f'Saving precomputed kernel at: {kernel_path}')
            with open(kernel_path, 'wb') as f:
                np.save(f, adhoc_matrix_train)

        elif feature_map == 'ZZFeatureMap':
            for entanglement in entanglements_list:
                print(60*'-')
                print(f'START ({datetime.now()}): {feature_map}, {reps}, {entanglement}')
                print(60*'-')

                adhoc_matrix_train, adhoc_kernel = precompute_quantum_kernel(adhoc_dimension=adhoc_dimension, feature_map=feature_map, reps=reps, entanglement=entanglement)
                adhoc_svc = fit_quantum_kernel(adhoc_matrix_train=adhoc_matrix_train, y=y)
                adhoc_score_train = train_score_calc(adhoc_svc=adhoc_svc, adhoc_matrix_train=adhoc_matrix_train, y=y)
                test_score = test_score_calc(adhoc_svc=adhoc_svc, adhoc_kernel=adhoc_kernel, x_val=x_val, y_val=y_val)

                print(60*'-')
                print(f'RESULT: {feature_map}, {reps}, {entanglement}, {adhoc_score_train}, {test_score}')
                print(60*'-')
                kernel_path = 'saved_kernels/' + f'{feature_map}-{reps}-{entanglement}-{str(uuid.uuid4())}'
                print(f'Saving precomputed kernel at: {kernel_path}')
                with open(kernel_path, 'wb') as f:
                    np.save(f, adhoc_matrix_train)
        else:
            print(f'Unknown feature map {feature_map}')

print(f'End run at {datetime.now()}')
