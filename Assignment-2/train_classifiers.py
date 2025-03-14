from sklearn.neural_network import MLPClassifier
from dataset import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch
import joblib
import os

dataset_list = ['moons', 'circles', 'manycircles', 'blobs', 'helix']

for i,dataset_name in enumerate(dataset_list):
    data_X, data_y = load_dataset(dataset_name)
    # MLPClassifier with better hyperparameters for multiclass problems
    # Use different parameters based on dataset complexity
    clf = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
    clf.fit(data_X, data_y)

    # Save the trained model to a file
    model_filename = f"classifier_models/{dataset_name}_mlp_model.pkl"
    joblib.dump(clf, model_filename)
    print(f"Model for {dataset_name} dataset saved to {model_filename}")
    print("Accuracy: ", clf.score(data_X, data_y))

    # training a classifier from sklearn, can be replaced with any torch classifier as long predict proba method is implemented
    # sklearn will be slower than torch
