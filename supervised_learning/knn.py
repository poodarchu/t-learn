from __future__ import print_function
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

from utils.data_manipulate import train_test_split, normalize
from utils.data_operate import euclidean_distance, accuracy_score

from unsupervised_learning import PCA

class kNN():
    def __init__(self, k=5):
        self.k = k

    # do a majority vote among neighbors
    def _majority_vote(self, neighbors, classes):
        max_count = 0
        most_common = None
        # Count class occurrences among neighbors
        for c in np.unique(classes):
            count = len(neighbors[neighbors[:-1] == c])
            if count > max_count:
                max_count = count
                most_common = c

        return  most_common

    def predict(self, X_test, X_train, y_train):
        n_classes = np.unique(y_train)
        y_pred = []

        for test_i in X_test:
            neighbors = []
            # Calculate the distance from each observed sample to the samples we wish to predict
            for j, observed_sample in enumerate(X_train):
                distance = euclidean_distance(test_i, observed_sample)
                label = y_train[j]
                # Add neighbor and its distance
                neighbors.append([label, distance])
            # Sort the list of observed samples from the lowest to highest distance
            # then select the first k elem
            k_nearest_neighbors = neighbors[neighbors[:, 1].argsort()][:self.k]
            # Do a majority vote among the k nearest neighbors and set prediction as the
            # class receiving the most votes
            label = self._majority_vote(k_nearest_neighbors, n_classes)
            y_pred.append(label)

        return np.array(y_pred)



if __name__ == "__main__":
    data = datasets.load_iris()
    X = normalize(data.data)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    classifier = kNN(3)

    y_pred = classifier.predict(X_test, X_train, y_train)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy:", accuracy)

    # Reduce dimensions to 2D using pca and plot the results

