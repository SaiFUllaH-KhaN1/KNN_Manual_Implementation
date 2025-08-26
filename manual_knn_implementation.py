import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label among the k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def evaluate(self, X_test, y_true):
        y_pred = self.predict(X_test)
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        return accuracy

# Example usage:
if __name__ == "__main__":
    # Create a sample dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])

    # Create a KNN classifier with k=2
    classifier = KNN(k=2)
    classifier.fit(X_train, y_train)

    # Test data and true labels
    X_test = np.array([[2.5, 3.5], [1.5, 2.5]])
    y_true = np.array([0, 0])

    # Make predictions and evaluate the model
    accuracy = classifier.evaluate(X_test, y_true)

    print("Accuracy:", accuracy)
