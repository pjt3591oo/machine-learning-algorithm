from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()

# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

# We only take the two corresponding features
X = iris.data
y = iris.target

# Train
clf = DecisionTreeClassifier().fit(X, y)
Z = clf.predict([X[0]])

print('예측값: ', Z, '실제값: ', y[0])