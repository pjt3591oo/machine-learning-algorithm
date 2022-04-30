import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

C = 1                         
clf= svm.SVC(kernel = "linear", C=C)
clf.fit(X,y)
print('예측라벨: ', clf.predict([X[0]]), '실제라벨', y[0])