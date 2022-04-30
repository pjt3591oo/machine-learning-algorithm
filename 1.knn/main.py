from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(3, 3)
fig.set_size_inches(15, 15)

for i in range(9):
    # 더미 데이터 생성
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2, n_redundant=0, random_state=30)

    blue = X[y==0]
    red = X[y==1]
    
    # 랜덤한 새로운 점 생성
    newcomer = np.random.randn(1, 2)
    
    # K
    K = 3*(i//3+1)
    
    axes[i//3, i%3].scatter(red[:,0], red[:, 1], 80, 'r', '^')
    axes[i//3, i%3].scatter(blue[:,0], blue[:, 1], 80, 'b', '^')
    axes[i//3, i%3].scatter(newcomer[:, 0], newcomer[:, 1], 80, 'g', 'o')
    
    # k=3
    knn = KNeighborsClassifier(n_neighbors=3*(i//3+1))
    knn.fit(X, y)
    pred = knn.predict(newcomer)
    
    # 표기
    axes[i//3, i%3].annotate('red' if pred == 1 else 'blue', xy=newcomer[0], xytext=(newcomer[0]), fontsize=12)

plt.tight_layout()
plt.show()