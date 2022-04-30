import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 1. toy 데이터
X, y = make_blobs(centers=4, random_state=8)
y = y%2

# 그래프 확인
plt.scatter(X[:,0],X[:,1],c=y,s=50,edgecolors="b")



# 2. 2번째 특성 제곱하여 추가
X_new = np.hstack([X,X[:,1:]**2]) # 제곱을 새로운 열로 만들어 줌
X_new

# 그래프 확인
figure = plt.figure()
ax = Axes3D(figure, elev=-162, azim=-26) # 각도 틀어주는 파라미터 
mask = y==0
ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c='b', s=60, edgecolor='k')
ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask,2], c='r', marker = '^', s=60, edgecolor='k')
ax.set_xlabel('feature0')
ax.set_ylabel('feature1')
ax.set_zlabel('feature0 ** 2')

linear_svm_3d = LinearSVC(max_iter=10000).fit(X_new,y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# 선형 결정 경계 그려주기

figure = plt.figure()
ax = Axes3D(figure, elev=-140, azim=-26)

xx = np.linspace(X_new[:,0].min() - 2, X_new[:,0].max()+2,50)
yy = np.linspace(X_new[:,1].min() - 2, X_new[:,1].max()+2,50)
# 평면 그려주기 위해 조금 범위 넓게 (+2) 만큼 해줌

XX,YY = np.meshgrid(xx,yy)
ZZ = (coef[0]*XX+coef[1]*YY+intercept)/(-coef[2])

ax.plot_surface(XX,YY,ZZ,rstride=8,cstride=8,alpha=0.3,cmap=cm.Accent)

ax.scatter(X_new[mask,0], X_new[mask,1], X_new[mask,2], c='b', s=60, edgecolor='k')
ax.scatter(X_new[~mask,0], X_new[~mask,1], X_new[~mask,2], c='r', marker = '^', s=60, edgecolor='k')
ax.set_xlabel('feature0')
ax.set_ylabel('feature1')
ax.set_zlabel('feature0 ** 2')

plt.show()