from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import datasets

# 데이터 로드
data = datasets.load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)

# 데이터와 라벨 분리
X = df[['mean radius', 'mean texture', 'mean area', 'mean symmetry']]
Y = data.target # 0: 사망, 1: 생존

model = LogisticRegression(penalty = 'l2')
model.fit(X, Y)
print(model.predict([[7.76, 24.54, 181.0, 0.1587]]))