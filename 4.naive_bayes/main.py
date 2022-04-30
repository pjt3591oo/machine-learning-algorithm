from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

news = fetch_20newsgroups()
X, y, labels = news.data, news.target, news.target_names
print(labels)
# 학습/테스트 데이터셋 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

vectorizer = CountVectorizer()
tfid = TfidfTransformer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

X_train_tfid = tfid.fit_transform(X_train_vec)
X_test_tfid = tfid.transform(X_test_vec)

nb = MultinomialNB()
nb.fit(X_train_tfid, y_train)

predicted_y = nb.predict(X_test_tfid)
for i in range(10):
    print('실제 값: {0}, 예측 값: {1}'.format(labels[y_test[i]], labels[predicted_y[i]]))