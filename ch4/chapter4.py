from itertools import combinations

import pandas as pd
import numpy as np
from io import StringIO

from matplotlib import pyplot as plt
from numpy import argmax
from sklearn import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''

df = pd.read_csv(StringIO(csv_data))
# print(df)
# print(df.isnull().sum())

# imr = SimpleImputer(missing_values=np.nan, strategy="mean")
# imr.fit(df.values)
# imputed_data = imr.transform(df.values)
# print(imputed_data)

df = df.fillna(df.mean())
# print(df)

df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']])
df.columns = ['color', 'size', 'price', 'classlabel']

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)

class_mapping = {label: num for num, label in
                 enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)

inv_class_mapping = {num: label for label, num in class_mapping.items()}

X = df[['color', 'size', 'price']].values
trans = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
])
X = trans.fit_transform(X).astype(float)
# print(X)
# X[:, 0] = color_le.fit_transform(X[:, 0])

s = 'D:\Tudor\Projects\ML-book-mirror\ch4\wine.data'
df_wine = pd.read_csv(s, header=None, encoding='utf-8')
X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    stratify=y)
print(X_train[:3, :])
print(y_train[:3])


class SBS:
    def __init__(self, k_features, model, score=accuracy_score, test_size=0.3, random_state=1):
        self._k_features = k_features
        self._model = clone(model)
        self._score = score
        self._test_size = test_size
        self._random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=self._test_size,
                                                            random_state=self._random_state,
                                                            stratify=y)
        length = X_train.shape[1]
        self._indices = tuple(range(length))
        initial_score = self._getScore(X_train, y_train, X_test, y_test, self._indices)
        self.score_history = [initial_score]
        self.subset_history = [self._indices]
        while length > self._k_features:
            subsets = []
            score = []
            for p in combinations(self._indices, length - 1):
                subsets.append(p)
                score.append(self._getScore(X_train, y_train, X_test, y_test, p))
            max_index = argmax(score)
            self._indices = subsets[max_index]
            length -= 1
            self.subset_history.append(self._indices)
            self.score_history.append(score[max_index])
        return self

    def transform(self, X):
        return X[:, self._indices]

    def _getScore(self, X_train, y_train, X_test, y_test, indices):
        self._model.fit(X_train[:, indices], y_train)
        y_predict = self._model.predict(X_test[:, indices])
        return self._score(y_test, y_predict)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
sbs = SBS(k_features=1, model=knn)
sbs.fit(X_train_std, y_train)

subset_lengths = [len(k) for k in sbs.subset_history]
plt.plot(subset_lengths, sbs.score_history, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel("Accuracy")
plt.xlabel("Number of features")
plt.grid()
plt.tight_layout()
plt.show()
