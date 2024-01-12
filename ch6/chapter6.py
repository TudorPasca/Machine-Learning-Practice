import numpy as np
import pandas as pd
import scipy.stats
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve, GridSearchCV, \
    RandomizedSearchCV, HalvingRandomSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

df = pd.read_csv("D:\Tudor\Projects\ML-book-mirror\ch6\wdbc.data", header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1, stratify=y)
pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression())
pipe_lr.fit(X_train, y_train)
y_hat = pipe_lr.predict(X_test)
test_accuracy = pipe_lr.score(X_test, y_test)
print("Test accuracy: ", test_accuracy)

scores = cross_val_score(estimator=pipe_lr,
                         X=X_train,
                         y=y_train,
                         cv=10,
                         n_jobs=1)
print(f'CV accuracy: { np.mean(scores) }')

pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
test_size, train_score, validation_score = learning_curve(pipe_lr,
                                                          X_train,
                                                          y_train,
                                                          train_sizes=np.linspace(0.1, 1.0, 10),
                                                          cv=10)
train_score_mean = np.mean(train_score, axis=1)
validation_score_mean = np.mean(validation_score, axis=1)
plt.plot(test_size, train_score_mean, color="blue", marker='x', label="Training accuracy")
plt.plot(test_size, validation_score_mean, color="green", marker='o', label="Validation accuracy")
plt.xlabel("Test size")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.ylim([0.8, 1.05])
plt.show()

param_range = [0.001, 0.01, 0.1, 1, 10, 100]
train_score, validation_score = validation_curve(pipe_lr,
                                                 X_train,
                                                 y_train,
                                                 param_name="logisticregression__C",
                                                 param_range=param_range,
                                                 cv=10,
                                                 n_jobs=2)
train_score_mean = np.mean(train_score, axis=1)
validation_score_mean = np.mean(validation_score, axis=1)
plt.plot(param_range, train_score_mean, color="blue", marker="x", label="Training accuracy")
plt.plot(param_range, validation_score_mean, color="green", marker="o", label="Validation accuracy")
plt.xscale("log")
plt.legend(loc="lower right")
plt.ylim([0.8, 1.05])
plt.xlabel("C value")
plt.ylabel("Accuracy")
plt.show()

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{"svc__C": param_range,
               "svc__kernel": ["linear"]},
              {"svc__C": param_range,
               "svc__gamma": param_range,
               "svc__kernel": ["rbf"]}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  n_jobs=-1,
                  cv=10)
gs = gs.fit(X_train, y_train)
print(gs.best_score_, gs.best_params_)
y_hat = gs.best_estimator_.predict(X_test)
print("Test accuracy: ", accuracy_score(y_test, y_hat))

param_range = scipy.stats.loguniform(0.0001, 1000.0)
param_grid = [{"svc__C": param_range,
               "svc__kernel": ["linear"]},
              {"svc__C": param_range,
               "svc__gamma": param_range,
               "svc__kernel": ["rbf"]}]

rs = RandomizedSearchCV(estimator=pipe_svc,
                        param_distributions=param_grid,
                        n_iter=20,
                        scoring='accuracy',
                        n_jobs=-1,
                        cv=10,
                        random_state=1)
rs = rs.fit(X_train, y_train)
print(rs.best_score_, rs.best_params_)

hs = HalvingRandomSearchCV(estimator=pipe_svc,
                           param_distributions=param_grid[0],
                           n_jobs=-1,
                           random_state=1,
                           factor=1.5)
hs = hs.fit(X_train, y_train)
print(hs.best_score_, hs.best_params_)

hs = HalvingRandomSearchCV(estimator=pipe_svc,
                           param_distributions=param_grid[1],
                           n_jobs=-1,
                           random_state=2,
                           factor=1.5)
hs = hs.fit(X_train, y_train)
print(hs.best_score_, hs.best_params_)

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{"svc__C": param_range,
               "svc__kernel": ["linear"]},
              {"svc__C": param_range,
               "svc__gamma": param_range,
               "svc__kernel": ["rbf"]}]
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)

scores = cross_val_score(gs, X_train, y_train,
                         scoring='accuracy', cv=5)
print(scores.mean())

pipe_svc.fit(X_train, y_train)
y_pred = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state=1, C=100.0))
X_train2 = X_train[:, [4, 14]]
