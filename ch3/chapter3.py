import matplotlib.pyplot
from matplotlib.colors import ListedColormap
from onedal.svm import SVC
from sklearn import datasets, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

y_pred = ppn.predict(X_test_std)
print("Accuracy score: ", accuracy_score(y_test, y_pred))


class LogisticRegressionGD:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses = []

        for _ in range(self.n_iter):
            z = self._z(X)
            y_hat = self._sigmoid(z)
            error = y - y_hat
            self.w_ += self.eta * X.T.dot(error) / X.shape[0]
            self.b_ += self.eta * error.mean()
            loss = (-y.dot(np.log(y_hat)) - (1 - y).dot(np.log(1 - y_hat))) / y.shape[0]
            self.losses.append(loss)

        return self

    def _z(self, X):
        return np.dot(X, self.w_) + self.b_

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        return np.where(self._sigmoid(self._z(X)) >= 0.5, 1, 0)


X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
X_test_01_subset = X_test_std[(y_test == 0) | (y_test == 1)]
y_test_01_subset = y_test[(y_test == 0) | (y_test == 1)]
# print(lrgd.predict(X_test_01_subset[:3, :]))
# print(y_test_01_subset[:3])

lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='ovr')
lr.fit(X_train_std, y_train)
# print(lr.predict(X_test_std[:3, :]))
# print(y_test[:3])

np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)
plt.scatter(X_xor[y_xor == 1, 0],
            X_xor[y_xor == 1, 1],
            color="blue",
            marker="s",
            label="True")
plt.scatter(X_xor[y_xor == 0, 0],
            X_xor[y_xor == 0, 1],
            color="red",
            marker="o",
            label="False")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

X_xor_train, X_xor_test, y_xor_train, y_xor_test = train_test_split(X_xor,
                                                                    y_xor,
                                                                    test_size=0.3,
                                                                    random_state=1,
                                                                    stratify=y_xor)
svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor_train, y_xor_train)
y_xor_predictions = svm.predict(X_xor_test)
print("XOR accuracy score:", accuracy_score(y_xor_test, y_xor_predictions))

tree_model = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
tree_model.fit(X_train, y_train)
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
tree.plot_tree(tree_model, feature_names=feature_names, filled=True)
plt.show()

random_forest_model = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
random_forest_model.fit(X_train, y_train)
y_forest = random_forest_model.predict(X_test)
print("Random forest accuracy: ", accuracy_score(y_test, y_forest))

knn = KNeighborsClassifier()
knn.fit(X_train_std, y_train)
y_knn = knn.predict(X_test_std)
print("K-nearest neighbor accuracy: ", accuracy_score(y_test, y_knn))
