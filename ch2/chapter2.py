import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        random_gen = np.random.RandomState(self.random_state)
        self.w_ = random_gen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.w_ += update * x_i
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, x_i):
        return np.where(self.z(x_i) >= 0.0, 1, 0)

    def z(self, x_i):
        return np.dot(x_i, self.w_) + self.b_


s = 'D:/Tudor/Projects/ML-book-mirror/ch2/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='Versicolor')
plt.xlabel('Sepal length [cm]')
plt.ylabel('Petal length [cm]')
plt.legend(loc='upper left')
plt.show()

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(X, y)
plt.plot(range(1, len(perceptron.errors_) + 1), perceptron.errors_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Nr. of updates")
plt.show()

testX = np.array([[1, 2], [3, 4]])
testW = np.array([1, 1])
testB = 3

print(testX.dot(testW) + testB)


class Adaline:

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        random_gen = np.random.RandomState(self.random_state)
        self.w_ = random_gen.normal(loc=0.0, scale=0.1, size=X.shape[1])
        self.b_ = np.float_(0)
        self.losses_ = []
        for i in range(self.n_iter):
            predictions = self.z(X)
            errors = y - predictions
            self.w_ += 2 * self.eta * X.T.dot(errors) / X.shape[0]
            self.b_ += 2 * self.eta * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self

    def z(self, X):
        return X.dot(self.w_) + self.b_


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = Adaline(n_iter=15, eta=0.1).fit(X, y)
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(MSE)')
ax[0].set_title('Learning rate = 0.1')

ada2 = Adaline(n_iter=15, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.losses_) + 1), np.log10(ada2.losses_), marker='x')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(MSE)')
ax[1].set_title('Learning rate = 0.0001')

plt.show()
