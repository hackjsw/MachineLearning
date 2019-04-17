# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# 感知器算法


class Perceptron(object):
    '''
    eta:学习率
    n_inter:权重向量训练次数
    w_:神经分叉权重向量
    errors_:用于记录神经元判断出错次数
    '''

    def __init__(self, eta=0.01, n_inter=10):
        self.eta = eta
        self.n_inter = n_inter
        pass

    def fit(self, X, y):
        '''
        输入训练数据，培训神经元，X输入样本向量，Y对应样本分类
        X：shape[n_samples, n_features]
        X:[[1, 2, 3],[4, 5, 6]]
        n_samples:2
        n_features:3
        y:[1, -1]
        '''

        # 初始化权重向量为0
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_inter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(update != 0.0)
                self.errors_.append(errors)

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


file = 'iris.data.csv'
df = pd.read_csv(file, header=None)
y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',
            marker='x', label='versicolor')
plt.xlabel('花瓣长度')
plt.ylabel('花径长度')
plt.legend(loc='upper left')
# plt.show()

ppn = Perceptron(eta=0.1, n_inter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('error numbers')


# plt.show()

def plot_decision_regions(X, y, classifier, resolution=0.02):
    marker = ('s', 'x', 'o', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max()
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    # print(np.arange(x2_min, x2_max, resolution).shape)
    # print(np.arange(x2_min, x2_max, resolution))
    print(xx2.shape)
    print(xx2)


plot_decision_regions(X, y, ppn, resolution=0.02)