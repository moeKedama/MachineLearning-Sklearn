import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import datasets
from sklearn import metrics
import pandas as pd

iris = datasets.load_iris()
X = iris.data[:, :4]

print(X.shape)
y = iris.target
np.random.seed(114514)

# 绘制数据分布图
plt.figure()
plt.scatter(X[:, 2], X[:, 3], c="red", label='see')
plt.xlabel('Petal.Length')
plt.ylabel('Petal.Width')
plt.legend(loc=2)
plt.show()

print("调整兰德系数")

# k-means
estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
print("kmeans: ", metrics.adjusted_rand_score(y, label_pred))
# 绘制k-means结果
plt.figure()
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 2], x0[:, 3], c="red", label='x0')
plt.scatter(x1[:, 2], x1[:, 3], c="blue", label='x1')
plt.scatter(x2[:, 2], x2[:, 3], c="green", label='x2')
plt.xlabel('Petal.Length')
plt.ylabel('Petal.Width')
plt.title('K-means')
plt.legend(loc=2)
plt.show()

# MiniBatchKMeans
estimator = MiniBatchKMeans(n_clusters=3)  # 构造聚类器
estimator.fit(X)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
print("MiniBatchKMeans: ", metrics.adjusted_rand_score(y, label_pred))

plt.figure()
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 2], x0[:, 3], c="red", label='x0')
plt.scatter(x1[:, 2], x1[:, 3], c="blue", label='x1')
plt.scatter(x2[:, 2], x2[:, 3], c="green", label='x2')
plt.xlabel('Petal.Length')
plt.ylabel('Petal.Width')
plt.title('MiniBatchKMeans')
plt.legend(loc=2)
plt.show()

# AGNES
irisdata = iris.data
clustering = AgglomerativeClustering(linkage='ward', n_clusters=3)
res = clustering.fit(irisdata)
label_pred = clustering.labels_
print("AGNES: ", metrics.adjusted_rand_score(y, label_pred))

plt.figure('AGNES')
d0 = irisdata[clustering.labels_ == 0]
d1 = irisdata[clustering.labels_ == 1]
d2 = irisdata[clustering.labels_ == 2]
plt.scatter(d0[:, 2], d0[:, 3], c="red", label='x1')
plt.scatter(d1[:, 2], d1[:, 3], c="blue", label='x2')
plt.scatter(d2[:, 2], d2[:, 3], c="green", label='x3')
plt.xlabel('Petal.Length')
plt.ylabel('Petal.Width')
plt.title("AGNES Clustering")
plt.legend(loc=2)
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.4, min_samples=5)
dbscan.fit(X)
label_pred = dbscan.labels_
print("DBSCAN: ", metrics.adjusted_rand_score(y, label_pred))

x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 2], x0[:, 3], c="red", label='x0')
plt.scatter(x1[:, 2], x1[:, 3], c="blue", label='x1')
plt.scatter(x2[:, 2], x2[:, 3], c="green", label='x2')
plt.xlabel('Petal.Length')
plt.ylabel('Petal.Width')
plt.title("DBSCAN")
plt.legend(loc=2)
plt.show()

# GMM
gmm = GaussianMixture(n_components=3).fit(X)
label_pred = gmm.predict(X)
print("GMM: ", metrics.adjusted_rand_score(y, label_pred))
plt.figure()
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0[:, 2], x0[:, 3], c="red", label='x0')
plt.scatter(x1[:, 2], x1[:, 3], c="blue", label='x1')
plt.scatter(x2[:, 2], x2[:, 3], c="green", label='x2')
plt.xlabel('Petal.Length')
plt.ylabel('Petal.Width')
plt.title('GMM')
plt.legend(loc=2)
plt.show()
