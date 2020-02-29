import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

with open('iris.csv')as file:
    title = file.readline()
    data = file.readlines()
    data = np.array(list(map(lambda x: x.strip('\n').split(','), data)))
    X = data[:, 1:5]
    y = data[:, 5]
    X = X.tolist()
    for i in range(len(X)):
        X[i] = list(map(float, X[i]))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

param_grid = [
    {
        'weights': ['uniform'],
        'n_neighbors': [i for i in range(1, 11)]
    },
    {
        'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11)],
        'p': [i for i in range(1, 6)]
    }
]
knn = kNN()
grid_search = GridSearchCV(knn, param_grid, n_jobs=-1, verbose=2, scoring='f1_micro')
grid_search.fit(X_train, y_train)

knn = grid_search.best_estimator_
y_pre = knn.predict(X_test)
print('f1:', knn.score(X_test, y_pre))

X_test = np.array(X_test)
x0 = X_test[y_pre == 'Iris-versicolor']
x1 = X_test[y_pre == 'Iris-setosa']
x2 = X_test[y_pre == 'Iris-virginica']
plt.figure()
plt.scatter(x0[:, 0], x0[:, 1])
plt.scatter(x1[:, 0], x1[:, 1])
plt.scatter(x2[:, 0], x2[:, 1])
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()