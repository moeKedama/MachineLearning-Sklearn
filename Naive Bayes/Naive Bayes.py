import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def convert_to_sparse_matrix(docs):
    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in docs:
        for term in d:
            index = vocabulary.setdefault(term, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    return csr_matrix((data, indices, indptr), dtype=int)


def get_class(spam_prob):
    if spam_prob[0] < 6.5859811930136635e-77:
        return 1
    else:
        return 0


data_dir = 'messages'
messages = []
y = []
for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename)) as file:
        subject = file.readline()
        skip = file.readline()
        content = file.readline()
        messages.append(content.split())
    if "legit" in filename:
        y.append("legit")
    else:
        y.append("spam")

X = convert_to_sparse_matrix(messages)
y = list(map(lambda x: 1 if x == 'legit' else 0, y))

y_predict = []
y_probs = []
y_true = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42 + i)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_probs = y_probs + model.predict_proba(X_test).tolist()
    y_true = y_true + y_test
#    print(len(y_probs), len(y_true))
# print(X_train, X_test, y_train, y_test)

"""
model = MultinomialNB()
model.fit(X_train, y_train)
"""

min_val = 12
for i in range(len(y_probs)):
    if (y_true[i] == 0 and y_probs[i][0] < min_val):
        min_val = y_probs[i][0]

print(min_val)
y_predict = (list(map(get_class, y_probs)))

print(confusion_matrix(y_true, y_predict))

y_probs = np.array(y_probs)

# fpr,tpr,thresholds=metrics.roc_curve(y_test,y_probs[:,0])
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probs[:, 1])
roc_auc = auc(fpr, tpr)
#
plt.plot(fpr, tpr, 'b', label='AUC=%0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('roc')
plt.show()
