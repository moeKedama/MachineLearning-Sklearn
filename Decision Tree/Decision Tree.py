import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

data_dir = "DT_csv"
x_train = []
y_train = []


def get_F1(x_test, y_test, DP):
    dtc = DecisionTreeClassifier(max_depth=DP)
    dtc.fit(x_train, y_train)
    y_pred = dtc.predict(x_test)
    f1 = f1_score(y_test, y_pred, average='micro')
    return f1


for filename in os.listdir(data_dir)[::-1]:
    print(filename)

    with open(os.path.join(data_dir, filename)) as file:

        if "train" in filename:
            X = []
            y = []
            title = file.readline()
            cnt = 0
            while True:
                tmp = file.readline()
                if not tmp:
                    break

                tmp = tmp.split(',')
                tmp = list(map(int, tmp))
                x_train.append(tmp[0:len(tmp) - 1])
                y_train.append(tmp[len(tmp) - 1])

                cnt = cnt + 1

        else:
            title = file.readline()
            tot = 0
            ac = 0
            x_test = []
            y_test = []
            while True:
                tmp = file.readline()
                if not tmp:
                    break
                tot = tot + 1
                tmp = tmp.split(',')
                tmp = list(map(int, tmp))
                x_test_tmp = tmp[0:len(tmp) - 1]
                y_test_tmp = tmp[len(tmp) - 1]

                x_test.append(x_test_tmp)
                y_test.append(y_test_tmp)

            x_test = np.array(x_test)
            y_test = np.array(y_test)

            aix = []
            aiy = []
            for i in range(1, 20):
                aix.append(i)
                F1 = get_F1(x_test, y_test, i)
                aiy.append(F1)
                print(i, '--> f1 =', F1)

            plt.figure(1)
            plt.plot(aix, aiy)
            plt.show()

            x_train.clear()
            y_train.clear()

            pass
