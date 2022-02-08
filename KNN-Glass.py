"""Prepare a model for glass classification using KNN

Data Description:

RI : refractive index

Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)

Mg: Magnesium

AI: Aluminum

Si: Silicon

K:Potassium

Ca: Calcium

Ba: Barium

Fe: Iron

Type: Type of glass: (class attribute)
1 -- building_windows_float_processed
 2 --building_windows_non_float_processed
 3 --vehicle_windows_float_processed
 4 --vehicle_windows_non_float_processed (none in this database)
 5 --containers
 6 --tableware
 7 --headlamps
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

glass = pd.read_csv('C:/ExcelrData/Data-Science_Assignments/KNN/glass.csv')

# to split train and test data
from sklearn.model_selection import train_test_split

train, test = train_test_split(glass, test_size=0.3, random_state=0)

# KNN
from sklearn.neighbors import KNeighborsClassifier as KNC

# to find best k value
acc = []
for i in range(3, 50, 2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:, 0:9], train.iloc[:, 9])
    train_acc = np.mean(neigh.predict(train.iloc[:, 0:9]) == train.iloc[:, 9])
    test_acc = np.mean(neigh.predict(test.iloc[:, 0:9]) == test.iloc[:, 9])
    acc.append([train_acc, test_acc])

plt.plot(np.arange(3, 50, 2), [i[0] for i in acc], 'bo-')
plt.plot(np.arange(3, 50, 2), [i[1] for i in acc], 'ro-')
plt.legend(['train', 'test'])
plt.show()

# from plots at k=5 we get best model
# model building at k=5
neigh = KNC(n_neighbors=5)
neigh.fit(train.iloc[:, 0:9], train.iloc[:, 9])
pred_train = neigh.predict(train.iloc[:, 0:9])
train_acc = np.mean(pred_train == train.iloc[:, 9])
print("train_acc: ", train_acc)  # 0.7651
pred_test = neigh.predict(test.iloc[:, 0:9])
test_acc = np.mean(pred_test == test.iloc[:, 9])
print("test_acc: ", test_acc)  # 0.661