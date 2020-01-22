import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from mpl_toolkits.mplot3d import axes3d

ds = pd.read_excel('mlr03.xls')
# print(ds)
X = ds.iloc[:, :-1].values
Y = ds.iloc[:, 3].values

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=0 )
model =LinearRegression()
model.fit(x_train, y_train)
print(x_train)

xTrainPre = model.predict(x_test)
print(xTrainPre)
yPre = model.predict(x_test)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(ds["EXAM1"], ds["EXAM2"], ds["EXAM3"], c='red',marker = 'o', alpha = 0.5)
ax.set_xlabel('exam 1')
ax.set_ylabel('exam 2')
ax.set_zlabel('exam 3')
plt.show()
