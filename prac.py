import pandas as p 
import numpy as np
import matplotlib.pyplot as plt

dataset = p.read_excel('mlr03.xls')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,3:4]

 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size= 0.2/3)

model = LinearRegression()
model.fit(x,y)

y_predictions = model.predict(x_train)






_