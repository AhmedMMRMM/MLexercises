import pandas as pd
from sklearn import linear_model
from numpy import reshape
import numpy as np


#import matplotlib.pyplot as plt

df = pd.read_csv('linear.csv',sep='\t')
#print(df)
#print(df[['area']],df[['price']],sep='\n')
reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
area = df[['area']]
price = df.price
#price.reshape(1, -1)
reg.fit(area,price)
reg.coef_


#print(reg.predict(3300))
print(reg.coef_)
print(reg.intercept_)
