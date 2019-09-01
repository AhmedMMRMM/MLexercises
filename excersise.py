from sklearn import linear_model
import pandas as pd
import numpy as np
from word2number import w2n

df = pd.read_csv('hiring.csv',sep=',')
#print(df)
df.experience=df.experience.fillna('zero')
df[['test_score']]=df[['test_score(out of 10)']]
df[['interview_score']]=df[['interview_score(out of 10)']]
df[['salary']]=df[['salary($)']]
df.drop(['test_score(out of 10)','interview_score(out of 10)','salary($)'],axis=1,inplace=True)
df[['test_score']]=df[['test_score']].fillna(0)
print(df)
df.experience = df.experience.apply(w2n.word_to_num)
reg=linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df.salary)
#print(reg.predict([[2,9,6]]))
#print(reg.coef_)
#print(reg.intercept_)
#print(reg.predict([[12,10,10]]))
x = df[['experience', 'test_score']].values
print(x)