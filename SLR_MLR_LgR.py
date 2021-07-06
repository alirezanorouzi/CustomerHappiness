#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:39:07 2018

@author: Alireza
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df=pd.read_csv('result.csv')

x=df[['horsepower','curb-weight','engine-size','highway-mpg','city-mpg']]
y=df[['price']]
 
lm=LinearRegression()
lm.fit(x,y)
yhat=lm.predict(x)

er=abs(y-yhat)
mae=er.mean()
erp=mae*100/(y.max()-y.min())

mse=mean_squared_error(y,yhat)

acc=lm.score(x,y)

#_______Split Train & Test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

lm=LinearRegression()
lm.fit(x_train,y_train)
acc=lm.score(x_test,y_test)

#______Crossvalidation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

lm=LinearRegression()
kfold=KFold(n_splits=3,random_state=7)
'''kfold=LeaveOneOut()'''
cvs=cross_val_score(lm,x,y,cv=kfold)
#----poly
from sklearn.preprocessing import PolynomialFeatures

x=df[['engine-size']]
y=df[['price']]

poly = PolynomialFeatures(degree=3)
x_trans = poly.fit_transform(x)

lm=LinearRegression()
lm.fit(x_trans,y)
yhat=lm.predict(x_trans)

er=abs(y-yhat)
mae=er.mean()
erp=mae*100/(y.max()-y.min())
#_____
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.scatter(x,yhat)

#-----Logistic
from sklearn.linear_model import LogisticRegression

x=df[['horsepower','curb-weight','engine-size','highway-mpg','city-mpg']]
h=df['price']>10000
df['b-price']=h
df['b-price']=df['b-price'].astype('int')
y=df['b-price']

lm=LogisticRegression()
lm.fit(x,y)
yhat=lm.predict(x)

er=abs(y-yhat)
mae=er.mean()
erp=mae*100/(y.max()-y.min())

#-----
import seaborn as sns

sns.regplot(x='horsepower',y='price',data=df)

sns.residplot(df['horsepower'],df['price'])


#------
import numpy as np

x=df['engine-size']
y=df['price']
f=np.polyfit(x,y,3)
p=np.poly1d(f)
print(p)


#______SVM
from sklearn import svm
import numpy as np

x=df[['horsepower','curb-weight','engine-size','highway-mpg','city-mpg']]
h=df['price']>10000
df['b-price']=h
df['b-price']=df['b-price'].astype('int')
y=df['b-price']+1

#x=np.asarray(x)
#y=np.asarray(y)

svmModel=svm.SVC(kernel='linear')
svmModel.fit(x,y)
yhat=lm.predict(x)

er=abs(y-yhat)
mae=er.mean()
erp=mae*100/(y.max()-y.min())

#______Scale
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

sc=StandardScaler()
sc.fit(x)
xsc=sc.transform(x)

lm=LinearRegression()
lm.fit(xsc,y)
yhat=lm.predict(xsc)

#______Pipline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

lin=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('LR',LinearRegression()),( 'fit',fit(x,y))]
pip=Pipeline(lin)
pip.fit(x,y)
yhat=pip.predict(x)




