#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 10:03:44 2019

@author: Alireza
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


def FeildLableEncoder(fld):
    lb_make = LabelEncoder() 
    df[fld+'bl'] = lb_make.fit_transform(df[fld]) 
    return

# Read Data ------------------------------------------------------------------------------------------------------------
#df=pd.read_excel('/Users/Alireza/Documents/Courses/Solar/CAD2019.xlsx',index_col=None)


df=pd.read_excel('/Users/Alireza/Documents/Courses/Solar/last3.xlsx',index_col=None)

df['_End_TerminationDate']=df['_End_TerminationDate'].str[0:7]

df=df[df['_End_Has_Left']=='Yes']
df=df.drop(columns=['_End_Has_Left'])
df=df.drop(columns=['_CustomerID'])
df=df.drop(columns=['_Data_From'])
df=df.drop(columns=['_Data_To'])
df=df.drop(columns=['_Sickness_AllCustomersSickHours'])
df=df.drop(columns=['_Sickness_AllCustomersWorkHours'])
df=df.drop(columns=['_Employee_AllCustomerChangesCount'])
df=df.drop(columns=['_Email_AllCustomersEmailCount'])


x=pd.DataFrame()
#---------------------------------------
for fld in df.columns:
    if (pd.api.types.is_integer_dtype(df[fld])):
        df[fld].replace(np.nan,0,inplace=True)
        x[fld]=df[fld]


for fld in df.columns:
    if (pd.api.types.is_float_dtype(df[fld])):
        df[fld].replace(np.nan,0,inplace=True)
        x[fld]=df[fld]
        
for fld in df.columns:
    if (pd.api.types.is_string_dtype(df[fld])):
        df[fld].replace(np.nan,'other',inplace=True)
        FeildLableEncoder(fld)
        x[fld+'bl']=df[fld+'bl']

#_______________________________________ Y
#_End_Has_Left  _End_Reason_Quality
dateCode=df[['_End_TerminationDate','_End_TerminationDatebl']]
target='_End_TerminationDate'
FeildLableEncoder(target) 
y=df[target+'bl']

#---------------------

if target+'bl' in x.columns:
    x=x.drop(columns=[target+'bl'])
if target in x.columns:
    x=x.drop(columns=[target])
#-------------------------------

fs= SelectKBest(f_classif, k=6)
x2 = fs.fit_transform(x,y) 
#-------------------------------
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x2, y)
yhat=clf.predict(x2)

import sklearn.metrics as mtr

print('accuracy_score',mtr.accuracy_score(y, yhat,normalize=True))
print('precision_score',mtr.precision_score(y, yhat,average=None))
#
#a=np.array(y)
#a2=a==14
#print(sum(yhat[a2]),sum(a2),sum(yhat[a2])/sum(a2))
#print(sum(yhat==1))

#-------------------------------
mask = fs.get_support() #list of booleans
new_features = [] # The list of your K best features
feature_names = list(x.columns.values)
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)

#After that, change the name of your features:

dataframe = pd.DataFrame(x, columns=new_features)

#---------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
 
clf = tree.DecisionTreeClassifier()

kf=KFold(n_splits=50,random_state=1)
cvs=cross_val_score(clf,x2,y,cv=kf,scoring='accuracy')
# neg_mean_absolute_error
print ('accuracy',cvs.mean())

clf.fit(x2,y)    
yhat=clf.predict(x2)


#--------------- importance feature
fim=pd.Series ( clf.feature_importances_)
fn=dataframe.columns.to_series()
fl=pd.concat([fn.reset_index(drop=True) ,fim],axis=1)
fl=fl.sort_values(by=[1])
fl=fl.reset_index()
print(fl)