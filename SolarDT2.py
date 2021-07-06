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

cols=['_QC_GradeAverage',
'_QC_MaxGrade',
'_Task_ClosedPlannedTaskCount',
'_Email_Count',
'_Task_PlannedTaskCount',
'_Manager_MonthWorkedAverage',
'_Manager_MinMonthWorked',
'_Customer_Type',
'_Manager_MaxMonthWorked',
'_End_Has_Left']

cols=['_Email_AllCustomersEmailFactor',
      '_Customer_MonthWithSolar',
      '_Sickness_AllCustomersSickHours',
      '_Sickness_RatioDelta',
      '_Task_PlannedTaskCount',
      '_Task_ClosedPlannedTaskCount',
      '_Email_Count','_End_Month To End','_Start_Has_Started','_month','_End_Has_Left']
df=pd.read_excel('/Users/Alireza/Documents/Courses/Solar/last.xlsx',index_col=None,columns=cols)
df=df[cols]

df=df[df['_month']==5]
df=df[df['_Start_Has_Started']=='Yes']
df['_End_Month To End'].replace(np.nan,0,inplace=True)
#df=df[df['_End_Month To End']<=0]
df=df[df['_End_Month To End']<=2]
#df=df[0<=df['_End_Month To End']]

for i in range(df.shape[0]):
    if df.iloc[i]['_End_Month To End']>0:
        print(df.iloc[i]['_End_Month To End'],df.iloc[i]['_End_Has_Left'])
        df._End_Has_Left.iloc[i]='Yes'

#df=df.drop(columns=['_End_TerminationDate'])
#df=df.drop(columns=['Customer'])
df=df.drop(columns=['_Start_Has_Started'])
df=df.drop(columns=['_month'])
df=df.drop(columns=['_End_Month To End'])


# Y
#_End_Has_Left  _End_Reason_Quality
target='_End_Has_Left'
FeildLableEncoder(target) 
y=df[target+'bl']

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
        x[fld].fillna(0, inplace = True)
for fld in df.columns:
    if (pd.api.types.is_string_dtype(df[fld])):
        df[fld].replace(np.nan,'other',inplace=True)
        FeildLableEncoder(fld)
        x[fld+'bl']=df[fld+'bl']

if target+'bl' in x.columns:
    x=x.drop(columns=[target+'bl'])
if target in x.columns:
    x=x.drop(columns=[target])
#-------------------------------

#fs= SelectKBest(f_classif, k=9) #6
#x2 = fs.fit_transform(x,y) 
x2=x
#-------------------------------
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x2, y)
yhat=clf.predict(x2)

import sklearn.metrics as mtr

print('accuracy_score',mtr.accuracy_score(y, yhat,normalize=True))
print('precision_score',mtr.precision_score(y, yhat,average=None))
print('recall_score',mtr.recall_score(y, yhat,average=None))

#-------------------------------
#mask = fs.get_support() #list of booleans
#new_features = [] # The list of your K best features
#feature_names = list(x.columns.values)
#for bool, feature in zip(mask, feature_names):
#    if bool:
#        new_features.append(feature)
#
##After that, change the name of your features:
#
#dataframe = pd.DataFrame(x, columns=new_features)

#---------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut

clf = tree.DecisionTreeClassifier()

kf=KFold(n_splits=7,random_state=0)
cvs=cross_val_score(clf,x2,y,cv=kf,scoring='accuracy')
print ('accuracy=:',cvs.mean())
cvs=cross_val_score(clf,x2,y,cv=kf,scoring='precision')
print ('precision',cvs.mean())
cvs=cross_val_score(clf,x2,y,cv=kf,scoring='recall')
print ('recall',cvs.mean())

clf.fit(x,y)    
yhat=clf.predict(x)
#

#--------------- importance feature
fim=pd.Series ( clf.feature_importances_)
fn=df.columns.to_series()
fl=pd.concat([fn.reset_index(drop=True) ,fim],axis=1)
fl=fl.sort_values(by=[1])
fl=fl.reset_index()
print(fl)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

#Train Different Model
models=[]
models.append(('SVC',SVC()))
models.append(('LR', LogisticRegression()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('Knn', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=10, p=2,
                     weights='uniform')))
models.append(('RF', RandomForestClassifier()))
models.append(('GBC', GradientBoostingClassifier()))

results=[]
name=[]

for name,model in models:
    print(name)
    kfold=KFold(n_splits=10,random_state=1)
    cvs=cross_val_score(model,x,y,cv=kfold,scoring='accuracy')
    model.fit(x,y)
    results.append((name,cvs.mean(),model))
    print('{}: {}'.format(name,cvs.mean()))
