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

#cols=['_QC_GradeAverage',
#'_QC_MaxGrade',
#'_Task_ClosedPlannedTaskCount',
#'_Email_Count',
#'_Task_PlannedTaskCount',
#'_Manager_MonthWorkedAverage',
#'_Manager_MinMonthWorked',
#'_Customer_Typebl',
#'_Manager_MaxMonthWorked']
df=pd.read_excel('/Users/Alireza/Documents/Courses/Solar/Last.xlsx',index_col=None)#,columns=cols)

#Extract One month
df=df[df['_month']==6]
df=df[df['_Start_Has_Started']=='Yes']
df['_End_Month To End'].replace(np.nan,0,inplace=True)
df=df[df['_End_Month To End']<=0]
#df=df[df['_End_Month To End']<=2]
#df=df[0<=df['_End_Month To End']]


df=df.drop(columns=['_End_TerminationDate'])
df=df.drop(columns=['Customer'])
df=df.drop(columns=['_Start_Has_Started'])
df=df.drop(columns=['_month'])
df=df.drop(columns=['_End_Month To End'])
#df=df.drop(columns=['_Sickness_AllCustomersSickHours'])
df=df.drop(columns=['_Sickness_AllCustomersWorkHours'])
#df=df.drop(columns=['_Employee_AllCustomerChangesCount'])
df=df.drop(columns=['_Email_AllCustomersEmailCount'])

df=df.drop(columns=['_Employee_Changes_Count'])
df=df.drop(columns=['_Employee_AllCustomerChangesCount'])
df=df.drop(columns=['_Employee_Change_Ratio'])
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

fs= SelectKBest(f_classif, k=9) #6
x2 = fs.fit_transform(x,y) 
#-------------------------------
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x2, y)
yhat=clf.predict(x2)

import sklearn.metrics as mtr

print('accuracy_score',mtr.accuracy_score(y, yhat,normalize=True))
print('precision_score',mtr.precision_score(y, yhat,average=None))
print('recall_score',mtr.recall_score(y, yhat,average=None))

a=np.array(y)
a2=a==1
print(sum(yhat[a2]),sum(a2),sum(yhat[a2])/sum(a2))
print(sum(yhat==1))

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
from sklearn.model_selection import LeaveOneOut

clf = tree.DecisionTreeClassifier()

kf=KFold(n_splits=7,random_state=0)
cvs=cross_val_score(clf,x2,y,cv=kf,scoring='accuracy')
print ('accuracy=:',cvs.mean())
cvs=cross_val_score(clf,x2,y,cv=kf,scoring='precision')
print ('precision',cvs.mean())
cvs=cross_val_score(clf,x2,y,cv=kf,scoring='recall')
print ('recall',cvs.mean())

clf.fit(x2,y)    
yhat=clf.predict(x2)


#--------------- importance feature
fim=pd.Series ( clf.feature_importances_)
fn=dataframe.columns.to_series()
fl=pd.concat([fn.reset_index(drop=True) ,fim],axis=1)
fl=fl.sort_values(by=[1])
fl=fl.reset_index()
print(fl)