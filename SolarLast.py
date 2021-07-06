#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:28:29 2019

@author: Alireza
"""
#cols=['_QC_GradeAverage',
#'_QC_MaxGrade',
#'_Task_ClosedPlannedTaskCount',
#'_Email_Count',
#'_Task_PlannedTaskCount',
#'_Manager_MonthWorkedAverage',
#'_Manager_MinMonthWorked',
#'_Customer_Type',
#'_Manager_MaxMonthWorked',
#'_End_Has_Left']
cols=['_Email_AllCustomersEmailFactor',
      '_Customer_MonthWithSolar',
      '_Sickness_AllCustomersSickHours',
      '_Sickness_RatioDelta',
      '_Task_PlannedTaskCount',
      '_Task_ClosedPlannedTaskCount',
      '_Email_Count',
      '_End_Month To End','_Start_Has_Started','_month','Customer','_End_TerminationDate',
      '_End_Has_Left']
df=pd.read_excel('/Users/Alireza/Documents/Courses/Solar/last.xlsx',index_col=None,columns=cols)
df=df[cols]

df=df[df['_month']==7]
df=df[df['_Start_Has_Started']=='Yes']
df['_End_Month To End'].replace(np.nan,0,inplace=True)
#df=df[df['_End_Month To End']<=0]
df=df[df['_End_Month To End']<=2]
#df=df[0<=df['_End_Month To End']]

for i in range(df.shape[0]):
    if df.iloc[i]['_End_Month To End']>0:
        print(df.iloc[i]['_End_Month To End'],df.iloc[i]['_End_Has_Left'])
        df._End_Has_Left.iloc[i]='Yes'

df=df.reset_index(drop=True)

td=df[['Customer','_End_TerminationDate','_End_Has_Left']]

df=df.drop(columns=['_End_TerminationDate'])
df=df.drop(columns=['Customer'])
df=df.drop(columns=['_Start_Has_Started'])
df=df.drop(columns=['_month'])
df=df.drop(columns=['_End_Month To End'])

#df=df.drop(columns=['_End_TerminationDate'])
#df=df.drop(columns=['_CustomerID'])
#df=df.drop(columns=['_Data_From'])
#df=df.drop(columns=['_Data_To'])
#df=df.drop(columns=['_Sickness_AllCustomersSickHours'])
#df=df.drop(columns=['_Sickness_AllCustomersWorkHours'])
#df=df.drop(columns=['_Employee_AllCustomerChangesCount'])
#df=df.drop(columns=['_Email_AllCustomersEmailCount'])

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
#x2=x
#ynew=clf.predict(x2)

import sklearn.metrics as mtr

y=y.reset_index(drop=True)
#Train Different Model
results2=[]
for name,r,model in results:
    yhat=model.predict(x)
    yhats=pd.Series(yhat)
    print('{}: {} ynews=={}  pre={}'.format(name,model.score(x,y),sum(yhats==1),sum(y[yhats==1]) ))
    #print('    accuracy_score',mtr.accuracy_score(y, yhat,normalize=True))
    print('    precision_score',mtr.precision_score(y, yhat,average=None))
    print('    recall_score',mtr.recall_score(y, yhat,average=None))    
    results2.append((name,model.score(x,y),yhat,model))
    #print(sum(ynews==0),sum(y[ynews==0]))
    
a=0.1*results2[5][2]+0.1*results2[3][2]+0.4*results2[2][2]+0.4*results2[1][2]
aa=td[a>0]
aa['p']=a[a>0]