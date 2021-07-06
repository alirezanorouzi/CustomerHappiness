#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 22:28:29 2019

@author: Alireza
"""

df=pd.read_excel('/Users/Alireza/Documents/Courses/Solar/LastRanks.xlsx',index_col=None)
df2=df
#Extract One month
#df=df[df['Year']==2019]
df=df[df['_month']==9]
df=df[df['_Start_Has_Started']=='Yes']
df['_End_Month To End'].replace(np.nan,0,inplace=True)
#df=df[df['_End_Month To End']<=0]
df=df[df['_End_Month To End']<=0]
#df=df[0<=df['_End_Month To End']]
#df=df[df['_End_Month To End']<=0]

#df=df[(df['_Customer_Rank']=='A') | (df['_Customer_Rank']=='B')]

#df=df.drop(columns=['Year'])
#df=df.drop(columns=['_HalfOfYear'])
#df=df.drop(columns=['_Quarter'])

for i in range(df.shape[0]):
    if df.iloc[i]['_End_Month To End']>0:
        print(df.iloc[i]['_End_Month To End'],df.iloc[i]['_End_Has_Left'])
        df._End_Has_Left.iloc[i]='Yes'

td=df[['Customer','_Customer_Rank','_End_TerminationDate','_End_Has_Left']]

df=df.drop(columns=['_End_TerminationDate'])
df=df.drop(columns=['Customer'])
df=df.drop(columns=['_Start_Has_Started'])
df=df.drop(columns=['_month'])
df=df.drop(columns=['_End_Month To End'])

df=df.drop(columns=['_Employee_AllCustomerChangesCount'])
##df=df.drop(columns=['_Sickness_AllCustomersSickHours'])
#df=df.drop(columns=['_Sickness_AllCustomersWorkHours'])
#df=df.drop(columns=['_Email_AllCustomersEmailCount'])
#df=df.drop(columns=['_Employee_Changes_Count'])
#df=df.drop(columns=['_Employee_AllCustomerChangesCount'])
#df=df.drop(columns=['_Employee_Change_Ratio'])

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
        #df[fld] = df.groupby("_Customer_Rank")[fld].transform(lambda x: x.fillna(x.mean()))
        NormFld(fld)
        x[fld]=df[fld]


for fld in df.columns:
    if (pd.api.types.is_float_dtype(df[fld])):
        df[fld].replace(np.nan,0,inplace=True)
        #df[fld] = df.groupby("_Customer_Rank")[fld].transform(lambda x: x.fillna(x.mean()))
        NormFld(fld)
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
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca.fit(x)  
x2=pca.transform(x)

#_____________________________________

import sklearn.metrics as mtr

y=y.reset_index(drop=True)
#Train Different Model
results2=[]
for name,r1,r2,r3,model in results:
    yhat=model.predict(x2)
    yhats=pd.Series(yhat)
    print('{}: {} ynews=={}  pre={}'.format(name,model.score(x2,y),sum(yhats==1),sum(y[yhats==1]) ))
    #print('    accuracy_score',mtr.accuracy_score(y, yhat,normalize=True))
    print('    precision_score',mtr.precision_score(y, yhat,average=None))
    print('    recall_score',mtr.recall_score(y, yhat,average=None))    
    results2.append((name,model.score(x2,y),yhat,model))
    #print(sum(ynews==0),sum(y[ynews==0]))
    
a=0.3*results2[5][2]+0.3*results2[3][2]+0.2*results2[2][2]+0.2*results2[1][2]
td['pr']=a
aa=td[a>0]


#aa['p']=a[a>0]
aa=aa.reset_index(drop=True)
for i in range(aa.shape[0]):
    a=df2[df2['Customer']==aa.iloc[i]['Customer']]
    b=a[a['_End_Has_Left']=='Yes']
    if b.shape[0]> 0 :
        b=b[b['_End_Month To End']==0]
#        print(i,b['_End_Month To End'])    
#        aa.at[i,'_End_TerminationDate'] = '*'
        if aa.at[i,'_End_Has_Left']=='No':
            aa.at[i,'_End_Has_Left'] ='Yes*' #b['_End_Has_Left']

aa=aa.sort_values(by=['_Customer_Rank','pr'],ascending=[True,False])
aa.to_csv('RankABC.csv')        
            
            
#for i in range(aa.shape[0]):
#    print(aa.iloc[i,0])
#    print(df2[df2['Customer']==aa.iloc[i,0]])
#    if item['_End_TerminationDate']==np.nan:
#        print(item['Customer']
#    

#
#def trans(p):
#    for s in p['pr']:
#        print (s)
#        
#aa.transform(lambda x: trans(x))