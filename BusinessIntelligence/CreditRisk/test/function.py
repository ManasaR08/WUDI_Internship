from __future__ import division
from datetime import datetime, timedelta
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

def credit_risk(path):
    df = pd.read_csv(path)
    df.rename(columns ={'Unnamed: 0': 'new column name'}, inplace = True)
    df.drop(["new column name"], axis = 1, inplace = True)
    df['gross_total'] = df['debit'] - df['credit']
    df['balance'] = df.groupby('customer_name')['gross_total'].cumsum()
    df.drop(["gross_total"], axis = 1, inplace = True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['days'] = -(df.groupby('customer_name')['Date'].diff(periods=-1))
    df['days'] = df['days'].apply(lambda x: x.days)
    df = df.fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    df = df.fillna(0)
    df_debit = df.groupby('customer_name')['debit'].sum()
    df_debit.to_csv('df_debit.csv')
    df_debit = pd.read_csv('df_debit.csv', names=["customer_name", "debit"])
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df_debit[['debit']])
    df_debit['debitCluster'] = kmeans.predict(df_debit[['debit']])
    df_count.to_csv('debit_count.csv')
    df_count = pd.read_csv('debit_count.csv',names=['cluster','count'])
    k=[]
    for value in df_count['count']:
        k.append(value)
    i=0
    while(i<=4):
        if(df_count['count'][i]==max(k)):
            c = df_count['cluster'][i]
            break
        else:
            i = i+1
    s = df_count['count'].sum()
    p = max(df_debit['debitCluster'].value_counts())
    if p<0.75*s:
        df_clusters = df_debit
    else:
        df_clusters = df_debit[df_debit['debitCluster']!=c]
        if c=0:
            df_clusters['debitCluster']=df_clusters['debitCluster'].replace({1:4,2:5,3:6})
        elif c=1:
            df_clusters['debitCluster']=df_clusters['debitCluster'].replace({2:5,3:6})
        elif c=2:
            df_clusters['debitCluster']=df_clusters['debitCluster'].replace({3:6})
        df_debitc = df_debit[df_debit['debitCluster']==c]
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(df_debitc[['debit']])
        df_debitc['debitClusterc'] = kmeans.predict(df_debitc[['debit']])
        if c=1:
            df_debitc['debitCluster']=df_debitc["debitClusterc"].replace({0:1,1:2,2:3,3:4})
        elif c=2:
            df_debitc['debitClusterc']=df_debitc["debitClusterc"].replace({0:2,1:3,2:4,3:5})
        elif c=3:
            df_debitc['debitClusterc']=df_debitc["debitClusterc"].replace({0:3,1:4,2:5,3:6})
        df_debitc.drop(['debitCluster'])
        df_debitc.rename({'debitClusterc':'debitCluster'})
        df_clusters = df.concat([df_clusters,df_debitc])
        df_clusters.to_csv("clusters.csv")
        os.system('rm debit_count.csv')
        os.system('rm df_debit.csv')

return

            





        
