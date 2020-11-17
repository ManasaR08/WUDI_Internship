from __future__ import division
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import os

def order_cluster(cluster_field_name, target_field_name,df_,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df_.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df_,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

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
    df['CEP'] = df['balance']/df['days']
    df.replace([np.inf, -np.inf], np.nan, inplace=True) 
    df = df.fillna(0)
    df_debit = df.groupby('customer_name')['debit'].sum()
    df_debit.to_csv('df_debit.csv')
    df_debit = pd.read_csv('df_debit.csv')
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(df_debit[['debit']])
    df_debit['debitCluster'] = kmeans.predict(df_debit[['debit']])
    df_debit = order_cluster('debitCluster', 'debit',df_debit,False)
    df_cep = df.groupby('customer_name')['CEP'].mean()
    df_cep.to_csv('df_cep.csv')
    df_cep = pd.read_csv('df_cep.csv')
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(df_cep[['CEP']])
    df_cep['CEPCluster'] = kmeans.predict(df_cep[['CEP']])
    df_cep = order_cluster('CEPCluster', 'CEP',df_cep,False)
    df_clusters=df_debit
    df_clusters['CEP'] = df_cep['CEP']
    df_clusters['CEPCluster'] = df_cep['CEPCluster']
    df_purchase = df[df.debit != 0]
    df_max_purchase = df_purchase.groupby('customer_name').Date.max().reset_index()
    df_max_purchase.columns = ['customer_name','MaxPurchaseDate']
    df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days
    df_clusters = pd.merge(df_clusters, df_max_purchase[['customer_name','Recency']], on='customer_name')
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(df_clusters[['Recency']])
    df_clusters['RecencyCluster'] = kmeans.predict(df_clusters[['Recency']])
    df_clusters = order_cluster('RecencyCluster', 'Recency',df_clusters,False)
    df_frequency = df_purchase.groupby('customer_name').Date.count().reset_index() #here total purchases are considedred intead of voucher_number based purchases
    df_frequency.columns = ['customer_name','Frequency']
    df_clusters = pd.merge(df_clusters, df_frequency, on='customer_name')
    kmeans = KMeans(n_clusters=8)
    kmeans.fit(df_clusters[['Frequency']])
    df_clusters['FrequencyCluster'] = kmeans.predict(df_clusters[['Frequency']])
    df_clusters = order_cluster('FrequencyCluster', 'Frequency',df_clusters,False)
    df_clusters['debitCluster']=df_clusters['debitCluster'].replace({0:7,1:6,2:5,3:4,4:3,5:2,6:1,7:0})
    df_clusters['CEPCluster']=df_clusters['CEPCluster'].replace({0:7,1:6,2:5,3:4,4:3,5:2,6:1,7:0})
    df_clusters['FrequencyCluster']=df_clusters['FrequencyCluster'].replace({0:7,1:6,2:5,3:4,4:3,5:2,6:1,7:0})
    df_clusters['overall_score'] = df_clusters['RecencyCluster'] + df_clusters['FrequencyCluster'] + df_clusters['CEPCluster'] +df_clusters['debitCluster']
    df_clusters['segment'] = 'Low-Value'
    df_clusters.loc[df_clusters['overall_score']>12,'segment'] = 'Mid-Value' 
    df_clusters.loc[df_clusters['overall_score']>16,'segment'] = 'High-Value' 
    df_clusters.to_csv('credit_risk.csv')
    os.remove("df_debit.csv")
    os.remove("df_cep.csv")
    return

credit_risk('result.csv')





        
