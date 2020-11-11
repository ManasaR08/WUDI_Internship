from __future__ import division
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans

def order_cluster(cluster_field_name, target_field_name,df_,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df_.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df_,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

def segmentation(path):
    extension = 'xlsx'
    files = [i for i in glob.glob('*.{}'.format(extension))]
    for x in files:
        df = pd.read_excel(files,encoding = "ISO-8859-1")
        df['date'] = pd.to_datetime(df['date'])
        df_user = pd.DataFrame(df['customer_name'].unique())
        df_user.columns = ['customer_name']
        df_max_purchase = df.groupby('customer_name').date.max().reset_index()
        df_max_purchase.columns = ['customer_name','MaxPurchaseDate']
        df_max_purchase['Recency'] = (df_max_purchase['MaxPurchaseDate'].max() - df_max_purchase['MaxPurchaseDate']).dt.days
        df_user = pd.merge(df_user, df_max_purchase[['customer_name','Recency']], on='customer_name')
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(df_user[['Recency']])
        df_user['RecencyCluster'] = kmeans.predict(df_user[['Recency']])
        df_user = order_cluster('RecencyCluster', 'Recency',df_user,False)
        df_frequency = df.groupby('customer_name').date.count().reset_index()
        df_frequency.columns = ['customer_name','Frequency']
        df_user_f = pd.merge(df_user, df_frequency, on='customer_name')
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(df_user_f[['Frequency']])
        df_user_f['FrequencyCluster'] = kmeans.predict(df_user_f[['Frequency']])
        df_user_f = order_cluster('FrequencyCluster', 'Frequency',df_user_f,True)
        df_profit = df.groupby('customer_name').profit.sum().reset_index()
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(df_user_m[['profit']])
        df_user_m['ProfitCluster'] = kmeans.predict(df_user_m[['profit']])
        df_user_m = order_cluster('ProfitCluster', 'profit',df_user_m,True)
        df_user_overall = df_user_m
        df_user_overall['OverallScore'] = df_user_overall['RecencyCluster'] + df_user_overall['FrequencyCluster'] + 2*(df_user_overall['ProfitCluster'])
        df_user_overall.groupby('OverallScore')['Recency','Frequency','profit'].mean()
        df_user_overall.groupby('OverallScore')['Recency'].count()
        df_user_overall['Segment'] = 'Low-Value'
        df_user_overall.loc[df_user_overall['OverallScore']>2,'Segment'] = 'Mid-Value' 
        df_user_overall.loc[df_user_overall['OverallScore']>4,'Segment'] = 'High-Value' 
        df_user_overall.to_csv('Cal_Overall_Score.csv', index = False)
        df_churn = df_user_f
        df_churn['RFScore'] = df_churn['RecencyCluster'] + 1.25*df_churn['FrequencyCluster']
        df_churn['RFpercentile'] = df_churn['RFScore']/6
        df_churn['ChurnPrediction'] = (1 - df_churn['RFpercentile'])
        df_churn['Churn_Probability'] = 'Loyal'
        df_churn.loc[df_churn['ChurnPrediction']>0.49,'Churn_Probability'] = 'Regular'
        df_churn.loc[df_churn['ChurnPrediction']>0.66,'Churn_Probability'] = 'Probable'
        df_churn.loc[df_churn['ChurnPrediction']>0.83,'Churn_Probability'] = 'Most Probable'
        df_churn.to_csv('Cal_Churn_prob.csv', index= False)
        df_combined = pd.merge(df_churn, df_user_overall, on='customer_name')
        df_combined.to_csv('Cal_Customer_Segmentation.csv', index = False)
        return
        
        