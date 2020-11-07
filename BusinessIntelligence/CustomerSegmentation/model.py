from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import os
import glob


def customer_seg(path):
    os.chdir(path)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')
    df = pd.read_csv('combined_csv.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df["InvoiceDate"] = df["InvoiceDate"].dt.date
    df["TotalSum"] = df["Quantity"]*df["UnitPrice"]
    snapshot_date = max(df.InvoiceDate) + datetime.timedelta(days=1)
    customers = df.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})
    customers.rename(columns = {'InvoiceDate': 'Recency',
                            'InvoiceNo': 'Frequency',
                            'TotalSum': 'MonetaryValue'}, inplace=True)
    customers_fix = pd.DataFrame()
    customers_fix["Recency"] = stats.boxcox(customers['Recency'])[0]
    customers_fix["Frequency"] = stats.boxcox(customers['Frequency'])[0]
    customers_fix["MonetaryValue"] = pd.Series(np.cbrt(customers['MonetaryValue'])).values   
    scaler = StandardScaler()
    scaler.fit(customers_fix)
    customers_normalized = scaler.transform(customers_fix)          
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(customers_normalized)
    customers["Cluster"] = model.labels_
    customers.groupby('Cluster').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean', 'count']}).round(2)
    df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
    df_normalized['ID'] = customers.index
    df_normalized['Cluster'] = model.labels_
    df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency','Frequency','MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')
    sns.lineplot('Attribute', 'Value', hue='Cluster', data=df_nor_melt)
    pickle.dump(model,open('model.pkl','wb'))
    model = pickle.load(open('model.pkl','rb'))
    df_nor_melt.to_csv('result.csv')
    return                 

path = 'D:\WUDI_Internship\BusinessIntelligence\CustomerSegmentation'
customer_seg(path)

