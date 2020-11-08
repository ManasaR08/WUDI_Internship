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
    files = [i for i in glob.glob('*.{}'.format(extension))]
    final_headers = ['InvoiceNo', 'StockCode', 'Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']
    merged_rows = set()
    for f in files:
        with open(f, 'r') as csv_in:
            csvreader = csv.reader(csv_in, delimiter=',')
            headers = dict((h, i) for i, h in enumerate(next(csvreader)))
            for row in csvreader:
                merged_rows.add(tuple(row[headers[x]] for x in final_headers))
    with open('output.csv', 'w') as csv_out:
        csvwriter = csv.writer(csv_out, delimiter=',')
        csvwriter.writerows(merged_rows)
        df = pd.read_csv("output.csv",encoding= 'unicode_escape', names=["InvoiceNo", "StockCode", "Description", "Quantity","InvoiceDate","UnitPrice","CustomerID","Country"])
        df.to_csv("combined_csv")
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
    df_nor_melt.to_csv('output/result.csv')
    return                 

path = 'D:\WUDI_Internship\BusinessIntelligence\CustomerSegmentation'
customer_seg(path)

