import os
import glob
import pandas as pd
import csv

def combineCSV():
    os.chdir("D:\WUDI_Internship\BusinessIntelligence\CustomerSegmentation")
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames])
    combined_csv.to_csv("combined_csv.csv")
    return

    
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