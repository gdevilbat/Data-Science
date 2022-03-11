from django.http import HttpResponse
from django.http import JsonResponse

import base64
from io import BytesIO

import math
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from contextlib import closing
import csv
from sklearn import preprocessing

def tugasMembacaData(request):
    csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data.csv")

    print('\r\n\r\n Full Data :')
    print(csv_data)

    print('\r\n\r\n Cuplikat Data :')
    print(csv_data.head())

    print('\r\n\r\n Melihat Column :')
    print(csv_data.columns)

    print('\r\n\r\n Akses Baris :')
    print(csv_data.iloc[5])

    print('\r\n\r\n Menampilkan suatu data dari baris dan kolom tertentu :')
    print(csv_data['Age'].iloc[1])

    print("\r\n\r\n Menampilkan data row ke 5 sampai kurang dari 10 :")
    print(csv_data.iloc[5:10])

    print("\r\n\r\n Menampilkan data Column Tertentu row ke 5 sampai kurang dari 10 :")
    print(csv_data['Age'].iloc[5:10])

    print("\r\n\r\n Menampilkan Informasi Statistic :")
    print(csv_data.describe(include='all'))

    print("\r\n\r\n Menampilkan Informasi Statistic Hanya yang Numeric :")
    print(csv_data.describe(exclude=['O']))

    return HttpResponse("Output On Console")

def tugasIsiDataKosongMeanMedian(request):
    csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data.csv")

    print('\r\n\r\n Pengecekan Nilai Null :')
    print(csv_data.isnull().values.any())
    
    csv_data = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/shopping_data_missingvalue.csv")

    print('\r\n\r\n Pengecekan Nilai Null :')
    print(csv_data.isnull().values.any())
    print(csv_data)

    #Lakukan Mean
    print("\r\n\r\n Dataset yang masih terdapat nilai kosong ! :")
    print(csv_data.head(10))

    csv_data=csv_data.fillna(csv_data.mean())
    print("\r\n\r\n Dataset yang sudah diproses Handling Missing Values dengan Mean :")
    print(csv_data.head(10))

    #Lakukan Median
    csv_data = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/shopping_data_missingvalue.csv")
    print("\r\n\r\n Dataset yang masih terdapat nilai kosong ! :")
    print(csv_data.head(10))

    csv_data=csv_data.fillna(csv_data.median())
    print("\r\n\r\n Dataset yang sudah diproses Handling Missing Values dengan Median :")
    print(csv_data.head(10))

    return HttpResponse("Output On Console")

def tugasNormalisasiData(request):
    # Metode MinMax, Zscore, Decimal Scaling, Sigmoid, dan Softmax

    csv_data = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/shopping_data.csv")
    array = csv_data.values

    X = array[:,2:5] #memisahkan fitur dari dataset. 
    Y = array[:,0:1]  #memisahkan class dari dataset

    print(csv_data.head(10));

    dataset=pd.DataFrame({'Customer ID':array[:,0],'Gender':array[:,1],'Age':array[:,2],'Income':array[:,3],'Spending Score':array[:,4]})
    print("\r\n\r\n dataset sebelum dinormalisasi :")
    print(dataset.head(10))

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1)) #inisialisasi normalisasi MinMax
    data = min_max_scaler.fit_transform(X) #transformasi MinMax untuk fitur
    dataset = pd.DataFrame({'Age':data[:,0],'Income':data[:,1],'Spending Score':data[:,2],'Customer ID':array[:,0],'Gender':array[:,1]})

    print("\r\n\r\n dataset setelah dinormalisasi :")
    print(dataset.head(10))

    return HttpResponse("Output On Console")

def tugasProffesionalBeginner(request):
    list_cash_flow = [
    2500000, 5000000, -1000000, -2500000, 5000000, 10000000,
    -5000000, 7500000, 10000000, -1500000, 25000000, -2500000
    ]
    total_pengeluaran, total_pemasukan = 0, 0
    for dana in list_cash_flow:
        if dana > 0:
            total_pemasukan += dana
        else:
            total_pengeluaran += dana
    total_pengeluaran *= -1

    return HttpResponse(str(total_pengeluaran)+"<br/>"+str(total_pemasukan))

def tugasMultiLine(request):
    # Baca dataset
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    # Buat kolom baru yang bertipe datetime dalam format '%Y-%m'
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    # Buat Kolom GMV
    dataset['gmv'] = dataset['item_price']*dataset['quantity']

    # Buat Multi-Line Chart
    #plt.switch_backend('AGG')
    # dataset.groupby(['order_month', 'brand'])['gmv'].sum().unstack().plot() # group by brand
    dataset.groupby(['order_month', 'province'])['gmv'].sum().unstack().plot(cmap='Set1') # cmap untuk set warna
    plt.title('Monthly GMV Year 2019 - Breakdown by Brand', loc='center', pad=30, fontsize=20, color='blue')
    plt.xlabel('Order Month', fontsize=15)
    plt.ylabel('Total Amount (in Billions)', fontsize = 15)
    plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
    plt.ylim(ymin=0)
    labels, locations = plt.yticks()
    #plt.yticks(labels, (labels/1000000000).astype(int))
    plt.legend(loc='right', bbox_to_anchor=(0.5, -0.5), shadow=True, ncol=2) # Customisation Legend
    plt.gcf().set_size_inches(12, 7)
    plt.tight_layout()

    graph = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />")

def tugasMultiLineTopProvinsi(request):
    # Baca dataset
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    # Buat Kolom GMV
    dataset['gmv'] = dataset['item_price']*dataset['quantity']

    # Buat variabel untuk 5 propinsi dengan GMV tertinggi
    top_provinces = (dataset.groupby('province')['gmv']
                            .sum()
                            .reset_index()
                            .sort_values(by='gmv', ascending=False).head(5)
                            )

    dataset['province_top'] = dataset['province'].apply(lambda x: x if(x in top_provinces['province'].to_list()) else 'other')

    dataset.groupby(['order_month', 'province_top'])['gmv'].sum().unstack().plot(marker='.', cmap='plasma') # cmap untuk set warna
    plt.title('Monthly GMV Year 2019 - Breakdown by Province', loc='center', pad=30, fontsize=20, color='blue')
    plt.xlabel('Order Month', fontsize=15)
    plt.ylabel('Total Amount (in Billions)', fontsize = 15)
    plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
    plt.ylim(ymin=0)
    labels, locations = plt.yticks()
    #plt.yticks(labels, (labels/1000000000).astype(int))
    plt.legend(loc='right', bbox_to_anchor=(0.5, -0.5), shadow=True, ncol=2) # Customisation Legend
    plt.gcf().set_size_inches(12, 7)
    plt.tight_layout()

    graph = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />")

def tugasMultiLineAnnotation(request):
     # Baca dataset
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    # Buat Kolom GMV
    dataset['gmv'] = dataset['item_price']*dataset['quantity']

    # Buat variabel untuk 5 propinsi dengan GMV tertinggi
    top_provinces = (dataset.groupby('province')['gmv']
                            .sum()
                            .reset_index()
                            .sort_values(by='gmv', ascending=False).head(5)
                            )

    dataset['province_top'] = dataset['province'].apply(lambda x: x if(x in top_provinces['province'].to_list()) else 'other')

    dataset.groupby(['order_month', 'province_top'])['gmv'].sum().unstack().plot(marker='.', cmap='plasma') # cmap untuk set warna
    plt.title('Monthly GMV Year 2019 - Breakdown by Province', loc='center', pad=30, fontsize=20, color='blue')
    plt.xlabel('Order Month', fontsize=15)
    plt.ylabel('Total Amount (in Billions)', fontsize = 15)
    plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
    plt.ylim(ymin=0)
    labels, locations = plt.yticks()
    plt.yticks(labels, (labels/1000000000).astype(int))
    plt.legend(loc='right', bbox_to_anchor=(0.5, -0.5), shadow=True, ncol=2) # Customisation Legend
    # Anotasi pertama
    plt.annotate('GMV other meningkat pesat', xy=(5,900000000), xytext=(4, 1700000000), weight='bold', color='red',arrowprops=dict(arrowstyle='fancy',connectionstyle="arc3", color='red'))
    # Anotasi kedua
    plt.annotate('DKI Jakarta mendominasi', xy=(3,3350000000), xytext=(0, 3800000000), weight='bold', color='red', arrowprops=dict(arrowstyle='->', connectionstyle="angle", color='red'))
    plt.gcf().set_size_inches(12, 7)
    plt.tight_layout()

    graph = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />")

def tugasPieBarChart(request):
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    dataset['gmv'] = dataset['item_price']*dataset['quantity']

    dataset_dki_q4 = dataset[(dataset['province']=='DKI Jakarta') & (dataset['order_month'] >= '2019-10')]

    gmv_per_city_dki_q4 = dataset_dki_q4.groupby('city')['gmv'].sum().reset_index()
    plt.figure(figsize=(6,6))
    plt.pie(gmv_per_city_dki_q4['gmv'],labels = gmv_per_city_dki_q4['city'], autopct='%1.2f%%')
    plt.title('GMV Contribution Per City - DKI Jakarta in Q4 2019', loc='center', pad=30, fontsize=15, color='blue')

    pie = getGraph()

    plt.clf()
    dataset_dki_q4.groupby('city')['gmv'].sum().sort_values(ascending=False).plot(kind='bar', color='green')
    plt.title('GMV Per City - DKI Jakarta in Q4 2019', loc='center', pad=30, fontsize=15, color='blue')
    plt.xlabel('City', fontsize=15)
    plt.ylabel('Total Amount (in Billions)', fontsize=15)
    plt.ylim(ymin=0)
    labels, locations = plt.yticks()
    plt.yticks(labels, (labels/1000000000).astype(float))
    plt.xticks(rotation=0)
    plt.gcf().set_size_inches(12, 7)

    bar = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+pie+"' />"+"<br/>"+"<img src='data:image/png;base64, "+bar+"' />")

def tugasMultiBarChart(request):
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    dataset['gmv'] = dataset['item_price']*dataset['quantity']
    dataset_dki_q4 = dataset[(dataset['province']=='DKI Jakarta') & (dataset['order_month'] >= '2019-10')]

    dataset_dki_q4.groupby(['city', 'order_month'])['gmv'].sum().unstack().plot(kind='bar')
    plt.title('GMV Per City - DKI Jakarta in Q4 2019', loc='center', pad=30, fontsize=15, color='blue')
    plt.xlabel('Province', fontsize=15)
    plt.ylabel('Total Amount (in Billions)', fontsize=15)
    plt.ylim(ymin=0)
    labels, locations = plt.yticks()
    plt.yticks(labels, (labels/1000000000).astype(float))
    plt.xticks(rotation=0)
    plt.gcf().set_size_inches(12, 7)

    graph = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />")

def tugasStackedChart(request):
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    dataset['gmv'] = dataset['item_price']*dataset['quantity']
    dataset_dki_q4 = dataset[(dataset['province']=='DKI Jakarta') & (dataset['order_month'] >= '2019-10')]

    dataset_dki_q4.groupby(['order_month', 'city'])['gmv'].sum().unstack().plot(kind='bar', stacked=True)
    plt.title('GMV Per City - DKI Jakarta in Q4 2019', loc='center', pad=30, fontsize=15, color='blue')
    plt.xlabel('Province', fontsize=15)
    plt.ylabel('Total Amount (in Billions)', fontsize=15)
    plt.ylim(ymin=0)
    labels, locations = plt.yticks()
    plt.yticks(labels, (labels/1000000000).astype(float))
    plt.xticks(rotation=0)
    plt.legend(loc='right', bbox_to_anchor=(1, 1), shadow=True, ncol=2, title='City') # Customisation Legend
    plt.tight_layout()
    plt.gcf().set_size_inches(12, 7)

    graph = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />")

def tugasHistogramScatterplot(request):
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    dataset['gmv'] = dataset['item_price']*dataset['quantity']
    dataset_dki_q4 = dataset[(dataset['province']=='DKI Jakarta') & (dataset['order_month'] >= '2019-10')]

    # bins: jumlah bin (kelompok nilai) yang diinginkan
    # range: nilai minimum dan maksimum yang ditampilkan
    # orientation: ‘horizontal’ atau ‘vertikal’
    # color: warna bar di histogram

    data_per_customer = (dataset_dki_q4.groupby('customer_id')
                                    .agg({'order_id':'nunique', 
                                            'quantity': 'sum', 
                                            'gmv':'sum'})
                                    .reset_index()
                                    .rename(columns={'order_id':'orders'})
                                    .sort_values(by='orders', ascending=False))

    plt.figure()
    plt.hist(data_per_customer['orders'], range=(1,5))
    plt.title('Distribution of Number of Orders per Customer\nDKI Jakarta in Q4 2019', fontsize=15,color='blue')
    plt.xlabel('Number of Orders', fontsize = 12)
    plt.ylabel('Number of Customers', fontsize = 12)

    orders = getGraph()

    plt.clf()
    plt.figure()
    plt.hist(data_per_customer['quantity'], bins=100, range=(1,200), color='brown')
    plt.title('Distribution of Total Quantity per Customer\nDKI Jakarta in Q4 2019', fontsize=15,color='blue')
    plt.xlabel('Quantity', fontsize = 12)
    plt.ylabel('Number of Customers', fontsize = 12)

    quantity = getGraph()

    plt.clf()
    plt.figure(figsize=(10,5))
    plt.hist(data_per_customer['gmv'], bins=100, range=(1,200000000), color='green')
    plt.title('Distribution of Total GMV per Customer\nDKI Jakarta in Q4 2019', fontsize=15, color='blue')
    plt.xlabel('GMV (in Millions)', fontsize = 12)
    plt.ylabel('Number of Customers', fontsize = 12)
    plt.xlim(xmin=0, xmax=200000000)
    labels, locations = plt.xticks()
    plt.xticks(labels, (labels/1000000).astype(int))

    gmv = getGraph()

    plt.clf()
    plt.figure(figsize=(10,8))
    plt.scatter(data_per_customer['quantity'], data_per_customer['gmv'], marker='+', color='red')
    plt.title('Correlation of Quantity and GMV per Customer\nDKI Jakarta in Q4 2019', fontsize=15, color='blue')
    plt.xlabel('Quantity', fontsize = 12)
    plt.ylabel('GMV (in Millions)', fontsize = 12)
    plt.xlim(xmin=0, xmax=300)
    plt.ylim(ymin=0, ymax=150000000)
    labels, locations = plt.yticks()
    plt.yticks(labels, (labels/1000000).astype(int))

    correlation = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+orders+"' />"+"<br/>"+"<img src='data:image/png;base64, "+quantity+"' />"+"<br/>"+"<img src='data:image/png;base64, "+gmv+"' />"+"<br/>"+"<img src='data:image/png;base64, "+correlation+"' />")

def tugasMiniProject(request):
    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced.csv')
    dataset['order_month'] = dataset['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").strftime('%Y-%m'))
    dataset['gmv'] = dataset['item_price']*dataset['quantity']

    top_brands = (dataset[dataset['order_month']=='2019-12'].groupby('brand')['quantity']
                .sum()
                .reset_index()
                .sort_values(by='quantity',ascending=False)
                .head(5))

    dataset_top5brand_dec = dataset[(dataset['order_month']=='2019-12') & (dataset['brand'].isin(top_brands['brand'].to_list()))]

    # Buat visualisasi multi-line chart untuk daily quantity terjualnya, breakdown per brand. Maka, akan terlihat 1 tanggal di mana ada salah satu brand yang mengalami lonjakan (quantity lebih tinggi dari tanggal-tanggal lain). Beri anotasi untuk titik lonjakan tersebut.

    dataset_top5brand_dec.groupby(['order_date','brand'])['quantity'].sum().unstack().plot(marker='.', cmap='plasma')
    plt.title('Daily Sold Quantity Dec 2019 - Breakdown by Brands',loc='center',pad=30, fontsize=15, color='blue')
    plt.xlabel('Order Date', fontsize = 12)
    plt.ylabel('Quantity',fontsize = 12)
    plt.grid(color='darkgray', linestyle=':', linewidth=0.5)
    plt.ylim(ymin=0)
    plt.legend(loc='upper center', bbox_to_anchor=(1.1,1), shadow=True, ncol=1)
    plt.annotate('Terjadi lonjakan', xy=(7, 310), xytext=(8, 300),
                weight='bold', color='red',
                arrowprops=dict(arrowstyle='->',
                                connectionstyle="arc3",
                                color='red'))
    plt.gcf().set_size_inches(10, 5)
    plt.tight_layout()


    multiline = getGraph()

    # Cari tahu jumlah product untuk masing-masing brand yang laku selama bulan Desember 2019. Gunakan barchart untuk visualisasinya, urutkan dengan yang kiri adalah brand dengan product lebih banyak.
    
    plt.clf()
    dataset_top5brand_dec.groupby('brand')['product_id'].nunique().sort_values(ascending=False).plot(kind='bar', color='green')
    plt.title('Number of Sold per Brand, December 2019',loc='center',pad=30, fontsize=15, color='blue')
    plt.xlabel('Brand', fontsize = 15)
    plt.ylabel('Number Of Product',fontsize = 15)
    plt.ylim(ymin=0)
    plt.xticks(rotation=0)

    bar = getGraph()

    # Gunakan stacked chart, untuk breakdown barchart yang di Case 3, antara product yang terjual >= 100 dan < 100 di bulan Desember 2019. Apakah ada pola yang menarik?

    dataset_top5brand_dec_per_product = dataset_top5brand_dec.groupby(['brand','product_id'])['quantity'].sum().reset_index()
    dataset_top5brand_dec_per_product['quantity_group'] = dataset_top5brand_dec_per_product['quantity'].apply(lambda x: '>= 100' if x>=100 else '< 100')
    dataset_top5brand_dec_per_product.sort_values('quantity',ascending=False,inplace=True)

    s_sort = dataset_top5brand_dec_per_product.groupby('brand')['product_id'].nunique().sort_values(ascending=False)

    plt.clf()
    dataset_top5brand_dec_per_product.groupby(['brand','quantity_group'])['product_id'].nunique().reindex(index=s_sort.index, level='brand').unstack().plot(kind='bar', stacked=True);
    plt.title('Number of Sold Products per Brand, December 2019',loc='center',pad=30, fontsize=15, color='blue')
    plt.xlabel('Brand', fontsize = 15)
    plt.ylabel('Number of Products',fontsize = 15)
    plt.ylim(ymin=0)
    plt.xticks(rotation=0)

    stack = getGraph()

    # Gunakan histogram untuk melihat distribusi harga product-product yang ada di top 5 brand tersebut (untuk tiap product_id, ambil median harganya). Bagaimana persebaran harga product-nya? Cenderung banyak yang murah atau yang mahal?

    plt.clf()
    plt.figure(figsize=(10,5))
    plt.hist(dataset_top5brand_dec.groupby('product_id')['item_price'].median(), bins=10, stacked=False, range=(1,2000000), color='green')
    plt.title('Distribution of Price Media per Product Top 5 Brands in Dec 2019',fontsize=15, color='blue')
    plt.xlabel('Price Median', fontsize = 12)
    plt.ylabel('Number of Product',fontsize = 12)
    plt.xlim(xmin=0,xmax=2000000)

    histogram = getGraph()

    #agregat per product
    data_per_product_top5brand_dec = dataset_top5brand_dec.groupby('product_id').agg({'quantity': 'sum', 'gmv':'sum', 'item_price':'sum'}).reset_index()

    #scatter plot
    plt.clf()
    plt.figure(figsize=(10,8))
    plt.scatter(data_per_product_top5brand_dec['quantity'],data_per_product_top5brand_dec['gmv'], marker='+', color='red')
    plt.title('Correlation of Quantity and GMV per Product\nTop 5 Brands in December 2019',fontsize=15, color='blue')
    plt.xlabel('Quantity', fontsize = 12)
    plt.ylabel('GMV (in Millions)',fontsize = 12)
    plt.xlim(xmin=0,xmax=300)
    plt.ylim(ymin=0,ymax=200000000)
    labels, locations = plt.yticks()
    plt.yticks(labels, (labels/1000000).astype(int))

    scatter = getGraph()

    print(data_per_product_top5brand_dec);

    plt.clf()
    plt.figure(figsize=(10,8))
    plt.scatter(data_per_product_top5brand_dec['item_price'],data_per_product_top5brand_dec['quantity'], marker='o', color='red')
    plt.title('Correlation of Quantity and GMV per Product\nTop 5 Brands in December 2019',fontsize=15, color='blue')
    plt.xlabel('Quantity', fontsize = 12)
    plt.ylabel('GMV (in Millions)',fontsize = 12)
    plt.xlim(xmin=0,xmax=2000000)
    plt.ylim(ymin=0,ymax=250)
    labels, locations = plt.yticks()
    plt.yticks(labels, (labels/1000000).astype(int))

    scatter_price = getGraph()

    print(data_per_product_top5brand_dec);

    return HttpResponse("<img src='data:image/png;base64, "+multiline+"' />"+"<br/>"+"<img src='data:image/png;base64, "+bar+"' />"+"<br/>"+"<img src='data:image/png;base64, "+stack+"' />"+"<br/>"+"<img src='data:image/png;base64, "+histogram+"' />"+"<br/>"+"<img src='data:image/png;base64, "+scatter+"' />"+"<br/>"+"<img src='data:image/png;base64, "+scatter_price+"' />")

def getGraph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph