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
import pandas_profiling
from contextlib import closing
import csv
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta
from scipy.stats import norm

def index(request):

    retail_raw = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced_data_quality.csv')

    # Cetak tipe data di setiap kolom retail_raw
    print(retail_raw.dtypes)

    print('')

    # Kolom city
    length_city = len(retail_raw['city'])
    print('Length kolom city:', length_city)

    # Tugas Praktek: Kolom product_id
    length_product_id = len(retail_raw['product_id'])
    print('Length kolom product_id:', length_product_id)

    # Count kolom city
    count_city = retail_raw['city'].count()
    print('Count kolom count_city', count_city)

    # Tugas praktek: count kolom product_id
    count_product_id = retail_raw['product_id'].count()
    print('Count kolom product_id:', count_product_id)

    # Missing value pada kolom city
    number_of_missing_values_city = length_city - count_city
    float_of_missing_values_city = float(number_of_missing_values_city/length_city)
    pct_of_missing_values_city = '{0:.1f}%'.format(float_of_missing_values_city * 100)
    print('Persentase missing value kolom city:', pct_of_missing_values_city)

    # Tugas praktek: Missing value pada kolom product_id
    number_of_missing_values_product_id = length_product_id - count_product_id
    float_of_missing_values_product_id = float(number_of_missing_values_product_id/length_product_id)
    pct_of_missing_values_product_id = '{0:.1f}%'.format(float_of_missing_values_product_id * 100)
    print('Persentase missing value kolom product_id:', pct_of_missing_values_product_id)

    print('')

    # Deskriptif statistics kolom quantity
    print('Kolom quantity')
    print('Minimum value: ', retail_raw['quantity'].min())
    print('Maximum value: ', retail_raw['quantity'].max())
    print('Mean value: ', retail_raw['quantity'].mean())
    print('Mode value: ', retail_raw['quantity'].mode())
    print('Median value: ', retail_raw['quantity'].median())
    print('Standard Deviation value: ', retail_raw['quantity'].std())

    # Tugas praktek: Deskriptif statistics kolom item_price
    print('')
    print('Kolom item_price')
    print('Minimum value: ', retail_raw['item_price'].min())
    print('Maximum value: ', retail_raw['item_price'].max())
    print('Mean value: ', retail_raw['item_price'].mean())
    print('Median value: ', retail_raw['item_price'].median())
    print('Standard Deviation value: ', retail_raw['item_price'].std())

    print('')

    # Quantile statistics kolom quantity
    print('Kolom quantity:')
    print(retail_raw['quantity'].quantile([0.25, 0.5, 0.75]))

    # Tugas praktek: Quantile statistics kolom item_price
    print('')
    print('Kolom item_price:')
    print(retail_raw['item_price'].quantile([0.25, 0.5, 0.75]))

    print('')

    print('Korelasi quantity dengan item_price')
    print(retail_raw[['quantity', 'item_price']].corr())

    pandas_profiling.ProfileReport(retail_raw)

    return HttpResponse("Output On Console")

def tugasProfiling(request):
    retail_raw = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/retail_raw_reduced_data_quality.csv')

    # Dataset yang ditemui di real-world biasanya akan memiliki banyak missing value. Kemampuan untuk treatment missing value sangat penting karena jika membiarkan missing value itu dapat memengaruhi analisis dan machine learning model. Sehingga jika menemukan nilai yang hilang dalam dataset, harus melakukan treatment sedemikian rupa.

    # Check kolom yang memiliki missing data
    print('Check kolom yang memiliki missing data:')
    print(retail_raw.isnull().any())

    # Filling the missing value (imputasi)
    print('\nFilling the missing value (imputasi):')
    print(retail_raw['quantity'].fillna(retail_raw.quantity.mean()))

    # Drop missing value
    print('\nDrop missing value:')
    print(retail_raw['quantity'].dropna())

    print(retail_raw['item_price'].fillna(retail_raw['item_price'].mean()))

    # Outliers merupakan data observasi yang muncul dengan nilai-nilai ekstrim. Yang dimaksud dengan nilai-nilai ekstrim dalam observasi adalah nilai yang jauh atau beda sama sekali dengan sebagian besar nilai lain dalam kelompoknya.

    # Cara treatment terhadap outliers antara lain:
    # Remove the outliers (dibuang)
    # Filling the missing value (imputasi)
    # Capping
    # Prediction

    # Q1, Q3, dan IQR
    Q1 = retail_raw['quantity'].quantile(0.25)
    Q3 = retail_raw['quantity'].quantile(0.75)
    IQR = Q3 - Q1

    # Check ukuran (baris dan kolom) sebelum data yang outliers dibuang
    print('Shape awal: ', retail_raw.shape)

    # Removing outliers
    retail_raw = retail_raw[~((retail_raw['quantity'] < (Q1 - 1.5 * IQR)) | (retail_raw['quantity'] > (Q3 + 1.5 * IQR)))]

    # Check ukuran (baris dan kolom) setelah data yang outliers dibuang
    print('Shape akhir: ', retail_raw.shape)

    # Check ukuran (baris dan kolom) sebelum data duplikasi dibuang
    print('Shape awal: ', retail_raw.shape)

    # Buang data yang terduplikasi
    retail_raw.drop_duplicates(inplace=True)

    # Check ukuran (baris dan kolom) setelah data duplikasi dibuang
    print('Shape akhir: ', retail_raw.shape)

    return HttpResponse("Output On Console")

def tugasMiniProject(request):
    # Baca dataset uncleaned_raw.csv
    uncleaned_raw = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/uncleaned_raw.csv')

    #inspeksi dataframe uncleaned_raw
    print('Lima data teratas:') 
    print(uncleaned_raw.head(5))

    #Check kolom yang mengandung missing value
    print('\nKolom dengan missing value:') 
    print(uncleaned_raw.isnull().any())

    #Persentase missing value
    length_qty = len(uncleaned_raw['Quantity'])
    count_qty = uncleaned_raw['Quantity'].count()

    #mengurangi length dengan count
    number_of_missing_values_qty = length_qty - count_qty

    #mengubah ke bentuk float
    float_of_missing_values_qty = float(number_of_missing_values_qty / length_qty) 

    #mengubah ke dalam bentuk persen
    pct_of_missing_values_qty = '{0:.1f}%'.format(float_of_missing_values_qty*100) 

    #print hasil percent dari missing value
    print('Persentase missing value kolom Quantity:', pct_of_missing_values_qty)

    #Mengisi missing value tersebut dengan mean dari kolom tersebut
    uncleaned_raw['Quantity'] = uncleaned_raw['Quantity'].fillna(uncleaned_raw['Quantity'].mean())

    #Mengetahui kolom yang memiliki outliers!
    uncleaned_raw.boxplot()
    plt.show()

    graph = getGraph()

    print(uncleaned_raw['Quantity'].std())
    print(uncleaned_raw['Quantity'].mean())

    plt.clf()
    # Plot between -10 and 10 with .001 steps.
    x_axis = np.arange(uncleaned_raw['Quantity'].min()-10000, uncleaned_raw['Quantity'].max()+100, 100)
    # Mean = 0, SD = 2.
    plt.plot(x_axis, norm.pdf(x_axis,uncleaned_raw['Quantity'].mean(),uncleaned_raw['Quantity'].std()))

    std = getGraph()

    #Check IQR
    Q1 = uncleaned_raw['UnitPrice'].quantile(0.25)
    Q3 = uncleaned_raw['UnitPrice'].quantile(0.75)
    IQR = Q3 - Q1

    #removing outliers
    uncleaned_raw = uncleaned_raw[~((uncleaned_raw[['UnitPrice']] < (Q1 - 1.5 * IQR)) | (uncleaned_raw[['UnitPrice']] > (Q3 + 1.5 * IQR)))]

    #check for duplication
    print(uncleaned_raw.duplicated(subset=None))

    #remove duplication
    uncleaned_raw = uncleaned_raw.drop_duplicates()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />"+"<br/>"+"<img src='data:image/png;base64, "+std+"' />")

def getGraph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph
