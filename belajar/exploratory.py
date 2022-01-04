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
from dateutil.relativedelta import relativedelta

def index(request):
    order_df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/order.csv")
    print(order_df.describe())
    print(order_df.shape)
    print(order_df.loc[:, "price"].median())
    print(order_df.loc[:, "price"].mean())
    print(order_df.loc[:, "product_weight_gram"].std())
    print(order_df.loc[:, "product_weight_gram"].var())

    #Menghitung Outliner
    # Hitung quartile 1
    Q1 = order_df[["product_weight_gram"]].quantile(0.25)
    # Hitung quartile 3
    Q3 = order_df[["product_weight_gram"]].quantile(0.75)
    # Hitung inter quartile range dan cetak ke console
    IQR = Q3 - Q1
    print(IQR)
    print((order_df < (Q1-1.5*IQR)) | (order_df > (Q3 + 1.5*IQR)))

    order_df.rename(columns={"freight_value": "shipping_cost"}, inplace=True)
    print(order_df)

    rata_rata = order_df["price"].groupby(order_df["payment_type"]).mean()
    print(rata_rata)

    sort_harga = order_df.sort_values(by="price", ascending=False)
    print(sort_harga)

    return HttpResponse("Output On Console")

def tugasMiniProject(request):
    order_df = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/order.csv")

    # Median price yang dibayar customer dari masing-masing metode pembayaran. 
    median_price = order_df["price"].groupby(order_df["payment_type"]).median()
    print(median_price)
    # Ubah freight_value menjadi shipping_cost dan cari shipping_cost 
    # termahal dari data penjualan tersebut menggunakan sort.
    order_df.rename(columns={"freight_value": "shipping_cost"}, inplace=True)
    sort_value = order_df.sort_values(by="price", ascending=0)
    print(sort_value)
    # Untuk product_category_name, berapa  rata-rata weight produk tersebut 
    # dan standar deviasi mana yang terkecil dari weight tersebut, 
    mean_value = order_df["product_weight_gram"].groupby(order_df["product_category_name"]).mean()
    print(mean_value.sort_values())
    std_value = order_df["product_weight_gram"].groupby(order_df["product_category_name"]).std()
    print(std_value.sort_values())
    # Buat histogram quantity penjualan dari dataset tersebutuntuk melihat persebaran quantity 
    # penjualan tersebut dengan bins = 5 dan figsize= (4,5)
    order_df[["quantity"]].hist(figsize=(4, 5), bins=5)

    hist = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+hist+"' />")

def getGraph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph
