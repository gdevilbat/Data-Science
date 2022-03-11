from django.http import HttpResponse
from django.http import JsonResponse

import base64
from io import BytesIO

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from contextlib import closing
import csv
import logging

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'console': {
            'format': '%(name)-12s %(levelname)-8s \r\n %(message)s'
        },
        'file': {
            'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'console'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'file',
            'filename': '/tmp/debug.log'
        }
    },
    'loggers': {
        '': {
            'level': 'DEBUG',
            'handlers': ['console']
        }
    }
})

logger = logging.getLogger(__name__)

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def intro(request):
    a = 'Hello World <br/>'
    b = 'Saya Aksara, baru belajar Python'
    return HttpResponse(a+b)

def tugasFundamentalTiga(request):
    angka = 5
    if(angka%2 == 0):
        hasil = 'angka termasuk bilangan genap'
    else:
        hasil = 'angka termasuk bilangan ganjil'
    return HttpResponse(hasil)

def tugasPerulangan(request):
    j = 0
    text = '';
    while j<6:
        text += 'ini adalah perulangan ke - '+str(j)+'<br/>'
        j+=1

    text+= '<br/>'

    for i in range (1,6): #perulangan for sebagai inisialisasi dari angka 1 hingga angka yang lebih kecil daripada 6.
        text += 'ini adalah perulangan ke - '+str(i)+'<br/>'

    count=[1,2,3,4,5] #elemen list

    text+= '<br/>'

    for number in count: #looping untuk menampilkan semua elemen pada count
        text += 'Ini adalah element count : '+str(number)+'<br/>'

    return HttpResponse(text)

def tugasMembacaCSV(request):
    # tentukan lokasi file, nama file, dan inisialisasi csv
    url = 'https://storage.googleapis.com/dqlab-dataset/penduduk_gender_head.csv'

    # baca file csv secara streaming 
    with closing(requests.get(url, stream=True)) as r:
        f = (line.decode('utf-8') for line in r.iter_lines())

        reader = csv.reader(f, delimiter=',')

        data = []
        # membaca baris per baris
        for row in reader:
            logging.debug(row)

    # # tentukan lokasi file, nama file, dan inisialisasi csv dri lokal file
    # f = open('https://storage.googleapis.com/dqlab-dataset/penduduk_gender_head.csv', 'r')
    # reader = csv.reader(f)

    # # membaca baris per baris
    # for row in reader:
    #     print (row)

    # # menutup file csv
    # f.close()

    logging.debug(data)

    return HttpResponse("Output On Console")

    ##return JsonResponse(data, safe=False)

def tugasCSVPandas(request):
    pd.set_option("display.max_columns",50)
    table = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/penduduk_gender_head.csv")
    table.head()

    logging.debug(table)

    return HttpResponse("Output On Console")

    ##return JsonResponse(table.to_dict())

def tugasBarSatu(request):
    table = pd.read_csv("https://academy.dqlab.id/dataset/penduduk_gender_head.csv")
    table.head()
    
    x_label = table['NAMA KELURAHAN']
    plt.switch_backend('Agg')
    plt.bar(x=np.arange(len(x_label)),height=table['LAKI-LAKI WNI'])
    plt.xticks(np.arange(len(x_label)), table['NAMA KELURAHAN'], rotation=90)
    plt.bar(x=np.arange(len(x_label)),height=table['LAKI-LAKI WNI'])
    plt.xlabel('Keluarahan di Jakarta pusat')
    plt.ylabel('Jumlah Penduduk Laki - Laki')
    plt.title('Persebaran Jumlah Penduduk Laki- Laki di Jakarta Pusat')

    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />")