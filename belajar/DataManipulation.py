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

    # Di Pandas terdapat 2 kelas data baru yang digunakan sebagai struktur dari spreadsheet:

    # Series: satu kolom bagian dari tabel dataframe yang merupakan 1 dimensional numpy array sebagai basis datanya, terdiri dari 1 tipe data (integer, string, float, dll).
    # DataFrame: gabungan dari Series, berbentuk rectangular data yang merupakan tabel spreadsheet itu sendiri (karena dibentuk dari banyak Series, tiap Series biasanya punya 1 tipe data, yang artinya 1 dataframe bisa memiliki banyak tipe data).

    # Series
    number_list = pd.Series([1,2,3,4,5,6])
    print("Series:")
    print(number_list)
    # DataFrame
    matrix = [[1,2,3],
            ['a', 'b', 'c'],
            [3,4,5],
            ['d',4,6]]
    matrix_list = pd.DataFrame(matrix)
    print("DataFrame:")
    print(matrix_list)

    # [1] attribute .info()
    print("[1] attribute .info()")
    print(matrix_list.info())
    # [2] attribute .shape
    print("\n[2] attribute .shape")
    print("    Shape dari number_list:", number_list.shape)
    print("    Shape dari matrix_list:", matrix_list.shape)
    # [3] attribute .dtypes
    print("\n[3] attribute .dtypes")
    print("    Tipe data number_list:", number_list.dtypes)
    print("    Tipe data matrix_list:", matrix_list.dtypes)
    # [4] attribute .astype()
    print("\n[4] attribute .astype()")
    print("    Konversi number_list ke str:", number_list.astype("str"))
    print("    Konversi matrix_list ke str:", matrix_list.astype("str"))

    return HttpResponse("Output On Console")

def getGraph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph
