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

def dataInterest(request):
    dataset = pd.read_excel("/code/Member_23-12-2021.xlsx")
    
    cln_interest_data = dataset[(dataset['interest'].notna()) & (dataset['interest'] .notnull())]

    # data = cln_interest_data[cln_interest_data.interest.str.match(pat='.*music.*', case=False)]
    # row, column = data.shape

    interests = pd.DataFrame({'interest': ['music', 'sport', 'travel', 'culinary', 'gadget & tech', 'fashion', 'enterpreneurship', 'design', 'education', 'movies']})
    interests['total'] = interests['interest'].apply(lambda x : cln_interest_data[cln_interest_data.interest.str.match(pat='.*'+x+'.*', case=False)].shape[0])

    interests.groupby('interest')['total'].sum().sort_values(ascending=False).plot(kind='bar', color='green')
    plt.title('Perbadingan Interest User MLDSPOT', loc='center', pad=30, fontsize=15, color='blue')
    plt.xlabel('Interest', fontsize=15)
    plt.ylabel('Total User', fontsize=15)
    plt.ylim(ymin=0, ymax=4000)
    # labels, locations = plt.yticks()
    # plt.yticks(labels, (labels/1000000000).astype(float))
    plt.xticks(rotation=10)
    plt.gcf().set_size_inches(12, 7)
    plt.annotate('Terlihat jika user mld banyak menyukai music, untuk menambah interaksi user\nbisa perbanyak content atau podcast yang membahas music.\nUntuk loyalty bisa kita tambahkan quiz tebak musik agar lebih menarik', xy=(0,3300), xytext=(1, 3500), weight='bold', color='red',arrowprops=dict(arrowstyle='fancy',connectionstyle="arc3", color='red'))
    plt.annotate('Kurangi kontent yang terlalu banyak\nkata/paragraf yang bisa membuat\nuser bosan', xy=(9,1800), xytext=(6, 2500), weight='bold', color='red',arrowprops=dict(arrowstyle='fancy',connectionstyle="arc3", color='red'))

    bar = getGraph()

    birthdays = dataset[(dataset['birthday'].notna()) & (dataset['birthday'] .notnull())]
    birthdays['age'] = birthdays['birthday'].apply(lambda x: relativedelta(datetime.datetime.now(), datetime.datetime.strptime(x, "%Y-%m-%d")).years)

    plt.clf()
    plt.figure(figsize=(10,5))
    plt.hist(birthdays['age'], bins=10, range=(1,80), color='green')
    plt.title('Distribution of Total GMV per Customer\nDKI Jakarta in Q4 2019', fontsize=15, color='blue')
    plt.xlabel('GMV (in Millions)', fontsize = 12)
    plt.ylabel('Number of Customers', fontsize = 12)

    hist = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+bar+"' />"+"<br/>"+"<img src='data:image/png;base64, "+hist+"' />")

def getGraph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph