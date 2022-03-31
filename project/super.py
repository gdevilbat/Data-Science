from re import I
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

def dataLogin(request):

    dataset = pd.read_csv("/code/super-login-2022-01.csv")

    total_login = dataset.groupby(['log']).nunique('id').rename(columns={'id':'total'}).sort_values(by='total',ascending=False)
    unique_login = dataset.groupby(['log']).nunique('user_id').rename(columns={'user_id':'total'}).sort_values(by='total',ascending=False)

    print(unique_login['total'])
    print(total_login['total'])


    return HttpResponse("Output On Console")

def dataDemographic(request):

    dataset = pd.read_csv("/code/super-user-2022-01.csv")

    dataset = dataset[(dataset['log'].notna()) & (dataset['log'] .notnull()) & (dataset['log'])]

    venue = dataset.loc[:, 'log'].unique()

    birthdays = dataset[(dataset['dob'].notna()) & (dataset['dob'] .notnull()) & (dataset['dob'])]
    birthdays['age'] = birthdays['dob'].apply(lambda x: relativedelta(datetime.datetime.now(), datetime.datetime.strptime(x, "%Y-%m-%d")).years)
    birthdays['age_group'] = birthdays['age'].apply(lambda x: '10-17' if x>=10 and x<=17 else ('18-34' if x>= 18 and x<=34 else ('35-44' if x>=35 and x<=44 else ('45-54' if x>=45 and x<=54 else '55+'))))

    user = birthdays.groupby(['log']).nunique('user_id').rename(columns={'user_id':'total'}).sort_values(by='total',ascending=False)

    print(user['total'])

    age = birthdays.groupby(['age_group']).nunique('user_id').rename(columns={'user_id':'total'}).sort_values(['age_group'],ascending=[True])
    age_venue = birthdays.groupby(['log', 'age_group']).nunique('user_id').rename(columns={'user_id':'total'}).sort_values(['log', 'age_group'],ascending=[True, True])

    print(age['total'])
    print(age_venue['total'])

    cln_interest_data = dataset[(dataset['interests'].notna()) & (dataset['interests'] .notnull())]

    cln_interest_data = cln_interest_data.groupby(['user_id']).tail(1)

    interest = ['music', 'adventure', 'soccer', 'other']
    data = getInterest(interest, venue)

    interests = pd.DataFrame({'interests': data[0],'venue': data[1]})

    interest.remove('other')
    interests['total'] = interests.apply(lambda x : 0)

    i = 0
    for x, y in zip(interests['interests'], interests['venue']):
        cln_interest_data_filtered = cln_interest_data[cln_interest_data['log'] == y ]
        if( x in interest):
            interests['total'][i] = cln_interest_data_filtered[cln_interest_data_filtered.interests.str.match(pat='^.*'+x+'.*$', case=False)].shape[0]
        else:
            interests['total'][i] = cln_interest_data_filtered[cln_interest_data_filtered.interests.str.match(pat='^(?!'+'|'.join(map(str, interest))+').*$', case=False)].shape[0]

        i = i + 1

    data_interest = interests.groupby(['interests']).sum('total').sort_values(by='total',ascending=False)
    data_interest_venue = interests.groupby(['venue', 'interests']).sum('total').sort_values(['venue', 'interests'],ascending=[True, True])


    print(data_interest)
    print(data_interest_venue)

    return HttpResponse("Output On Console")

def getInterest(interest, venue):
    data_int = []

    for x in interest:
        for y in venue:
            data_int.append(x)

    data_ven = []

    for x in interest:
        for y in venue:
            data_ven.append(y)


    return [data_int, data_ven]

def getPrint(data):
    print(data[0])

    return 3


def getGraph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph