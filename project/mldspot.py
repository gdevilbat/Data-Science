from ast import pattern
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
import re
from contextlib import closing
import csv
from sklearn import preprocessing
from dateutil.relativedelta import relativedelta

def dataLogin(request, date:str):
    dataset = pd.read_csv("/code/mldspot-user-"+date+".csv")

    pattern = "(ref|utm_source)(%253D|%3D|=)(([A-z\-\.0-9]|(%20|%20%2520))+)((%|\&)*)"
    dataset['log'] = dataset['url_path'].apply(lambda x: re.compile(pattern).search(x).group(3) if (bool(re.search(pattern, x))) else 'Organic' )

    pattern = "(utm_medium)(%253D|%3D|=)(([A-z\-\.0-9]|(%20|%20%2520))+)((%|\&)*)"
    dataset['medium'] = dataset['url_path'].apply(lambda x: re.compile(pattern).search(x).group(3) if (bool(re.search(pattern, x))) else 'None' )

    total_login = dataset.groupby(['log', 'medium']).nunique('id.1').rename(columns={'id.1':'total'}).sort_values(by='log',ascending=False)
    unique_login = dataset.groupby(['log', 'medium']).nunique('member_id').rename(columns={'member_id':'total'}).sort_values(by='log',ascending=False)

    print("\r\n")

    print("Unique Login")
    print(unique_login['total'])
    print("\r\n")

    print("Total Login")
    print(total_login['total'])
    print("\r\n")

    print("\r\n")

    return HttpResponse("Output On Console")

def dataDemographic(request, date:str):
    dataset = pd.read_csv("/code/mldspot-user-"+date+".csv")

    pattern = "(ref|utm_source)(%253D|%3D|=)(([A-z\-\.0-9]|(%20|%20%2520))+)((%|\&)*)"
    dataset['log'] = dataset['url_path'].apply(lambda x: re.compile(pattern).search(x).group(3) if (bool(re.search(pattern, x))) else 'Organic' )

    pattern = "(utm_medium)(%253D|%3D|=)(([A-z\-\.0-9]|(%20|%20%2520))+)((%|\&)*)"
    dataset['medium'] = dataset['url_path'].apply(lambda x: re.compile(pattern).search(x).group(3) if (bool(re.search(pattern, x))) else 'None' )

    venue = dataset.loc[:, 'log'].unique()

    birthdays = dataset[(dataset['meta_value'].notna()) & (dataset['meta_value'] .notnull()) & (dataset['meta_key'] == 'birthday')]
    birthdays['age'] = birthdays['meta_value'].apply(lambda x: relativedelta(datetime.datetime.now(), datetime.datetime.strptime(x, "%Y-%m-%d")).years)
    birthdays['age_group'] = birthdays['age'].apply(lambda x: '10-17' if x>=10 and x<=17 else ('18-24' if x>=18 and x<=24 else('25-34' if x>= 25 and x<=34 else ('35-44' if x>=35 and x<=44 else ('45-54' if x>=45 and x<=54 else '55+')))))

    age = birthdays.groupby(['age_group']).nunique('member_id').rename(columns={'member_id':'total'}).sort_values(['age_group'],ascending=[True])
    age_venue = birthdays.groupby(['log', 'age_group']).nunique('member_id').rename(columns={'member_id':'total'}).sort_values(['log', 'age_group'],ascending=[True, True])

    print("\r\n")

    print("Age Total")
    print(age['total'])
    print("\r\n")

    print("Age Venue")
    print(age_venue['total'])
    print("\r\n")

    otp = dataset.groupby(['int_phone_verified']).nunique('member_id').rename(columns={'member_id':'total'}).sort_values(['int_phone_verified'],ascending=[True])
    otp_venue = dataset.groupby(['log', 'int_phone_verified']).nunique('member_id').rename(columns={'member_id':'total'}).sort_values(['log', 'int_phone_verified'],ascending=[True, True])

    print("OTP Total")
    print(otp['total'])
    print("\r\n")

    print("OTP Venue")
    print(otp_venue['total'])
    print("\r\n")

    data_gender = dataset[(dataset['meta_value'].notna()) & (dataset['meta_value'] .notnull()) & (dataset['meta_key'] == 'gender')]
    data_gender['gender'] = dataset['meta_value'].apply(lambda x:  x)

    gender = data_gender.groupby(['gender']).nunique('member_id').rename(columns={'member_id':'total'}).sort_values(['gender'],ascending=[True])
    gender_venue = data_gender.groupby(['log', 'gender']).nunique('member_id').rename(columns={'member_id':'total'}).sort_values(['log', 'gender'],ascending=[True, True])

    print("Gender Total")
    print(gender['total'])
    print("\r\n")

    print("Gender Venue")
    print(gender_venue['total'])

    cln_interest_data = dataset[(dataset['meta_value'].notna()) & (dataset['meta_value'] .notnull()) & (dataset['meta_key'] == 'interest')]
    cln_interest_data = cln_interest_data.groupby(['member_id']).tail(1)

    interest = ['music', 'sport', 'travel', 'culinary', 'gadget & tech', 'fashion & streetwear', 'enterpreneurship', 'design', 'education', 'movies']
    data = getInterest(interest, venue)

    interests = pd.DataFrame({'interests': data[0],'venue': data[1]})

    interests['total'] = interests.apply(lambda x : 0)

    i = 0
    for x, y in zip(interests['interests'], interests['venue']):
        cln_interest_data_filtered = cln_interest_data[cln_interest_data['log'] == y ]
        if( x in interest):
            interests['total'][i] = cln_interest_data_filtered[cln_interest_data_filtered.meta_value.str.match(pat='^.*'+x+'.*$', case=False)].shape[0]
        else:
            interests['total'][i] = cln_interest_data_filtered[cln_interest_data_filtered.meta_value.str.match(pat='^(?!'+'|'.join(map(str, interest))+').*$', case=False)].shape[0]

        i = i + 1

    data_interest = interests.groupby(['interests']).sum('total').sort_values(by='total',ascending=False)
    data_interest_venue = interests.groupby(['venue', 'interests']).sum('total').sort_values(['venue', 'interests'],ascending=[True, True])

    print("Interest Total")
    print(data_interest.to_string())
    print("\r\n")

    print("Interest Venue")
    print(data_interest_venue.to_string())
    print("\r\n")

    print("\r\n")

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

def dataInterest(request):
    # dataset = pd.read_excel("/code/Member_13-01-2022.xlsx")
    
    # cln_interest_data = dataset[(dataset['interest'].notna()) & (dataset['interest'] .notnull())]

    # # data = cln_interest_data[cln_interest_data.interest.str.match(pat='.*music.*', case=False)]
    # # row, column = data.shape

    # interests = pd.DataFrame({'interest': ['music', 'sport', 'travel', 'culinary', 'gadget & tech', 'fashion', 'enterpreneurship', 'design', 'education', 'movies']})
    # interests['total'] = interests['interest'].apply(lambda x : cln_interest_data[cln_interest_data.interest.str.match(pat='.*'+x+'.*', case=False)].shape[0])

    # print(interests.groupby('interest')['total'].sum().sort_values(ascending=False))
    # interests.groupby('interest')['total'].sum().sort_values(ascending=False).plot(kind='bar', color='green')
    # plt.title('Perbadingan Interest User MLDSPOT', loc='center', pad=30, fontsize=15, color='blue')
    # plt.xlabel('Interest', fontsize=15)
    # plt.ylabel('Total User', fontsize=15)
    # plt.ylim(ymin=0, ymax=4000)
    # # labels, locations = plt.yticks()
    # # plt.yticks(labels, (labels/1000000000).astype(float))
    # plt.xticks(rotation=10)
    # plt.gcf().set_size_inches(12, 7)
    # plt.annotate('Terlihat jika user mld banyak menyukai music, untuk menambah interaksi user\nbisa perbanyak content atau podcast yang membahas music.\nUntuk loyalty bisa kita tambahkan quiz tebak musik agar lebih menarik', xy=(0,3300), xytext=(1, 3500), weight='bold', color='red',arrowprops=dict(arrowstyle='fancy',connectionstyle="arc3", color='red'))
    # plt.annotate('Kurangi kontent yang terlalu banyak\nkata/paragraf yang bisa membuat\nuser bosan', xy=(9,1800), xytext=(6, 2500), weight='bold', color='red',arrowprops=dict(arrowstyle='fancy',connectionstyle="arc3", color='red'))

    # bar = getGraph()

    # birthdays = dataset[(dataset['birthday'].notna()) & (dataset['birthday'] .notnull())]
    # birthdays['age'] = birthdays['birthday'].apply(lambda x: relativedelta(datetime.datetime.now(), datetime.datetime.strptime(x, "%Y-%m-%d")).years)

    # plt.clf()
    # plt.figure(figsize=(10,5))
    # plt.hist(birthdays['age'], bins=10, range=(1,80), color='green')
    # plt.title('Distribution of Total GMV per Customer\nDKI Jakarta in Q4 2019', fontsize=15, color='blue')
    # plt.xlabel('GMV (in Millions)', fontsize = 12)
    # plt.ylabel('Number of Customers', fontsize = 12)

    # hist = getGraph()

    # return HttpResponse("<img src='data:image/png;base64, "+bar+"' />"+"<br/>"+"<img src='data:image/png;base64, "+hist+"' />")

    dataset = pd.read_csv("/code/core_logs_point.csv")

    objective = dataset.groupby(['objective_name']).nunique('id_log_point').rename(columns={'id_log_point':'total'}).sort_values(by='total',ascending=False)

    print(objective['total'])

    print(dataset.columns)

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