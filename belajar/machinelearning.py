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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from sklearn.cluster import KMeans
from dateutil.relativedelta import relativedelta

def index(request):
    # supervised untuk label/output yang telah diketahui menggunakan classification untuk diskrit/non-numerik, regression untuk numerik. contoh: harga rumah, saham
    # unsupervised untuk label/output yang belum diketahui menggunakan clustering contoh segmentasi pasar

    dataset = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv")

    print('Shape dataset:', dataset.shape)
    print('\nLima data teratas:\n', dataset.head())
    print('\nInformasi dataset:')
    print(dataset.info())
    print('\nStatistik deskriptif:\n', dataset.describe())

    dataset_corr = dataset.corr()
    print('Korelasi dataset:\n', dataset.corr())
    print('Distribusi Label (Revenue):\n', dataset['Revenue'].value_counts())
    # Tugas praktek
    print('\nKorelasi BounceRates-ExitRates:', dataset_corr.loc['BounceRates', 'ExitRates'])
    print('\nKorelasi Revenue-PageValues:', dataset_corr.loc['Revenue', 'PageValues'])
    print('\nKorelasi TrafficType-Weekend:', dataset_corr.loc['TrafficType', 'Weekend'])

    # checking the Distribution of customers on Revenue
    plt.rcParams['figure.figsize'] = (12,5)
    plt.subplot(1, 2, 1)
    sns.countplot(dataset['Revenue'], palette = 'pastel')
    plt.title('Buy or Not', fontsize =20)
    plt.xlabel('Revenue or not', fontsize = 14)
    plt.ylabel('count', fontsize = 14)
    # checking the Distribution of customers on Weekend
    plt.subplot(1, 2, 2)
    sns.countplot(dataset['Weekend'], palette = 'inferno')
    plt.title('Purchase on Weekends', fontsize = 20)
    plt.xlabel('Weekend or not', fontsize = 14)
    plt.ylabel('count', fontsize =14)

    graph = getGraph()

    plt.clf()
    # visualizing the distribution of customers around the Region
    plt.hist(dataset['Region'], color = 'lightblue')
    plt.title('Distribution of Customers', fontsize = 20)
    plt.xlabel('Region Codes', fontsize = 14)
    plt.ylabel('Count Users', fontsize = 14)

    hist = getGraph()

    #checking missing value for each feature  
    print('Checking missing value for each feature:')
    print(dataset.isnull().sum())
    #Counting total missing value
    print('\nCounting total missing value:')
    print(dataset.isnull().sum().sum())

    #Drop rows with missing value   
    dataset_clean = dataset.dropna()  
    print('Ukuran dataset_clean:', dataset_clean.shape)

    print("Before imputation:")
    # Checking missing value for each feature  
    print(dataset.isnull().sum())
    # Counting total missing value  
    print(dataset.isnull().sum().sum())

    print("\nAfter imputation:")
    # Fill missing value with mean of feature value  
    dataset.fillna(dataset.mean(), inplace = True)
    # Checking missing value for each feature  
    print(dataset.isnull().sum())
    # Counting total missing value  
    print(dataset.isnull().sum().sum())

    dataset1 = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

    print("Before imputation:")
    # Checking missing value for each feature  
    print(dataset1.isnull().sum())
    # Counting total missing value  
    print(dataset1.isnull().sum().sum())

    print("\nAfter imputation:")
    # Fill missing value with median of feature value  
    dataset1.fillna(dataset1.median(), inplace = True)
    # Checking missing value for each feature  
    print(dataset1.isnull().sum())
    # Counting total missing value  
    print(dataset1.isnull().sum().sum())

    # “Beberapa machine learning seperti K-NN dan gradient descent mengharuskan semua variabel memiliki rentang nilai yang sama, karena jika tidak sama, feature dengan rentang nilai terbesar misalnya ProductRelated_Duration otomatis akan menjadi feature yang paling mendominasi dalam proses training/komputasi, sehingga model yang dihasilkan pun akan sangat bias. Oleh karena itu, sebelum memulai training model, kita terlebih dahulu perlu melakukan data rescaling ke dalam rentang 0 dan 1, sehingga semua feature berada dalam rentang nilai tersebut, yaitu nilai max = 1 dan nilai min = 0. Data rescaling ini dengan mudah dapat dilakukan di Python menggunakan .MinMaxScaler( ) dari Scikit-Learn library.”

    #Define MinMaxScaler as scaler  
    scaler = MinMaxScaler()  
    #list all the feature that need to be scaled  
    scaling_column = ['Administrative','Administrative_Duration','Informational','Informational_Duration','ProductRelated','ProductRelated_Duration','BounceRates','ExitRates','PageValues']
    #Apply fit_transfrom to scale selected feature  
    dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
    #Cheking min and max value of the scaling_column
    print(dataset[scaling_column].describe().T[['min','max']])

    # "Aksara, kita memiliki dua kolom yang bertipe object yang dinyatakan dalam tipe data str, yaitu kolom 'Month' dan 'VisitorType'. Karena setiap algoritma machine learning bekerja dengan menggunakan nilai numeris, maka kita perlu mengubah kolom dengan tipe pandas object atau str ini ke bertipe numeris. Untuk itu, kita list terlebih dahulu apa saja label unik di kedua kolom ini," jelas Senja. Lalu Senja pun mulai mempraktikkan sambil menunjukkannya padaku. 

    # Convert feature/column 'Month'
    LE = LabelEncoder()
    dataset['Month'] = LE.fit_transform(dataset['Month'])
    print(LE.classes_)
    print(np.sort(dataset['Month'].unique()))
    print('')

    # Convert feature/column 'VisitorType'
    LE = LabelEncoder()
    dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])
    print(LE.classes_)
    print(np.sort(dataset['VisitorType'].unique()))

    # Dalam dataset user online purchase, label target sudah diketahui, yaitu kolom Revenue yang bernilai 1 untuk user yang membeli dan 0 untuk yang tidak membeli, sehingga pemodelan yang dilakukan ini adalah klasifikasi. Nah, untuk melatih dataset menggunakan Scikit-Learn library, dataset perlu dipisahkan ke dalam Features dan Label/Target. Variabel Feature akan terdiri dari variabel yang dideklarasikan sebagai X dan [Revenue] adalah variabel Target yang dideklarasikan sebagai y. Gunakan fungsi drop() untuk menghapus kolom [Revenue] dari dataset.

    # removing the target column Revenue from dataset and assigning to X
    X = dataset.drop(['Revenue'], axis = 1)
    # assigning the target column Revenue to y
    y = dataset['Revenue']
    # checking the shapes
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)


    # “Well done, Aksara! Nah, sebelum kita melatih model dengan suatu algorithm machine , seperti yang saya jelaskan sebelumnya, dataset perlu kita bagi ke dalam training dataset dan test dataset dengan perbandingan 80:20. 80% digunakan untuk training dan 20% untuk proses testing.”
    # “Perbandingan lain yang biasanya digunakan adalah 75:25. Hal penting yang perlu diketahui adalah scikit-learn tidak dapat memproses dataframe dan hanya mengakomodasi format data tipe Array. Tetapi kalian tidak perlu khawatir, fungsi train_test_split( ) dari Scikit-Learn, otomatis mengubah dataset dari dataframe ke dalam format array. Apakah kamu paham. Aksara? Atau ada pertanyaan?”
    # “Kenapa perlu ada Training dan Testing, Nja?”
    # “Fungsi Training adalah melatih model untuk mengenali pola dalam data, sedangkan testing berfungsi untuk memastikan bahwa model yang telah dilatih tersebut mampu dengan baik memprediksi label dari new observation dan belum dipelajari oleh model sebelumnya. Lebih baik kita praktik saja ya, tampaknya kalau praktik kamu lebih paham.”
    # “Aksara silahkan bagi dataset ke dalam Training dan Testing dengan melanjutkan coding yang  sudah kukerjakan ini. Gunakan test_size = 0.2 dan tambahkan argumen random_state = 0,  pada fungsi train_test_split( ). Dicoba saja dulu yah, saya yakin kamu bisa.”

    # splitting the X, and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # checking the shapes
    print("Shape of X_train :", X_train.shape)
    print("Shape of y_train :", y_train.shape)
    print("Shape of X_test :", X_test.shape)
    print("Shape of y_test :", y_test.shape)

    # “Good Job, Aksara! Sekarang saatnya kita melatih model atau training. Dengan Scikit-Learn, proses ini menjadi sangat sederhana. Kita cukup memanggil nama algorithm yang akan kita gunakan, biasanya disebut classifier untuk problem klasifikasi, dan regressor untuk problem regresi.”
    # Aku selalu suka ketika Senja mengapresiasiku sesederhana apapun itu, karena selalu berhasil mendorong semangat belajarku. Aku jadi lebih berani untuk mencoba dan menyimak hal baru. Dan satu lagi, Senja selalu mau aku repotkan dengan meminta contoh.
    # “Begini, sebagai contoh, kita akan menggunakan Decision Tree. Kita hanya perlu memanggil fungsi DecisionTreeClassifier() yang kita namakan “model”. Kemudian menggunakan fungsi .fit() dan X_train, y_train untuk melatih classifier tersebut dengan training dataset, seperti ini:”

    # Call the classifier
    model = DecisionTreeClassifier()
    # Fit the classifier to the training data
    model = model.fit(X_train, y_train)

    # “Yang tadi sudah cukup saya rasa. Setelah model/classifier terbentuk, selanjutnya kita menggunakan model ini untuk memprediksi LABEL dari testing dataset (X_test), menggunakan fungsi .predict(). Fungsi ini akan mengembalikan hasil prediksi untuk setiap data point dari X_test dalam bentuk array. Proses ini kita kenal dengan TESTING,” sambung Senja.

    # Apply the classifier/model to the test data
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    # ukuran y_pred harus sama dengan y_test


    # Aku menelusuri ulang susunan kodeku, aku merasa ini sudah lengkap dan siap. “Nja, ini semua sudah selesai menurutku, ada tahap akhir khusus kah?”
    # “Tentu saja. sekarang kita melanjutkan di tahap terakhir dari modelling yaitu evaluasi hasil model. Untuk evaluasi model performance, setiap algorithm mempunyai metrik yang berbeda-beda. Sekarang saya akan menjelaskan sedikit metrik apa saja yang umumnya digunakan. Metrik paling sederhana untuk mengecek performansi model adalah accuracy.”
    # “Kita bisa munculkan dengan fungsi .score( ). Tetapi, di banyak real problem, accuracy saja tidaklah cukup. Metode lain yang digunakan adalah dengan Confusion Matrix. Confusion Matrix merepresentasikan perbandingan prediksi dan real LABEL dari test dataset yang dihasilkan oleh algoritma ML,” tukas Senja sambil membuka  template dari confusion Matrix untukku:

    # True Positive (TP): Jika user diprediksi (Positif) membeli ([Revenue] = 1]), dan memang benar(True) membeli.
    # True Negative (TN): Jika user diprediksi tidak (Negatif) membeli dan aktualnya user tersebut memang (True) membeli.
    # False Positive (FP): Jika user diprediksi Positif membeli, tetapi ternyata tidak membeli (False).
    # False Negatif (FN): Jika user diprediksi tidak membeli (Negatif), tetapi ternyata sebenarnya membeli.

    # Untuk menampilkan confusion matrix cukup menggunakan fungsi confusion_matrix() dari Scikit-Learn

    # evaluating the model
    print('Training Accuracy :', model.score(X_train, y_train))
    print('Testing Accuracy :', model.score(X_test, y_test))

    # confusion matrix
    print('\nConfusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # classification report
    print('\nClassification report:')
    cr = classification_report(y_test, y_pred)
    print(cr)

    # Jika dataset memiliki jumlah data False Negatif dan False Positif yang seimbang (Symmetric), maka bisa gunakan Accuracy, tetapi jika tidak seimbang, maka sebaiknya menggunakan F1-Score.
    # Dalam suatu problem, jika lebih memilih False Positif lebih baik terjadi daripada False Negatif, misalnya: Dalam kasus Fraud/Scam, kecenderungan model mendeteksi transaksi sebagai fraud walaupun kenyataannya bukan, dianggap lebih baik, daripada transaksi tersebut tidak terdeteksi sebagai fraud tetapi ternyata fraud. Untuk problem ini sebaiknya menggunakan Recall.
    # Sebaliknya, jika lebih menginginkan terjadinya False Negatif dan sangat tidak menginginkan terjadinya False Positif, sebaiknya menggunakan Precision.
    # Contohnya adalah pada kasus klasifikasi email SPAM atau tidak. Banyak orang lebih memilih jika email yang sebenarnya SPAM namun diprediksi tidak SPAM (sehingga tetap ada pada kotak masuk email kita), daripada email yang sebenarnya bukan SPAM tapi diprediksi SPAM (sehingga tidak ada pada kotak masuk email).

    # Logistic Regression merupakan salah satu algoritma klasifikasi dasar yang cukup popular. Secara sederhana, Logistic regression hampir serupa dengan linear regression tetapi linear regression digunakan untuk Label atau Target Variable yang berupa numerik atau continuous value, sedangkan Logistic regression digunakan untuk Label atau Target yang berupa categorical/discrete value.
    # Contoh continuous value adalah harga rumah, harga saham, suhu, dsb; dan contoh dari categorical value adalah prediksi SPAM or NOT SPAM (1 dan 0) atau prediksi customer SUBSCRIBE atau UNSUBSCRIBED (1 dan 0).
    # Umumnya Logistic Regression dipakai untuk binary classification (1/0; Yes/No; True/False) problem, tetapi beberapa data scientist juga menggunakannya untuk multiclass classification problem. Logistic regression adalah salah satu linear classifier, oleh karena itu, Logistik regression juga menggunakan rumus atau fungsi yang sama seperti linear regression

    # Call the classifier
    logreg = LogisticRegression()
    # Fit the classifier to the training data  
    logreg = logreg.fit(X_train, y_train)
    #Training Model: Predict 
    y_pred = logreg.predict(X_test)

    #Evaluate Model Performance
    print('Training Accuracy :', logreg.score(X_train, y_train))  
    print('Testing Accuracy :', logreg.score(X_test, y_test))  

    # confusion matrix
    print('\nConfusion matrix')  
    cm = confusion_matrix(y_test, y_pred)  
    print(cm)

    # classification report  
    print('\nClassification report')  
    cr = classification_report(y_test, y_pred)  
    print(cr)

    # Decision Tree merupakan salah satu metode klasifikasi yang populer dan banyak diimplementasikan serta mudah diinterpretasi. Decision tree adalah model prediksi dengan struktur pohon atau struktur berhierarki. Decision Tree dapat digunakan untuk classification problem dan regression problem.

    # splitting the X, and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # Call the classifier
    decision_tree = DecisionTreeClassifier()
    # Fit the classifier to the training data
    decision_tree = decision_tree.fit(X_train, y_train)

    # evaluating the decision_tree performance
    print('Training Accuracy :', decision_tree.score(X_train, y_train))
    print('Testing Accuracy :', decision_tree.score(X_test, y_test))

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />"+"<br/>"+"<img src='data:image/png;base64, "+hist+"' />")

def supervisedLearning(request):
    # Regression merupakan metode statistik dan machine learning yang paling banyak digunakan. Seperti yang dijelaskan sebelumnya, regresi digunakan untuk memprediksi output label yang berbentuk numerik atau continuous value. Dalam proses training, model regresi akan menggunakan variabel input (independent variables atau features) dan variabel output (dependent variables atau label) untuk mempelajari bagaimana hubungan/pola dari variabel input dan output

    # Model regresi terdiri atas 2 tipe yaitu :

    # Simple regression model → model regresi paling sederhana, hanya terdiri dari satu feature (univariate) dan 1 target.
    # Multiple regression model → sesuai namanya, terdiri dari lebih dari satu feature (multivariate).
    # Adapun model regresi yang paling umum digunakan adalah Linear Regression.

    # Perlu diketahui bahwa tidak semua problem dapat diselesaikan dengan linear regression. Untuk pemodelan dengan linear regression, terdapat beberapa asumsi yang harus dipenuhi, yaitu :

    # Terdapat hubungan linear antara variabel input (feature) dan variabel output(label). Untuk melihat hubungan linear feature dan label, dapat menggunakan chart seperti scatter chart. Untuk mengetahui hubungan dari variabel umumnya dilakukan pada tahap eksplorasi data.
    # Tidak ada multicollinearity antara features. Multicollinearity artinya terdapat dependency (ketergantungan) antara feature, misalnya saja hanya bisa mengetahui nilai feature B jika nilai feature A sudah diketahui.
    # Tidak ada autocorrelation dalam data, contohnya pada time-series data.

    housing = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/housing_boston.csv')
    #Data rescaling
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    housing[['RM','LSTAT','PTRATIO','MEDV']] = data_scaler.fit_transform(housing[['RM', 'LSTAT', 'PTRATIO', 'MEDV']])
    # getting dependent and independent variables
    X = housing.drop(['MEDV'], axis = 1)
    y = housing['MEDV']
    # checking the shapes
    print('Shape of X:', X.shape)
    print('Shape of y:', y.shape)

    # splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    # checking the shapes  
    print('Shape of X_train :', X_train.shape)
    print('Shape of y_train :', y_train.shape)
    print('Shape of X_test :', X_test.shape)
    print('Shape of y_test :', y_test.shape)

    ##import regressor from Scikit-Learn
    # Call the regressor
    reg = LinearRegression()
    # Fit the regressor to the training data  
    reg = reg.fit(X_train, y_train)
    # Apply the regressor/model to the test data  
    y_pred = reg.predict(X_test)

    # “Untuk model regression, kita menghitung selisih antara nilai aktual (y_test) dan nilai prediksi (y_pred) yang disebut error, adapun beberapa metric yang umum digunakan. Coba kamu ke mari, aku jelaskan langkah-langkahnya.”
    # Semakin kecil nilai MSE, RMSE, dan MAE, semakin baik pula performansi model regresi. Untuk menghitung nilai MSE, RMSE dan MAE dapat dilakukan dengan menggunakan fungsi mean_squared_error () ,  mean_absolute_error () dari scikit-learn.metrics dan untuk RMSE sendiri tidak terdapat fungsi khusus di scikit-learn tapi dapat dengan mudah kita hitung dengan terlebih dahulu menghitung MSE kemudian menggunakan numpy module yaitu, sqrt() untuk memperoleh nilai akar kuadrat dari MSE.

    #Calculating MSE, lower the value better it is. 0 means perfect prediction
    mse = mean_squared_error(y_test, y_pred)
    print('Mean squared error of testing set:', mse)
    #Calculating MAE
    mae = mean_absolute_error(y_test, y_pred)
    print('Mean absolute error of testing set:', mae)
    #Calculating RMSE
    rmse = np.sqrt(mse)
    print('Root Mean Squared Error of testing set:', rmse)

    #Plotting y_test dan y_pred
    plt.scatter(y_test, y_pred, c = 'green')
    plt.xlabel('Price Actual')
    plt.ylabel('Predicted value')
    plt.title('True value vs predicted value : Linear Regression')

    graph = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />")

def unsupervisedLearning(request):
    # Unsupervised Learning adalah teknik machine learning dimana tidak terdapat label atau output yang digunakan untuk melatih model. Jadi, model dengan sendirinya akan bekerja untuk menemukan pola atau informasi dari dataset yang ada. Metode unsupervised learning yang dikenal dengan clustering. Sesuai dengan namanya, Clustering memproses data dan mengelompokkannya atau mengcluster objek/sample berdasarkan kesamaan antar objek/sampel dalam satu kluster, dan objek/sample ini cukup berbeda dengan objek/sample di kluster yang lain.

    # “K-Means merupakan tipe clustering dengan centroid based (titik pusat). Artinya kesamaan dari objek/sampel dihitung dari seberapa dekat objek itu dengan centroid atau titik pusat.”

    # “Untuk menentukan centroid, pada awalnya kita perlu mendefinisikan jumlah centroid (K) yang diinginkan, semisalnya kita menetapkan jumlah K = 3; maka pada awal iterasi, algorithm akan secara random menentukan 3 centroid. Setelah itu, objek/sample/data point yang lain akan dikelompokkan sebagai anggota dari salah satu centroid yang terdekat, sehingga terbentuk 3 cluster data. Sampai sini cukup dipahami?”

    # “Iterasi selanjutnya, titik-titik centroid diupdate atau berpindah ke titik yang lain, dan jarak dari data point yang lain ke centroid yang baru dihitung kembali, kemudian dikelompokkan kembali berdasarkan jarak terdekat ke centroid yang baru. Iterasi akan terus berlanjut hingga diperoleh cluster dengan error terkecil, dan posisi centroid tidak lagi berubah.”

    dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/mall_customers.csv')   
    X = dataset[['annual_income','spending_score']]  

    cluster_model = KMeans(n_clusters = 5, random_state = 24)  
    labels = cluster_model.fit_predict(X)

    #convert dataframe to array
    X = X.values
    #Separate X to xs and ys --> use for chart axis
    xs = X[:,0]
    ys = X[:,1]
    # Make a scatter plot of xs and ys, using labels to define the colors
    plt.scatter(xs,ys,c=labels, alpha=0.5)

    # Assign the cluster centers: centroids
    centroids = cluster_model.cluster_centers_
    # Assign the columns of centroids: centroids_x, centroids_y
    centroids_x = centroids[:,0]
    centroids_y = centroids[:,1]
    # Make a scatter plot of centroids_x and centroids_y
    plt.scatter(centroids_x,centroids_y,marker='D', s=50)
    plt.title('K Means Clustering', fontsize = 20)
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')

    graph = getGraph()

    # “Segmentasinya udah jadi nih, Nja. Tapi, bagaimana kita tahu bahwa membagi segmentasi ke dalam 5 cluster adalah segmentasi yang paling optimal? Karena jika dilihat pada gambar beberapa data point masih cukup jauh jaraknya dengan centroidnya.”
    # “Clustering yang baik adalah cluster yang data point-nya saling rapat/sangat berdekatan satu sama lain dan cukup berjauhan dengan objek/data point di cluster yang lain. Jadi, objek dalam satu cluster tidak tersebut berjauhan. Nah, untuk mengukur kualitas dari clustering, kita bisa menggunakan inertia,” jawab Senja langsung.
    # “Inertia sendiri mengukur seberapa besar penyebaran object/data point data dalam satu cluster, semakin kecil nilai inertia maka semakin baik. Kita tidak perlu bersusah payah menghitung nilai inertia karena secara otomatis, telah dihitung oleh KMeans( ) ketika algorithm di fit ke dataset. Untuk mengecek nilai inertia cukup dengan print fungsi .inertia_ dari model yang sudah di fit ke dataset.”
    # “Kalau begitu,   bagaimana caranya mengetahui nilai K yang paling baik dengan inertia yang paling kecil? Apakah harus trial Error dengan mencoba berbagai jumlah cluster?”
    # “Benar, kita perlu mencoba beberapa nilai, dan memplot nilai inertia-nya. Semakin banyak cluster maka inertia semakin kecil. Sini deh, saya tunjukkan gambarnya.”
    # Meskipun suatu clustering dikatakan baik jika memiliki inertia yang kecil tetapi secara praktikal in real life, terlalu banyak cluster juga tidak diinginkan. Adapun rule untuk memilih jumlah cluster yang optimal adalah dengan memilih jumlah cluster yang terletak pada “elbow” dalam intertia plot, yaitu ketika nilai inertia mulai menurun secara perlahan. Jika dilihat pada gambar maka jumlah cluster yang optimal adalah K = 3.

    # “Kamu coba latihan saja. Aksara. Coba kamu membuat inertia plot untuk melihat apakah K = 5 merupakan jumlah cluster yang optimal. Saya kirim ya berkas latihannya.”

    #Elbow Method - Inertia plot
    inertia = []
    #looping the inertia calculation for each k
    for k in range(1, 10):
        #Assign KMeans as cluster_model
        cluster_model = KMeans(n_clusters = k, random_state = 24)
        #Fit cluster_model to X
        cluster_model.fit(X)
        #Get the inertia value
        inertia_value = cluster_model.inertia_
        #Append the inertia_value to inertia list
        inertia.append(inertia_value)
        
    ##Inertia plot
    plt.clf()
    plt.plot(range(1, 10), inertia)
    plt.title('The Elbow Method - Inertia plot', fontsize = 20)
    plt.xlabel('No. of Clusters')
    plt.ylabel('inertia')

    elbow = getGraph()

    return HttpResponse("<img src='data:image/png;base64, "+graph+"' />"+"<br/>"+"<img src='data:image/png;base64, "+elbow+"' />")

def tugasMiniProject(request):
    # Baca data 'ecommerce_banner_promo.csv'
    data = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/ecommerce_banner_promo.csv')

    #1. Data eksplorasi dengan head(), info(), describe(), shape
    print("\n[1] Data eksplorasi dengan head(), info(), describe(), shape")
    print("Lima data teratas:")
    print(data.head())
    print("Informasi dataset:")
    print(data.info())
    print("Statistik deskriptif dataset:")
    print(data.describe())
    print("Ukuran dataset:")
    print(data.shape)

    #2. Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()
    print("\n[2] Data eksplorasi dengan dengan mengecek korelasi dari setiap feature menggunakan fungsi corr()")
    print(data.corr())

    #3. Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()
    print("\n[3] Data eksplorasi dengan mengecek distribusi label menggunakan fungsi groupby() dan size()")
    print(data.groupby('Clicked on Ad'))

    # Seting: matplotlib and seaborn
    sns.set_style('whitegrid')  
    plt.style.use('fivethirtyeight')

    #4. Data eksplorasi dengan visualisasi
    #4a. Visualisasi Jumlah user dibagi ke dalam rentang usia (Age) menggunakan histogram (hist()) plot
    plt.figure(figsize=(10, 5))
    plt.hist(data['Age'], bins = data.Age.nunique())
    plt.xlabel('Age')
    plt.tight_layout()

    hist = getGraph()

    #4b. Gunakan pairplot() dari seaborn (sns) modul untuk menggambarkan hubungan setiap feature.
    plt.clf()
    plt.figure()
    sns.pairplot(data)
    plt.show()

    pairplot = getGraph()

    elbow = getGraph()

    #5. Cek missing value
    print("\n[5] Cek missing value")
    print(data.isnull().sum().sum())

    #6.Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing
    print("\n[6] Lakukan pemodelan dengan Logistic Regression, gunakan perbandingan 80:20 untuk training vs testing")
    #6a.Drop Non-Numerical (object type) feature from X, as Logistic Regression can only take numbers, and also drop Target/label, assign Target Variable to y.   
    X = data.drop(['Ad Topic Line','City','Country','Timestamp','Clicked on Ad'], axis = 1)
    y = data['Clicked on Ad']

    #6b. splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    #6c. Modelling
    # Call the classifier
    logreg = LogisticRegression()
    # Fit the classifier to the training data
    logreg = logreg.fit(X_train,y_train)
    # Prediksi model
    y_pred = logreg.predict(X_test)

    #6d. Evaluasi Model Performance
    print("Evaluasi Model Performance:")
    print("Training Accuracy :", logreg.score(X_train, y_train))
    print("Testing Accuracy :", logreg.score(X_test, y_test))

    #7. Print Confusion matrix dan classification report
    print("\n[7] Print Confusion matrix dan classification report")

    #apply confusion_matrix function to y_test and y_pred
    print("Confusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    #apply classification_report function to y_test and y_pred
    print("Classification report:")
    cr = classification_report(y_test,y_pred)
    print(cr)

    return HttpResponse("<img src='data:image/png;base64, "+hist+"' />"+"<br/>"+"<img src='data:image/png;base64, "+pairplot+"' />")

def getGraph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png');
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()

    return graph
