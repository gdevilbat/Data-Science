from django.urls import path

from . import persiapan
from . import fundamentalscience
from . import exploratory
from . import mldspot
from . import dataquality
from . import machinelearning

urlpatterns = [
    path('persiapan/tugas-fundamental-3', persiapan.tugasFundamentalTiga),
    path('persiapan/tugas-perulangan', persiapan.tugasPerulangan),
    path('persiapan/tugas-membaca-csv', persiapan.tugasMembacaCSV),
    path('persiapan/tugas-csv-pandas', persiapan.tugasCSVPandas),
    path('persiapan/tugas-bar-1', persiapan.tugasBarSatu),

    path('fundamental-science/tugas-membaca-data', fundamentalscience.tugasMembacaData),
    path('fundamental-science/tugas-isi-data-kosong-mean-median', fundamentalscience.tugasIsiDataKosongMeanMedian),
    path('fundamental-science/tugas-normalisasi-data', fundamentalscience.tugasNormalisasiData),
    path('fundamental-science/tugas-proffesional-beginner', fundamentalscience.tugasProffesionalBeginner),
    path('fundamental-science/tugas-multi-line', fundamentalscience.tugasMultiLine),
    path('fundamental-science/tugas-multi-line-top-provinsi', fundamentalscience.tugasMultiLineTopProvinsi),
    path('fundamental-science/tugas-multi-line-annotation', fundamentalscience.tugasMultiLineAnnotation),
    path('fundamental-science/tugas-pie-bar-chart', fundamentalscience.tugasPieBarChart),
    path('fundamental-science/tugas-multi-bar-chart', fundamentalscience.tugasMultiBarChart),
    path('fundamental-science/tugas-stacked-bar', fundamentalscience.tugasStackedChart),

    path('fundamental-science/tugas-histogram-scatterplot', fundamentalscience.tugasHistogramScatterplot),
    path('fundamental-science/tugas-mini-project', fundamentalscience.tugasMiniProject),

    path('exploratory/', exploratory.index),
    path('exploratory/tugas-mini-project', exploratory.tugasMiniProject),

    path('data-quality/', dataquality.index),
    path('data-quality/tugas-profiling', dataquality.tugasProfiling),
    path('data-quality/tugas-mini-project', dataquality.tugasMiniProject),

    path('machine-learning/', machinelearning.index),
    path('machine-learning/supervised-learning', machinelearning.supervisedLearning),
    path('machine-learning/unsupervised-learning', machinelearning.unsupervisedLearning),
    path('machine-learning/tugas-mini-project', machinelearning.tugasMiniProject),

    path('mldspot/data-mldspot', mldspot.dataInterest),

]