from django.urls import path

from . import persiapan
from . import FundamentalScience
from . import exploratory
from . import mldspot
from . import super
from . import DataQuality
from . import MachineLearning
from . import FundamentalVisualization
from . import DataManipulation

urlpatterns = [
    path('persiapan/tugas-fundamental-3', persiapan.tugasFundamentalTiga),
    path('persiapan/tugas-perulangan', persiapan.tugasPerulangan),
    path('persiapan/tugas-membaca-csv', persiapan.tugasMembacaCSV),
    path('persiapan/tugas-csv-pandas', persiapan.tugasCSVPandas),
    path('persiapan/tugas-bar-1', persiapan.tugasBarSatu),

    path('fundamental-science/tugas-membaca-data', FundamentalScience.tugasMembacaData),
    path('fundamental-science/tugas-isi-data-kosong-mean-median', FundamentalScience.tugasIsiDataKosongMeanMedian),
    path('fundamental-science/tugas-normalisasi-data', FundamentalScience.tugasNormalisasiData),
    path('fundamental-science/tugas-proffesional-beginner', FundamentalScience.tugasProffesionalBeginner),
    path('fundamental-science/tugas-multi-line', FundamentalScience.tugasMultiLine),
    path('fundamental-science/tugas-multi-line-top-provinsi', FundamentalScience.tugasMultiLineTopProvinsi),
    path('fundamental-science/tugas-multi-line-annotation', FundamentalScience.tugasMultiLineAnnotation),
    path('fundamental-science/tugas-pie-bar-chart', FundamentalScience.tugasPieBarChart),
    path('fundamental-science/tugas-multi-bar-chart', FundamentalScience.tugasMultiBarChart),
    path('fundamental-science/tugas-stacked-bar', FundamentalScience.tugasStackedChart),

    path('fundamental-science/tugas-histogram-scatterplot', FundamentalScience.tugasHistogramScatterplot),
    path('fundamental-science/tugas-mini-project', FundamentalScience.tugasMiniProject),

    path('exploratory/', exploratory.index),
    path('exploratory/tugas-mini-project', exploratory.tugasMiniProject),

    path('data-quality/', DataQuality.index),
    path('data-quality/tugas-profiling', DataQuality.tugasProfiling),
    path('data-quality/tugas-mini-project', DataQuality.tugasMiniProject),

    path('fundamental-visualization/tugas-mini-project', FundamentalVisualization.tugasMiniProject),

    path('machine-learning/', MachineLearning.index),
    path('machine-learning/supervised-learning', MachineLearning.supervisedLearning),
    path('machine-learning/unsupervised-learning', MachineLearning.unsupervisedLearning),
    path('machine-learning/tugas-mini-project', MachineLearning.tugasMiniProject),

    path('data-manipulation/introduction', DataManipulation.introduction),
    path('data-manipulation/dataset-io', DataManipulation.datasetIO),
    path('data-manipulation/transforming', DataManipulation.transforming),
    path('data-manipulation/mysql', DataManipulation.mysql),
]