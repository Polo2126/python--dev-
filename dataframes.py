#!/usr/bin/env python
# coding: utf-8

# # Intrusion detection on NSL-KDD

# This is my try with [NSL-KDD](http://www.unb.ca/research/iscx/dataset/iscx-NSL-KDD-dataset.html) dataset, which is an improved version of well-known [KDD'99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) dataset. I've used Python, Scikit-learn and PySpark via [ready-to-run Jupyter applications in Docker](https://github.com/jupyter/docker-stacks).
# 
# I've tried a variety of approaches to deal with this dataset. Here are presented some of them.
# 
# To be able to run this notebook, use `make nsl-kdd-pyspark` command. It'll download the latest jupyter/pyspark-notebook docker image and start a container with Jupyter available at `8889` port.

# ## Contents

# 1. [Task description summary](#1.-Task-description-summary)
# 2. [Data loading](#2.-Data-loading)
# 3. [Exploratory Data Analysis](#3.-Exploratory-Data-Analysis)
# 4. [One Hot Encoding for categorical variables](#4.-One-Hot-Encoding-for-categorical-variables)
# 5. [Feature Selection using Attribute Ratio](#5.-Feature-Selection-using-Attribute-Ratio)
# 6. [Data preparation](#6.-Data-preparation)
# 7. [Visualization via PCA](#7.-Visualization-via-PCA)
# 8. [KMeans clustering with Random Forest Classifiers](#8.-KMeans-clustering-with-Random-Forest-Classifiers)
# 9. [Gaussian Mixture clustering with Random Forest Classifiers](#9.-Gaussian-Mixture-clustering-with-Random-Forest-Classifiers)
# 10. [Supervised approach for dettecting each type of attacks separately](#10.-Supervised-approach-for-dettecting-each-type-of-attacks-separately)
# 11. [Ensembling experiments](#11.-Ensembling-experiments)
# 12. [Results summary](#12.-Results-summary)

# ## 1. Task description summary

# Software to detect network intrusions protects a computer network from unauthorized users, including perhaps insiders. The intrusion detector learning task is to build a predictive model (i.e. a classifier) capable of distinguishing between bad connections, called intrusions or attacks, and good normal connections.
# 
# A connection is a sequence of TCP packets starting and ending at some well defined times, between which data flows to and from a source IP address to a target IP address under some well defined protocol. Each connection is labeled as either normal, or as an attack, with exactly one specific attack type. Each connection record consists of about 100 bytes.
# 
# Attacks fall into four main categories:
# 
# - DOS: denial-of-service, e.g. syn flood;
# - R2L: unauthorized access from a remote machine, e.g. guessing password;
# - U2R: unauthorized access to local superuser (root) privileges, e.g., various ''buffer overflow'' attacks;
# - probing: surveillance and other probing, e.g., port scanning.
# 
# It is important to note that the test data is not from the same probability distribution as the training data, and it includes specific attack types not in the training data. This makes the task more realistic. Some intrusion experts believe that most novel attacks are variants of known attacks and the "signature" of known attacks can be sufficient to catch novel variants.  The datasets contain a total of 24 training attack types, with an additional 14 types in the test data only.
# 
# The complete task description could be found [here](http://kdd.ics.uci.edu/databases/kddcup99/task.html).

# ### NSL-KDD dataset description

# [NSL-KDD](http://www.unb.ca/research/iscx/dataset/iscx-NSL-KDD-dataset.html) is a data set suggested to solve some of the inherent problems of the [KDD'99](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) data set.
# 
# The NSL-KDD data set has the following advantages over the original KDD data set:
# - It does not include redundant records in the train set, so the classifiers will not be biased towards more frequent records.
# - There is no duplicate records in the proposed test sets; therefore, the performance of the learners are not biased by the methods which have better detection rates on the frequent records.
# - The number of selected records from each difficultylevel group is inversely proportional to the percentage of records in the original KDD data set. As a result, the classification rates of distinct machine learning methods vary in a wider range, which makes it more efficient to have an accurate evaluation of different learning techniques.
# - The number of records in the train and test sets are reasonable, which makes it affordable to run the experiments on the complete set without the need to randomly select a small portion. Consequently, evaluation results of different research works will be consistent and comparable.

# ## 2. Data loading

# In[1]:


# Here are some imports that are used along this notebook
import os
import math
import itertools
import multiprocessing
import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from collections import OrderedDict
get_ipython().run_line_magic('matplotlib', 'inline')
gt0 = time()


# In[2]:


from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row


# In[3]:


conf = SparkConf()    .setMaster(f"local[{multiprocessing.cpu_count()}]")    .setAppName("PySpark NSL-KDD")    .setAll([("spark.driver.memory", "8g"), ("spark.default.parallelism", f"{multiprocessing.cpu_count()}")])

# Creating local SparkContext with specified SparkConf and creating SQLContext based on it
sc = SparkContext.getOrCreate(conf=conf)
sc.setLogLevel('INFO')
sqlContext = SQLContext(sc)


# In[4]:


from pyspark.sql.types import *
from pyspark.sql.functions import udf, split, col
import pyspark.sql.functions as sql

train20_nsl_kdd_dataset_path = os.path.join("NSL_KDD_Dataset", "KDDTrain+_20Percent.txt")
train_nsl_kdd_dataset_path = os.path.join("NSL_KDD_Dataset", "KDDTrain+.txt")
test_nsl_kdd_dataset_path = os.path.join("NSL_KDD_Dataset", "KDDTest+.txt")

col_names = np.array(["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","labels"])

nominal_inx = [1, 2, 3]
binary_inx = [6, 11, 13, 14, 20, 21]
numeric_inx = list(set(range(41)).difference(nominal_inx).difference(binary_inx))

nominal_cols = col_names[nominal_inx].tolist()
binary_cols = col_names[binary_inx].tolist()
numeric_cols = col_names[numeric_inx].tolist()


# In[5]:


# Function to load dataset and divide it into 8 partitions
def load_dataset(path):
    dataset_rdd = sc.textFile(path, 8).map(lambda line: line.split(','))
    dataset_df = (dataset_rdd.toDF(col_names.tolist()).select(
                    col('duration').cast(DoubleType()),
                    col('protocol_type').cast(StringType()),
                    col('service').cast(StringType()),
                    col('flag').cast(StringType()),
                    col('src_bytes').cast(DoubleType()),
                    col('dst_bytes').cast(DoubleType()),
                    col('land').cast(DoubleType()),
                    col('wrong_fragment').cast(DoubleType()),
                    col('urgent').cast(DoubleType()),
                    col('hot').cast(DoubleType()),
                    col('num_failed_logins').cast(DoubleType()),
                    col('logged_in').cast(DoubleType()),
                    col('num_compromised').cast(DoubleType()),
                    col('root_shell').cast(DoubleType()),
                    col('su_attempted').cast(DoubleType()),
                    col('num_root').cast(DoubleType()),
                    col('num_file_creations').cast(DoubleType()),
                    col('num_shells').cast(DoubleType()),
                    col('num_access_files').cast(DoubleType()),
                    col('num_outbound_cmds').cast(DoubleType()),
                    col('is_host_login').cast(DoubleType()),
                    col('is_guest_login').cast(DoubleType()),
                    col('count').cast(DoubleType()),
                    col('srv_count').cast(DoubleType()),
                    col('serror_rate').cast(DoubleType()),
                    col('srv_serror_rate').cast(DoubleType()),
                    col('rerror_rate').cast(DoubleType()),
                    col('srv_rerror_rate').cast(DoubleType()),
                    col('same_srv_rate').cast(DoubleType()),
                    col('diff_srv_rate').cast(DoubleType()),
                    col('srv_diff_host_rate').cast(DoubleType()),
                    col('dst_host_count').cast(DoubleType()),
                    col('dst_host_srv_count').cast(DoubleType()),
                    col('dst_host_same_srv_rate').cast(DoubleType()),
                    col('dst_host_diff_srv_rate').cast(DoubleType()),
                    col('dst_host_same_src_port_rate').cast(DoubleType()),
                    col('dst_host_srv_diff_host_rate').cast(DoubleType()),
                    col('dst_host_serror_rate').cast(DoubleType()),
                    col('dst_host_srv_serror_rate').cast(DoubleType()),
                    col('dst_host_rerror_rate').cast(DoubleType()),
                    col('dst_host_srv_rerror_rate').cast(DoubleType()),
                    col('labels').cast(StringType())))

    return dataset_df


# The first part of data preparation is deviding connections into normal and attack classes based on 'labels' column. Then attacks are splitted to four main categories: DoS, Probe, R2L and U2R. After this, all of those categories are indexed. Also, ID column is added to simplify work with clustered data.

# In[6]:


from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer
from pyspark import keyword_only
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param

# Dictionary that contains mapping of various attacks to the four main categories
attack_dict = {
    'normal': 'normal',
    
    'back': 'DoS',
    'land': 'DoS',
    'neptune': 'DoS',
    'pod': 'DoS',
    'smurf': 'DoS',
    'teardrop': 'DoS',
    'mailbomb': 'DoS',
    'apache2': 'DoS',
    'processtable': 'DoS',
    'udpstorm': 'DoS',
    
    'ipsweep': 'Probe',
    'nmap': 'Probe',
    'portsweep': 'Probe',
    'satan': 'Probe',
    'mscan': 'Probe',
    'saint': 'Probe',

    'ftp_write': 'R2L',
    'guess_passwd': 'R2L',
    'imap': 'R2L',
    'multihop': 'R2L',
    'phf': 'R2L',
    'spy': 'R2L',
    'warezclient': 'R2L',
    'warezmaster': 'R2L',
    'sendmail': 'R2L',
    'named': 'R2L',
    'snmpgetattack': 'R2L',
    'snmpguess': 'R2L',
    'xlock': 'R2L',
    'xsnoop': 'R2L',
    'worm': 'R2L',
    
    'buffer_overflow': 'U2R',
    'loadmodule': 'U2R',
    'perl': 'U2R',
    'rootkit': 'U2R',
    'httptunnel': 'U2R',
    'ps': 'U2R',    
    'sqlattack': 'U2R',
    'xterm': 'U2R'
}

attack_mapping_udf = udf(lambda v: attack_dict[v])

class Labels2Converter(Transformer):

    @keyword_only
    def __init__(self):
        super(Labels2Converter, self).__init__()

    def _transform(self, dataset):
        return dataset.withColumn('labels2', sql.regexp_replace(col('labels'), '^(?!normal).*$', 'attack'))
     
class Labels5Converter(Transformer):
    
    @keyword_only
    def __init__(self):
        super(Labels5Converter, self).__init__()

    def _transform(self, dataset):
        return dataset.withColumn('labels5', attack_mapping_udf(col('labels')))
    
labels2_indexer = StringIndexer(inputCol="labels2", outputCol="labels2_index")
labels5_indexer = StringIndexer(inputCol="labels5", outputCol="labels5_index")

labels_mapping_pipeline = Pipeline(stages=[Labels2Converter(), Labels5Converter(), labels2_indexer, labels5_indexer])

labels2 = ['normal', 'attack']
labels5 = ['normal', 'DoS', 'Probe', 'R2L', 'U2R']
labels_col = 'labels2_index'


# In[7]:


# Loading train data
t0 = time()
train_df = load_dataset(train_nsl_kdd_dataset_path)

# Fitting preparation pipeline
labels_mapping_model = labels_mapping_pipeline.fit(train_df)

# Transforming labels column and adding id column
train_df = labels_mapping_model.transform(train_df).withColumn('id', sql.monotonically_increasing_id())

train_df = train_df.cache()
print(f"Number of examples in train set: {train_df.count()}")
print(f"Time: {time() - t0:.2f}s")


# In[8]:


# Loading test data
t0 = time()
test_df = load_dataset(test_nsl_kdd_dataset_path)

# Transforming labels column and adding id column
test_df = labels_mapping_model.transform(test_df).withColumn('id', sql.monotonically_increasing_id())

test_df = test_df.cache()
print(f"Number of examples in test set: {test_df.count()}")
print(f"Time: {time() - t0:.2f}s")


# ## 2. Exploratory Data Analysis

# Here are some descriptive statistics of available features.

# In[9]:


# Labels columns
(train_df.groupby('labels2').count().show())
(train_df.groupby('labels5').count().sort(sql.desc('count')).show())


# In[10]:


(test_df.groupby('labels2').count().show())
(test_df.groupby('labels5').count().sort(sql.desc('count')).show())


# In[11]:


# 'protocol_type' nominal column
(train_df.crosstab(nominal_cols[0], 'labels2').sort(sql.asc(nominal_cols[0] + '_labels2')).show())
(train_df.crosstab(nominal_cols[0], 'labels5').sort(sql.asc(nominal_cols[0] + '_labels5')).show())


# In[12]:


# 'service' nominal column
print(train_df.select(nominal_cols[1]).distinct().count())
(train_df.crosstab(nominal_cols[1], 'labels2').sort(sql.asc(nominal_cols[1] + '_labels2')).show(n=70))
(train_df.crosstab(nominal_cols[1], 'labels5').sort(sql.asc(nominal_cols[1] + '_labels5')).show(n=70))


# In[13]:


# 'flag' nominal column
print(train_df.select(nominal_cols[2]).distinct().count())
(train_df.crosstab(nominal_cols[2], 'labels2').sort(sql.asc(nominal_cols[2] + '_labels2')).show())
(train_df.crosstab(nominal_cols[2], 'labels5').sort(sql.asc(nominal_cols[2] + '_labels5')).show())


# In[14]:


# Binary columns
(train_df.select(binary_cols).describe().toPandas().transpose())


# In[15]:


# 'su_attempted' should be a binary feature, but has 3 values
(train_df.crosstab('su_attempted', 'labels2').show())


# In[16]:


# '2.0' value is replaced to '0.0' for both train and test datasets
train_df = train_df.replace(2.0, 0.0, 'su_attempted')
test_df = test_df.replace(2.0, 0.0, 'su_attempted')


# In[17]:


# Numeric columns
print(len(numeric_cols))
(train_df.select(numeric_cols).describe().toPandas().transpose())


# In[18]:


(train_df.crosstab('num_outbound_cmds', 'labels2').show())


# As 'num_outbound_cmds' feature takes only 0.0 values, so it is dropped as redundant.

# In[19]:


train_df = train_df.drop('num_outbound_cmds')
test_df = test_df.drop('num_outbound_cmds')
numeric_cols.remove('num_outbound_cmds')


# Commented code below is related to removing highly correlated features. However, it hasen't been tested a lot yet.

# In[20]:


# from pyspark.mllib.stat import Statistics
# from pyspark.mllib.linalg import Vectors
# from pyspark.ml.feature import VectorAssembler

# t0 = time()
# stat_assembler = VectorAssembler(
#                 inputCols=numeric_cols,
#                 outputCol='features')

# stat_rdd = stat_assembler.transform(train_df).rdd.map(lambda row: row['features'].toArray())

# pearson_corr = Statistics.corr(stat_rdd, method='pearson')
# spearman_corr = Statistics.corr(stat_rdd, method='spearman')

# print(time() - t0)


# In[21]:


# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# ax1.set_title("Pearson")
# ax2.set_title("Spearman")
# sns.heatmap(pearson_corr, ax=ax1)
# sns.heatmap(spearman_corr, ax=ax2)


# In[22]:


# inx_correlated_to_delete = [8, 15, 28, 17, 29]

# for inx in inx_correlated_to_delete:
#     train_df = train_df.drop(numeric_cols[inx])
#     test_df = test_df.drop(numeric_cols[inx])

# numeric_cols = [col for inx, col in enumerate(numeric_cols) if inx not in inx_correlated_to_delete]

# train_df = train_df.cache()
# test_df = test_df.cache()
# print(train_df.count())
# print(test_df.count())
# print(len(numeric_cols))


# ## 4. One Hot Encoding for categorical variables

# One Hot Encoding (OHE) is used for treating categorical variables. Custom function is created for demonstration purposes. However, it could be easily replaced by PySpark OneHotEncoder.

# In[23]:


def ohe_vec(cat_dict, row):
    vec = np.zeros(len(cat_dict))
    vec[cat_dict[row]] = float(1.0)
    return vec.tolist()

def ohe(df, nominal_col):
    categories = (df.select(nominal_col)
                    .distinct()
                    .rdd.map(lambda row: row[0])
                    .collect())
    
    cat_dict = dict(zip(categories, range(len(categories))))
    
    udf_ohe_vec = udf(lambda row: ohe_vec(cat_dict, row), 
                      StructType([StructField(cat, DoubleType(), False) for cat in categories]))
    
    df = df.withColumn(nominal_col + '_ohe', udf_ohe_vec(col(nominal_col))).cache()
    
    nested_cols = [nominal_col + '_ohe.' + cat for cat in categories]
    ohe_cols = [nominal_col + '_' + cat for cat in categories]
        
    for new, old in zip(ohe_cols, nested_cols):
        df = df.withColumn(new, col(old))

    df = df.drop(nominal_col + '_ohe')
                   
    return df, ohe_cols


# In[24]:


t0 = time()
train_ohe_cols = []

train_df, train_ohe_col0 = ohe(train_df, nominal_cols[0])
train_ohe_cols += train_ohe_col0

train_df, train_ohe_col1 = ohe(train_df, nominal_cols[1])
train_ohe_cols += train_ohe_col1

train_df, train_ohe_col2 = ohe(train_df, nominal_cols[2])
train_ohe_cols += train_ohe_col2

binary_cols += train_ohe_cols

train_df = train_df.cache()
print(f"Number of examples in train set: {train_df.count()}")
print(f"Time: {time() - t0:.2f}s")


# Custom list of test binary cols is used as test dataset could contain additional categories for 'service' and 'flag' features. However, those additional categories aren't used below.

# In[25]:


t0 = time()
test_ohe_cols = []

test_df, test_ohe_col0_names = ohe(test_df, nominal_cols[0])
test_ohe_cols += test_ohe_col0_names

test_df, test_ohe_col1_names = ohe(test_df, nominal_cols[1])
test_ohe_cols += test_ohe_col1_names

test_df, test_ohe_col2_names = ohe(test_df, nominal_cols[2])
test_ohe_cols += test_ohe_col2_names

test_binary_cols = col_names[binary_inx].tolist() + test_ohe_cols

test_df = test_df.cache()
print(f"Number of examples in test set: {test_df.count()}")
print(f"Time: {time() - t0:.2f}s")


# ## 5. Feature Selection using Attribute Ratio

# Attribute Ratio approach is used for feature selection purposes. This approach was described by Hee-su Chae and Sang Hyun Choi in [Feature Selection for efficient Intrusion Detection using Attribute Ratio](http://www.naun.org/main/UPress/cc/2014/a102019-106.pdf) and [Feature Selection for Intrusion Detection using NSL-KDD](http://www.wseas.us/e-library/conferences/2013/Nanjing/ACCIS/ACCIS-30.pdf)
# 
# This approach is also used for nominal variables as they were encoded as binary variables above.

# As it is a possible to have 'null' values because binary features could have Frequency(0) = 0, those 'null' values are replaced with 1000.0 (magic number). For NSL KDD dataset it is related only for 'protocol_type_tcp' ohe variable.

# In[26]:


def getAttributeRatio(df, numericCols, binaryCols, labelCol):
    ratio_dict = {}
    
    if numericCols:
        avg_dict = (df
                .select(list(map(lambda c: sql.avg(c).alias(c), numericCols)))
                .first()
                .asDict())

        ratio_dict.update(df
                .groupBy(labelCol)
                .avg(*numericCols)
                .select(list(map(lambda c: sql.max(col('avg(' + c + ')')/avg_dict[c]).alias(c), numericCols)))
                .fillna(0.0)
                .first()
                .asDict())
    
    if binaryCols:
        ratio_dict.update((df
                .groupBy(labelCol)
                .agg(*list(map(lambda c: (sql.sum(col(c))/(sql.count(col(c)) - sql.sum(col(c)))).alias(c), binaryCols)))
                .fillna(1000.0)
                .select(*list(map(lambda c: sql.max(col(c)).alias(c), binaryCols)))
                .first()
                .asDict()))
        
    return OrderedDict(sorted(ratio_dict.items(), key=lambda v: -v[1]))

def selectFeaturesByAR(ar_dict, min_ar):
    return [f for f in ar_dict.keys() if ar_dict[f] >= min_ar]


# In[27]:


t0 = time()
ar_dict = getAttributeRatio(train_df, numeric_cols, binary_cols, 'labels5')

print(f"Number of features in Attribute Ration dict: {len(ar_dict)}")
print(f"Time: {time() - t0:.2f}s")
ar_dict


# ## 6. Data preparation

# Standartization is necessary as a lot of distance based algorithms are used below. Custom standartization is created for demonstration purposes, so it could be easily replaced by PySpark StandardScaler. Note that data is sparse, so it is reasonable to not substract mean for avoiding violating sparsity. 

# In[28]:


t0 = time()
avg_dict = (train_df.select(list(map(lambda c: sql.avg(c).alias(c), numeric_cols))).first().asDict())
std_dict = (train_df.select(list(map(lambda c: sql.stddev(c).alias(c), numeric_cols))).first().asDict())

def standardizer(column):
    return ((col(column) - avg_dict[column])/std_dict[column]).alias(column)

# Standardizer without mean
# def standardizer(column):
#     return (col(column)/std_dict[column]).alias(column)

train_scaler = [*binary_cols, *list(map(standardizer, numeric_cols)), *['id', 'labels2_index', 'labels2', 'labels5_index', 'labels5']]
test_scaler = [*test_binary_cols, *list(map(standardizer, numeric_cols)), *['id', 'labels2_index', 'labels2', 'labels5_index', 'labels5']]

scaled_train_df = (train_df.select(train_scaler).cache())
scaled_test_df = (test_df.select(test_scaler).cache())

print(scaled_train_df.count())
print(scaled_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# VectorAssembler is used for combining a given list of columns into a single vector column. Then VectorIndexer is used for indexing categorical (binary) features. Indexing categorical features allows algorithms to treat them appropriately, improving performance.

# In[29]:


from pyspark.ml.feature import VectorIndexer, VectorAssembler
assembler = VectorAssembler(inputCols=selectFeaturesByAR(ar_dict, 0.01), outputCol='raw_features')
indexer = VectorIndexer(inputCol='raw_features', outputCol='indexed_features', maxCategories=2)

prep_pipeline = Pipeline(stages=[assembler, indexer])
prep_model = prep_pipeline.fit(scaled_train_df)


# In[30]:


t0 = time()
scaled_train_df = (prep_model
        .transform(scaled_train_df)
        .select('id', 'indexed_features', 'labels2_index', 'labels2', 'labels5_index', 'labels5')
        .cache())

scaled_test_df = (prep_model 
        .transform(scaled_test_df)
        .select('id', 'indexed_features','labels2_index', 'labels2', 'labels5_index', 'labels5')
        .cache())

print(scaled_train_df.count())
print(scaled_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[31]:


# Setting seed for reproducibility
seed = 4667979835606274383
print(seed)


# The train dataset is splitted into 80% train and 20% cross-validation sets.

# In[32]:


split = (scaled_train_df.randomSplit([0.8, 0.2], seed=seed))

scaled_train_df = split[0].cache()
scaled_cv_df = split[1].cache()

print(scaled_train_df.count())
print(scaled_cv_df.count())


# Additional "result" dataframes are used to collect probabilities and predictions from different approaches.

# In[33]:


res_cv_df = scaled_cv_df.select(col('id'), col('labels2_index'), col('labels2'), col('labels5')).cache()
res_test_df = scaled_test_df.select(col('id'), col('labels2_index'), col('labels2'), col('labels5')).cache()
prob_cols = []
pred_cols = []

print(res_cv_df.count())
print(res_test_df.count())


# Different metrics from sklearn are used for evaluating results. The most important from them for this task are False positive Rate, Detection Rate and F1 score. 
# As evaluating via sklearn requires to collect predicted and label columns to the driver, it will be replaced with PySpark metrics later.

# In[34]:


import sklearn.metrics as metrics

def printCM(cm, labels):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels])
    # Print header
    print(" " * columnwidth, end="\t")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end="\t")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("%{0}s".format(columnwidth) % label1, end="\t")
        for j in range(len(labels)):
            print("%{0}d".format(columnwidth) % cm[i, j], end="\t")
        print()

def getPrediction(e):
    return udf(lambda row: 1.0 if row >= e else 0.0, DoubleType())
        
def printReport(resDF, probCol, labelCol='labels2_index', e=None, labels=['normal', 'attack']):
    if (e):
        predictionAndLabels = list(zip(*resDF.rdd
                                       .map(lambda row: (1.0 if row[probCol] >= e else 0.0, row[labelCol]))
                                       .collect()))
    else:
        predictionAndLabels = list(zip(*resDF.rdd
                                       .map(lambda row: (row[probCol], row[labelCol]))
                                       .collect()))
    
    cm = metrics.confusion_matrix(predictionAndLabels[1], predictionAndLabels[0])
    printCM(cm, labels)
    print(" ")
    print("Accuracy = %g" % (metrics.accuracy_score(predictionAndLabels[1], predictionAndLabels[0])))
    print("AUC = %g" % (metrics.roc_auc_score(predictionAndLabels[1], predictionAndLabels[0])))
    print(" ")
    print("False Alarm Rate = %g" % (cm[0][1]/(cm[0][0] + cm[0][1])))
    print("Detection Rate = %g" % (cm[1][1]/(cm[1][1] + cm[1][0])))
    print("F1 score = %g" % (metrics.f1_score(predictionAndLabels[1], predictionAndLabels[0], labels)))
    print(" ")
    print(metrics.classification_report(predictionAndLabels[1], predictionAndLabels[0]))
    print(" ")


# ## 7. Visualization via PCA

# PCA algorithm is used for visualization purposes. It's also used later as preprocessing for Gaussian Mixture clustering.
# 
# First graph shows 'attack' vs 'normal' labels, second graph shows 4 different types of attacks vs normal connections.

# In[35]:


from pyspark.ml.feature import VectorSlicer
from pyspark.ml.feature import PCA

t0 = time()
pca_slicer = VectorSlicer(inputCol="indexed_features", outputCol="features", names=selectFeaturesByAR(ar_dict, 0.05))

pca = PCA(k=2, inputCol="features", outputCol="pca_features")
pca_pipeline = Pipeline(stages=[pca_slicer, pca])

pca_train_df = pca_pipeline.fit(scaled_train_df).transform(scaled_train_df)
print(f"Time: {time() - t0:.2f}s")


# In[36]:


t0 = time()
viz_train_data = np.array(pca_train_df.rdd.map(lambda row: [*row['pca_features'], row['labels2_index'], row['labels5_index']]).collect())
plt.figure()
plt.scatter(x=viz_train_data[:,0], y=viz_train_data[:,1], c=viz_train_data[:,2], cmap="Set1")
plt.figure()
plt.scatter(x=viz_train_data[:,0], y=viz_train_data[:,1], c=viz_train_data[:,3], cmap="Set1")
plt.show()
print(time() - t0)


# ## 8. KMeans clustering with Random Forest Classifiers

# The idea of the first approach is to clusterize data into clusters and then train different Random Forest classifiers for each of the clusters. As Random Forest returns probabilities, it is possible to improve detection rate for a new types of attacks by adjusting threshold.

# As KMeans cannot truly handle binary/categorical features only numeric features are used for clustarization.

# In[37]:


kmeans_prob_col = 'kmeans_rf_prob'
kmeans_pred_col = 'kmeans_rf_pred'

prob_cols.append(kmeans_prob_col)
pred_cols.append(kmeans_pred_col)


# In[38]:


# KMeans clustrering
from pyspark.ml.clustering import KMeans

t0 = time()
kmeans_slicer = VectorSlicer(inputCol="indexed_features", outputCol="features", 
                             names=list(set(selectFeaturesByAR(ar_dict, 0.1)).intersection(numeric_cols)))

kmeans = KMeans(k=8, initSteps=25, maxIter=100, featuresCol="features", predictionCol="cluster", seed=seed)

kmeans_pipeline = Pipeline(stages=[kmeans_slicer, kmeans])

kmeans_model = kmeans_pipeline.fit(scaled_train_df)

kmeans_train_df = kmeans_model.transform(scaled_train_df).cache()
kmeans_cv_df = kmeans_model.transform(scaled_cv_df).cache()
kmeans_test_df = kmeans_model.transform(scaled_test_df).cache()

print(f"Time: {time() - t0:.2f}s")


# In[39]:


# Function for describing the contents of the clusters 
def getClusterCrosstab(df, clusterCol='cluster'):
    return (df.crosstab(clusterCol, 'labels2')
              .withColumn('count', col('attack') + col('normal'))
              .withColumn(clusterCol + '_labels2', col(clusterCol + '_labels2').cast('int'))
              .sort(col(clusterCol +'_labels2').asc()))


# In[40]:


kmeans_crosstab = getClusterCrosstab(kmeans_train_df).cache()
kmeans_crosstab.show(n=30)


# Clustres are splitted into two categories. Frist category contains clusters that have both 'attack' and 'normal' connections and have more than 25 connections. For the first category Random Forest classifiers are aplied. Second category contains all other clusters and maps cluster to 'attack' or 'normal' based on majority. All clusters that contains less or equal than 25 connections are treated as outliers and are mapped to 'attack' type.

# In[41]:


# Function for splitting clusters
def splitClusters(crosstab):
    exp = ((col('count') > 25) & (col('attack') > 0) & (col('normal') > 0))

    cluster_rf = (crosstab
        .filter(exp).rdd
        .map(lambda row: (int(row['cluster_labels2']), [row['count'], row['attack']/row['count']]))
        .collectAsMap())

    cluster_mapping = (crosstab
        .filter(~exp).rdd
        .map(lambda row: (int(row['cluster_labels2']), 1.0 if (row['count'] <= 25) | (row['normal'] == 0) else 0.0))
        .collectAsMap())
    
    return cluster_rf, cluster_mapping

kmeans_cluster_rf, kmeans_cluster_mapping = splitClusters(kmeans_crosstab)

print(len(kmeans_cluster_rf), len(kmeans_cluster_mapping))
print(kmeans_cluster_mapping)
kmeans_cluster_rf


# In[42]:


from pyspark.ml.classification import RandomForestClassifier

# This function returns Random Forest models for provided clusters
def getClusterModels(df, cluster_rf):
    cluster_models = {}

    labels_col = 'labels2_cl_index'
    labels2_indexer.setOutputCol(labels_col)

    rf_slicer = VectorSlicer(inputCol="indexed_features", outputCol="rf_features", 
                             names=selectFeaturesByAR(ar_dict, 0.05))

    for cluster in cluster_rf.keys():
        t1 = time()
        rf_classifier = RandomForestClassifier(labelCol=labels_col, featuresCol='rf_features', seed=seed,
                                               numTrees=500, maxDepth=20, featureSubsetStrategy="sqrt")
        
        rf_pipeline = Pipeline(stages=[labels2_indexer, rf_slicer, rf_classifier])
        cluster_models[cluster] = rf_pipeline.fit(df.filter(col('cluster') == cluster))
        print("Finished %g cluster in %g s" % (cluster, time() - t1))
        
    return cluster_models


# In[43]:


# This utility function helps to get predictions/probabilities for the new data and return them into one dataframe
def getProbabilities(df, probCol, cluster_mapping, cluster_models):
    pred_df = (sqlContext.createDataFrame([], StructType([
                    StructField('id', LongType(), False),
                    StructField(probCol, DoubleType(), False)])))
    
    udf_map = udf(lambda cluster: cluster_mapping[cluster], DoubleType())
    pred_df = pred_df.union(df.filter(col('cluster').isin(list(cluster_mapping.keys())))
                            .withColumn(probCol, udf_map(col('cluster')))
                            .select('id', probCol))

                                       
    for k in cluster_models.keys():
        maj_label = cluster_models[k].stages[0].labels[0]
        udf_remap_prob = udf(lambda row: float(row[0]) if (maj_label == 'attack') else float(row[1]), DoubleType())

        pred_df = pred_df.union(cluster_models[k]
                         .transform(df.filter(col('cluster') == k))
                         .withColumn(probCol, udf_remap_prob(col('probability')))
                         .select('id', probCol))

    return pred_df


# In[44]:


# Training Random Forest classifiers for each of the clusters
t0 = time()
kmeans_cluster_models = getClusterModels(kmeans_train_df, kmeans_cluster_rf)
print(f"Time: {time() - t0:.2f}s")


# In[45]:


# Getting probabilities for CV data
t0 = time()
res_cv_df = (res_cv_df.drop(kmeans_prob_col)
             .join(getProbabilities(kmeans_cv_df, kmeans_prob_col, kmeans_cluster_mapping, kmeans_cluster_models), 'id')
             .cache())

print(res_cv_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[46]:


# Getting probabilities for Test data
t0 = time()
res_test_df = (res_test_df.drop(kmeans_prob_col)
               .join(getProbabilities(kmeans_test_df, kmeans_prob_col, kmeans_cluster_mapping, kmeans_cluster_models), 'id')
               .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# As CV data is from the same distribution as the train data it isn't needed to adjust threshold.

# In[47]:


printReport(res_cv_df, kmeans_prob_col, e=0.5, labels=labels2)


# Because test data is from the different distribution and it is expected to face unseen attack types, it makes sence to adjust a probability threshold to something like 0.01 for attack connections (0.99 for normal connections). For this approach it gives around ~98-99% Detection Rate with around ~14-15% of False Alarm Rate.

# In[48]:


printReport(res_test_df, kmeans_prob_col, e=0.01, labels=labels2)


# In[49]:


# Adding prediction columns based on chosen thresholds into result dataframes
t0 = time()
res_cv_df = res_cv_df.withColumn(kmeans_pred_col, getPrediction(0.5)(col(kmeans_prob_col))).cache()
res_test_df = res_test_df.withColumn(kmeans_pred_col, getPrediction(0.01)(col(kmeans_prob_col))).cache()

print(res_cv_df.count())
print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# ## 9. Gaussian Mixture clustering with Random Forest Classifiers

# The idea of this approach is to clusterize data into clusters via Gaussian Mixture and then train different Random Forest classifiers for each of the clusters. Gaussian Mixture produces a diffirent clustering than KMeans, so results from both approaches could be combine for improving performance. As Gaussian Mixture clustering doesn't work well on high-demensional data PCA algorithm is used for preprocessing.

# In[50]:


gm_prob_col = 'gm_rf_prob'
gm_pred_col = 'gm_rf_pred'

prob_cols.append(gm_prob_col)
pred_cols.append(gm_pred_col)


# In[51]:


# Gaussian Mixture clustering
from pyspark.ml.clustering import GaussianMixture

t0 = time()
gm = GaussianMixture(k=8, maxIter=150, seed=seed, featuresCol="pca_features", 
                     predictionCol="cluster", probabilityCol="gm_prob")

gm_pipeline = Pipeline(stages=[pca_slicer, pca, gm])
gm_model = gm_pipeline.fit(scaled_train_df)

gm_train_df = gm_model.transform(scaled_train_df).cache()
gm_cv_df = gm_model.transform(scaled_cv_df).cache()
gm_test_df = gm_model.transform(scaled_test_df).cache()

gm_params = (gm_model.stages[2].gaussiansDF.rdd
                  .map(lambda row: [row['mean'].toArray(), row['cov'].toArray()])
                  .collect())
gm_weights = gm_model.stages[2].weights

print(gm_train_df.count())
print(gm_cv_df.count())
print(gm_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[52]:


# Description of the contents of the clusters 
gm_crosstab = getClusterCrosstab(gm_train_df).cache()
gm_crosstab.show(n=30)


# In[53]:


# Splitting clusters
gm_cluster_rf, gm_cluster_mapping = splitClusters(gm_crosstab)

print(len(gm_cluster_rf), len(gm_cluster_mapping))
print(gm_cluster_mapping)
gm_cluster_rf


# In[54]:


# Training Random Forest classifiers for each of the clusters
t0 = time()
gm_cluster_models = getClusterModels(gm_train_df, gm_cluster_rf)
print(f"Time: {time() - t0:.2f}s")


# In[55]:


# Getting probabilities for CV data
t0 = time()
res_cv_df = (res_cv_df.drop(gm_prob_col)
             .join(getProbabilities(gm_cv_df, gm_prob_col, gm_cluster_mapping, gm_cluster_models), 'id')
             .cache())

print(res_cv_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[56]:


# Getting probabilities for Test data
t0 = time()
res_test_df = (res_test_df.drop(gm_prob_col)
               .join(getProbabilities(gm_test_df, gm_prob_col, gm_cluster_mapping, gm_cluster_models), 'id')
               .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[57]:


printReport(res_cv_df, gm_prob_col, e=0.5, labels=labels2)


# In[58]:


printReport(res_test_df, gm_prob_col, e=0.01, labels=labels2)


# In[59]:


# Adding prediction columns based on chosen thresholds into result dataframes
t0 = time()
res_cv_df = res_cv_df.withColumn(gm_pred_col, getPrediction(0.5)(col(gm_prob_col))).cache()
res_test_df = res_test_df.withColumn(gm_pred_col, getPrediction(0.01)(col(gm_prob_col))).cache()

print(res_cv_df.count())
print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# ## 10. Supervised approach for dettecting each type of attacks separately

# The idea of the following approach is training Random Forest Classifiers for each of four major 'attack' categories separately.

# ### 10.1 DoS and normal

# In[60]:


dos_prob_col = 'dos_prob'
dos_pred_col = 'dos_pred'

prob_cols.append(dos_prob_col)
pred_cols.append(dos_pred_col)


# In[61]:


dos_exp = (col('labels5') == 'DoS') | (col('labels5') == 'normal')
dos_train_df = (scaled_train_df.filter(dos_exp).cache())

print(dos_train_df.count())
(dos_train_df
     .groupby('labels5')
     .count()
     .sort(sql.desc('count'))
     .show())


# Commented code below is related to undersampling 'normal' connections. It could give better results. However, it hasen't been tested a lot yet.

# In[62]:


# dos_train_df = dos_train_df.sampleBy('labels5', fractions={'normal': 45927./67343, 'DoS': 1.0}).cache()

# print(dos_train_df.count())
# (dos_train_df
#      .groupby('labels5')
#      .count()
#      .sort(sql.desc('count'))
#      .show())


# Diffirent AR feature selection is used as only normal and DoS connections are treated. Note that train dataframe without standartization is used for getting Attribute Ratio dictionary.

# In[63]:


t0 = time()
dos_ar_dict = getAttributeRatio(train_df.filter(dos_exp), numeric_cols, binary_cols, 'labels5')

print(f"Time: {time() - t0:.2f}s")
dos_ar_dict


# In[64]:


t0 = time()
dos_slicer = VectorSlicer(inputCol="indexed_features", outputCol="features", 
                          names=selectFeaturesByAR(dos_ar_dict, 0.05))

dos_rf = RandomForestClassifier(labelCol=labels_col, featuresCol='features', featureSubsetStrategy='sqrt',
                                numTrees=500, maxDepth=20, seed=seed)

dos_rf_pipeline = Pipeline(stages=[dos_slicer, dos_rf])
dos_rf_model = dos_rf_pipeline.fit(dos_train_df)

dos_cv_df = dos_rf_model.transform(scaled_cv_df).cache()
dos_test_df = dos_rf_model.transform(scaled_test_df).cache()
print(dos_cv_df.count())
print(dos_test_df.count())

print(f"Time: {time() - t0:.2f}s")


# In[65]:


# Getting probabilities for CV data
t0 = time()
res_cv_df = (res_cv_df.drop(dos_prob_col)
             .join(dos_cv_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', dos_prob_col]),
                    'id')
                    .cache())

print(res_cv_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[66]:


# Getting probabilities for Test data
t0 = time()
res_test_df = (res_test_df.drop(dos_prob_col)
               .join(dos_test_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', dos_prob_col]),
                    'id')
                    .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# The first report shows performance of classification for 'normal' and 'DoS' labels, the second report shows performance for the whole data with adjusted threshold.

# In[67]:


printReport(res_cv_df.filter(dos_exp), probCol=dos_prob_col, e=0.5, labels=['normal', 'DoS'])
printReport(res_cv_df, probCol=dos_prob_col, e=0.05)


# In[68]:


printReport(res_test_df.filter(dos_exp), probCol=dos_prob_col, e=0.5, labels=['normal', 'DoS'])
printReport(res_test_df, probCol=dos_prob_col, e=0.01)


# In[69]:


# Adding prediction columns based on chosen thresholds into result dataframes
t0 = time()
res_cv_df = res_cv_df.withColumn(dos_pred_col, getPrediction(0.05)(col(dos_prob_col))).cache()
res_test_df = res_test_df.withColumn(dos_pred_col, getPrediction(0.01)(col(dos_prob_col))).cache()

print(res_cv_df.count())
print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# ### 10.2 Probe and normal

# In[70]:


probe_prob_col = 'probe_prob'
probe_pred_col = 'probe_pred'

prob_cols.append(probe_prob_col)
pred_cols.append(probe_pred_col)


# In[71]:


probe_exp = (col('labels5') == 'Probe') | (col('labels5') == 'normal')
probe_train_df = (scaled_train_df.filter(probe_exp).cache())

print(probe_train_df.count())
(probe_train_df
     .groupby('labels5')
     .count()
     .sort(sql.desc('count'))
     .show())


# Commented code below is related to undersampling 'normal' connections. It could give better results. However, it hasen't been tested a lot yet.

# In[72]:


# probe_train_df = probe_train_df.sampleBy('labels5', fractions={'normal': 9274./53789, 'Probe': 1.0}).cache()

# print(probe_train_df.count())
# (probe_train_df
#      .groupby('labels5')
#      .count()
#      .sort(sql.desc('count'))
#      .show())


# Diffirent AR feature selection is used as only normal and Probe connections are treated. Note that train dataframe without standartization is used for getting Attribute Ratio dictionary.

# In[73]:


t0 = time()
probe_ar_dict = getAttributeRatio(train_df.filter(probe_exp), numeric_cols, binary_cols, 'labels5')

print(f"Time: {time() - t0:.2f}s")
probe_ar_dict


# In[74]:


t0 = time()
probe_slicer = VectorSlicer(inputCol="indexed_features", outputCol="features",
                            names=selectFeaturesByAR(probe_ar_dict, 0.05))

probe_rf = RandomForestClassifier(labelCol=labels_col, featuresCol='features', featureSubsetStrategy='sqrt',
                                  numTrees=500, maxDepth=20, seed=seed)
probe_rf_pipeline = Pipeline(stages=[probe_slicer, probe_rf])

probe_rf_model = probe_rf_pipeline.fit(probe_train_df)

probe_cv_df = probe_rf_model.transform(scaled_cv_df).cache()
probe_test_df = probe_rf_model.transform(scaled_test_df).cache()

print(probe_cv_df.count())
print(probe_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[75]:


# Getting probabilities for CV data
t0 = time()
res_cv_df = (res_cv_df.drop(probe_prob_col)
             .join(probe_cv_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', probe_prob_col]), 'id')
                    .cache())

print(res_cv_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[76]:


# Getting probabilities for Test data
t0 = time()
res_test_df = (res_test_df.drop(probe_prob_col)
               .join(probe_test_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', probe_prob_col]), 'id')
                    .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# The first report shows performance of classification for 'normal' and 'Probe' labels, the second report shows performance for the whole data with adjusted threshold.

# In[77]:


printReport(res_cv_df.filter(probe_exp), probCol=probe_prob_col, e=0.5, labels=['normal', 'Probe'])
printReport(res_cv_df, probCol=probe_prob_col, e=0.05)


# In[78]:


printReport(res_test_df.filter(probe_exp), probCol=probe_prob_col, e=0.5, labels=['normal', 'Probe'])
printReport(res_test_df, probCol=probe_prob_col, e=0.01)


# In[79]:


# Adding prediction columns based on chosen thresholds into result dataframes
t0 = time()
res_cv_df = res_cv_df.withColumn(probe_pred_col, getPrediction(0.05)(col(probe_prob_col))).cache()
res_test_df = res_test_df.withColumn(probe_pred_col, getPrediction(0.01)(col(probe_prob_col))).cache()

print(res_cv_df.count())
print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# ### 10.3 R2L, U2R and normal types

# As there are a few examples of both R2L and U2R attack types and they have similar behaviour, they are combined into one group.

# In[80]:


r2l_u2r_prob_col = 'r2l_u2r_prob'
r2l_u2r_pred_col = 'r2l_u2r_pred'

prob_cols.append(r2l_u2r_prob_col)
pred_cols.append(r2l_u2r_pred_col)


# In[81]:


r2l_u2r_exp = (col('labels5') == 'R2L') | (col('labels5') == 'U2R') | (col('labels5') == 'normal')
r2l_u2r_train_df = (scaled_train_df.filter(r2l_u2r_exp).cache())

print(r2l_u2r_train_df.count())
(r2l_u2r_train_df
     .groupby('labels5')
     .count()
     .sort(sql.desc('count'))
     .show())


# Diffirent AR feature selection is used as only normal, R2L and U2R connections are treated. Note that train dataframe without standartization is used for getting Attribute Ratio dictionary.

# In[82]:


t0 = time()
r2l_u2r_ar_dict = getAttributeRatio(train_df.filter(r2l_u2r_exp), numeric_cols, binary_cols, 'labels5')

print(f"Time: {time() - t0:.2f}s")
r2l_u2r_ar_dict


# In[83]:


t0 = time()
r2l_u2r_slicer = VectorSlicer(inputCol="indexed_features", outputCol="features",
                              names=selectFeaturesByAR(r2l_u2r_ar_dict, 0.05))

r2l_u2r_rf = RandomForestClassifier(labelCol=labels_col, featuresCol='features', featureSubsetStrategy='sqrt',
                                    numTrees=500, maxDepth=20, seed=seed)
r2l_u2r_rf_pipeline = Pipeline(stages=[r2l_u2r_slicer, r2l_u2r_rf])

r2l_u2r_rf_model = r2l_u2r_rf_pipeline.fit(r2l_u2r_train_df)

r2l_u2r_cv_df = r2l_u2r_rf_model.transform(scaled_cv_df).cache()
r2l_u2r_test_df = r2l_u2r_rf_model.transform(scaled_test_df).cache()
print(r2l_u2r_cv_df.count())
print(r2l_u2r_test_df.count())

print(f"Time: {time() - t0:.2f}s")


# In[84]:


# Getting probabilities for CV data
t0 = time()
res_cv_df = (res_cv_df.drop(r2l_u2r_prob_col)
             .join(r2l_u2r_cv_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', r2l_u2r_prob_col]), 'id')
                    .cache())

print(res_cv_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[85]:


# Getting probabilities for Test data
t0 = time()
res_test_df = (res_test_df.drop(r2l_u2r_prob_col)
               .join(r2l_u2r_test_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', r2l_u2r_prob_col]), 'id')
                    .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# The first report shows performance of classification for 'normal' and 'R2L&U2R' labels, the second report shows performance for the whole data with adjusted threshold.

# In[86]:


printReport(res_cv_df.filter(r2l_u2r_exp), probCol=r2l_u2r_prob_col, e=0.5, labels=['normal', 'R2L&U2R'])
printReport(res_cv_df, probCol=r2l_u2r_prob_col, e=0.05, labels=labels2)


# In[87]:


printReport(res_test_df.filter(r2l_u2r_exp), probCol=r2l_u2r_prob_col, e=0.5, labels=['normal', 'R2L&U2R'])
printReport(res_test_df, probCol=r2l_u2r_prob_col, e=0.01, labels=labels2)


# In[88]:


# Adding prediction columns based on chosen thresholds into result dataframes
t0 = time()
res_cv_df = res_cv_df.withColumn(r2l_u2r_pred_col, getPrediction(0.05)(col(r2l_u2r_prob_col))).cache()
res_test_df = res_test_df.withColumn(r2l_u2r_pred_col, getPrediction(0.01)(col(r2l_u2r_prob_col))).cache()

print(res_cv_df.count())
print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# ### 10.4 Combining results

# In[89]:


sup_prob_col = 'sup_prob'
sup_pred_col = 'sup_pred'

prob_cols.append(sup_prob_col)
pred_cols.append(sup_pred_col)


# In[90]:


res_cv_df = res_cv_df.withColumn(sup_prob_col, 
                                 (col(dos_prob_col) + col(probe_prob_col) + col(r2l_u2r_prob_col))/3).cache()

printReport(res_cv_df, sup_prob_col, e=0.05, labels=labels2)


# In[91]:


res_cv_df = res_cv_df.withColumn(sup_pred_col, col(dos_pred_col).cast('int')
                                        .bitwiseOR(col(probe_pred_col).cast('int'))
                                        .bitwiseOR(col(r2l_u2r_pred_col).cast('int'))).cache()

printReport(res_cv_df, sup_pred_col, labels=labels2)


# In[92]:


res_test_df = res_test_df.withColumn(sup_prob_col, 
                                 (col(dos_prob_col) + col(probe_prob_col) + col(r2l_u2r_prob_col))/3).cache()

printReport(res_test_df, sup_prob_col, e=0.005, labels=labels2)


# In[93]:


res_test_df = res_test_df.withColumn(sup_pred_col, col(dos_pred_col).cast('int')
                                            .bitwiseOR(col(probe_pred_col).cast('int'))
                                            .bitwiseOR(col(r2l_u2r_pred_col).cast('int'))).cache()

printReport(res_test_df, sup_pred_col, labels=labels2)


# ## 11. Ensembling experiments

# Here are some experiments with ensembling and stacking results from different approaches.

# ### 11.1 Linear combination of all models

# In[94]:


# Printing report of the best single model for comparison
printReport(res_test_df, kmeans_pred_col)


# In[95]:


# Linear combination of all models 
printReport(res_test_df
            .select('labels2_index', ((3 * col(kmeans_prob_col) \
                                        + col(gm_prob_col) \
                                        + col(dos_prob_col) \
                                        + col(probe_prob_col) \
                                        + col(r2l_u2r_prob_col))/7)
                    .alias('voting')), 
            'voting', e=0.005, labels=labels2)


# In[96]:


printReport(res_test_df
            .select('labels2_index', ((2 * col(kmeans_prob_col) \
                                        + col(gm_prob_col) \
                                        + col(sup_prob_col))/4)
                    .alias('voting')), 
            'voting', e=0.005, labels=labels2)


# In[97]:


printReport(res_test_df
            .select('labels2_index', (col(kmeans_pred_col).cast('int')
                                      .bitwiseOR(col(gm_pred_col).cast('int'))
                                      .bitwiseOR(col(sup_pred_col).cast('int')))
                    .alias('voting')), 
                    'voting', labels=labels2)


# ### 11.2 Logistic Regression and Random Forest Classifier 

# In[98]:


from pyspark.ml.classification import LogisticRegression

t0 = time()
lr_assembler = VectorAssembler(inputCols=[
                            kmeans_prob_col, 
                            gm_prob_col, 
                            dos_prob_col, 
                            probe_prob_col, 
                            r2l_u2r_prob_col
                            ], 
                            outputCol="features")

lr = LogisticRegression(maxIter=100, labelCol="labels2_index", standardization=False, weightCol='weights')
lr_pipeline = Pipeline(stages=[lr_assembler, lr])

weights_dict = {
    'normal': 1.0,
    'DoS': 100.0,
    'Probe': 100.0,
    'R2L': 100.0,
    'U2R': 100.0
}

udf_weight = udf(lambda row: weights_dict[row], DoubleType())
lr_model = lr_pipeline.fit(res_cv_df.withColumn('weights', udf_weight('labels5')))
lr_test_df = lr_model.transform(res_test_df).cache()

res_test_df = (res_test_df.drop('lr_prob')
                    .join(lr_test_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', 'lr_prob']), 'id')
                    .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[99]:


printReport(res_test_df, 'lr_prob', e=0.01, labels=labels2)


# In[100]:


t0 = time()
rf_assembler = VectorAssembler(inputCols=[
                            kmeans_pred_col, 
                            gm_pred_col, 
                            dos_pred_col, 
                            probe_pred_col, 
                            r2l_u2r_pred_col
                            ],
                            outputCol='features')

rf_indexer =  VectorIndexer(inputCol='features', outputCol='indexed_features', maxCategories=2)

rf = RandomForestClassifier(labelCol='labels2_index', featuresCol='features', seed=seed,
                            numTrees=250, maxDepth=5, featureSubsetStrategy='auto')
rf_pipeline = Pipeline(stages=[rf_assembler, 
                               rf_indexer,
                               rf])
rf_model = rf_pipeline.fit(res_cv_df)
rf_test_df = rf_model.transform(res_test_df).cache()

res_test_df = (res_test_df.drop('rf_prob')
                    .join(rf_test_df.rdd
                    .map(lambda row: (row['id'], float(row['probability'][1])))
                    .toDF(['id', 'rf_prob']), 'id')
                    .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[101]:


printReport(res_test_df, 'rf_prob', e=0.01, labels=labels2)


# In[102]:


# Adding prediction columns based on chosen thresholds into result dataframes
t0 = time()
res_test_df = res_test_df.withColumn('lr_pred', getPrediction(0.01)(col('lr_prob'))).cache()
res_test_df = res_test_df.withColumn('rf_pred', getPrediction(0.01)(col('rf_prob'))).cache()

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[103]:


printReport(res_test_df
            .select('labels2_index', ((col('lr_prob') + col('rf_prob'))/2)
                    .alias('voting')), 
                    'voting', e=0.01, labels=labels2)


# In[104]:


printReport(res_test_df
            .select('labels2_index', (col('lr_pred').cast('int').bitwiseOR(col('rf_pred').cast('int')))
                    .alias('voting')), 
                    'voting', labels=labels2)


# ### 11.3 Stacking with Random Forest Classifier

# In[105]:


stack_cv_df = scaled_cv_df.join(res_cv_df.select('id', *[
                            kmeans_pred_col, 
                            gm_pred_col, 
                            dos_pred_col, 
                            probe_pred_col, 
                            r2l_u2r_pred_col,
                            sup_pred_col
                            ]), 'id').cache()

stack_test_df = scaled_test_df.join(res_test_df.select('id', *[
                            kmeans_pred_col, 
                            gm_pred_col, 
                            dos_pred_col, 
                            probe_pred_col, 
                            r2l_u2r_pred_col,
                            sup_pred_col
                            ]), 'id').cache()

print(stack_cv_df.count())
print(stack_test_df.count())


# In[106]:


t0 = time()
pred_assembler = VectorAssembler(inputCols=[
                            kmeans_pred_col, 
                            gm_pred_col, 
                            dos_pred_col, 
                            probe_pred_col, 
                            r2l_u2r_pred_col,
                            sup_pred_col
                            ], outputCol='pred_features')
pred_indexer = VectorIndexer(inputCol='pred_features', outputCol='indexed_pred_features', maxCategories=2)

rf_stack_slicer = VectorSlicer(inputCol='indexed_features', outputCol='selected_features', 
                               names=selectFeaturesByAR(ar_dict, 1.5))

rf_stack_assembler = VectorAssembler(inputCols=['selected_features', 'indexed_pred_features'], outputCol='rf_features')

rf_stack_classifier = RandomForestClassifier(labelCol=labels_col, featuresCol='rf_features', seed=seed,
                                             numTrees=500, maxDepth=20, featureSubsetStrategy="auto")

stack_pipeline = Pipeline(stages=[pred_assembler, 
                                  pred_indexer, 
                                  rf_stack_slicer, 
                                  rf_stack_assembler,
                                  rf_stack_classifier
                                 ])
stack_model = stack_pipeline.fit(stack_cv_df)

pred_stack_cv_df = stack_model.transform(stack_cv_df).cache()
pred_stack_test_df = stack_model.transform(stack_test_df).cache()
print(pred_stack_cv_df.count())
print(pred_stack_test_df.count())

print(f"Time: {time() - t0:.2f}s")


# In[107]:


t0 = time()
res_cv_df = res_cv_df.drop('prob_stack_rf')
res_cv_df = (res_cv_df.join(pred_stack_cv_df.rdd
                            .map(lambda row: (row['id'], float(row['probability'][1])))
                            .toDF(['id', 'prob_stack_rf']),
                            'id')
                        .cache())

print(res_cv_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[108]:


t0 = time()
res_test_df = res_test_df.drop('prob_stack_rf')
res_test_df = (res_test_df.join(pred_stack_test_df.rdd
                            .map(lambda row: (row['id'], float(row['probability'][1])))
                            .toDF(['id', 'prob_stack_rf']),
                            'id')
                        .cache())

print(res_test_df.count())
print(f"Time: {time() - t0:.2f}s")


# In[109]:


printReport(res_test_df, 'prob_stack_rf', e=0.01, labels=labels2)


# In[111]:


print(f"Time: {time() - gt0:.2f}s")


# ## 12. Results summary

# The best result from a single approach was achieved by KMeans Clustering with Random Forest Classifiers. It gives 
# around ~98-99% of detection rate with reasonable ~14-15% of false alarm rate. F1 score is 0.94, weighted F1 score is 0.93.
# 
# For improving detection rate ensembling approaches are used. The best of them gives ~99.5-99.6% of detection rate with ~16.1-16.6% of false alarm rate. So there are only about 40-90 attack connections from 12833 (including unknown before) which haven't been recognized.
