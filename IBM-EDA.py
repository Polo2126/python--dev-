#!/usr/bin/env python
# coding: utf-8

# # IBM Exploratory Analysis for Employees Data

# # Part 1 - Pandas Dataframe

# In[1]:


import pandas as pd

path = 's3://686-assignment1/IBM-Employee/IBM_Employee.csv'
dataframe = pd.read_csv(path)
dataframe.head()


# In[2]:


dataframe.describe()


# In[3]:


dupl = dataframe.duplicated()
dupl.sum()


# In[4]:


dataframe.drop_duplicates(inplace = True)
dpl = dataframe.duplicated()
dpl.sum()


# In[5]:


dataframe.drop_duplicates(inplace = True)
dpl = dataframe.duplicated()
dpl.sum()


# # Part 2 - Pyspark Dataframe

# In[6]:


dataframe.dtypes


# In[7]:


import pyspark
from pyspark import SparkContext
from pyspark import SQLContext
sc = SparkContext('local','IBMSparkApp')
sqlContext = SQLContext(sc)


# In[8]:


from pyspark.sql.types import *
IBMSchema = StructType([ StructField("Age", FloatType(), True)                       ,StructField("Attrition", StringType(), True)                       ,StructField("DailyRate", IntegerType(), True)                       ,StructField("Department", StringType(), True)                       ,StructField("DistanceFromHome", FloatType(), True)                       ,StructField("Education", FloatType(), True)                       ,StructField("EducationField", StringType(), True)                       ,StructField("EmployeeCount", IntegerType(), True)                       ,StructField("EmployeeNumber", IntegerType(), True)])
pyDataframe = sqlContext.createDataFrame(dataframe,schema=IBMSchema)


# In[9]:


pyDataframe.printSchema()


# In[10]:


pyDataframe.show(5)


# In[11]:


from pyspark.sql.functions import isnan, when, count, col
pyDataframe.select([count(when(isnan(c), c)).alias(c) for c in pyDataframe.columns]).show()


# In[12]:


from pyspark.sql.functions import col, isnan, when, trim
def to_null(c):
    return when(~(col(c).isNull() | isnan(col(c)) | (trim(col(c)) == "")), col(c))
cleaned_pydf = pyDataframe.select([to_null(c).alias(c) for c in pyDataframe.columns]).na.drop()
cleaned_pydf.show()


# In[13]:


cleaned_pydf.count()


# # The min, max and median value of Age column is 18, 60, and 36 respectively.

# # The binning Range should be from 15 to 65. Thus, dividing the gap between the range of values 50/4 = 12.5. Therefore, 15-27.5 belongs to group 1, 27.5-40 belongs to group 2, 40 - 52.5 belongs to group 3 and 52.5 - 65 belongs to group 4.

# In[14]:


import numpy as np
splits = [float(i) for i in np.arange(15,65,12.5)]
splits


# In[15]:


expr = """CASE 
          WHEN Age BETWEEN 14.9 AND 27.49 THEN 1 
          WHEN Age BETWEEN 27.5 AND 39.99 THEN 2
          WHEN Age BETWEEN 40.0 AND 52.49 THEN 3
          WHEN Age BETWEEN 52.5 AND 64.49 THEN 4
          END AS Age_Bins"""
cleaned_pydf = cleaned_pydf.selectExpr("*",expr)
cleaned_pydf.show()


# In[16]:





# In[ ]:




