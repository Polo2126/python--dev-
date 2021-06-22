#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pydot')
get_ipython().system('pip install graphviz')
import pandas as pd
import numpy as np
import seaborn as sns
import pydot
import matplotlib.pyplot as plt
from pandas import DataFrame

df = pd.read_csv('/Users/pixie/Downloads/creditcard.csv')
df.head()


# In[2]:


df.describe()


# In[3]:


print(df.shape)


# In[4]:


amount = [df['Amount'].values]
sns.distplot(amount)


# In[5]:


time = df['Time'].values
sns.distplot(time)


# In[6]:


from matplotlib import gridspec

# distribution of anomalous features
features = df.iloc[:,0:28].columns

plt.figure(figsize=(12,28*4))
gs = gridspec.GridSpec(28, 1)
for i, c in enumerate(df[features]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[c][df.Class == 1], bins=50)
    sns.distplot(df[c][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(c))
plt.show()


# In[7]:


# Plot histograms of each parameter 
df.hist(figsize = (20, 20))
plt.show()


# In[8]:


fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]


# In[9]:


#describe fraud
print(fraud.Amount.describe())


# In[10]:


#describe normal
print(valid.Amount.describe())


# In[11]:


#correlation to understand which features are relevant for prediction
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
corrmat = df.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# #observation:
# v2 and v5 have high negative correlation with Amount
# slight correlation of v20 and amount
# 
# 

# In[12]:


#use of stratify 
#For example, if variable y is a binary categorical variable 
#with values 0 and 1 and there are 25% of zeros and 75% of ones,
#stratify=y will make sure that your random split has 25% of 0's and 
#75% of 1's


# In[13]:


from sklearn.model_selection import train_test_split 
x = df.drop("Class", axis = 1)
y = df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42, stratify = y)


# In[14]:


y_train.value_counts()


# In[15]:


y_test.value_counts()


# In[16]:


len(df[df['Class']==0])


# In[17]:


len(df[df['Class']==1])


# In[18]:


from sklearn.ensemble import RandomForestClassifier
# random forest model creation
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
# predictions
y_pred = rfc.predict(x_test)


# In[19]:


# Evaluating the classifier
# printing every score of the classifier
# scoring in anything
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
  
n_outliers = len(fraud)
n_errors = (y_pred != y_test).sum()
print("The model used is Random Forest classifier")
  
acc = accuracy_score(y_test, y_pred)
print("The accuracy is {}".format(acc))
  
prec = precision_score(y_test, y_pred)
print("The precision is {}".format(prec))
  
rec = recall_score(y_test, y_pred)
print("The recall is {}".format(rec))
  
f1 = f1_score(y_test, y_pred)
print("The F1-Score is {}".format(f1))
  
MCC = matthews_corrcoef(y_test, y_pred)
print("The Matthews correlation coefficient is{}".format(MCC))


# In[20]:


# printing the confusion matrix
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize =(12, 12))
sns.heatmap(conf_matrix, xticklabels = LABELS, 
            yticklabels = LABELS, annot = True, fmt ="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


# In[21]:


# Run classification metrics
plt.figure(figsize=(9, 7))
print('{}: {}'.format("Random Forest", n_errors))
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# As you can see with our Random Forest Model we are getting a better result even for the recall which is the most tricky part.
# 
# 

# In[ ]:




