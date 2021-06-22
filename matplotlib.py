#!/usr/bin/env python
# coding: utf-8

# In[8]:


import codecs
import csv
import boto3

client = boto3.client("s3")

data = client.get_object(Bucket='686bucket', Key='airline data/airline-safety.csv')

#make a new dictionary
airline = {}


##the key is going to be the airline name and the value is the sum of incidents_85_99 + fatal_accidents_85_99
for row in csv.DictReader(codecs.getreader("utf-8")(data["Body"])):
    #print(row)
    airline_name = row['airline']
    airline[airline_name] = int(row['incidents_85_99']) + int(row['fatal_accidents_85_99'])

print(airline)


# In[ ]:


print(airline)


# In[10]:


lsSorted = sorted(airline.items(), key = lambda x: x[1], reverse = True)


# In[11]:


print(lsSorted)


# In[13]:


dPlotly={'airline':[],'number problems':[]}
for item in lsSorted:
    name = item[0]
    num = item[1]
    dPlotly['airline'].append(name)
    dPlotly['number problems'].append(num)
print(dPlotly)


# In[14]:


import plotly.express as px
fig = px.bar(dPlotly, x='airline', y='number problems')
fig.show()


# In[ ]:




