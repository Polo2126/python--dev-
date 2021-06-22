#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install plotly
#read a csv file
import pandas as pd
import numpy as np
import plotly.express as px


# In[4]:


path = 's3://686-balaj2p/Car_sales.csv'


# In[5]:


df = pd.read_csv(path)


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


dimension = df.shape
rows = dimension[0]
cols = dimension[1]
print("total rows",rows)
print("total columns", cols)


# In[11]:


#scatter plot 
#extra feature: used update layout removed the axis lines and added title
fuel_Eff = df['Fuel_efficiency']
power = df['Horsepower']

fig = px.scatter(x= fuel_Eff, y = power, title = 'Fuel and horsepower co-relation')
fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)',})

fig.show()


# In[12]:


#line plot
#extra feature added: used update layout and removed axis, added title
fig = px.line(df, x = fuel_Eff, y = power, title = 'line plot')

fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)', 'paper_bgcolor': 'rgba(0,0,0,0)',})
fig.show()


# In[13]:


#pie chart is so varied as there are too many categories
#extra feature: added bgcolor
fig = px.pie(df, values = 'Fuel_efficiency', names = 'Horsepower', title = 'pie chart')
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue")


fig.show() 


# In[14]:


#bar chart 
#extra feature added: added stack in barmode
fig = px.bar(df, x = 'Fuel_efficiency', y = 'Horsepower')
fig.update_layout(barmode='stack')
fig.show()


# In[ ]:




