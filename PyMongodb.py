#!/usr/bin/env python
# coding: utf-8

# In[1]:


#install pymongo
#install dnspython
#!pip install pymongo
#!pip install dnspython


# In[2]:


import pymongo


# In[3]:


#connect to our mongo database on our atlas server

myclient = pymongo.MongoClient("mongodb+srv://kaza:PASSWORD@cluster0.db4vu.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")


# In[4]:


#look at database names
print(myclient.list_database_names())


# In[5]:


#creating or accessing a new database
mydb = myclient["mydatabase"]

#make a collection called airline_data 
#remember the collection is loosely equivalent
#to a sql table
mycol = mydb["airline_data"]


# In[6]:


import codecs
import csv
import boto3

client = boto3.client("s3")

data = client.get_object(Bucket='686-assignment1', Key='airline/airline-safety.csv')

#make a new dictionary
airline = {}

#[{'first_name':'Lisa'},{'first_name':'Pravalika'}]
lsData = []
##the key is going to be the airline name and the value is the sum of incidents_85_99 + fatal_accidents_85_99
for row in csv.DictReader(codecs.getreader("utf-8")(data["Body"])):
    #example of casting avail_seat_km_per_week to int before saving back to dictionary
    row['avail_seat_km_per_week'] = int(row['avail_seat_km_per_week'])
    lsData.append(row)


# In[7]:


#insert "documents" aka dictionaries into the airline_data collection
lsIDS = mycol.insert_many(lsData)


# In[8]:


print(lsIDS)


# In[9]:


#just get the first row
x = mycol.find_one()

print(x)


# In[1]:


#do a query on airline
myquery = { "airline": "Delta" }

mydoc = mycol.find(myquery)

for x in mydoc:
  print(x)


# # 2 Statements regarding update of MongoDB Database with Python

# In[11]:


myquery2 = { "airline": "Air Canada" }

mydoc2 = mycol.find(myquery2)

for x in mydoc2:
  print(x)


# In[13]:


mycol.update_many({"airline":"Air Canada"},{"$set":{"airline": "Updated Air Canada"}})
myquery3 = { "airline": "Updated Air Canada" }

mydoc3 = mycol.find(myquery3)

for x in mydoc3:
  print(x)


# In[14]:


myquery4 = { "airline": "Condor" }

mydoc4 = mycol.find(myquery4)

for x in mydoc4:
  print(x)


# In[16]:


mycol.update_many({"airline":"Condor"},{"$set":{"airline": "Updated Condor"}})
myquery5 = { "airline": "Updated Condor" }

mydoc5 = mycol.find(myquery5)

for x in mydoc5:
  print(x)


# # 1 Delete statement regarding MongoDB Database with Python

# In[17]:


myquery6 = { "airline": "Saudi Arabian" }

mydoc6 = mycol.find(myquery6)

for x in mydoc6:
  print(x)


# In[18]:


mycol.remove({"airline":"Saudi Arabian"})
myquery7 = { "airline": "Saudi Arabian" }

mydoc7 = mycol.find(myquery7)

for x in mydoc7:
  print(x)


# # Thus, the update_many() did update on all matching elements of the document, changed the Air Canada airline to Updated Air Canada airline and Condor to Updated Condor.
# # And, also the remove() did delete on all matching elements of the document, removed airlines with matching name as Saudi Arabian. Therefore, 7 Saudi Arabian Documents with airline name have been removed from the database.

# In[ ]:




