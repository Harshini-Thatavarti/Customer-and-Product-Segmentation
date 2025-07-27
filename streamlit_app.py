#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('C:/Users/harsh/Final Year Project/joblib')
import joblib
import streamlit as st
import pandas as pd


# In[2]:


# Load the online retail dataset
data = pd.read_excel("online_retail_II.xlsx")


# In[3]:


# Load the pre-trained clustering model
model = joblib.load("clustering_model.pkl")


# In[4]:


# Select the features used for clustering
features = ["Quantity", "UnitPrice", "CustomerID"]


# In[5]:



# Define a function to predict the cluster for a given row
def predict_cluster(row):
    X = row[features].values.reshape(1, -1)
    cluster = model.predict(X)[0]
    return cluster


# In[6]:


# Define the Streamlit app
def app():
    # Add a file uploader to select a row from the dataset
    file = st.file_uploader("Upload a CSV file", type="csv")
    if file is not None:
        data = pd.read_csv(file)

        # Add a dropdown menu to select a row from the dataset
        selected_row = st.selectbox("Select a row", data.index)

        # Get the selected row and predict its cluster
        row = data.loc[selected_row]
        cluster = predict_cluster(row)

        # Display the predicted cluster
        st.write(f"The selected row belongs to cluster {cluster}")


# In[7]:


import os


# In[8]:


print(os.getcwd())


# In[ ]:
