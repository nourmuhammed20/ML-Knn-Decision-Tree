#!/usr/bin/env python
# coding: utf-8

# # Decision Trees and K-nn (Assignment-2)

# ## Team Name :
# 1. Nour Muhammed                     20200605
# 2. Ahmed Mohamed Eid                 20200042
# 3. Mohamed Samy Atwa                 20200446
# 4. Beshoy Ashraf                     20200118
# 5. John Fady                         20200133

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ## Part 1 Decision Trees

# ## Part 2 K-nn

# In[2]:


# Loading File to DataFrame
df2=pd.read_csv("diabetes.csv")
df2.head()


# In[3]:


df2.describe()


# In[4]:


missing_values = df2.isnull().sum()
print(missing_values)


# In[5]:


# Separate features and targets
X = df2.drop(['Outcome'], axis=1)
y = df2['Outcome']  # Use double square brackets to select multiple columns

# Shuffle and split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[6]:


# Helper Function
def min_max_scaling(data):
    """    
    Note:
    data: numpy array or pandas DataFrame for the numerical data to be scaled.
    """
    min_val = data.min()
    max_val = data.max()
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


def euclidean_distance(point1, point2):
    #Note points are array, coordinates 

    point1 = np.array(point1)
    point2 = np.array(point2)
    distance = np.sqrt(np.sum((point1 - point2)**2))
    return distance

