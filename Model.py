#!/usr/bin/env python
# coding: utf-8

# # Recommendation System Notebook
# - User based recommendation
# - User based prediction & evaluation
# - Item based recommendation
# - Item based prediction & evaluation

# Different Approaches to develop Recommendation System -
# 
# 1. Demographich based Recommendation System
# 
# 2. Content Based Recommendation System
# 
# 3. Collaborative filtering Recommendation System

# In[1]:


import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import matplotlib.colors as colors
#%matplotlib inline

## Warnings
import warnings
from scipy import stats
warnings.filterwarnings('ignore')


# In[2]:

def prod_predict(user_name):
    # Reading ratings file from GitHub. # MovieLens
    ratings = pd.read_csv('sample30.csv' , encoding='latin-1')
    ratings.head()


    # In[3]:


    #Dropping columns not adding value to analysis
    rv_recom=ratings[['name','reviews_username','reviews_rating','reviews_date']]
    rv_recom = rv_recom.rename(columns={'reviews_username': 'userID', 'name': 'prod_name','reviews_rating': 'rating' })


    # In[4]:


    rv_recom.head()


    # In[5]:


    rv_recom.isnull().sum()


    # In[6]:


    #drop rows having any blank cell
    rv_recom=rv_recom[~rv_recom.userID.isnull()==True]
    rv_recom=rv_recom[~rv_recom.reviews_date.isnull()==True]


    # In[7]:


    rv_recom.shape


    # In[8]:


    counts1=rv_recom['userID'].value_counts() 
    counts=rv_recom['prod_name'].value_counts()


    # In[9]:


    #print(len(counts1[counts1>=2].index),len(counts[counts>=10].index))


    # In[10]:


    #df1=rv_recom[rv_recom['userID'].isin(counts1[counts1 >=2].index)]
    #df1=df1[df1['prod_name'].isin(counts[counts >=10].index)]
    df1=rv_recom
    df1.shape


    # In[11]:


    df1.drop_duplicates()
    df1.shape


    # ## Dividing the dataset into train and test

    # In[12]:


    # Test and Train split of the dataset.
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df1, test_size=0.30, random_state=31)


    # In[13]:


    #print(train.shape)
    #print(test.shape)


    # In[14]:


    # Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
    df_pivot = train.pivot_table(
        index=['userID'],
        columns='prod_name',
        values='rating'
    ).fillna(0)

    df_pivot


    # ### Creating dummy train & dummy test dataset
    # These dataset will be used for prediction 
    # - Dummy train will be used later for prediction of the Products which has not been rated by the user. To ignore the Products rated by the user, we will mark it as 0 during prediction. The Products not rated by user is marked as 1 for prediction in dummy train dataset. 
    # 
    # - Dummy test will be used for evaluation. To evaluate, we will only make prediction on the Products rated by the user. So, this is marked as 1. This is just opposite of dummy_train.

    # In[15]:


    # Copy the train dataset into dummy_train
    dummy_train = train.copy()


    # In[16]:


    # The Product not rated by user is marked as 1 for prediction. 
    dummy_train['rating'] = dummy_train['rating'].apply(lambda x: 0 if x>=1 else 1)


    # In[17]:


    # Convert the dummy train dataset into matrix format.
    dummy_train = dummy_train.pivot_table(
        index=['userID'],
        columns='prod_name',
        values='rating'
    ).fillna(1)


    # In[18]:


    dummy_train.head()


    # **Cosine Similarity**
    # 
    # Cosine Similarity is a measurement that quantifies the similarity between two vectors [Which is Rating Vector in this case] 
    # 
    # **Adjusted Cosine**
    # 
    # Adjusted cosine similarity is a modified version of vector-based similarity where we incorporate the fact that different users have different ratings schemes. In other words, some users might rate items highly in general, and others might give items lower ratings as a preference. To handle this nature from rating given by user , we subtract average ratings for each user from each user's rating for different movies.
    # 
    # 

    # # User Similarity Matrix

    # ## Using Cosine Similarity

    # In[19]:


    from sklearn.metrics.pairwise import pairwise_distances

    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    user_correlation


    # In[20]:


    user_correlation.shape


    # ## Using adjusted Cosine 

    # ### Here, we are not removing the NaN values and calculating the mean only for the products rated by the user

    # In[21]:


    # Create a user-product matrix.
    df_pivot = train.pivot_table(
        index=['userID'],
        columns='prod_name',
        values='rating'
    )


    # In[22]:


    df_pivot.head()


    # ### Normalising the rating of the product for each user around 0 mean

    # In[23]:


    mean = np.nanmean(df_pivot, axis=1)
    df_subtracted = (df_pivot.T-mean).T


    # In[24]:


    df_subtracted.head()


    # ### Finding cosine similarity

    # In[25]:


    from sklearn.metrics.pairwise import pairwise_distances


    # In[26]:


    # Creating the User Similarity Matrix using pairwise_distance function.
    user_correlation = 1 - pairwise_distances(df_subtracted.fillna(0), metric='cosine')
    user_correlation[np.isnan(user_correlation)] = 0
    #print(user_correlation)


    # ## Prediction - User User

    # Doing the prediction for the users which are positively related with other users, and not the users which are negatively related as we are interested in the users which are more similar to the current users. So, ignoring the correlation for values less than 0. 

    # In[27]:


    user_correlation[user_correlation<0]=0
    user_correlation


    # Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the product rating (as present in the rating dataset). 

    # In[28]:


    user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))
    user_predicted_ratings


    # In[29]:


    user_predicted_ratings.shape


    # Since we are interested only in the products not rated by the user, we will ignore the products rated by the user by making it zero. 

    # In[30]:


    user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
    user_final_rating.head()


    # ### Finding the top 5 recommendation for the *user*

    # In[31]:


    # Take the user ID as input.
    #user_input = input("Please Enter the user name for recommendation : ")
    #print(user_input)


    # In[32]:


    user_final_rating.head(2)


    # In[33]:


    d = user_final_rating.loc[user_name].sort_values(ascending=False)[0:5]
    #print("Recommened products are as below: ",list(d.index))
    return list(d.index)

    # In[ ]:




