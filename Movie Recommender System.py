#!/usr/bin/env python
# coding: utf-8

# # Step#0: Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Step#1: Import Dataset

# In[2]:


#Get the Movie Titles
movie_titles = pd.read_csv('movieIdTitles.csv')
movie_titles.head()


# In[3]:


movie_titles.tail(5)


# In[4]:


movies_rating_df = pd.read_csv('dataset.csv', sep = '\t', names = ['user_id','item_id','rating','timestamp'])


# In[5]:


movies_rating_df


# In[6]:


movies_rating_df.drop (['timestamp'], axis = 1, inplace = True)


# In[7]:


movies_rating_df


# In[8]:


movies_rating_df.describe()


# In[9]:


movies_rating_df.info()


# In[10]:


movies_rating_df = pd.merge(movies_rating_df, movie_titles, on = 'item_id')


# In[11]:


movies_rating_df


# In[12]:


movies_rating_df.shape


# # Step#2: Visualize Dataset

# In[13]:


movies_rating_df


# In[14]:


movies_rating_df.groupby('title').describe()


# In[15]:


movies_rating_df.groupby('title')['rating'].describe()


# In[16]:


ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']


# In[17]:


ratings_df_mean


# In[18]:


ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']


# In[19]:


ratings_df_count


# In[20]:


rating_mean_count_df = pd.concat([ratings_df_count, ratings_df_mean], axis = 1)


# In[21]:


rating_mean_count_df


# In[22]:


rating_mean_count_df.reset_index()


# In[23]:


rating_mean_count_df['mean'].plot(bins = 100, kind = 'hist', color = 'red')


# In[24]:


rating_mean_count_df['count'].plot(bins = 100, kind = 'hist', color = 'red')


# In[25]:


rating_mean_count_df[rating_mean_count_df['mean'] == 5 ]


# In[26]:


rating_mean_count_df.sort_values('count', ascending = False).head(100)


# In[27]:


rating_mean_count_df.sort_values('count', ascending = True).head(400)


# # Step#3: Perform Item-Based Collaborative Filtering On One Movie Sample

# In[28]:


movies_rating_df


# In[29]:


userid_movietitle_matrix = movies_rating_df.pivot_table(index = 'user_id',columns = 'title', values = 'rating')


# In[30]:


userid_movietitle_matrix


# In[31]:


titanic = userid_movietitle_matrix['Titanic (1997)']


# In[32]:


titanic


# In[33]:


starwars = userid_movietitle_matrix['Star Wars (1977)']


# In[34]:


starwars


# In[35]:


starwars_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(starwars), columns = ['correlation'])


# In[36]:


starwars_correlations = starwars_correlations.join(rating_mean_count_df['count'])


# In[37]:


starwars_correlations


# In[38]:


starwars_correlations.dropna(inplace = True)


# In[39]:


starwars_correlations


# In[40]:


starwars_correlations.sort_values('correlation', ascending = False)


# In[41]:


starwars_correlations[starwars_correlations['count']>80].sort_values('correlation', ascending = False).head()


# # Step#4: Create An Item-Based Collaborative Filter On The Entire Dataset

# In[42]:


userid_movietitle_matrix


# In[43]:


movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80)


# In[44]:


movie_correlations


# In[45]:


myRatings = pd.read_csv('My_Ratings.csv')


# In[46]:


myRatings


# In[47]:


myRatings['Movie Name'][0]


# In[48]:


similar_movies_list = pd.Series()

for i in range(0,2):
    similar_movie = movie_correlations[ myRatings['Movie Name'][i]].dropna()
    similar_movie = similar_movie.map(lambda x: x* myRatings['Ratings'][i])
    similar_movies_list=similar_movies_list.append(similar_movie)


# In[49]:


similar_movies_list.sort_values(inplace = True, ascending = False)
print(similar_movies_list.head(9))


# In[50]:


similar_movies_list=pd.concat([similar_movies_list,similar_movie])


# In[51]:


similar_movies_list

