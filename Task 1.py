#!/usr/bin/env python
# coding: utf-8

# # Predicting the percentage of student based on the number of study hours using Linear Regression 

# # Author : Sathwik J R

# In[5]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


link="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data=pd.read_csv(link)
data.head()


# In[7]:


data.describe()


# In[8]:


data.shape


# # Visualization using matplotlib

# In[9]:


plt.scatter(data.Hours,data.Scores,c='g')
plt.xlabel('Hours studied')
plt.ylabel('Percentage scored')
plt.title('Relation Between Hours & Scores')


# From the above scatter plot it is clear that hours studied by a student are directly proportional to the marks scored

# In[10]:


plt.hist(data.Scores, bins=[15,35,50,80,100], rwidth=0.86)
plt.xlabel('Scores')
plt.ylabel('Y axis')
plt.title('Score Ranges')


# From the above histogram we can say that there are many students who scored more than 50%, but there are many more students with very low percentange between 15% to 35% and there are few people who scored above 35% and below 50%

# # Preparing the data for training it

# In[11]:


x=data.drop(columns='Scores')
y=data.drop(columns='Hours')
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y, 
                            test_size=0.2, random_state=0)


# # Trainng data using linear regression

# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


reg = LinearRegression()  
reg.fit(X_train, y_train)


# # Plotting the regressor line on scatter plot

# In[14]:


line = reg.coef_*x+reg.intercept_
# Plotting for the test data
plt.scatter(x, y, c='g')
plt.plot(x, line);
plt.show()


# # Predicting the scores using model

# In[15]:


reg.predict([[2.5]])


# # Predicting the score of a student who studied for 9.25 hours
# 

# In[17]:


reg.predict([[9.25]])


# # Comparing Actual Vs Predicted

# In[18]:


y_predicted=reg.predict(X_test)
y_predicted


# In[19]:


y_test


# # Accuracy & Mean absolute Error of the model

# In[20]:


reg.score(X_test,y_test)


# In[21]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_predicted))


# # By this we can conclude that our model is giving 94.5% accuacy with mean absolute error of 4.183859899002975

# # Thank You !

# In[ ]:




