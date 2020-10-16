#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np


# In[9]:


import os
data=pd.read_csv('C:\\Users\\Mahishekar\\Documents\\car data.csv')


# In[10]:


data


# In[11]:


data.describe()


# In[12]:


data['Selling_Price'].isnull().sum()


# In[13]:


print(data['Selling_Price'].unique().values())


# In[14]:


data['current_year']=2020


# In[15]:


#data['current year']


# In[16]:


data['no_yrs']=data['current_year']-data['Year']


# In[17]:


data.columns


# In[18]:


data=data[[ 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner',
       'no_yrs']]


# In[19]:


data.head()


# In[ ]:





# In[20]:


#data.columns


# In[21]:


#data=data[[]]


# In[22]:


data=pd.get_dummies(data,drop_first=True)


# In[23]:


data.head()


# In[24]:


data.corr()


# In[25]:


import seaborn as sns
sns.pairplot(data)


# In[26]:


import matplotlib.pyplot as plt


# In[27]:


#heat=data.corr()
#corre=heat.index
plt.figure(figsize=(18,17))
h=sns.heatmap(data.corr(),annot=True)


# In[28]:


X=data.iloc[:,1:]
Y=data.iloc[:,0]


# In[29]:


Y


# In[36]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
X.head()


# In[31]:


model.fit(X,Y)


# In[32]:


model.feature_importances_


# In[33]:


aa=pd.Series(model.feature_importances_,index=X.columns)


# In[35]:


aa.nlargest(8).plot(kind='barh')
plt.show()


# In[38]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=1)


# In[40]:


print(X_train.head())


# In[42]:


from sklearn .ensemble import RandomForestRegressor
model=RandomForestRegressor()


# In[68]:


model.fit(X_train,y_train)
pred=model.predict(X_test)


# In[74]:


sns.distplot(y_test-pred)


# ###hyperparameter 

# In[59]:


n_estimators=[int(x) for x in np.linspace(100,1500,10)]
max_depth=[4,5,6,7,8,10,12,14]
min_samples_split=[2,5,10,15]
max_features=['auto','sqrt']
min_samples_leaf=[1,3,4,5,6]



# In[61]:


random_grid={'n_estimators':n_estimators,
             'max_depth':max_depth,
              'min_samples_split':min_samples_split,
            'max_features':max_features,
              'min_samples_leaf':min_samples_leaf}


# In[62]:


random_grid


# In[64]:


from sklearn.model_selection import RandomizedSearchCV()


# In[88]:


m=RandomizedSearchCV(estimator=model,
            param_distributions=random_grid,
            cv=5,n_jobs=1,
                    scoring='neg_mean_squared_error',
                    verbose=2)
m.fit(X_train,y_train)


# In[98]:


pred1=m.predict(X_test)


# In[99]:


print(m.best_params_)
print(m.best_score_)


# In[100]:


sns.distplot(y_test-pred1)


# In[107]:


plt.scatter(y_test,pred1)


# In[108]:


import pickle
file=open('car prediction.pkl','wb')


# In[110]:


pickle.dump(m,file)


# In[ ]:




