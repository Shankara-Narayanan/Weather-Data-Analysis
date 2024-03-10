#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ### Data Exploration

# In[2]:


df = pd.read_csv('weather.csv')
df.head(5)


# In[3]:


df.info()
df.describe()


# ### Visualization

# In[4]:


sns.pairplot(df[['MinTemp','MaxTemp','Rainfall']])


# In[17]:


plt.figure(figsize = (10,5))
plt.plot(df['MinTemp'].index, df['MaxTemp'].values,marker = 'o')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.title('Line Plot of MinTemp vs MaxTemp')
plt.grid(True)
plt.show()


# ### Analysis

# In[6]:


corr = df.corr()
corr


# In[24]:


plt.figure(figsize=(14, 14))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


# In[8]:


df.head(2)


# In[9]:


stats = df.agg(['mean','median','min','max']).transpose()
stats


# In[10]:


cov_mat = df.cov()
cov_mat


# In[11]:


spearman_corr_mat = df.corr()
spearman_corr_mat


# In[28]:


# data preparation
features = ['MinTemp', 'MaxTemp', 'Evaporation', 'WindGustSpeed', 'Humidity9am', 'Pressure9am', 'RainToday']
target = 'Rainfall'

df_model = df[features + [target]].dropna()

x_train,x_test,y_train,y_test = train_test_split(
df_model[features],
df_model[target],
test_size=0.2,
random_state=42)
"""
Upon examining the features, it is observed that the 'RainToday' column is a categorical variable
with values 'Yes' or 'No'. Since machine learning models typically operate with numerical data,
encoding categorical variables becomes necessary before incorporating them into the model.
"""
# preprocessing 
preprocessor = ColumnTransformer(
    transformers=[
        ('rain_today', OneHotEncoder(), ['RainToday'])
    ],
    remainder='passthrough'
)
# model building
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])   
# model Training 
model.fit(x_train, y_train)
# prediction
y_pred = model.predict(x_test)
y_pred


# In[14]:


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[30]:


print('Predicted Values:', y_pred)


# In[31]:


print('Actual Values:', y_test)


# In[16]:


print('Coefficients:', model.named_steps['regressor'].coef_)
print('Intercept:', model.named_steps['regressor'].intercept_)


# ### Conclusion

# In[27]:


''''
In visualization
The distribution of minimum temperature is right-skewed.
This means that there are more data points on the left side of the distributionthan on the right side. 
The distribution of maximum temperature isleft-skewed, with more data points on the right side than on the left side.

LINE PLOT
The plot suggests that there is a strong, positive relationship between minimum and maximum temperatures.
This means that days with higher minimum temperatures tend to have higher maximum temperatures as well.
The upward curvature of the line suggests that this relationship might be slightly stronger 
for higher minimum temperatures.

HEAT MAP 
The Heat Map shows that there is a strong relation ship between Minimum Temperatue,Maximum Temperature and 
Temperature at 9 AM and 3 PM
There is a weak relationship between Wind Gust Speed to Pressure 9AM and 3PM

Coefficients
Increasing the first feature by 1 unit decreases predicted rainfall by 2.946 millimeters.
Increasing the second feature by 1 unit increases predicted rainfall by 2.946 millimeters.
Increasing the third feature by 1 unit increases predicted rainfall by 0.038 millimeters, and so on.

Value: 113.02658245599746
Interpretation: Represents the predicted rainfall when all features are 0. 
In this case, the model predicts 113.026 millimeters of rainfall when all features have a value of 0.
'''


# In[ ]:




