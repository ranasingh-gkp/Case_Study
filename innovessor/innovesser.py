#help from
#https://www.hackerearth.com/practice/machine-learning/machine-learning-projects/python-project/tutorial/

#Loading libraries 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns
from scipy import stats
from scipy.stats import norm


#loading data

train=pd.read_excel('data_new.xlsx')
train.head()
train.info()


#check missing values
train.columns[train.isnull().any()]

#missing value counts in each of these columns
miss = train.isnull().sum()/len(train)
miss = miss[miss > 0]
miss.sort_values(inplace=True)
miss

#visualising missing values
miss = miss.to_frame()
miss.columns = ['count']
miss.index.names = ['Name']
miss['Name'] = miss.index

#plot the missing value count
sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'count', data=miss)
plt.xticks(rotation = 90)
sns.plt.show()


#Let's proceed and check the distribution of the target variable. 


train_ID = train['aco_num']
train_name = train['aco_name']
train_state = train['aco_state']


#-------Data Pre-Processing
#since some numeric data have very high missing data so remove them

del train['aco12']
del train['aco44']
del train['aco43']
del train['aco_num']
del train['aco_name']
del train['aco_state']



del train['aco18']
del train['aco42']
del train['dm_comp']
del train['aco27']
del train['aco41']
del train['aco28']
del train['aco30']
del train['aco19']
del train['aco20']
del train['aco40']
del train['py']

#create new data
train_new = train[train['per_capita_exp_total_py'].notnull()]
test_new = train[train['per_capita_exp_total_py'].isnull()]

#imputing missing values
test_new.fillna(train_new.mean(), inplace=True)
train_new.fillna(train_new.mean(), inplace=True)

#get numeric features
numeric_features = [f for f in train_new.columns if train_new[f].dtype != object]

#transform the numeric features using log(x + 1)
from scipy.stats import skew
skewed = train_new[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
train_new[skewed] = np.log1p(train_new[skewed])


#Now, we'll standardize the numeric features. 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_new[numeric_features])
scaled = scaler.transform(train_new[numeric_features])

for i, col in enumerate(numeric_features):
       train_new[col] = scaled[:,i]



