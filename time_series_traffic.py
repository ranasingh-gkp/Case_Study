import pandas as pd          
import numpy as np          # For mathematical calculations
import matplotlib.pyplot as plt  # For plotting graphs
from datetime import datetime    # To access datetime
from pandas import Series        # To work on series
%matplotlib inline
import warnings                   # To ignore the warnings
warnings.filterwarnings("ignore")

train=pd.read_csv("Train.csv")
test=pd.read_csv("Test.csv")

train_original=train.copy()
test_original=test.copy()

train.columns, test.columns
#Let’s look at the data types of each feature.

train.dtypes, test.dtypes

#=====Feature Extraction
#We will extract the time and date from the Datetime. We have seen earlier that the data type of Datetime is object. So first of all we have to change the data type to datetime format otherwise we can not extract features from it.

train['Datetime'] = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
test['Datetime'] = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test_original['Datetime'] = pd.to_datetime(test_original.Datetime,format='%d-%m-%Y %H:%M')
train_original['Datetime'] = pd.to_datetime(train_original.Datetime,format='%d-%m-%Y %H:%M')

#let’s extract the year, month, day and hour from the Datetime.

for i in (train, test, test_original, train_original):
    i['year']=i.Datetime.dt.year 
    i['month']=i.Datetime.dt.month 
    i['day']=i.Datetime.dt.day
    i['Hour']=i.Datetime.dt.hour 

#We will first extract the day of week from Datetime and then based on the values we will assign whether the day is a weekend or not.

#Values of 5 and 6 represents that the days are weekend.

train['day of week']=train['Datetime'].dt.dayofweek
temp = train['Datetime']

#Let’s assign 1 if the day of week is a weekend and 0 if the day of week in not a weekend.

def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0

temp2 = train['Datetime'].apply(applyer)
train['weekend']=temp2

#Let’s look at the time series.

train.index = train['Datetime'] # indexing the Datetime to get the time period on the x-axis.
df=train.drop('ID',1)           # drop ID variable to get only the Datetime on x-axis.
ts = df['Count']
plt.figure(figsize=(16,8))
plt.plot(ts, label='Passenger Count')
plt.title('Time Series')
plt.xlabel("Time(year-month)")
plt.ylabel("Passenger count")
plt.legend(loc='best')

#Here we can infer that there is an increasing trend in the series, i.e., the number of count is increasing with respect to time. We can also see that at certain points there is a sudden increase in the number of counts. The possible reason behind this could be that on particular day, due to some event the traffic was high.

#=====Exploratory Analysis
#our hypothesis given below
#Lets recall the hypothesis that we made earlier:

#Traffic will increase as the years pass by
#Traffic will be high from May to October
#Traffic on weekdays will be more
#Traffic during the peak hours will be high
#After having a look at the dataset, we will now try to validate our hypothesis and make other inferences from the dataset.

#1)Our first hypothesis was traffic will increase as the years pass by. So let’s look at yearly passenger count.
train.groupby('year')['Count'].mean().plot.bar()
#2)Our second hypothesis was about increase in traffic from May to October. So, let’s see the relation between count and month.

train.groupby('month')['Count'].mean().plot.bar()
#Here we see a decrease in the mean of passenger count in last three months. This does not look right. Let’s look at the monthly mean of each year separately.
temp=train.groupby(['year', 'month'])['Count'].mean()
temp.plot(figsize=(15,5), title= 'Passenger Count(Monthwise)', fontsize=14)

#In the above line plot we can see an increasing trend in monthly passenger count and the growth is approximately exponential.
train.groupby('day')['Count'].mean().plot.bar()
#We also made a hypothesis that the traffic will be more during peak hours. So let’s see the mean of hourly passenger count.

train.groupby('Hour')['Count'].mean().plot.bar()
#It can be inferred that the peak traffic is at 7 PM and then we see a decreasing trend till 5 AM.
#After that the passenger count starts increasing again and peaks again between 11AM and 12 Noon.

#4)Let’s try to validate our hypothesis in which we assumed that the traffic will be more on weekdays.
train.groupby('weekend')['Count'].mean().plot.bar()
#It can be inferred from the above plot that the traffic is more on weekdays as compared to weekends which validates our hypothesis.

#5)Now we will try to look at the day wise passenger count.

#Note - 0 is the starting of the week, i.e., 0 is Monday and 6 is Sunday.
train.groupby('day of week')['Count'].mean().plot.bar()
#From the above bar plot, we can infer that the passenger count is less for saturday and sunday as compared to the other days of the week

#drop the ID variable as it has nothing to do with the passenger count.
train=train.drop('ID',1)

#we will aggregate the hourly time series to daily, weekly, and monthly time series to reduce the noise and make it more stable and hence would be easier for a model to learn.
train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp

# Hourly time series
hourly = train.resample('H').mean()

# Converting to daily mean
daily = train.resample('D').mean()

# Converting to weekly mean
weekly = train.resample('W').mean()

# Converting to monthly mean
monthly = train.resample('M').mean()

#Let’s look at the hourly, daily, weekly and monthly time series.

fig, axs = plt.subplots(4,1)

hourly.Count.plot(figsize=(15,8), title= 'Hourly', fontsize=14, ax=axs[0])
daily.Count.plot(figsize=(15,8), title= 'Daily', fontsize=14, ax=axs[1])
weekly.Count.plot(figsize=(15,8), title= 'Weekly', fontsize=14, ax=axs[2])
monthly.Count.plot(figsize=(15,8), title= 'Monthly', fontsize=14, ax=axs[3])

plt.show()

#But it would be difficult to convert the monthly and weekly predictions to hourly predictions, as first we have to convert the monthly predictions to weekly, weekly to daily and daily to hourly predictions, which will become very expanded process. So, we will work on the daily time series.
test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp 

# Converting to daily mean
test = test.resample('D').mean()

train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp

# Converting to daily mean
train = train.resample('D').mean()

#=====Splitting the data into training and validation part
#The starting date of the dataset is 25-08-2012 as we have seen in the exploration part and the end date is 25-09-2014.
Train=train.ix['2012-08-25':'2014-06-24']
valid=train.ix['2014-06-25':'2014-09-25']
#Now we will look at how the train and validation part has been divided.

Train.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='train')
valid.Count.plot(figsize=(15,8), title= 'Daily Ridership', fontsize=14, label='valid')
plt.xlabel("Datetime")
plt.ylabel("Passenger count")
plt.legend(loc='best')
plt.show()
