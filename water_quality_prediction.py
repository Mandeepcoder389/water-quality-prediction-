# To install basic/necessary libraries
import pandas as pd # data manipulation
import numpy as np# numerical python - linear algebra

# Import necessary libraries
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# load the dataset
df = pd.read_csv('PB_All_2000_2021.csv', sep=';')
print(df)

#datasate info
df.info()

# how mach row & columns
print(df.shape)

# Statistics of the data
print(df.describe().T)

# Missing values
print(df.isnull().sum())

# date is in object - date format
df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
print(df)

#data info in current updated data
print(df.info)

df = df.sort_values(by=['id', 'date'])
print(df.head())

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
print(df.head())

print(df.columns)

pollutants = ['O2', 'NO3', 'NO2', 'SO4','PO4', 'CL']
