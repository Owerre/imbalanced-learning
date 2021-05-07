import numpy as np
import pandas as pd

# load data
df = pd.read_csv('../../data/clean_data.csv') 

# Calculate first and third quartile
QR_1st = df['latitude'].describe()['25%']
QR_3rd = df['latitude'].describe()['75%']

# Interquartile range
IQR = QR_3rd - QR_1st

# Lower and upper bounds
lower_bound = QR_1st - 3*IQR
upper_bound = QR_3rd + 3*IQR

# Remove outliers in the X attribute
df['latitude'] =  df['latitude'][(df['latitude'] > lower_bound)&(df['latitude'] < upper_bound)] 