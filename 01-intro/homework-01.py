## Homework 01

### Set up the environment

import sys, io, os, textwrap
import numpy as np
import pandas as pd

URL = "https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv"
CSV_PATH = "car_fuel_efficiency.csv"

### Q1. Pandas version

# What's the version of Pandas that you installed?

pd.__version__ 

### Getting the data 

# For this homework, we'll use the Car Fuel Efficiency dataset. Download it from <a href='https://raw.githubusercontent.com/alexeygrigorev/datasets/master/car_fuel_efficiency.csv'>here</a>.

df = pd.read_csv(URL)
df.head()

### Q2. Records count

# How many records are in the dataset?

# - 4704
# - 8704
# - 9704
# - 17704

len(df)
# 9704

### Q3. Fuel types

# How many fuel types are presented in the dataset?

# - 1
# - 2
# - 3
# - 4


df['fuel_type'].nunique()
# 2

### Q4. Missing values

# How many columns in the dataset have missing values?

# - 0
# - 1
# - 2
# - 3
# - 4

missing_cols = df.columns[df.isna().any()].tolist()

print("Number of columns with missing values:", len(missing_cols))
print("Columns:", missing_cols)

# Number of columns with missing values: 4
# Columns: ['num_cylinders', 'horsepower', 'acceleration', 'num_doors']


### Q5. Max fuel efficiency

# What's the maximum fuel efficiency of cars from Asia?

# - 13.75
# - 23.75
# - 33.75
# - 43.75

df.columns

# Q5. Maximum fuel efficiency for cars from Asia
max_asia = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()

print("Maximum fuel efficiency (Asia):", max_asia)

# Maximum fuel efficiency (Asia): 23.759122836520497

### Q6. Median value of horsepower

# 1. Find the median value of `horsepower` column in the dataset.
# 2. Next, calculate the most frequent value of the same `horsepower` column.
# 3. Use `fillna` method to fill the missing values in `horsepower` column with the most frequent value from the previous step.
# 4. Now, calculate the median value of `horsepower` once again.

# Has it changed?


# - Yes, it increased
# - Yes, it decreased
# - No

# Q6. Median value of horsepower before and after filling NAs

# Original median
median_before = df['horsepower'].median()

# Most frequent value (mode)
most_freq = df['horsepower'].mode()[0]

# Fill missing values with the mode
df_filled = df.copy()
df_filled['horsepower'] = df_filled['horsepower'].fillna(most_freq)

# New median
median_after = df_filled['horsepower'].median()

print("Median before:", median_before)
print("Most frequent value:", most_freq)
print("Median after:", median_after)

if median_after > median_before:
    print("Answer: Yes, it increased")
elif median_after < median_before:
    print("Answer: Yes, it decreased")
else:
    print("Answer: No")



### Q7. Sum of weights

# 1. Select all the cars from Asia
# 2. Select only columns `vehicle_weight` and `model_year`
# 3. Select the first 7 values
# 4. Get the underlying NumPy array. Let's call it `X`.
# 5. Compute matrix-matrix multiplication between the transpose of `X` and `X`. To get the transpose, use `X.T`. Let's call the result `XTX`.
# 6. Invert `XTX`.
# 7. Create an array `y` with values `[1100, 1300, 800, 900, 1000, 1100, 1200]`.
# 8. Multiply the inverse of `XTX` with the transpose of `X`, and then multiply the result by `y`. Call the result `w`.
# 9. What's the sum of all the elements of the result?

# > **Note**: You just implemented linear regression. We'll talk about it in the next lesson.

# - 0.051
# - 0.51
# - 5.1
# - 51


# Q7. Sum of weights

# 1. Select all cars from Asia
asia = df[df['origin'] == 'Asia']

# 2. Keep only vehicle_weight and model_year
asia_subset = asia[['vehicle_weight', 'model_year']]

# 3. Take the first 7 rows
X = asia_subset.iloc[:7].to_numpy()

# 4. Compute X^T * X
XTX = X.T @ X

# 5. Invert XTX
XTX_inv = np.linalg.inv(XTX)

# 6. Create y
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

# 7. Compute w = (XTX_inv * X^T) * y
w = XTX_inv @ X.T @ y

# 8. Sum of weights
w_sum = w.sum()

print("Sum of weights:", w_sum)

# 9. What's the sum of all the elements of the result?

# Sum of weights: 0.5187709081074016

## Submit the results

# * Submit your results here: https://courses.datatalks.club/ml-zoomcamp-2025/homework/hw01
# * If your answer doesn't match options exactly, select the closest one