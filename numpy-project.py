import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('temperature_data.csv')
years = df['YEAR'].values
max_temp = df['MAX TEMP'].values
day = df['DAY'].values
precipitation = df['PRECIPITATION'].values
min_temp = df['MIN TEMP'].values
mean_temp = df['MEAN TEMP'].values

# Create a "design matrix" with a column of ones and a column for each year
X = np.vstack([np.ones_like(years), years]).T

# Use the pseudo-inverse to perform the linear regression
beta = np.linalg.pinv(X).dot(max_temp)

# Now beta contains the intercept (average temperature) and the slope (change in temperature per year)
intercept, slope = beta

plt.scatter(years, max_temp, label='Data')
plt.plot(years, intercept + slope * years, 'r', label='Fit')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.title('Temperature Trend Over Time')
plt.legend()
plt.show()
