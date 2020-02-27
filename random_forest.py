import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Position_Salaries.csv')
x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x,y)

from numpy import array
y_pred = regressor.predict(array(6.5).reshape(-1, 1))

x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid))
plt.title('Salary to Position regression with Random Forest')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()