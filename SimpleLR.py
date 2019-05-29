from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Salary_data.csv')

xData = np.array(df['YearsExperience'])
yData = np.array(df['Salary'])


def best_fit_line(xs, ys):
    m = ((mean(xs)*mean(ys)) - mean(xs*ys))/((mean(xs)*mean(xs)) - (mean(xs*xs)))
    b = mean(ys) - (m*mean(xs))
    return m, b


def squared_error(ys, yline):
    return sum((ys - yline)**2)


def coeff_of_deter(ys, regression_line):
    y_mean_line = [mean(ys) for y in ys]
    squared_error_mean = squared_error(yData, y_mean_line)
    squared_error_regression = squared_error(yData, regression_line)
    return 1 - (squared_error_regression/squared_error_mean)


m, b = best_fit_line(xData, yData)
regression_line = [m*x + b for x in xData]
coeff = coeff_of_deter(yData, regression_line)

print(m, b, coeff)

plt.scatter(xData, yData)
plt.plot(xData, regression_line)
plt.show()