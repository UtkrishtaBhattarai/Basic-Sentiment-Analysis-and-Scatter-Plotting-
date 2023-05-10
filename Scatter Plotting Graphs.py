import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# Define a custom converter function to remove the parentheses from the data
def converter(x):
    return float(x.decode('UTF-8').strip('()'))

# Load the data from the file using np.genfromtxt with the custom converter
data = np.genfromtxt('Part1_x_y_Values.txt', delimiter=',', dtype=float, converters={0: converter, 1: converter}, skip_header=1)


X = data[:, 0]  # select all rows from the first column
Y = data[:, 1]  # select all rows from the second column



# Compute the coefficients of the line using the least squares method
a, b = np.polyfit(X, Y, 1)

print(a, b)

# Plot the data and the line
plt.scatter(X, Y)

# Making use of linear regression line
plt.plot(X, a*X + b, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
