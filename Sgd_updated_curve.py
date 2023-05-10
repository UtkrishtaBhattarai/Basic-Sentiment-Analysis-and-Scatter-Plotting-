import random
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = []
with open('Part1_x_y_Values.txt', 'r') as file:
    next(file)
    for line in file:
        x, y = line.strip()[1:-1].split(', ')
        data.append((float(x), float(y)))

# Initialize feature weights
a = random.uniform(0, 1)
b = random.uniform(0, 1)
c = random.uniform(0, 1)

# Define learning rate and number of epochs
learning_rate = 0.0001
num_epochs = 10

# Initialize history of feature weights update
history = []

# Define three feature model
def three_feature_model(a, b, c, X):
    return a * X**2 + b * X + c

# Define cost function
def cost_function(data, a, b, c):
    cost = 0
    for X, Y in data:
        cost += (three_feature_model(a, b, c, X) - Y)**2
    return cost / len(data)

# Stochastic gradient descent
for epoch in range(num_epochs):
    # Shuffle data for each epoch
    random.shuffle(data)
    
    # Update feature weights for each data point
    for X, Y in data:
        # Calculate gradients
        grad_a = 2 * (three_feature_model(a, b, c, X) - Y) * X**2
        grad_b = 2 * (three_feature_model(a, b, c, X) - Y) * X
        grad_c = 2 * (three_feature_model(a, b, c, X) - Y)
        
        # Update feature weights
        a -= learning_rate * grad_a
        b -= learning_rate * grad_b
        c -= learning_rate * grad_c
        
        # Store history of feature weights update
        history.append((a, b, c))
    
    # Print cost and feature weights for every epoch
    cost = cost_function(data, a, b, c)
    print(f"Epoch {epoch+1}: Cost = {cost}, Feature Weights = ({a}, {b}, {c})")
    
    # Plot updated curve and scattered data
    X = [x for x, y in data]
    Y = [y for x, y in data]
    plt.figure(figsize=(8,6))
    plt.scatter(X, Y, color='blue')
    x_range = np.linspace(min(X), max(X), 100)
    y_range = [three_feature_model(a, b, c, x) for x in x_range]
    plt.plot(x_range, y_range, color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Epoch {epoch+1}: Cost = {cost:.4f}, Feature Weights = ({a:.4f}, {b:.4f}, {c:.4f})')
    plt.show()
