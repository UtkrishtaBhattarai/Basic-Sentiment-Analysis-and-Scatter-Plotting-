import random
import math

data = []
with open('Part1_x_y_Values.txt', 'r') as file:
    next(file)
    for line in file:
        x, y = line.strip()[1:-1].split(', ')
        data.append((float(x), float(y)))

# Normalize the data
x_mean = sum([x for x, _ in data]) / len(data)
x_std = math.sqrt(sum([(x - x_mean)**2 for x, _ in data]) / len(data))
y_mean = sum([y for _, y in data]) / len(data)
y_std = math.sqrt(sum([(y - y_mean)**2 for _, y in data]) / len(data))
data = [((x - x_mean) / x_std, (y - y_mean) / y_std) for x, y in data]

def three_feature_model(a, b, c, X):
    return a * X**2 + b * X + c

# Initialize feature weights
a = random.uniform(0, 1)
b = random.uniform(0, 1)
c = random.uniform(0, 1)

# Define learning rate and number of epochs
learning_rate = 0.01
num_epochs = 50

# Initialize history of feature weights update
history = []

# Define cost function
def cost_function(data, a, b, c):
    cost = 0
    for X, Y in data:
        cost += (three_feature_model(a, b, c, X) - Y)**2
    cost /= len(data)
    if math.isnan(cost) or math.isinf(cost):
        return float('nan')
    else:
        return cost

# Stochastic Gradient Descent
for epoch in range(num_epochs):
    # Shuffle data
    random.shuffle(data)
    
    # Iterate over each data point
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
