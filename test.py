
import numpy as np

#entirely training data

# Generate x-axis values from 0 to 0.3 (exclusive) with a step of 0.01
x_axis_values_1 = np.arange(0, 0.31, 0.01)

# Create y-axis values filled with 0 (same length as x_axis_values_1)
y_axis_values_1 = np.zeros_like(x_axis_values_1)

# Combine x and y axis values into the first part of the training data array
trainingData_1 = np.column_stack((x_axis_values_1, y_axis_values_1))

# Generate x-axis values from 0.31 to 0.35 (inclusive) with a step of 0.01
x_axis_values_2 = np.arange(0.31, 0.35, 0.01)

# Create y-axis values filled with 0.95 (same length as x_axis_values_2)
y_axis_values_2 = np.full_like(x_axis_values_2, 0.95)

# Combine x and y axis values into the second part of the training data array
trainingData_2 = np.column_stack((x_axis_values_2, y_axis_values_2))

# Generate x-axis values from 0.35 to 0.40 (inclusive) with a step of 0.01
x_axis_values_3 = np.arange(0.35, 0.41, 0.01)

# Create y-axis values with 1 for 0.35 and 0.95 for values from 0.36 to 0.39 and 0 for 0.4
y_axis_values_3 = np.where(x_axis_values_3 == 0.35, 1, np.where((x_axis_values_3 >= 0.36) & (x_axis_values_3 <= 0.39), 0.95, 0))

# Combine x and y axis values into the third part of the training data array
trainingData_3 = np.column_stack((x_axis_values_3, y_axis_values_3))

# Generate x-axis values from 0.41 to 1 (inclusive) with a step of 0.01
x_axis_values_4 = np.arange(0.41, 1.001, 0.01)

# Create y-axis values filled with 0 (same length as x_axis_values_4)
y_axis_values_4 = np.zeros_like(x_axis_values_4)

# Combine x and y axis values into the fourth part of the training data array
trainingData_4 = np.column_stack((x_axis_values_4, y_axis_values_4))

# Concatenate all four training data arrays to create the final training data
trainingData = np.concatenate((trainingData_1, trainingData_2, trainingData_3, trainingData_4))

# Find the index of the data point with x=0.3 and set its y-value to 0
index_x_03 = np.where(trainingData[:, 0] == 0.3)
trainingData[index_x_03, 1] = 0


trainingData = np.array([[0,0], [0.5,1], [1,0]])



# Initial values
w1 = 2.74
b1 = 0
w3 = 0.36
w2 = -1.13
b2 = 0
w4 = 0.63
b3 = 0

def softplus(inputval):
    return np.log(1 + np.exp(inputval))

iterator = 0
oterator = 0
bluelinearr = []
orangelinearr = []
# sets preliminary values for blue line
while iterator <= 1:  # Loop 3 times
    userInput1 = trainingData[0,0] + iterator  
    equation1 = w1 * userInput1 + b1
    yval1 = softplus(equation1)
    yval1f = yval1*w3
    bluelinearr.append(yval1f)
    iterator += 0.01
    iterator = round(iterator, 2)

while oterator <= 1:
    userInput1b = trainingData[0,0] + oterator
    equation1b = w2 * userInput1b + b2
    yval1b = softplus(equation1b)
    yval1bf = yval1b*w4
    orangelinearr.append(yval1bf)
    oterator += 0.01
    oterator = round(oterator, 2)

interator = 0
sumlist = []
while interator <= 100:
    sumlist.append(orangelinearr[interator] + bluelinearr[interator])
    interator += 1