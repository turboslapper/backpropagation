
import numpy as np

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
    return np.log(1+np.exp(inputval))


userInput2 = trainingData[1,0]
equation2 = w1*userInput2 + b1
yval2 = softplus(equation2)
yval2 = yval2*w3

print(equation2)

def derW1(observed ,output, equation, userinput):
    return -2 * (observed - output) * w3 * (np.exp(equation)/(1 + np.exp(equation))) * userinput

print(derW1(0,0.68,0,0) + derW1(1,0.85,1.37,0.5) + derW1(0,1.19,2.74,1))


for i in range(3):  # Loop 3 times

    if i == 0:
        userInput1 = trainingData[0,0]
        equation1 = w1 * userInput1 + b1
        yval1 = softplus(equation1)
        yval1f = yval1*w3
    elif i == 1:
        userInput2 = trainingData[1,0]
        equation2 = w1*userInput2 + b1
        yval2 = softplus(equation2)
        yval2f = yval2*w3
    else: 
        userInput3 = trainingData[2,0]
        equation3 = w1*userInput3 + b1
        yval3 = softplus(equation3)
        yval3f = yval3*w3 

for i in range(3):  # Loop 3 times

    if i == 0:
        userInput1b = trainingData[0,0]
        equation1b = w2 * userInput1b + b2
        yval1b = softplus(equation1b)
        yval1bf = yval1b*w4
    elif i == 1:
        userInput2b = trainingData[1,0]
        equation2b = w2*userInput2b + b2
        yval2b = softplus(equation2b)
        yval2bf = yval2b*w4
    else: 
        userInput3b = trainingData[2,0]
        equation3b = w2*userInput3b + b2
        yval3b = softplus(equation3b)
        yval3bf = yval3b *w4 

output1 = yval1f + yval1bf + b3
output2 = yval2f + yval2bf + b3
output3 = yval3f + yval3bf + b3

print(yval1)

def derW3(observed, output, f):
    return -2 * (observed - output) * f

print(derW3(0, 0.72, 0.21))
