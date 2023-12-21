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

# sets preliminary values for blue line
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

# sets preliminary values for orange line
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
        yval3bf = yval3b * w4 

output1 = yval1f + yval1bf + b3
output2 = yval2f + yval2bf + b3
output3 = yval3f + yval3bf + b3

# sets derivative definitions
def derW1(observed ,output, equation, userinput):
    return -2 * (observed - output) * w3 * (np.exp(equation)/(1 + np.exp(equation))) * userinput

def derW2(observed ,output, equation, userinput):
    return -2 * (observed - output) * w4 * (np.exp(equation)/(1 + np.exp(equation))) * userinput
#f represents the values prior to yval"f" so just yval
def derW3(observed, output, f):
    return -2 * (observed - output) * f
#bf represents the values prior to the yval"bf" so just yvalb
def derW4(observed, output, bf):
    return -2 * (observed - output) * bf

def derb1(observed, output, equation):
    return -2 * (observed - output) * w3 * (np.exp(equation)/(1 + np.exp(equation)))  

def derb2(observed, output, equation):
    return -2 * (observed - output) * w4 * (np.exp(equation)/(1 + np.exp(equation))) 

def derb3(observed, output):
    return -2 * (observed - output) 


# blue line derivative 
derivativew1 = derW1(trainingData[0,1],output1,equation1,userInput1) + derW1(trainingData[1,1], output2, equation2, userInput2) + derW1(trainingData[2,1], output3, equation3, userInput3)
stepsizew1 = derivativew1 * 0.095
w1 = w1 - stepsizew1

derivativeb1 = derb1(trainingData[0,1], output1, equation1) + derb1(trainingData[1,1], output2, equation2) + derb1(trainingData[2,1], output3, equation3)
stepsizeb1 = derivativeb1 * 0.09
b1 = b1 - stepsizeb1

derivativew3 = derW3(trainingData[0,1], output1, yval1) + derW3(trainingData[1,1], output2, yval2) + derW3(trainingData[2,1], output3, yval3)
stepsizew3 = derivativew3 * 0.099
w3 = w3 - stepsizew3

# orange line derivatives
derivativew2 = derW2(trainingData[0,1], output1, equation1b, userInput1b) + derW2(trainingData[1,1], output2, equation2b, userInput2b) + derW2(trainingData[2,1], output3, equation3b, userInput3b)
stepsizew2 = derivativew2 * 0.095
w2 = w2 - stepsizew2

derivativeb2 = derb2(trainingData[0,1], output1, equation1b) + derb2(trainingData[1,1], output2, equation2b) + derb2(trainingData[2,1], output3, equation3b)
stepsizeb2 = derivativeb2 * 0.09
b2 = b2 - stepsizeb2

derivativew4 = derW4(trainingData[0,1], output1, yval1b) + derW4(trainingData[1,1], output2, yval2b) + derW4(trainingData[2,1], output3, yval3b)
stepsizew4 = derivativew4 * 0.09
w4 = w4 - stepsizew4

# final bias calculation
derivativeb3 = derb3(trainingData[0,1], output1) + derb3(trainingData[1,1], output2) + derb3(trainingData[2,1], output3)
stepsizeb3 = derivativeb3 * 0.09
b3 = b3 - stepsizeb3


for i in range(100000):

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

    # sets preliminary values for orange line
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
            yval3bf = yval3b * w4 

    output1 = yval1f + yval1bf + b3
    output2 = yval2f + yval2bf + b3
    output3 = yval3f + yval3bf + b3

    
    
    # blue line derivative 
    derivativew1 = derW1(trainingData[0,1],output1,equation1,userInput1) + derW1(trainingData[1,1], output2, equation2, userInput2) + derW1(trainingData[2,1], output3, equation3, userInput3)
    stepsizew1 = derivativew1 * 0.095
    w1 = w1 - stepsizew1

    derivativeb1 = derb1(trainingData[0,1], output1, equation1) + derb1(trainingData[1,1], output2, equation2) + derb1(trainingData[2,1], output3, equation3)
    stepsizeb1 = derivativeb1 * 0.09
    b1 = b1 - stepsizeb1

    derivativew3 = derW3(trainingData[0,1], output1, yval1) + derW3(trainingData[1,1], output2, yval2) + derW3(trainingData[2,1], output3, yval3)
    stepsizew3 = derivativew3 * 0.099
    w3 = w3 - stepsizew3

    # orange line derivatives
    derivativew2 = derW2(trainingData[0,1], output1, equation1b, userInput1b) + derW2(trainingData[1,1], output2, equation2b, userInput2b) + derW2(trainingData[2,1], output3, equation3b, userInput3b)
    stepsizew2 = derivativew2 * 0.095
    w2 = w2 - stepsizew2

    derivativeb2 = derb2(trainingData[0,1], output1, equation1b) + derb2(trainingData[1,1], output2, equation2b) + derb2(trainingData[2,1], output3, equation3b)
    stepsizeb2 = derivativeb2 * 0.09
    b2 = b2 - stepsizeb2

    derivativew4 = derW4(trainingData[0,1], output1, yval1b) + derW4(trainingData[1,1], output2, yval2b) + derW4(trainingData[2,1], output3, yval3b)
    stepsizew4 = derivativew4 * 0.09
    w4 = w4 - stepsizew4

    # final bias calculation
    derivativeb3 = derb3(trainingData[0,1], output1) + derb3(trainingData[1,1], output2) + derb3(trainingData[2,1], output3)
    stepsizeb3 = derivativeb3 * 0.09
    b3 = b3 - stepsizeb3


    for i in range(1000):
        if i % 100 == 0:
            print(output1, output2, output3)
    