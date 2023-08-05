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