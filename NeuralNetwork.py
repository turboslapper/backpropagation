import numpy as np
import matplotlib.pyplot as plt

def softplus(inputval):
    return np.log(1 + np.exp(inputval))

def derW1(observed, output, equation, userinput):
    return -2 * (observed - output) * w3 * (np.exp(equation) / (1 + np.exp(equation))) * userinput

def derW2(observed, output, equation, userinput):
    return -2 * (observed - output) * w4 * (np.exp(equation) / (1 + np.exp(equation))) * userinput

def derW3(observed, output, f):
    return -2 * (observed - output) * f

def derW4(observed, output, bf):
    return -2 * (observed - output) * bf

def derb1(observed, output, equation):
    return -2 * (observed - output) * w3 * (np.exp(equation) / (1 + np.exp(equation)))

def derb2(observed, output, equation):
    return -2 * (observed - output) * w4 * (np.exp(equation) / (1 + np.exp(equation)))

def derb3(observed, output):
    return -2 * (observed - output)

# Your initial setup
trainingData = np.array([[0, 0], [0.5, 1], [1, 0]])
w1, b1 = 2.74, 0
w2, b2 = -1.13, 0
w3, b3 = 0.36, 0
w4 = 0.63

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
line1, = ax.plot([], [], 'r-', label='Output 1')
line2, = ax.plot([], [], 'g-', label='Output 2')
line3, = ax.plot([], [], 'b-', label='Output 3')
ax.set_title('Neural Network Outputs Over Training Iterations')
ax.set_xlabel('Iteration')
ax.set_ylabel('Output Values')
ax.legend()

# Main training loop
for iteration in range(100000):
    sum_derivatives_w1 = sum_derivatives_b1 = sum_derivatives_w3 = 0
    sum_derivatives_w2 = sum_derivatives_b2 = sum_derivatives_w4 = sum_derivative_b3 = 0

    for j in range(3):  # Loop over each data point
        userInput = trainingData[j, 0]
        observed = trainingData[j, 1]

        # Blue line calculations
        equation1 = w1 * userInput + b1
        yval1 = softplus(equation1)
        yval1f = yval1 * w3

        # Orange line calculations
        equation2 = w2 * userInput + b2
        yval2 = softplus(equation2)
        yval2f = yval2 * w4

        output = yval1f + yval2f + b3

        # Update sums of derivatives
        sum_derivatives_w1 += derW1(observed, output, equation1, userInput)
        sum_derivatives_b1 += derb1(observed, output, equation1)
        sum_derivatives_w3 += derW3(observed, output, yval1)

        sum_derivatives_w2 += derW2(observed, output, equation2, userInput)
        sum_derivatives_b2 += derb2(observed, output, equation2)
        sum_derivatives_w4 += derW4(observed, output, yval2)

        sum_derivative_b3 += derb3(observed, output)

        if j == 0:
            output1 = output
        elif j == 1:
            output2 = output
        else:
            output3 = output

    # Update weights and biases
    learning_rate = 0.01  # Example learning rate, adjust as needed
    w1 -= learning_rate * sum_derivatives_w1
    b1 -= learning_rate * sum_derivatives_b1
    w3 -= learning_rate * sum_derivatives_w3

    w2 -= learning_rate * sum_derivatives_w2
    b2 -= learning_rate * sum_derivatives_b2
    w4 -= learning_rate * sum_derivatives_w4

    b3 -= learning_rate * sum_derivative_b3

    # Update plot data
    line1.set_xdata(np.append(line1.get_xdata(), iteration))
    line1.set_ydata(np.append(line1.get_ydata(), output1))
    line2.set_xdata(np.append(line2.get_xdata(), iteration))
    line2.set_ydata(np.append(line2.get_ydata(), output2))
    line3.set_xdata(np.append(line3.get_xdata(), iteration))
    line3.set_ydata(np.append(line3.get_ydata(), output3))

    # Redraw the plot
    ax.relim()  # Recalculate limits
    ax.autoscale_view(True, True, True)  # Rescale the view
    plt.pause(0.001)  # Pause briefly to allow the plot to update

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
