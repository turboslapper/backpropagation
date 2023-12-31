# Neural Network Training with Real-Time Plotting

This script demonstrates a simple neural network training process using gradient descent. The network processes a set of training data, adjusts its weights and biases based on the error, and visualizes the outputs in real-time.

## Features

- **Neural Network Training**: Utilizes a basic form of a neural network to process training data.
- **Gradient Descent**: Employs gradient descent to optimize the network's weights and biases.
- **Real-Time Plotting**: Visualizes the network's output for each training iteration in real-time using Matplotlib.

## Code Description

The script is structured as follows:

1. **Softplus Activation Function**: Defines a `softplus` function used as the activation function in the neural network.

2. **Derivative Functions**: Includes derivative functions (`derW1`, `derW2`, etc.) to compute gradients for gradient descent.

3. **Initial Setup**: Sets up the training data and initializes the weights and biases.

4. **Plotting Setup**: Configures a Matplotlib plot for real-time visualization.

5. **Training Loop**:
   - The main loop runs for a predetermined number of iterations.
   - In each iteration, the network processes the training data.
   - The derivatives of the weights and biases are calculated.
   - The weights and biases are updated using the calculated derivatives.
   - The outputs are visualized in real-time on the plot.

6. **Finalization**: Turns off the interactive mode and displays the final plot.

## Usage

- Ensure `numpy` and `matplotlib` are installed in your Python environment.
- Adjust the `learning_rate` and the number of iterations as per your requirements.
- Run the script in an environment that supports real-time plotting (like VS Code).

## Note

Real-time plotting can slow down the training process, especially for a large number of iterations. Adjust the frequency of plot updates for optimal performance.
