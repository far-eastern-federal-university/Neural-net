import numpy as np
import matplotlib.pyplot as plt

"""функция 1/(1+e^(-x))""" 
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))
 
"""производная функции e^(-x))/((1+e^(-x))^2)"""    
def sigmoid_derivative(x):
    return (np.exp(-x))/((1 + np.exp(-x))**2)
 
def compute_loss(y_hat, y):
    return ((y_hat - y)**2).sum()
 
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4) 
        self.weights2   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)
    """функция прямого прохода (от входного слоя к выходному)"""
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))
 
    def backprop(self):
        """цепное правило вычисления производной для функции потерь относительно весов"""
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
 
        """Обновление весов на основании производной"""
        self.weights1 += d_weights1
        self.weights2 += d_weights2
 
 
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
y = np.array([[0],[1],[1],[0]])
nn = NeuralNetwork(X,y)
 
loss_values = []
 
for i in range(1500):
    nn.feedforward()
    nn.backprop()
    loss = compute_loss(nn.output, y)
    loss_values.append(loss)
 
print(nn.output)
print(loss)
"""Визуализация функции потерь""" 
plt.plot(loss_values)
