import numpy as np

class NN:
    def __init__(self, layer_dims, beta=0.9, learning_rate=0.01):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
        self.sdW = {f'sdW{l}': np.zeros_like(self.parameters[f'W{l}']) for l in range(1, len(self.layer_dims))}
        self.sdb = {f'sdb{l}': np.zeros_like(self.parameters[f'b{l}']) for l in range(1, len(self.layer_dims))}
        self.beta = beta
        self.learning_rate = learning_rate
        self.L = len(layer_dims) - 1  # number of layers excluding input

    def initialize_parameters(self):
        np.random.seed(1)
        parameters = {}
        for l in range(1, len(self.layer_dims)):
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * np.sqrt(2. / self.layer_dims[l-1])
            parameters[f'b{l}'] = np.zeros((self.layer_dims[l], 1))
        return parameters

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_derivative(Z):
        return Z > 0

    @staticmethod
    def softmax(Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

    def forward_propagation(self, X):
        caches = []
        A = X
        
        for l in range(1, self.L + 1):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            
            Z = np.dot(W, A_prev) + b
            
            if l == self.L:  # Output layer
                A = self.softmax(Z)
            else:
                A = self.relu(Z)
            
            cache = (A_prev, W, b, Z)
            caches.append(cache)
        return A, caches

    def backward_propagation(self, AL, Y, caches):
        grads = {}
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # Output layer
        dZL = AL - Y
        A_prev, WL, bL, ZL = caches[self.L - 1]
        grads[f'dW{self.L}'] = (1/m) * np.dot(dZL, A_prev.T)
        grads[f'db{self.L}'] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
        dA_prev = np.dot(WL.T, dZL)

        # Hidden layers
        for l in reversed(range(self.L - 1)):
            A_prev, W, b, Z = caches[l]
            dZ = dA_prev * self.relu_derivative(Z)
            grads[f'dW{l+1}'] = (1/m) * np.dot(dZ, A_prev.T)
            grads[f'db{l+1}'] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 0:
                dA_prev = np.dot(W.T, dZ)

        return grads

    def update_parameters(self, grads):
        epsilon = 1e-8  # Small value to avoid division by zero

        for l in range(1, self.L + 1):
            # Update the exponentially weighted averages of the squared gradients
            self.sdW[f'sdW{l}'] = (self.beta * self.sdW[f'sdW{l}']) + (1 - self.beta) * np.square(grads[f'dW{l}'])
            self.sdb[f'sdb{l}'] = (self.beta * self.sdb[f'sdb{l}']) + (1 - self.beta) * np.square(grads[f'db{l}'])

            # Update weights and biases using RMSprop-like update rule
            self.parameters[f'W{l}'] -= self.learning_rate * grads[f'dW{l}'] / (np.sqrt(self.sdW[f'sdW{l}']) + epsilon)
            self.parameters[f'b{l}'] -= self.learning_rate * grads[f'db{l}'] / (np.sqrt(self.sdb[f'sdb{l}']) + epsilon)

    def train(self, X, Y, num_iterations):
        costs = []
        for i in range(num_iterations):
            # Forward propagation
            AL, caches = self.forward_propagation(X)
            
            # Compute cost
            cost = -np.sum(Y * np.log(AL + 1e-8)) / Y.shape[1]
            
            # Backward propagation
            grads = self.backward_propagation(AL, Y, caches)
            
            # Update parameters with RMSprop
            self.update_parameters(grads)
            
            if i % 100 == 0:
                costs.append(cost)
                print(f"Cost after iteration {i}: {cost}")
        
        return costs

    def predict(self, X):
        AL, _ = self.forward_propagation(X)
        return np.argmax(AL, axis=0)