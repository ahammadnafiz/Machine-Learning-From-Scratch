import numpy as np

class NN:
    def __init__(self, layer_sizes):
        self.params = self.initialize_params(layer_sizes)
        
    def initialize_params(self, layer_sizes):
        np.random.seed(3)
        params = {}
        num_layers = len(layer_sizes)
        for layer in range(1, num_layers):
            params[f"W{layer}"] = np.random.randn(layer_sizes[layer], layer_sizes[layer - 1]) * np.sqrt(2. / layer_sizes[layer-1])
            params[f"b{layer}"] = np.zeros((layer_sizes[layer], 1))
        return params

    def forward_activation(self, A_prev, W, b, activation_type):
        Z = np.dot(W, A_prev) + b
        if activation_type == "relu":
            A = np.maximum(0, Z)
        elif activation_type == "softmax":
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        cache = (A_prev, W, b, Z)
        return A, cache

    def forward_pass(self, X):
        caches = []
        A = X
        num_layers = len(self.params) // 2
        for layer in range(1, num_layers):
            A, cache = self.forward_activation(A, self.params[f"W{layer}"], self.params[f"b{layer}"], "relu")
            caches.append(cache)
        AL, cache = self.forward_activation(A, self.params[f"W{num_layers}"], self.params[f"b{num_layers}"], "softmax")
        caches.append(cache)
        return AL, caches
    
    def compute_loss(self, predictions, true_labels):
        num_samples = true_labels.shape[1]
        return -1/num_samples * np.sum(true_labels * np.log(predictions + 1e-8))

    def backward_activation(self, dA, cache, activation_type):
        A_prev, W, b, Z = cache
        num_samples = A_prev.shape[1]
        if activation_type == "relu":
            dZ = np.where(Z > 0, dA, 0)
        elif activation_type == "softmax":
            dZ = dA  # Softmax derivative
        dW = 1/num_samples * np.dot(dZ, A_prev.T)
        db = 1/num_samples * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        return dA_prev, dW, db

    def backward_pass(self, predictions, true_labels, caches):
        gradients = {}
        num_layers = len(caches)
        true_labels = true_labels.reshape(predictions.shape)
        dAL = predictions - true_labels
        
        gradients[f"dA{num_layers-1}"], gradients[f"dW{num_layers}"], gradients[f"db{num_layers}"] = self.backward_activation(dAL, caches[num_layers-1], "softmax")
        
        for layer in reversed(range(num_layers-1)):
            dA_prev, dW, db = self.backward_activation(gradients[f"dA{layer+1}"], caches[layer], "relu")
            gradients[f"dA{layer}"] = dA_prev
            gradients[f"dW{layer+1}"] = dW
            gradients[f"db{layer+1}"] = db
        return gradients

    def update_params(self, gradients, learning_rate):
        num_layers = len(self.params) // 2
        for layer in range(num_layers):
            self.params[f"W{layer+1}"] -= learning_rate * gradients[f"dW{layer+1}"]
            self.params[f"b{layer+1}"] -= learning_rate * gradients[f"db{layer+1}"]

    def train(self, X, Y, num_iterations, learning_rate, print_cost=False):
        costs = []
        for iteration in range(num_iterations):
            predictions, caches = self.forward_pass(X)
            cost = self.compute_loss(predictions, Y)
            gradients = self.backward_pass(predictions, Y, caches)
            self.update_params(gradients, learning_rate)
            
            if print_cost and iteration % 100 == 0:
                print(f"Cost after iteration {iteration}: {cost}")
                costs.append(cost)
        return costs

    def predict(self, X):
        predictions, _ = self.forward_pass(X)
        return np.argmax(predictions, axis=0)
