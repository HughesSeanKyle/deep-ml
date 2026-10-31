# --- cell 1 ---
import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Required architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''
    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.1):
        '''
        Initialize network parameters using He (Kaiming) initialization.
        '''
        self.lr = lr
        
        # He initialization for ReLU layer: variance = 2 / input_size
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        # Xavier/He initialization for output layer: variance = 2 / hidden_size
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Dictionary to cache values for backpropagation
        self.cache = {}
        
    def forward(self, X):
        '''
        Forward pass through the network.
        '''
        # Layer 1: Linear Transformation
        z1 = X @ self.W1 + self.b1
        
        # Activation 1: ReLU
        h1 = np.maximum(0, z1)
        
        # Layer 2: Linear Transformation
        z2 = h1 @ self.W2 + self.b2
        
        # Activation 2: Numerically Stable Softmax
        # Subtract max for stability against overflow
        shift_z2 = z2 - np.max(z2, axis=1, keepdims=True)
        exps = np.exp(shift_z2)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        # Cache internal activations needed for the backward pass
        self.cache = {'X': X, 'z1': z1, 'h1': h1, 'z2': z2}
        
        return probs
    
    def backward(self, X, y, probs):
        '''
        Backward pass - compute analytical gradients and update parameters.
        '''
        N = X.shape[0]
        
        # 1. Compute Cross-Entropy Loss (with epsilon to prevent log(0))
        core_probs = probs[np.arange(N), y]
        loss = -np.mean(np.log(core_probs + 1e-12))
        
        # 2. Gradient of loss w.r.t. pre-softmax logits (z2)
        # dL/dz2 = (probs - y_one_hot) / N
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1.0
        dz2 /= N  # Average gradient over the batch size
        
        # 3. Backprop through Layer 2 (Output Linear Layer)
        h1 = self.cache['h1']
        dW2 = h1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Gradient flowing back to hidden activations
        dh1 = dz2 @ self.W2.T
        
        # 4. Backprop through ReLU activation
        z1 = self.cache['z1']
        dz1 = dh1.copy()
        dz1[z1 <= 0] = 0.0  # Derivative of ReLU is 0 where input was negative/zero
        
        # 5. Backprop through Layer 1 (Hidden Linear Layer)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # 6. Parameter Updates via Stochastic Gradient Descent (SGD)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        
        return loss
    
    def train_step(self, X, y):
        '''
        Complete training step: forward + backward + update.
        '''
        probs = self.forward(X)
        loss = self.backward(X, y, probs)
        return loss
    
    def predict(self, X):
        '''
        Predict class labels.
        '''
        probs = self.forward(X)
        return np.argmax(probs, axis=1)