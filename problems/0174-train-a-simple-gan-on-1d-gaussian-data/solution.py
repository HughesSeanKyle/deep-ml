import numpy as np

def train_gan(mean_real: float, std_real: float, latent_dim: int = 1, hidden_dim: int = 16, learning_rate: float = 0.001, epochs: int = 5000, batch_size: int = 128, seed: int = 42):
    """
    Train a simple GAN to learn a 1D Gaussian distribution.

    Args:
        mean_real: Mean of the target Gaussian
        std_real: Std of the target Gaussian
        latent_dim: Dimension of the noise input to the generator
        hidden_dim: Hidden layer size for both networks
        learning_rate: Learning rate for gradient descent
        epochs: Number of training epochs
        batch_size: Training batch size
        seed: Random seed for reproducibility

    Returns:
        gen_forward: A function that takes z and returns (x_gen, a1, h1)
    """
    # 1. Set seed and initialize weights in the exact order requested
    np.random.seed(seed)
    
    W1_G = np.random.normal(0, 0.01, (latent_dim, hidden_dim))
    W2_G = np.random.normal(0, 0.01, (hidden_dim, 1))
    W1_D = np.random.normal(0, 0.01, (1, hidden_dim))
    W2_D = np.random.normal(0, 0.01, (hidden_dim, 1))
    
    # Biases start at zero
    b1_G = np.zeros((1, hidden_dim))
    b2_G = np.zeros((1, 1))
    b1_D = np.zeros((1, hidden_dim))
    b2_D = np.zeros((1, 1))

    # Helper function for Sigmoid
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # 2. Main Training Loop
    for epoch in range(epochs):
        # --- GENERATE DATA FOR DISCRIMINATOR STEP ---
        x_real = np.random.normal(mean_real, std_real, batch_size).reshape((batch_size, 1))
        z_D = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Generator Forward Pass (for D step)
        h1_G_D = np.matmul(z_D, W1_G) + b1_G
        a1_G_D = np.maximum(0, h1_G_D)
        x_fake_D = np.matmul(a1_G_D, W2_G) + b2_G
        
        # Discriminator Forward Pass on Real
        h1_D_real = np.matmul(x_real, W1_D) + b1_D
        a1_D_real = np.maximum(0, h1_D_real)
        out_D_real = sigmoid(np.matmul(a1_D_real, W2_D) + b2_D)
        
        # Discriminator Forward Pass on Fake
        h1_D_fake = np.matmul(x_fake_D, W1_D) + b1_D
        a1_D_fake = np.maximum(0, h1_D_fake)
        out_D_fake = sigmoid(np.matmul(a1_D_fake, W2_D) + b2_D)
        
        # Discriminator Backward Pass
        # Gradient of loss w.r.t out_D_real: -1 / (B * out_D_real)
        # Gradient of out_D_real w.r.t pre-activation: out_D_real * (1 - out_D_real)
        # combined: d(out_D_real_pre) = (out_D_real - 1) / B
        dout_D_real = (out_D_real - 1.0) / batch_size
        dW2_D_real = np.matmul(a1_D_real.T, dout_D_real)
        db2_D_real = np.sum(dout_D_real, axis=0, keepdims=True)
        da1_D_real = np.matmul(dout_D_real, W2_D.T)
        dh1_D_real = da1_D_real * (h1_D_real > 0)
        dW1_D_real = np.matmul(x_real.T, dh1_D_real)
        db1_D_real = np.sum(dh1_D_real, axis=0, keepdims=True)
        
        # Gradient of loss w.r.t out_D_fake: 1 / (B * (1 - out_D_fake))
        # Gradient of out_D_fake w.r.t pre-activation: out_D_fake * (1 - out_D_fake)
        # combined: d(out_D_fake_pre) = out_D_fake / B
        dout_D_fake = out_D_fake / batch_size
        dW2_D_fake = np.matmul(a1_D_fake.T, dout_D_fake)
        db2_D_fake = np.sum(dout_D_fake, axis=0, keepdims=True)
        da1_D_fake = np.matmul(dout_D_fake, W2_D.T)
        dh1_D_fake = da1_D_fake * (h1_D_fake > 0)
        dW1_D_fake = np.matmul(x_fake_D.T, dh1_D_fake)
        db1_D_fake = np.sum(dh1_D_fake, axis=0, keepdims=True)
        
        # Update Discriminator Parameters via Plain Gradient Descent
        W2_D -= learning_rate * (dW2_D_real + dW2_D_fake)
        b2_D -= learning_rate * (db2_D_real + db2_D_fake)
        W1_D -= learning_rate * (dW1_D_real + dW1_D_fake)
        b1_D -= learning_rate * (db1_D_real + db1_D_fake)
        
        # --- GENERATE FRESH NOISE FOR GENERATOR STEP ---
        z_G = np.random.normal(0, 1, (batch_size, latent_dim))
        
        # Generator Forward Pass
        h1_G = np.matmul(z_G, W1_G) + b1_G
        a1_G = np.maximum(0, h1_G)
        x_fake_G = np.matmul(a1_G, W2_G) + b2_G
        
        # Discriminator Forward Pass on fresh Fake
        h1_D_G = np.matmul(x_fake_G, W1_D) + b1_D
        a1_D_G = np.maximum(0, h1_D_G)
        out_D_G = sigmoid(np.matmul(a1_D_G, W2_D) + b2_D)
        
        # Generator Backward Pass (using updated Discriminator)
        # Non-saturating Loss LG = -1/B * sum(log(D(G(z))))
        # Combined gradient w.r.t out_D_G pre-activation: (out_D_G - 1) / B
        dout_D_G = (out_D_G - 1.0) / batch_size
        da1_D_G = np.matmul(dout_D_G, W2_D.T)
        dh1_D_G = da1_D_G * (h1_D_G > 0)
        dx_fake_G = np.matmul(dh1_D_G, W1_D.T)
        
        # Backprop through Generator layers
        dW2_G = np.matmul(a1_G.T, dx_fake_G)
        db2_G = np.sum(dx_fake_G, axis=0, keepdims=True)
        da1_G = np.matmul(dx_fake_G, W2_G.T)
        dh1_G = da1_G * (h1_G > 0)
        dW1_G = np.matmul(z_G.T, dh1_G)
        b1_G_grad = np.sum(dh1_G, axis=0, keepdims=True)
        
        # Update Generator Parameters
        W2_G -= learning_rate * dW2_G
        b2_G -= learning_rate * db2_G
        W1_G -= learning_rate * dW1_G
        b1_G -= learning_rate * b1_G_grad

    # 3. Define the return generator forward function
    def gen_forward(z):
        h1 = np.matmul(z, W1_G) + b1_G
        a1 = np.maximum(0, h1)
        x_gen = np.matmul(a1, W2_G) + b2_G
        return x_gen, a1, h1

    return gen_forward