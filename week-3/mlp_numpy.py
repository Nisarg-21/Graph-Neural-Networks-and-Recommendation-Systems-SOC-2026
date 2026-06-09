import numpy as np
from torchvision.datasets import MNIST

train_data = MNIST(root='./data', train=True, download=True)
test_data = MNIST(root='./data', train=False, download=True)

X_train = train_data.data.numpy().reshape(60000, 784) / 255.0
y_train = train_data.targets.numpy()

X_test = test_data.data.numpy().reshape(10000, 784) / 255.0
y_test = test_data.targets.numpy()

def one_hot(y, num_classes=10):
    m = len(y)
    Y = np.zeros((m, num_classes))
    Y[np.arange(m), y] = 1
    return Y

Y_train = one_hot(y_train)  
Y_test = one_hot(y_test)    

np.random.seed(42)

W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))

W2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros((1, 64))

W3 = np.random.randn(64, 10) * 0.01
b3 = np.zeros((1, 10))

def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def cross_entropy_loss(A3, Y):
    m = Y.shape[0]
    A3_clipped = np.clip(A3, 1e-8, 1 - 1e-8)
    loss = -np.sum(Y * np.log(A3_clipped)) / m
    return loss

def forward(X):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)

    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)

    return Z1, A1, Z2, A2, Z3, A3

def backward(X, Y, Z1, A1, Z2, A2, Z3, A3):
    m = X.shape[0]

    dZ3 = A3 - Y
    dW3 = (A2.T @ dZ3) / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

def update(dW1, db1, dW2, db2, dW3, db3, lr):
    global W1, b1, W2, b2, W3, b3
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3

def accuracy(A3, y):
    predictions = np.argmax(A3, axis=1)
    return np.mean(predictions == y)

epochs = 20
lr = 0.01
batch_size = 32

for epoch in range(epochs):
    indices = np.random.permutation(60000)
    X_shuffled = X_train[indices]
    Y_shuffled = Y_train[indices]
    y_shuffled = y_train[indices]

    epoch_loss = 0
    num_batches = 60000 

    for i in range(num_batches):
        X_batch = X_shuffled[i*batch_size:(i+1)*batch_size]
        Y_batch = Y_shuffled[i*batch_size:(i+1)*batch_size]

        
        Z1, A1, Z2, A2, Z3, A3 = forward(X_batch)

        
        loss = cross_entropy_loss(A3, Y_batch)
        epoch_loss += loss

        dW1, db1, dW2, db2, dW3, db3 = backward(X_batch, Y_batch, Z1, A1, Z2, A2, Z3, A3)

        
        update(dW1, db1, dW2, db2, dW3, db3, lr)

    _, _, _, _, _, A3_train = forward(X_train)
    _, _, _, _, _, A3_test = forward(X_test)

    train_acc = accuracy(A3_train, y_train)
    test_acc = accuracy(A3_test, y_test)
    avg_loss = epoch_loss / num_batches

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")

print("\nTraining Complete!")
print(f"Final Test Accuracy: {test_acc*100:.2f}%")