import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_csv('/Users/muzarttuman/Downloads/digit-recognizer/train.csv').values
X = data[:, 1:] / 255.0  # Normalize pixel values
Y = data[:, 0]

# Train-validation split
X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_dev = torch.tensor(X_dev, dtype=torch.float32)
Y_dev = torch.tensor(Y_dev, dtype=torch.long)

# Create data loaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
dev_loader = DataLoader(TensorDataset(X_dev, Y_dev), batch_size=batch_size)

# Neural network model
class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # Input to hidden layer
        self.relu = nn.ReLU() # Activation function
        self.fc2 = nn.Linear(128, 10)  # Hidden to output layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Raw logits; softmax applied later for numerical stability

# Initialize the model, loss, and optimizer
model = DigitRecognizer()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with best model saving
best_accuracy = 0.0
best_model_state = None  # To save the best model's state

# Training loop
epochs = 40
runs = 1  # Number of times to train the model
for run in range(1, runs + 1):
    print(f"Run {run}/{runs}")
    # Reinitialize the model, optimizer, and loss for each run
    model = DigitRecognizer()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

        # Evaluate on validation set after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in dev_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()

        accuracy = correct / total
        print(f"Validation Accuracy after Epoch {epoch+1}: {accuracy * 100:.2f}%")

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = model.state_dict()  # Save the state of the best model
            print("New Best Model Saved!")

# Load the best model for evaluation or further use
model.load_state_dict(best_model_state)
print(f"Best Validation Accuracy: {best_accuracy * 100:.2f}%")

# Test predictions on some samples
def test_prediction(index):
    model.eval()
    image = X_dev[index].unsqueeze(0)
    label = Y_dev[index].item()
    with torch.no_grad():
        output = model(image)
        _, prediction = torch.max(output, 1)
    print(f"Prediction: {prediction.item()}, Label: {label}")

    plt.imshow(image.numpy().reshape(28, 28), cmap='gray')
    plt.title(f"Prediction: {prediction.item()}, Label: {label}")
    plt.show()

# Test on 4 images
test_prediction(0)
test_prediction(1)
test_prediction(2)
test_prediction(3) # Hit 97.60% accuracy
