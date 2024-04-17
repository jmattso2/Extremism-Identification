import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output, _ = self.lstm(input)
        output = self.fc(output[:, -1, :])
        output = self.sigmoid(output)
        return output

# Load data from CSV
data = pd.read_csv('final_data.csv')

# Extract embeddings from stored strings
data['embeddings'] = data['embeddings'].apply(lambda x: np.fromstring(x[1:-1], sep=', '))

# Convert embeddings to tensors
X = torch.tensor(data['embeddings'].tolist(), dtype=torch.float32)

# Define labels
y = torch.tensor(data['label'], dtype=torch.bool)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# Define hyperparameters
input_size = X.shape[1]
hidden_size = 128
output_size = 1
learning_rate = 0.001
num_epochs = 10

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.unsqueeze(2))  # LSTM expects input with shape (seq_len, batch, input_size)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluation
model.eval()
with torch.no_grad():
    outputs = model(X_test.unsqueeze(2))
    predicted = (outputs.squeeze() > 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    print(f'Test Accuracy: {accuracy.item()}')