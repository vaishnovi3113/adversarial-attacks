import torch
import torch.nn as nn
import torch.optim as optim
import os
from models import get_model
from utils import get_dataloaders

def train_model(epochs=10, batch_size=128, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load Data
    trainloader, testloader, classes = get_dataloaders(batch_size=batch_size)

    # Model
    model = get_model().to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}")

    print("Training Finished.")
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    save_path = 'models/cifar10_cnn.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train_model()
