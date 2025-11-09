"""
Train a simple MNIST MLP and export weights for the WebGPU demo

Requirements:
  pip install torch torchvision

Usage:
  python train_mnist.py

This will:
1. Train a 784→128→10 MLP on MNIST
2. Export weights to ../public/models/mnist-mlp.json
3. Takes ~2-3 minutes on CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
from pathlib import Path

class MNISTModel(nn.Module):
    """Same architecture as the WebGPU demo"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train():
    print("🔥 Training MNIST MLP Model")
    print("=" * 50)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    print("📥 Downloading MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Model
    model = MNISTModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("\n🏋️  Training (3 epochs)...")
    for epoch in range(3):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}/3 - Batch {batch_idx+1}/{len(train_loader)} - "
                      f"Loss: {loss.item():.4f} - Acc: {100*correct/total:.2f}%")

        print(f"✅ Epoch {epoch+1} complete - Avg Loss: {total_loss/len(train_loader):.4f} - "
              f"Train Acc: {100*correct/total:.2f}%")

    # Test
    print("\n🧪 Testing...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_acc = 100 * correct / total
    print(f"✅ Test Accuracy: {test_acc:.2f}%\n")

    return model, test_acc

def export_weights(model, test_acc):
    print("💾 Exporting weights...")

    # Extract weights
    fc1_weight = model.fc1.weight.detach().cpu().numpy().flatten().tolist()
    fc1_bias = model.fc1.bias.detach().cpu().numpy().tolist()
    fc2_weight = model.fc2.weight.detach().cpu().numpy().flatten().tolist()
    fc2_bias = model.fc2.bias.detach().cpu().numpy().tolist()

    # Create export structure
    weights = {
        "model_type": "mnist-mlp",
        "architecture": {
            "input_size": 784,
            "hidden_size": 128,
            "output_size": 10
        },
        "weights": {
            "fc1_weight": fc1_weight,  # Shape: 128x784 (flattened)
            "fc1_bias": fc1_bias,      # Shape: 128
            "fc2_weight": fc2_weight,  # Shape: 10x128 (flattened)
            "fc2_bias": fc2_bias       # Shape: 10
        },
        "metadata": {
            "test_accuracy": test_acc,
            "source": "pytorch_training",
            "framework": "PyTorch",
            "training_epochs": 3,
            "optimizer": "Adam",
            "learning_rate": 0.001
        }
    }

    # Save to file
    output_path = Path(__file__).parent.parent / 'public' / 'models' / 'mnist-mlp.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)

    print(f"✅ Weights exported to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"   Test accuracy: {test_acc:.2f}%")

if __name__ == '__main__':
    model, test_acc = train()
    export_weights(model, test_acc)

    print("\n🎉 Done! You can now use the trained model in the WebGPU demo.")
    print("   Run: bun run dev")
