import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 1. PRUNABLE LINEAR LAYER
# =========================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Gate scores (same shape as weights)
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features) - 2)

    def forward(self, x):
        gates = torch.sigmoid(10 * self.gate_scores)
        pruned_weights = self.weight * gates
        return torch.matmul(x, pruned_weights.t()) + self.bias

    def get_gate_values(self):
        return torch.sigmoid(self.gate_scores)


# =========================
# 2. MODEL
# =========================
class PrunableNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def get_all_gates(self):
        gates = []
        for layer in [self.fc1, self.fc2, self.fc3]:
            gates.append(layer.get_gate_values().view(-1))
        return torch.cat(gates)


# =========================
# 3. DATA
# =========================
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    return trainloader, testloader


# =========================
# 4. SPARSITY LOSS
# =========================
def sparsity_loss(model):
    gates = model.get_all_gates()
    return torch.mean(torch.abs(gates))


# =========================
# 5. TRAINING
# =========================
def train_model(lambda_val, epochs=10):
    model = PrunableNN().to(device)

    trainloader, testloader = load_data()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            cls_loss = criterion(outputs, labels)
            sp_loss = sparsity_loss(model)

            loss = cls_loss + lambda_val * sp_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    acc = evaluate(model, testloader)
    sparsity = calculate_sparsity(model)

    return model, acc, sparsity


# =========================
# 6. EVALUATION
# =========================
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

    return accuracy


# =========================
# 7. SPARSITY CALCULATION
# =========================
def calculate_sparsity(model, threshold=1e-2):
    gates = model.get_all_gates().detach().cpu().numpy()

    total = len(gates)
    pruned = np.sum(gates < threshold)

    sparsity = (pruned / total) * 100
    print(f"Sparsity: {sparsity:.2f}%")

    return sparsity


# =========================
# 8. PLOT
# =========================
def plot_gates(model):
    gates = model.get_all_gates().detach().cpu().numpy()

    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Values")
    plt.ylabel("Frequency")
    plt.show()


# =========================
# 9. MAIN
# =========================
if __name__ == "__main__":
    lambda_values = [0.1, 1, 5]

    results = []

    for lam in lambda_values:
        print(f"\nTraining with lambda = {lam}")

        model, acc, sparsity = train_model(lam, epochs=5)

        results.append((lam, acc, sparsity))

        if lam == 5:
            plot_gates(model)

    print("\nFinal Results:")
    print("Lambda | Accuracy | Sparsity")

    for r in results:
        print(f"{r[0]} | {r[1]:.2f}% | {r[2]:.2f}%")