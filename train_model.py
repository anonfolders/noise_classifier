import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # for progress bars



def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Train Loss: {train_loss:.3f}, Accuracy: {acc:.2%}")

        if val_loader:
            validate(model, val_loader, device)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    print(f"Validation Accuracy: {correct / total:.2%}")

