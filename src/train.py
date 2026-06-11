import torch
import torch.nn as nn

def train(model, train_loader, val_loader, device, epochs=20):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_acc_list = []
    val_acc_list = []

    for epoch in range(epochs):

        # TRAIN
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = correct / total * 100

        # VALIDATION
        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)

                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total * 100

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch+1}: Train {train_acc:.2f}% | Val {val_acc:.2f}%")

    return train_acc_list, val_acc_list