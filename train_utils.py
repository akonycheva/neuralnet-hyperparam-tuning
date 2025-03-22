import torch
import matplotlib.pyplot as plt

device = torch.cuda.is_available() and torch.device("cuda") or torch.device("cpu")

def train_model(model, trainloader, testloader, optimizer, criterion, epochs=20, patience=3):
    train_losses, val_accuracies = [], []
    best_acc = 0.0
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        val_acc = evaluate_model(model, testloader)
        val_accuracies.append(val_acc)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Learning Curves: Loss and Accuracy")
    plt.legend()
    plt.savefig("learning_curves.png")
    plt.close()

    return best_acc

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    return acc