import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from train_utils import device

def get_confusion_matrix(model, testloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix of Best Model')
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix.png'")

def plot_sample_predictions(model, testloader, num_samples=5):
    model.eval()
    images, labels, preds = [], [], []
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            images.extend(data.cpu().numpy())
            labels.extend(target.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
            if len(images) >= num_samples:
                break

    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap="gray")
        plt.title(f"True: {class_names[labels[i]]}\nPred: {class_names[preds[i]]}")
        plt.axis("off")
    plt.savefig("sample_predictions.png")
    plt.close()
    print("Sample predictions saved as 'sample_predictions.png'")