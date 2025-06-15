import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def show_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

def show_misclassified(model, test_loader, device, max_examples=5):
    misclassified = []

    model.eval()
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)

            for i in range(len(preds)):
                if preds[i] != labels[i] and len(misclassified) < max_examples:
                    misclassified.append((data[i].cpu(), preds[i].cpu(), labels[i].cpu()))

    for img, pred, true in misclassified:
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Predicted: {pred}, True: {true}")
        plt.axis('off')
        plt.show()