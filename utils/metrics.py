import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_accuracy_plot(train_acc, val_acc):
    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/accuracy.png")
    plt.close()


def save_confusion_matrix(cm, classes):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=classes,
                yticklabels=classes,
                cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()