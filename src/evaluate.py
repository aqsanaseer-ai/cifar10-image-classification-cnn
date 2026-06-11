import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from utils.metrics import save_confusion_matrix

def evaluate(model, loader, device, classes):

    model.eval()
    preds_list, labels_list = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()

            preds_list.extend(preds)
            labels_list.extend(y.numpy())

    acc = accuracy_score(labels_list, preds_list)
    print("\nFinal Accuracy:", acc)
    print(classification_report(labels_list, preds_list))

    cm = confusion_matrix(labels_list, preds_list)
    save_confusion_matrix(cm, classes)