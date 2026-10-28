import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluateModel(model, validation_dataset, class_names):

    y_true, y_pred = [], []
    for images, labels in validation_dataset:
        probs = model.predict_on_batch(images)
        y_pred.append(np.argmax(probs, axis = 1))
        y_true.append(labels.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    print(classification_report(
        y_true, y_pred,
        labels=range(len(class_names)),
        target_names = class_names,
        zero_division = 0,
    ))

    fig, ax = plt.subplots(figsize = (16, 16))
    
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels = range(len(class_names)),
        display_labels = class_names,
        xticks_rotation = 45,
        normalize = "true",
        cmap = "Blues",
        ax = ax,
    )
    plt.tight_layout()
    plt.show()