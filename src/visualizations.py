import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

def plot_graph(results, size, name = 'None'):

    accuracy = results.history['accuracy']
    val_accuracy = results.history['val_accuracy']
    loss = results.history['loss']
    val_loss = results.history['val_loss']
    epochs = range(len(accuracy))

    temp = name.replace(' ', '_')
    plt.plot(epochs, accuracy, 'bx', label='Training_accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title(label= f'{name} Accuracy vs Epochs')
    plt.legend()
    plt.savefig(f'Logs/{size}/{temp}_accuracy.png')
    plt.figure()

    plt.plot(epochs, loss, 'bx', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(label= f'{name} Loss vs Epochs')
    plt.legend()
    plt.savefig(f'Logs/{size}/{temp}_loss.png')
    plt.figure()


def plot_cm(model, name, test):
    predictions = (model.predict(test) >= 0.5).astype(int)
    cm = confusion_matrix(test.labels, predictions)
    temp = name.replace(' ', '_')

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'Logs/64x64/{temp}_cm.png')
    plt.show()
    return predictions
