import keras
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.metrics import confusion_matrix
import os


class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.j = 0
        self.x = []
        self.y = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        self.j += 1
        self.y.append(self.j)

        plt.plot(self.x, self.losses, label="Training loss")
        plt.plot(self.x, self.val_losses, label="Validation loss")
        plt.legend(['Training Loss','Validation Loss'])
        plt.grid()
        plt.savefig('model_loss.png')
        plt.close()
        plt.plot(self.y, self.acc, label="Training accuracy")
        plt.plot(self.y, self.val_acc, label="Validation accuracy")
        plt.legend(['Training Accuracy','Validation Accuracy'])
        plt.grid()
        plt.savefig('model_accuracy.png')
        plt.close()

    @staticmethod
    def plot_history(history, result_dir):
        plt.plot(history.history['acc'], marker='.')
        plt.plot(history.history['val_acc'], marker='.')
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.grid()
        plt.legend(['acc', 'val_acc'], loc='lower right')
        plt.savefig(os.path.join(result_dir, 'model_accuracy_history.png'))
        plt.close()

        plt.plot(history.history['loss'], marker='.')
        plt.plot(history.history['val_loss'], marker='.')
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.grid()
        plt.legend(['loss', 'val_loss'], loc='upper right')
        plt.savefig(os.path.join(result_dir, 'model_loss_history.png'))
        plt.close()

    @staticmethod
    def generate_confusion_matrix(y_true, y_pred, labels=None):
        cmat = confusion_matrix(y_true, y_pred, labels)     # labels: array of shape n_classes
        fig_size=(10,10)
        font_size = 20

        df_cm = pd.DataFrame(cmat,index=labels,columns=labels)
        fig = plt.figure(figsize=fig_size)

        try:
            heatmap = sb.heatmap(df_cm, annot=True, fmt="d")
        except ValueError:
            raise ValueError("Confusion matrix values must be integers")

        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=font_size)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=font_size)
        plt.ylabel('Ground Truth Labels')
        plt.xlabel('Predicted Labels')
        plt.savefig('confusion_matrix.png')
        return fig



