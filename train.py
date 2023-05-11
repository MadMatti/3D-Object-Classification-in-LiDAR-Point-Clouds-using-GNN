import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
import time
from tqdm import tqdm
from torch import optim
from torch.utils.data import random_split
from torchsummary import summary
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn import metrics as sk_metrics
from model import *
from dataset import Dataset

CLASSES = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

import torch_geometric
print(torch_geometric.__version__)
import torch_geometric
print(torch_geometric.__version__)

def plot_curves(train_loss_list, val_loss_list, acc_list, title, training_time):
    # plot train and validation loss on the same plot and accurcy on another
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.tight_layout(pad=3.0)

    axs[0].plot(train_loss_list, label="Train Loss", color='seagreen')
    axs[0].plot(val_loss_list, label="Validation Loss", color='indianred')
    axs[0].set_title("Train and Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(acc_list, label="Accuracy", color='seagreen')
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    plt.figtext(0.5, 0.01, "Training time: {:.2f} seconds".format(training_time), ha="center", fontsize=12)


    plt.suptitle("Learning curves for " + title, y=0.98)
    plt.subplots_adjust(top=0.85)
    plt.show()

def evaluate(model, dataset_test, device):
    test_loader = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)
    
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        model.eval()
        for data_batch in test_loader:
            x = data_batch
            y_true = data_batch.y
            y_pred = model(x.to(device))

            y_true_all.extend(
                y_true.to(device).flatten().cpu().numpy())
            
            y_pred_all.extend(
                y_pred.argmax(dim=1, keepdim=True)
                    .flatten().cpu().numpy())

    accuracy = sk_metrics.accuracy_score(y_true_all, y_pred_all)
    precision = sk_metrics.precision_score(y_true_all, y_pred_all, average='weighted')
    recall = sk_metrics.recall_score(y_true_all, y_pred_all, average='weighted')
    f1_score = sk_metrics.f1_score(y_true_all, y_pred_all, average='weighted')

    return accuracy, precision, recall, f1_score

def save_model():
    path = "./last.pt"
    torch.save(model.state_dict(), path)


def accuracy(x, y):
    if torch.argmax(y) == torch.argmax(x):
        return True
    return False


def predict(model, x):
    return torch.argmax(model(x))


# Training Function
def train(model, num_epochs, dataset, device):
    plt.show(block=False)
    # fig = plt.figure(figsize=(10, 10))

    class_weights = dataset.get_class_weights()

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_acc_value = 0.0

    train_size = int(0.75 * len(dataset))
    val_test_size = len(dataset) - train_size
    val_size = test_size = int(val_test_size / 2)
    train_dataset, val_test_dataset = random_split(dataset, [train_size, val_test_size])
    val_dataset, test_dataset = random_split(val_test_dataset, [val_size, test_size])

    print("Training set size:", len(train_dataset))
    print("Validation set size:", len(val_dataset))

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)

    train_loss_list = []
    val_loss_list = []
    acc_list = []

    print("Begin training...")
    start_time = time.time()
    for epoch in tqdm(range(1, num_epochs + 1)):
        x_all = []
        y_true_all = []
        y_pred_all = []
        y_conf_all = []

        # Reset the losses
        running_train_loss = 0.0
        running_vall_loss = 0.0

        # Training Loop
        model.train()

        for data_batch in train_loader:
            x = data_batch
            y_true = data_batch.y
            # for data in enumerate(train_loader, 0):
            optimizer.zero_grad()  # zero the parameter gradients

            # predict output from the model
            y_pred = model(x.to(device))

            # calculate loss for the predicted output
            train_loss = loss_fn(y_pred, y_true.to(device))

            train_loss.backward()  # backpropagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
            running_train_loss += train_loss.item()  # track the loss value

        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_loader)

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for data_batch in valid_loader:
                x = data_batch
                y_true = data_batch.y
                y_pred = model(x.to(device))
                val_loss = loss_fn(y_pred, y_true.to(device))
                running_vall_loss += val_loss.item()
                x_all.extend(x.x.cpu().numpy())
                #print(y_true)

                y_pred_all.extend(
                    y_pred.argmax(dim=1, keepdim=True)
                        .flatten().cpu().numpy())
                y_conf_all.extend(
                    y_pred.cpu().numpy().max(axis=1))

                y_true_all.extend(
                    y_true.to(device).flatten().cpu().numpy())

        val_loss_value = running_vall_loss / len(valid_loader)
        # print(len(y_true_all), len(y_pred_all))
        acc_value = sk_metrics.accuracy_score(y_true_all, y_pred_all)

        # cf_matrix = sk_metrics.confusion_matrix(y_true_all, y_pred_all, normalize="true")
        # df_cm = pd.DataFrame(cf_matrix, index=CLASSES, columns=CLASSES)

        # plt.subplot(2, 1, 1)

        # sn.heatmap(df_cm, annot=True)

        wrong = []
        correct = []
        #print(len(x_all), len(y_pred_all), len(y_true_all), len(y_conf_all))
        for i in range(0, len(y_pred_all)):
            if y_pred_all[i] != y_true_all[i]:
                wrong.append(i)
            else:
                correct.append(i)

        mean_conf_f = np.mean(np.array(y_conf_all)[wrong])
        mean_conf_t = np.mean(np.array(y_conf_all)[correct])
        # plt.title(f"Confusion Matrix {mean_conf_f:.2f} {mean_conf_t:.2f}")

        # for i in wrong[:7]:
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(x_all[i].squeeze())
        #     plt.title(f"true:{CLASSES[y_true_all[i]]} pred:{CLASSES[y_pred_all[i]]} {y_conf_all[i]:.2f}")

        #     plt.show(block=False)
        #     plt.pause(0.5)


        # Save the model if the accuracy is the best
        if best_acc_value < acc_value:
#            save_model()
            best_acc_value = acc_value

            # Print the statistics of the epoch
        print('Completed training epoch', epoch, 'Training Loss is: %.4f' % train_loss_value,
            'Validation Loss is: %.4f' % val_loss_value, 'Accuracy is: %.4f' % acc_value)
        train_loss_list.append(train_loss_value)
        val_loss_list.append(val_loss_value)
        acc_list.append(acc_value)

    training_time = time.time() - start_time

    # Plot the loss and accuracy values
    title = "GraphSage"
    plot_curves(train_loss_list, val_loss_list, acc_list, title, training_time)

    accuracy, precision, recall, f1 = evaluate(model, test_dataset, device)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    plt.close()


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    # open log.txt in append mode

    device = torch.device('cpu')

    DATASET_PATH = '/Users/mattiaevangelisti/Documents/'
    dataset = Dataset(DATASET_PATH)

    model = GraphClassifier(hidden_dim=64, output_dim=len(CLASSES))
    model2 = GraphSage(hidden_dim=64, output_dim=len(CLASSES))

    print("The model will be running on", device, "device\n")
    #summary(model, (input_dim,))

    train(model2, 10, dataset, device)
