import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
from torch import optim
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn import metrics as sk_metrics
from model import *
from dataset import Dataset

CLASSES = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]

import torch_geometric
print(torch_geometric.__version__)

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
    fig = plt.figure(figsize=(10, 10))

    class_weights = dataset.get_class_weights()

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    best_acc_value = 0.0

    dataset_train, dataset_valid = dataset, dataset
    print("Training set size:", len(dataset_train))
    print("Validation set size:", len(dataset_valid))
    # print a sample of the dataset
    print(dataset_train[0])


    print("Training set size:", len(dataset_train))
    print("Validation set size:", len(dataset_valid))

    train_loader = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=64, shuffle=True)

    print("Begin training...")
    for epoch in range(1, num_epochs + 1):
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
        print(len(y_true_all), len(y_pred_all))
        acc_value = sk_metrics.accuracy_score(y_true_all, y_pred_all)

        cf_matrix = sk_metrics.confusion_matrix(y_true_all, y_pred_all, normalize="true")
        df_cm = pd.DataFrame(cf_matrix, index=CLASSES, columns=CLASSES)

        plt.subplot(2, 1, 1)

        sn.heatmap(df_cm, annot=True)

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
        plt.title(f"Confusion Matrix {mean_conf_f:.2f} {mean_conf_t:.2f}")

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

    plt.close()


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    # open log.txt in append mode

    device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DATASET_PATH = '/Users/hamzaali/Workspace/3D-Object-Detection/3D-Object-Detection-in-LiDAR-Point-Clouds-using-GNN'
    dataset = Dataset(DATASET_PATH)

    model = GraphClassifier(hidden_dim=64, output_dim=len(CLASSES))

    print("The model will be running on", device, "device\n")
    #summary(model, (input_dim,))

    train(model, 1000, dataset, device)