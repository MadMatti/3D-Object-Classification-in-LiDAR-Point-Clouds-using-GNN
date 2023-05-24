import random
import time
import torch
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sn
from tqdm import tqdm
from torch import optim
from matplotlib import pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn import metrics as sk_metrics
from sklearn.model_selection import train_test_split
from model import *
from dataset import Dataset
from datasets.kitti import Dataset as KittiDataset
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

SEED = 42

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
def train(model, num_epochs, dataset, device, classes, lr=0.001, scheduler=None, batch_size=64, weight_decay=0.0001):
    plt.show(block=False)
    fig = plt.figure(figsize=(10, 10))

    class_weights = dataset.get_class_weights()

    # Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Define the scheduler
    if scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    elif scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        lr_scheduler = None

    best_acc_value = 0.0

    # Split the dataset into training, validation and test sets
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.15, random_state=42)
    dataset_train, dataset_valid = train_test_split(dataset_train, test_size=0.15, random_state=42)


    print("Training set size:", len(dataset_train))
    print("Validation set size:", len(dataset_valid))
    print("Test set size:", len(dataset_test))
    # print a sample of the dataset
    print(dataset_train[0])

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

    very_start_time = time.time()
    print("Begin training...")
    for epoch in tqdm(range(1, num_epochs + 1)):
        start_time = time.time()
        x_all = []
        y_true_all = []
        y_pred_all = []
        y_conf_all = []

        # Reset the losses
        running_train_loss = 0.0
        running_vall_loss = 0.0

        model.double()
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
            train_loss = loss_fn(y_pred.float(), y_true.to(device))

            train_loss.backward()  # backpropagate the loss
            optimizer.step()  # adjust parameters based on the calculated gradients
        
            running_train_loss += train_loss.item()  # track the loss value

        if lr_scheduler is not None:
            if scheduler == 'ReduceLROnPlateau':
                lr_scheduler.step(train_loss)
            else:
                lr_scheduler.step()

        # Calculate training loss value
        train_loss_value = running_train_loss / len(train_loader)

        # Validation Loop
        with torch.no_grad():
            model.eval()
            for data_batch in valid_loader:
                x = data_batch
                y_true = data_batch.y
                y_pred = model(x.to(device))
                val_loss = loss_fn(y_pred.float(), y_true.to(device))
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
        acc_value = sk_metrics.accuracy_score(y_true_all, y_pred_all)

        cf_matrix = sk_metrics.confusion_matrix(y_true_all, y_pred_all, normalize="true")
        df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)

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
            # save_model()
            best_acc_value = acc_value

            # Print the statistics of the epoch
        print('Completed training epoch', epoch, 'Training Loss is: %.4f' % train_loss_value,
            'Validation Loss is: %.4f' % val_loss_value, 'Accuracy is: %.4f' % acc_value)

        # Print the time required for the epoch
        print('Time taken for epoch %d is %.2f sec\n' % (epoch, time.time() - start_time))
    
    # Print total training time
    print('Training complete in %.2f sec' % (time.time() - very_start_time))

    # Test the model
    with torch.no_grad():
        model.eval()
        x_all = []
        y_true_all = []
        y_pred_all = []
        y_conf_all = []
        for data_batch in test_loader:
            x = data_batch
            y_true = data_batch.y
            y_pred = model(x.to(device))
            x_all.extend(x.x.cpu().numpy())
            y_pred_all.extend(
                y_pred.argmax(dim=1, keepdim=True)
                    .flatten().cpu().numpy())
            y_conf_all.extend(
                y_pred.cpu().numpy().max(axis=1))

            y_true_all.extend(
                y_true.to(device).flatten().cpu().numpy())

        # calculate test accuracy, recall, precision and f1 score
        acc_value_test = sk_metrics.accuracy_score(y_true_all, y_pred_all)
        recall_value_test = sk_metrics.recall_score(y_true_all, y_pred_all, average='weighted')
        precision_value_test = sk_metrics.precision_score(y_true_all, y_pred_all, average='weighted')
        f1_value_test = sk_metrics.f1_score(y_true_all, y_pred_all, average='weighted')

        # print the test statistics
        print('Test Accuracy is: %.4f' % acc_value_test)
        print('Test Recall is: %.4f' % recall_value_test)
        print('Test Precision is: %.4f' % precision_value_test)
        print('Test F1 Score is: %.4f' % f1_value_test)

    plt.close()

    return best_acc_value

def grid_search(num_epochs, dataset, device, classes):
    # define hyperparameters to search
    param_grid = {
        'scheduler': ['StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR'],
        'batch_size': [16, 32, 64, 128],
        'hidden_nodes': [32, 64, 128],
    }

    results = pd.DataFrame(columns=['scheduler', 'batch_size', 'hidden_nodes', 'accuracy'])

    for scheduler in param_grid['scheduler']:
        for batch_size in param_grid['batch_size']:
            for hidden_nodes in param_grid['hidden_nodes']:
                random.seed(SEED)
                np.random.seed(SEED)
                torch.manual_seed(SEED)
                print(f"Testing {scheduler} {batch_size} {hidden_nodes}")
                model = GraphSage(hidden_dim=hidden_nodes, output_dim=len(classes))
                accuracy = train(model, num_epochs, dataset, device, classes, scheduler=scheduler, batch_size=batch_size)
                results = pd.concat([results, pd.DataFrame([[scheduler, batch_size, hidden_nodes, accuracy]], columns=['scheduler', 'batch_size', 'hidden_nodes', 'accuracy'])], ignore_index=True)
                # substiture nan values with 'None'
                results = results.fillna('None')
                results.to_csv('GraphSage_results2.csv', index=False)

    
        


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    matplotlib.use('TkAgg')
    # open log.txt in append mode

    device = torch.device('cpu')

    DATASET_PATH = '/Users/mattiaevangelisti/Documents/KITTI/processed'
    dataset = KittiDataset(DATASET_PATH)
    classes = dataset.classes
    print(classes)

    # GNN = GraphClassifier(hidden_dim=64, output_dim=len(classes))
    # graphSage = GraphSage(hidden_dim=64, output_dim=len(classes))

    print("The model will be running on", device, "device\n")
    #summary(model, (input_dim,))

    # train(graphSage, 10, dataset, device, classes)
    # grid_search(10, dataset, device, classes)

    best_params = {'scheduler': 'ReduceLROnPlateau', 'batch_size': 32, 'hidden_nodes': 32}
    model = GraphSage(hidden_dim=best_params['hidden_nodes'], output_dim=len(classes))
    train(model, 10, dataset, device, classes, scheduler=best_params['scheduler'], batch_size=best_params['batch_size'])