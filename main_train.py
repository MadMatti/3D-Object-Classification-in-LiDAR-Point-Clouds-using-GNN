import datetime
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
from datasets.modelnet import Dataset as ModelNetDataset
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

SEED = 42

def train(model, num_epochs, dataset, device, scheduler=None, batch_size=64, weight_decay=1e-2, dtype=torch.float32):
    # Save the start time of the training
    very_start_time = time.time()

    # Set the seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Set the device
    device = torch.device(device)

    # Set the dtype
    torch.set_default_dtype(dtype)
    model = model.to(dtype=dtype)
    
    # Set the model
    model = model.to(device)

    # Set the loss function, optimizer and scheduler
    loss_fn = nn.CrossEntropyLoss(weight=dataset.get_class_weights().to(device))
    #loss_fn = nn.NLLLoss(weight=dataset.get_class_weights().to(device))
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay)

    if scheduler == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    elif scheduler == "StepLR":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif scheduler == "CosineAnnealingLR":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    else:
        lr_scheduler = None

    best_acc_value = 0.0

    dataset_train, dataset_test = train_test_split(dataset, test_size=0.15, random_state=42, shuffle=True)
    dataset_train, dataset_valid = train_test_split(dataset_train, test_size=0.15, random_state=42, shuffle=True)

    print("Training set size:", len(dataset_train))
    print("Validation set size:", len(dataset_valid))
    print("Test set size:", len(dataset_test))
    print("Sample from the dataset:", dataset_train[0])

    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True)

    train_loss_list = []
    valid_loss_list = []

    print("Begin training...")
    for epoch in tqdm(range(1, num_epochs + 1)):
        y_true_all = []
        y_pred_all = []
        y_conf_all = []

        running_train_loss = 0.0
        running_val_loss = 0.0

        model.train()
        for data_batch in train_loader:
            data_batch.x = data_batch.x.to(dtype)
            x = data_batch.to(device)
            #x.x = x.x.to(dtype)

            y_pred = model(x)
            train_loss = loss_fn(y_pred, x.y)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_train_loss += train_loss.item()

        if lr_scheduler is not None:
            if scheduler == 'ReduceLROnPlateau':
                lr_scheduler.step(train_loss)
            else:
                lr_scheduler.step()

        train_loss_value = running_train_loss / len(train_loader)
        train_loss_list.append(train_loss_value)

        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                for data_batch in valid_loader:
                    x = data_batch.to(device)
                    x.x = x.x.to(dtype)
                    
                    y_pred = model(x)
                    val_loss = loss_fn(y_pred, x.y)
                    running_val_loss += val_loss.item()
                    
                    y_pred_all.extend(y_pred.argmax(dim=1, keepdim=True).flatten().cpu().numpy())
                    y_conf_all.extend(y_pred.cpu().numpy().max(axis=1))
                    y_true_all.extend(x.y.flatten().cpu().numpy())
            
            val_loss_value = running_val_loss / len(valid_loader)
            valid_loss_list.append(val_loss_value)
            acc_value = sk_metrics.accuracy_score(y_true_all, y_pred_all)


            # Save the model if the accuracy is the best
            if best_acc_value < acc_value:
                # save_model()
                best_acc_value = acc_value

            tqdm.write(f"Completed training epoch {epoch:02d} | " +
                f"Train loss {train_loss_value:.4f} | " +
                f"Valid loss {val_loss_value:.4f} | " +
                f"Accuracy {acc_value:.4f}")

    # Print total training time
    print('Training complete in %.2f sec' % (time.time() - very_start_time))

    # Test the model
    with torch.no_grad():
        model.eval()
        y_true_all = []
        y_pred_all = []
        y_conf_all = []
        for data_batch in test_loader:
            x = data_batch.to(device)
            x.x = x.x.to(dtype)

            y_pred = model(x)

            y_pred_all.extend(
                y_pred.argmax(dim=1, keepdim=True)
                    .flatten().cpu().numpy())
            y_conf_all.extend(
                y_pred.cpu().numpy().max(axis=1))

            y_true_all.extend(
                x.y.flatten().cpu().numpy())

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

    # plt.close()

    # plot the training and validation loss
    #plt.plot(train_loss_list, label='Training Loss', color='seagreen')
    #plt.plot(valid_loss_list, label='Validation Loss', color='indianred')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.legend()
    #plt.grid()
    #plt.show()


    return best_acc_value

def grid_search(epochs, dataset, device, classes, model_class):
    # define hyperparameters to search
    param_grid = {
        'scheduler': [None, 'ReduceLROnPlateau', 'CosineAnnealingLR'],
        'batch_size': [128],
        'hidden_nodes': [256, 128, 64],
        'weight_decay': [0.1, 0.01, 0.001]
    }

    # initialize the results dataframe
    results = pd.DataFrame(columns=list(param_grid.keys()) + ['accuracy'])

    # Expand the grid search
    from sklearn.model_selection import ParameterGrid
    params = list(ParameterGrid(param_grid))

    for param in params:
        print(f"Currrent parameters: {param}")

        # initialize random seed
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # initialize model
        model = model_class(hidden_dim=param['hidden_nodes'], output_dim=len(classes))

        # train the model
        accuracy = train(model, epochs, dataset, device, scheduler=param['scheduler'], batch_size=param['batch_size'], weight_decay=param['weight_decay'])
        
        # save the results
        results = results.append({**param, 'accuracy': accuracy}, ignore_index=True)

        # substiture nan values with 'None'
        results = results.fillna('None')
        results.to_csv('GraphSage_results2.csv', index=False)

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    matplotlib.use('TkAgg')
    # open log.txt in append mode

    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"
    device = torch.device('mps')

    if True:
        DATASET_PATH = '/Users/mattiaevangelisti/Documents/KITTI/processed'
        dataset = KittiDataset(DATASET_PATH)
        classes = dataset.classes
        print(classes)
    
    else:
        DATASET_PATH = '/tmp_workspace/modelnet10_hdf5_2048'
        dataset = ModelNetDataset(DATASET_PATH)
        classes = dataset.classes
        print(classes)

    model_class = GraphSage
    #model_class = GCN

    print("The model will be running on", device, "device\n")

    grid_search(10, dataset, device, classes, model_class)

    #best_params = {'scheduler': 'ReduceLROnPlateau', 'batch_size': 32, 'hidden_nodes': 32}
    #model = GraphSage(hidden_dim=best_params['hidden_nodes'], output_dim=len(classes))
   # train(model, 100, dataset, device, classes, scheduler=None, batch_size=best_params['batch_size'])