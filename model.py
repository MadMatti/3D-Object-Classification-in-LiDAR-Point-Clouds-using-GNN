from torch import nn
from torch_geometric import nn as gnn
from torch.functional import F
from torch_geometric.nn import global_mean_pool as gnn_global_mean_pool

class Yofoold(nn.Module):
    """
    Yofo model
        YO, Find Orientation
    """

    def __init__(self, n_classes):
        super(Yofoold, self).__init__()
        self.n_classes = n_classes
        self.deeply_connected = nn.Sequential(
            # input
            nn.BatchNorm2d(1),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),

            # /2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),

            # /2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),

            # /2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),

            # /2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),

            # /2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),

            # /2
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),

            # flatten
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(p=0.3),
            nn.Flatten(),
            nn.Linear(8*8*16, 32),

            # deep
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, n_classes),

            # output
            nn.Softmax(dim=1)
        )
        self.float()

    def forward(self, x):
        x = self.deeply_connected(x)
        return x

class Yofo(nn.Module):
    """
    Yofo model
        YO, Find Orientation
    """

    def __init__(self, n_classes):
        super(Yofo, self).__init__()
        self.n_classes = n_classes
        self.deeply_connected = nn.Sequential(
            # input
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=0),
            nn.LayerNorm((64, 128, 128)),

            nn.Conv2d(64, 32, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=0),
            nn.LayerNorm((32, 32, 32)),

            nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4), padding=0),
            nn.LayerNorm((16, 8, 8)),

            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.LayerNorm((1, 8, 8)),

            # flatten
            nn.Flatten(),
            nn.Linear(8*8, 32),
            nn.ReLU(),
            nn.LayerNorm(32),

            # deep
            nn.Linear(32, self.n_classes),

            # output
            nn.Sigmoid()
        )
        self.float()

    def forward(self, x):
        x = self.deeply_connected(x)
        return x
    
class GraphClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GraphClassifier, self).__init__()

        self.gnn1 = gnn.GCNConv(-1, hidden_dim)
        self.gnn2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.gnn3 = gnn.GCNConv(hidden_dim, output_dim)

        #self.classifier = nn.Sequential(
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(hidden_dim, output_dim),
        #    nn.Softmax(dim=1)
        #s)
    
    def forward(self, x):
        x, edge_index, batch = x.x, x.edge_index, x.batch

        # Embedding
        x = self.gnn1(x, edge_index)
        x = F.relu(x)
        x = self.gnn2(x, edge_index)
        x = F.relu(x)
        x = self.gnn3(x, edge_index)
        x = gnn_global_mean_pool(x, batch)
        x = F.softmax(x, dim=1)

        # Pooling
        #x = gnn.global_mean_pool(x, batch)

        # Classification
        #x = self.classifier(x)
        return x

class GraphSage(nn.Module):
    '''GraphSAGE'''
    def __init__(self, hidden_dim, output_dim):
        super(GraphSage, self).__init__()

        self.conv1 = gnn.SAGEConv(-1, hidden_dim)
        self.conv2 = gnn.SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = gnn.SAGEConv(hidden_dim, output_dim)

    def forward(self, x):
        x, edge_index, batch = x.x, x.edge_index, x.batch

        # Embedding
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = gnn_global_mean_pool(x, batch)
        x = F.softmax(x, dim=1)

        return x