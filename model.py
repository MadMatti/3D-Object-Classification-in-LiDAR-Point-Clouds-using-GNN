from torch import nn
from torch_geometric import nn as gnn
from torch.functional import F
from torch_geometric.nn import global_mean_pool as gnn_global_mean_pool
    
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