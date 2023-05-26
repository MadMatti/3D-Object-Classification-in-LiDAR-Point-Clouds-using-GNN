from torch import nn
from torch_geometric import nn as gnn
from torch.functional import F
from torch.cuda.amp import autocast
    
class GraphClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(GraphClassifier, self).__init__()

        self.gnn1 = gnn.GCNConv(-1, hidden_dim)
        self.gnn2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.gnn3 = gnn.GCNConv(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x, edge_index, batch = x.x, x.edge_index, x.batch

        # Embedding
        x = self.gnn1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.gnn2(x, edge_index)
        x = F.leaky_relu(x)
        x = self.gnn3(x, edge_index)
        x = F.leaky_relu(x)

        x = gnn_global_mean_pool(x, batch)

        x = self.classifier(x)
        
        return x

class GraphSage(nn.Module):
    '''GraphSAGE'''
    def __init__(self, hidden_dim, output_dim):
        super(GraphSage, self).__init__()

        # Normalization
        #self.norm = gnn.GraphNorm(3)

        self.norm = gnn.BatchNorm(3)

        # GraphSAGE
        self.conv1 = gnn.SAGEConv(-1, hidden_dim)
        self.conv2 = gnn.SAGEConv(hidden_dim, hidden_dim//4)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//4),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim//4, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x, edge_index, batch = x.x, x.edge_index, x.batch

        # Normalization
        x = self.norm(x)

        # Embedding
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        # Pooling
        x = gnn.global_max_pool(x, batch)

        x = self.classifier(x)

        return x