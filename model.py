from torch import nn

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