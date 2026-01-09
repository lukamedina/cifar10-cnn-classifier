from torch import nn

class NeuronalNewtork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatter = nn.Flatten()
        
        self.ConvLayer = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
            
        self.Linear = nn.Sequential(
            nn.Linear(128*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

        
    def forward(self,x):
        x = self.ConvLayer(x)
        x = self.flatter(x)
        x = self.Linear(x)
        return x