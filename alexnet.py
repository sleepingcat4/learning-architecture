from torch import nn, flatten

# defining the classs and treat the data as a param and retrive the output after 
# performing the calculations

class AlexNet(nn.Module):
    
    def __init__(self, channels=3, first_fc_in_features=9216):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=96,
                kernel_size=(11, 11),
                stride=4,
                padding=0,

            ),
            nn.ReLU(),
            # now Local Response Norm like original paper
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            
            # second conv layer
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            # Third layer
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3)),
            nn.ReLU(),

            # Fourth conv layer
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3)), 
            nn.ReLU(),  

            # fifth Conv layer
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),

            # First Fully Connected layer
            nn.Linear(in_features=first_fc_in_features, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(),

            # Second fully connected layer
            nn.Linear(in_features=4096, out_features=4096),
            nn.Dropout(p=0.5),
            nn.ReLU(),

            # Thrid fully connected layer
            nn.Linear(in_features=4096, out_features=1000),
            nn.LogSoftmax(dim=1),



        ) # it helps the model architecture binding simple
        # we use a kernel of 11X11 and stride 4 (each step kernel moves by 4 pixels)


        def forward(self, x):
            # forward determines how the computation will be performed
            x = self.model(x)