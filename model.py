from __init__ import *
# Define model for CNN in paper

class CNN_I5(nn.Module):

    # xavier initialization of weight
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)


    def __init__(self, num_classes):
        super(CNN_I5, self).__init__()

        """
        CNN building block:
        1. 5x3 conv2d layer, kernel_size=(height, width)
        2. Leaky ReLU
        3. 2x1 max pool
        """
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, padding=(2,1), kernel_size=(5,3), stride=(3,1), dilation=(2,1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, padding=(2,1), kernel_size=(5,3)),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(19200, num_classes)

        self.all_layers = nn.Sequential(
            self.Block1,
            self.Block2,
            self.dropout,
            self.flatten,
            self.fc
        )
        self.all_layers.apply(self.init_weights)
        

    def forward(self, x):
        x = self.all_layers(x)
        #output = F.softmax(x, dim=1)
        return x


class CNN_I5_B3(nn.Module):

    # xavier initialization of weight
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)


    def __init__(self, num_classes):
        super(CNN_I5_B3, self).__init__()

        """
        CNN building block:
        1. 5x3 conv2d layer, kernel_size=(height, width)
        2. Leaky ReLU
        3. 2x1 max pool
        """
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, padding=(2,1), kernel_size=(5,3), stride=(3,1), dilation=(2,1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, padding=(2,1), kernel_size=(5,3)),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, padding=(2,1), kernel_size=(5,3)),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(15360, num_classes)

        self.all_layers = nn.Sequential(
            self.Block1,
            self.Block2,
            self.Block3,
            self.dropout,
            self.flatten,
            self.fc
        )
        self.all_layers.apply(self.init_weights)
        

    def forward(self, x):
        x = self.all_layers(x)
        #output = F.softmax(x, dim=1)
        return x
    
class CNN_I20(nn.Module):

    # xavier initialization of weight
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)


    def __init__(self, num_classes):
        super(CNN_I20, self).__init__()

        """
        CNN building block:
        1. 5x3 conv2d layer, kernel_size=(height, width)
        2. Leaky ReLU
        3. 2x1 max pool
        """
        self.Block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, padding=(3,1), kernel_size=(5,3), stride=(3,1), dilation=(2,1)),
            nn.BatchNorm2d(64, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.Block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, padding=(3,1), kernel_size=(5,3)),
            nn.BatchNorm2d(128, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.Block3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, padding=(2,1), kernel_size=(5,3)),
            nn.BatchNorm2d(256, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1))
        ) 

        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(46080, num_classes)

        self.all_layers = nn.Sequential(
            self.Block1,
            self.Block2,
            self.Block3,
            self.dropout,
            self.flatten,
            self.fc
        )
        self.all_layers.apply(self.init_weights)
        

    def forward(self, x):
        x = self.all_layers(x)
        #output = F.softmax(x, dim=1)
        return x




if __name__ == '__main__':
    # test
    model = CNN_I5(num_classes=2)
    random_data = torch.rand((1, 1, 32, 30))
    result = model(random_data)
    print(result)
    