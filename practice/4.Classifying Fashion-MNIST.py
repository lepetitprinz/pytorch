import os

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.path = os.path.join('~', '.pytorch', 'F_MNIST_data')
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.criterion = None
        self.optimizer = None

        self.lr = 0.003
        self.epochs = 5

    def load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))]
        )

        # Download and load the train data
        trainset = datasets.FashionMNIST(self.path, download=True, train=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

        # Download and load the test data
        testset = datasets.FashionMNIST(self.path, download=True, train=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

        return train_loader, test_loader

    def show_sample_image(self, data_loader):
        image, label = next(iter(data_loader))

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

    def define_loss_optimizer(self, lr):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def train_network(self, data_loader, optimizer):
        for e in range(self.epochs):
            running_loss = 0
            for images, labels in data_loader:
                logit = self.forward(images)
                loss = self.criterion(logit, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print(f"Training loss: {running_loss / len(data_loader)}")


# Run
model = Classifier()

train_loader, test_loader = model.load()
model.define_loss_optimizer(lr=model.lr)
model.train_network(
    data_loader=train_loader,
    optimizer=model.optimizer
)