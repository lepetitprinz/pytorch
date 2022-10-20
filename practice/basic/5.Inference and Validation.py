import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5,), (0.5,))]
	)

# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


class Classifier(nn.Module):
	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 64)
		self.fc4 = nn.Linear(64, 10)

	def forward(self, x):
		# make sure input tensor is flattened
		x = x.view(x.shape[0], -1)

		x = F.relu(self,fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.log_softmax(self.fc4(x), dim=1)

		return x


model = Classifier()

images, labels = next(iter(testloader))

# get the class probabilities
ps = torch.exp(model(images))

# make sure the shape is appropriate
print(ps.shape)

top_p, top_class = ps.topk(1, dim=1)

# look at the most likely classes for the first 10 examples
print(top_class[:10, :])

equals = top_class == labels.view(*top_class.shape)

accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item() * 100}%')

#

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        ## TODO: Implement the validation pass and print out the validation accuracy
        print(f'Accuracy: {accuracy.item()*100}%')
