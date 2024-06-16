import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm

from ..utils.utils import get_device

DEVICE = get_device()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_data_loaders():
    batch_size = 32

    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader


def train_cnn(cnn, trainloader):
    learning_rate = 1e-3
    epochs = 10

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    cnn.train()
    for epoch in tqdm(range(epochs)):
        total_loss_cnn = 0
        for batch_idx, (data, labels) in enumerate(trainloader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            # make label is equal to 3
            labels[labels != 3] = 0
            labels[labels == 3] = 1

            optimizer.zero_grad()
            output = cnn(data)

            loss_cnn = loss(output, labels)

            loss_cnn.backward()
            optimizer.step()
            total_loss_cnn += loss_cnn.item()
        print('Epoch: {} Average loss: {:.4f}'.format(epoch + 1, total_loss_cnn / len(trainloader.dataset)))


def test_cnn(cnn, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            labels[labels != 3] = 0
            labels[labels == 3] = 1
            outputs = cnn(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on the test set: {(100 * correct / total):.2f}%')


def main():
    cnn = Net().to(DEVICE)
    trainloader, testloader = get_data_loaders()
    model_path = os.path.join("/accounts/campus/omer_ronen/projects/lso_splines/results/models/cnn")

    w_file = os.path.join(model_path, "w.pth")
    print(w_file)
    if os.path.exists(w_file):
        cnn.load_state_dict(torch.load(w_file))
        cnn.eval()
        test_cnn(cnn, testloader)
        return

    train_cnn(cnn, trainloader)
    test_cnn(cnn, testloader)
    # save the model weights
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(cnn.state_dict(), os.path.join(model_path, "w.pth"))


if __name__ == '__main__':
    main()
