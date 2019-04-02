import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data_processing import *


class FcBaseLine(nn.Module):

    def __init__(self, size):
        super(FcBaseLine, self).__init__()
        self.size = size
        self.fc1 = nn.Linear(4, size)
        self.fc2 = nn.Linear(size, 3)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return self.fc2(x)


def train(train_data, eta=0.01, epoch_num=100, test_data=None):
    net = FcBaseLine(10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=eta)
    x = Variable(torch.Tensor(train_data[:, :-1]).float())
    y = Variable(torch.Tensor(train_data[:, -1]).long())
    if test_data is not None:
        test_x = Variable(torch.Tensor(test_data[:, :-1]).float())
        test_y = torch.Tensor(test_data[:, -1]).long()
    net.train()

    for epoch in range(epoch_num):
        optimizer.zero_grad()
        out = net(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        out = torch.max(out.data, 1)[1]
        
        result = torch.max(net(test_x).data, 1)[1]
        out = Variable(out)
        if test_data is not None:
            print('Epoch: ', epoch, '\tloss: ', loss.data[0],
                '\ttrain accuracy: ', sum(out==y).data[0] / len(y),
                '\ttest accuracy: ', sum(result==test_y) / len(test_y))
        else:
            print('Epoch: ', epoch, '\tloss: ', loss.data[0],
                '\ttrain accuracy: ', sum(out==y) / len(y))


def main():
    labelmap = {'Iris-setosa': 0,
                'Iris-versicolor': 1,
                'Iris-virginica': 2}
    data = load_data('data/iris.csv', labelmap=labelmap, header=None)
    train_data, test_data = train_test_split(data)
    train(train_data, 0.1, 1000, test_data)


if __name__ == '__main__':
    main()