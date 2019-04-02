import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import net
import utils

# set some global parameters
learning_rate = 0.01
batch_size = 32
epoch = 1000
optimizer = optim.Adam(lr=learning_rate)
loss_function = nn.CrossEntropyLoss()
root_dir = 'xxxxx/xxx/'
use_cuda = torch.cuda.is_available()
# load your train and test data 
train_data = utils.Loader(root_dir, train_infor, batch_size) 
test_data = utils.Loader(root_dir, test_infor, batch_size) 



def train(save_path):
    """
    train your network.

    Args:
        save_path: the path where your model saved
    """

    #set your network  net = yournet()
    net = vgg19()
    if use_cuda:
        net = net.cuda()
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
    for i in range(epoch):
        total_loss = 0
        for batch_index,(image,label) in enumerate(train_data):
            if use_cuda:
                image, label = image.cuda(), label.cuda()
            optimizer.zero_grad()
            image = Variable(image)
            label = Variable(label)
            outputs = net(image)
            loss = loss_function(outputs,label)
            total_loss += loss
            loss.backward()
            optimizer.step()
        if (i+1)%100 == 0:
            print('loss: '+ str(total_loss))
    torch.save(net.state_dict(),save_path)

     