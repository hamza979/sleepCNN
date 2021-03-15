#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
Epochs=torch.load('SleepEDF/Epochs.pt') 
Labels=torch.load('SleepEDF/Labels.pt') 


# In[4]:


Epochs.shape


# In[5]:


Labels.shape


# In[6]:


import torch.utils.data as data_utils

train = data_utils.TensorDataset(Epochs, Labels.long())
train_loader = data_utils.DataLoader(train, batch_size=747, shuffle=True)


# In[7]:


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 5)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 747, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 747)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net().cuda()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)


# In[ ]:


for epoch in range(64):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].cuda(),data[1].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 190 == 189:
            print('[Epoch %d] loss: %.3f' %
                  (epoch + 1, running_loss / 190))
            running_loss = 0.0

print('Finished Training')


# In[ ]:




