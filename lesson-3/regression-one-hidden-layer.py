#!/usr/bin/env python
# coding: utf-8

# # Regression with one hidden layer
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
get_ipython().run_line_magic('matplotlib', 'inline')

import pprint as pp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ## Test data

# In[2]:


np.random.seed(0)

n_samples = 10
x = np.arange(n_samples)
y = np.sin(2 * np.pi * x / n_samples) * 4

plt.figure(figsize=(4,4))
plt.plot(x, y, 'o')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1,10)
plt.ylim(-5,5)


# ## Torch dataset
# 
# We will create a dataset class that will be used by dataloader to present batches during training.

# In[3]:


from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        sample = {
            'feature': torch.tensor([self.x[idx]], dtype=torch.float32), 
            'label': torch.tensor(np.array([self.y[idx]]), dtype=torch.float32)}
        return sample


# Testing our dataset.  

# In[4]:


import pprint as pp

dataset = MyDataset(x, y)
print('length: ', len(dataset))
for i in range(5):
    pp.pprint(dataset[i])


# Using dataloader to construct batches for training purposes

# In[5]:


dataset = MyDataset(x, y)
batch_size = 4
shuffle = True
num_workers = 4
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
for i_batch, samples in enumerate(dataloader):
    print('\nbatch# = %s' % i_batch)
    print('samples: ')
    pp.pprint(samples)
    break # Otherwise it prints too much stuff


# ## Logistic regression model

# In[6]:


class Regression(nn.Module):
    def __init__(self, input_size):
        super(Regression, self).__init__()
        
        # input layer
        self.linear1 = nn.Linear(input_size, 10)
        self.tan1 = nn.Tanh()
        
        # hidden layer
        self.linear2 = nn.Linear(10, 10)
        self.tan2 = nn.Tanh()
        
        # output layer -- Sigmoid since we are interested in classification
        self.linear3 = nn.Linear(10, 1)
    
    def forward(self, x):
       # input layer
        out = self.tan1(self.linear1(x))
        
        # hidden layer
        out = self.tan2(self.linear2(out))

        # output layer -- No Sigmoid since we are interested in regression
        out = self.linear3(out)
        
        return out


# ## Loss

# In[7]:


import torch.nn as nn
class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        
    def forward(self, predictions, targets):
#        print(predictions.shape)
#        print(targets.shape)
        d = torch.sub(predictions, targets)
        d2 = torch.pow(d, 2)
        d2sum = torch.sum(d2)
        
        return d2sum


# ## Accuracy
# 
# Counting how many predictions were correct.

# In[8]:


def accuracy(predictions, targets):
    d = torch.sub(predictions, targets)
    d2 = torch.pow(d, 2)
    d2sum = torch.sum(d2)
    return d2sum.item()


# ## Training

# In[9]:


import torch.nn.functional as F

model = Regression(1)
criterion = MyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  

dataset = MyDataset(x, y)
batch_size = 16
shuffle = True
num_workers = 4
training_sample_generator = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

num_epochs = 5000
for epoch in range(num_epochs):
    n = 0
    for batch_i, samples in enumerate(training_sample_generator):
        predictions = model(samples['feature'])
        error = criterion(predictions, samples['label'])
        n += accuracy(predictions, samples['label'])
        optimizer.zero_grad()
        error.backward()        
        optimizer.step()
    if epoch % 200 == 0:
        print('epoch %d:' % epoch, error.item())
        print('accuracy', n)


# ## Visualizing results

# In[11]:


x_try = torch.tensor(x, dtype=torch.float32)
print(x_try.unsqueeze(1))

y_try = model(x_try.unsqueeze(1))
yy_try = y_try.detach().squeeze().numpy()
print(yy_try)

plt.figure(figsize=(4,4))
plt.plot(x, y, 'o', label='Ground truth')
plt.plot(x, yy_try, 'x', label='Prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-1,10)
plt.ylim(-5,5)
plt.legend()


# In[ ]:




