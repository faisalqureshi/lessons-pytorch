#!/usr/bin/env python
# coding: utf-8

# # Semantic Segmentation
# 
# Faisal Qureshi  
# http://www.vclab.ca    
# faisal.qureshi@uoit.ca
# 
# Check out the companion [sseg.py](./sseg.py) file.

# In[56]:


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import sseg_dataset100 as dataset
from torchvision import transforms
from utils import split_train_and_validation
from torch.utils.data import DataLoader, SubsetRandomSampler


# ## Loading data
# 
# Constructs SemanticSegmentationDataset100 from images available in a folder.
# 
# This assumes that images are 'folder/image/*.jpg' and masks are 'folder/ground-truth/*.png'. Images and masks are matched using their filenames.

# In[58]:


folder = '/Users/faisal/Google Drive File Stream/My Drive/Datasets/a-benchmark-for-semantic-image-segmentation/Semantic dataset100'
transform = transforms.Compose([transforms.ToTensor()])

dataset = dataset.SemanticSegmentationDataset100(folder=folder, size=(256,256), transform=transform)


# Inspecting data

# In[18]:


i = 10

image = np.transpose(dataset[i]['img_data'].numpy(), (1,2,0))
mask_raw = dataset[i]['mask_raw'].numpy().squeeze()
mask = dataset[i]['mask'].squeeze()
plt.figure(figsize=(5,5))
plt.subplot(131)
plt.title('Image')
plt.imshow(image)
plt.subplot(132)
plt.title('Mask raw')
plt.imshow(mask_raw)
plt.subplot(133)
plt.title('Mask classes')
plt.imshow(mask)
plt.show()


# Setting up train and validation data loaders

# In[15]:


train_indices, validation_indices = split_train_and_validation(dataset)
train_sampler = SubsetRandomSampler(train_indices)
validation_sampler = SubsetRandomSampler(validation_indices)

train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
valiation_loader = DataLoader(dataset, batch_size=8, sampler=validation_sampler)


# In[55]:


i = 10
x = dataset[i]['img_data']
print(x.shape)
y = x.view(-1, 256*256)
print(y.shape)
m = torch.tensor([10.,100.,1000.]).unsqueeze(-1)
print(m.shape)
torch.sum(torch.pow(y - m, 2), dim=(1))/(256*256)

#y[0,:] = y[0,:] - m[0]
#y[1,:] = y[1,:] - m[1]
#y[2,:] = y[2,:] - m[2]


# ## Model construction

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

class SSegNet(nn.Module):
    """Assumes that input images are 3-channel 256x256."""
    
    def __init__(self):
        super(SSegNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv1 = nn.Conv2d(3, 16, 5)
        
    def forward(self, x):
        xx



        

