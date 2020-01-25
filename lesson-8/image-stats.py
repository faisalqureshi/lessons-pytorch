#!/usr/bin/env python
# coding: utf-8

# # Computing per channel mean and standard deviation of a collection of RGB images
# 
# Faisal Qureshi
# http://www.vclab.ca    
# faisal.qureshi@uoit.ca
# 
# See also `normalize.py`

# In[26]:


import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse

from compute_image_stats import Accum, mk_filelist, compute_stats


# Specify the folder containing the RGB images.  We assume that we are interested in jpg files.

# In[27]:


folder = '../../../Datasets/a-benchmark-for-semantic-image-segmentation/Semantic dataset100/image'
ext = 'jpg'

#folder = '../../../Datasets/a-benchmark-for-semantic-image-segmentation/Semantic dataset100/ground-truth'
#ext = 'png'

files = mk_filelist(folder, ext)
nfiles = len(files[0])
print('Found {} images'.format(nfiles))


# Compute per-channel mean and standard deviation

# In[28]:


acc = compute_stats(folder, files[0])
print('Mean:\n\t', acc.mean().squeeze())
print('Standard deviation:\n\t', acc.stdev().squeeze())


# No lets *whiten* the data using `torchvision.transforms.Normalize()`.  Whitened data has 0 mean and unit standard deviation.  We will whiten the data whenever we plan one training a network from scratch or fine-tuning a network.

# In[29]:


print('Computing mean and standard deviation of whitened data.')
t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(acc.mean(), acc.stdev())])
acc2 = compute_stats(folder, files[0], t)
print('Mean (0):\n\t', acc2.mean().squeeze())
print('Standard deviation (1):\n\t', acc2.stdev().squeeze())


# Saving mean, variance and standard deviation to a file

# In[30]:


image_stats = { 'folder': folder, 'extension': 'jpg', 'mean': acc.mean(), 'var': acc.var(), 'stdev': acc.stdev() }
print(image_stats)
torch.save(image_stats, 'image_stats.pt')


# Lets try loading them back to see if saving worked

# In[31]:


image_stats_loaded = torch.load('image_stats.pt')
print(image_stats_loaded)


# In[ ]:




