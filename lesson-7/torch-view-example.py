#!/usr/bin/env python
# coding: utf-8

# # Using view to change the dimensions of torch tensors
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  

# In[1]:


import torch


# Consider a 4x28x28 tensor.  This is akin to having a batch of 4 single channel MNIST image.  Recall that each MNIST image is 28x28. 

# In[2]:


x = torch.ones(4,28,28)
print(x.shape)


# Now lets conver these 4 1D vectors.  Both options below yield the same result.

# In[3]:


y1 = x.view(-1, 28*28) # Option 1
print(y1.shape)

y2 = x.view(4, -1)     # Option 2
print(y2.shape)


# In[ ]:




