#!/usr/bin/env python
# coding: utf-8

# # Visualizing a tensor
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  

# In[1]:


import torch

t = torch.randn(128,256,12)


# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


plt.figure()
plt.imshow(t[:,:,3], cmap='gray')


# In[ ]:




