#!/usr/bin/env python
# coding: utf-8

# # Solving a system of linear equations using Pytorch
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  

# Solving $A \mathbf{x} = \mathbf{b}$

# In[1]:


import torch

torch.manual_seed(0)

A = torch.randn(4,4)
b = torch.randn(4,1)

print('A:\n', A)
print('b:\n', b)

x = torch.mm(torch.inverse(A), b)
print('x:\n', x)


# Check to see if $\mathbf{x}$ satisfies the equation

# In[2]:


print(torch.mm(A,x))


# In[ ]:




