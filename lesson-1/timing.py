#!/usr/bin/env python
# coding: utf-8

# # Timing in Python
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  

# In[41]:


import time
import torch

a = torch.ones((300000))


# In[42]:


start_time = time.time()
print(a.sum().item())
end_time = time.time()

print('It took {} seconds'.format(end_time - start_time))
print('a is sitting on', a.device)


# In[43]:


start_time = time.time()
sum = 0
for i in range(len(a)):
    sum = sum + a[i]
end_time = time.time()

print('It took {} seconds'.format(end_time - start_time))


# In[44]:


is_cuda = torch.cuda.is_available()

if not is_cuda:
    print('Nothing to do here')
else:
    a_ = a.cuda()
    start_time = time.time()
    print(a_.sum())
    end_time = time.time()
    print('It took {} seconds'.format(end_time - start_time))
    print('a is sitting on', a_.device)


# In[ ]:




