#!/usr/bin/env python
# coding: utf-8

# # The basics
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  
# 
# You can find excellent documentation for Pytorch at [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

# In[1]:


import torch


# In[2]:


t = torch.tensor([[1,2,3],[4,5,6]])
t


# In[3]:


t.t()


# In[4]:


print(t.permute(-1,0))


# In[5]:


print('shape:',t.shape)
print('size:',t.size())
print('dim:',t.dim())
print('type:',t.type())
print('num elements:', torch.numel(t))
print('device (cpu or gpu):', t.device)


# ## Changing tensor views

# In[6]:


t = torch.tensor([[1,2,3],[4,5,6]])

print(t)
print('View example:\n', t.view(1,-1))
print('View example:\n', t.view(-1,1))
print('View example:\n', t.view(3,2))


# ## Slicing

# In[7]:


# First row
print('Matlab or numpy style slicing:\n',t[1,:])

# Second column
print('Matlab or numpy style slicing:\n',t[:,1])

# Lower right most element
print('Matlab or numpy style slicing:\n',t[-1,-1])

# Lower right most 1 x 1 submatrix
print('Matlab or numpy style slicing:\n',t[-1:,-1:])

# Lower right most 2 x 2submatrix
print('Matlab or numpy style slicing:\n',t[-2:,-2:])


# ## Torch and Numpy

# In[8]:


import numpy as np

a = np.random.randn(2, 4)

# Constructing a torch tensor from a numpy array
t = torch.from_numpy(a)

# Back to numpy
b = t.numpy()

print('numpy:\n', a)
print('torch:\n', t)
print(type(a))
print(type(t))
print(type(b))


# ## Some common schemes for tensor creation

# In[9]:


# A zero tensor
print('Zero tensor:\n', torch.zeros(2,3,4))

# A one tensor
print('Ones tensor:\n', torch.ones(2,3,4))

# Some random tensors
print('Random - Uniform, between 0 and 1):\n', torch.rand(2,3,4))
print('Random - Normal, mean 0 and standard deviation 1 :\n', torch.randn(2,3,4))


# ## Tensor concatenation

# In[10]:


t1 = torch.tensor([[1,2,3],[4,5,7]])
t2 = torch.tensor([[8,9,10],[11,12,13]])

print('t1:\n', t1)
print('t2:\n', t2)

# Concatenating two tensors along 0 (first, rows in this case) dimension
print(torch.cat((t1,t2),0))

# Concatenating two tensors along 1 (second, columns in this case) dimension
print(torch.cat((t1,t2),1))


# In[11]:


t = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

# Computing cummulative sum
print(t)
print(t.cumsum(-1))
print(t.cumsum(-2))


# ## Adding a dimension to a tensor
# 
# ![Tensor Concatnation](tensor-concatination.png)

# In[12]:


# First, the numpy way
x = np.random.rand(3,4)
print('Before', x.shape)
x = x[None,:,:]
print('After', x.shape)

# Next, the torch way
t = torch.rand(3,4)
print('Before:', t.shape)
t1 = t.unsqueeze(0)
print('After:', t1.shape)

# Say we get another 3x4 matrix (say a grayscale image or a frame)
t2 = torch.rand(3,4)

# Say we want to combine t1 and t2, such that the first dimension
# iterates over the frames

t.unsqueeze_(0) # inplace unsqueeze, we just added a dimension
t2.unsqueeze_(0)

t_and_t2 = torch.cat((t1,t2),0) # The first dimension is the 
print(t_and_t2.shape)


# ## Testing for equality

# In[13]:


t1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
t2 = torch.tensor([[1,20,3],[40,5,6],[7,8,9]])

# Element wise equality test
print(torch.eq(t1,t2))


# In[14]:


t = torch.rand(2,3)
print('t:\n', t)

# Log
print('log t:\n', torch.log(t))

# Negative
print('neg t:\n', torch.neg(t))

# Power
print('power t:\n', torch.pow(t, 2))

# Reciprocal
print('reciprocal t:\n', torch.reciprocal(t))

# Round
print('round t:\n', torch.round(t))

# Sigmoid
print('sigmoid t:\n', torch.sigmoid(t))

# Sign
print('sign t:\n', torch.sign(t))

# sqrt
print('sqrt t:\n', torch.sqrt(t))

# argmax, along 0-th dimension (that moves along the rows)
print('argmax t:\n', torch.argmax(t, 0))

# mean, along 1-th dimension (that moves along the columns)
print('mean t:\n', torch.mean(t, 1))


# ## Vector and Matrix products

# In[15]:


t1 = torch.tensor([0,1,0])
t2 = torch.tensor([1,0,0])
print(t1.cross(t2))


# In[16]:


t1 = torch.randn(4,3)
t2 = torch.randn(4,3)

# Row-wise vector cross product 
t1_cross_t2 = t1.cross(t2)

# Confirm that the dot products of the result with 
# the corresponding vectors in t1 and t2 is 0
for i in range(t1.size(0)):
    print('Row %d' % i, t1[i,:].dot(t1_cross_t2[i,:]))


# In[17]:


m1 = torch.randn(4,3)
m2 = torch.randn(3, 2)

print('m1:\n', m1)
print('m2:\n', m2)
# Matrix multiplication
print('Matrix multiplication:\n', m1.mm(m2))


# In[18]:


m1 = torch.tensor([[1,2],[3,4]], dtype=torch.float32)
m2 = torch.tensor([[2,4],[-1,6]], dtype=torch.float32)
print('m1:\n', m1)
print('m2:\n', m2)
# Element-wise multiplication
print('Element-wise multiplication:\n', m1.mul(m2))


# ## CUDA GPU Support

# In[19]:


# Checking if CUDA GPU is available
result = torch.cuda.is_available()
print('CUDA available (T/F):', result)

# How many CUDA devices are available?
result = torch.cuda.device_count()
print('Number of CUDA devices available:', result)


# ## Loading an image using PIL
# 
# We load an image usng PIL library

# In[20]:


from PIL import Image

filename = './3063.jpg'
image = Image.open(filename)
print(image)


# Converting PIL image to numpy

# In[21]:


import numpy as np
image_np = np.array(image, dtype='float32')/255.
print(image_np.shape)


# Converting numpy to Torch tensor

# In[22]:


image_tensor = torch.tensor(image_np)
print(image_tensor.shape)


# Displaying image using matplotlib

# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.title('PIL image')
plt.imshow(image)
plt.subplot(132)
plt.title('Numpy array')
plt.imshow(image_np)
plt.subplot(133)
plt.title('Torch tensor')
plt.imshow(image_np)


# Computing mean and variance of red, blue and green channels
# 
# Before we do that, we will convert our (w x h x 3) image to (3 x w x h).

# In[24]:


print('Shape of image_tensor', image_tensor.shape)
x = image_tensor.transpose(0,2).transpose(1,2)
print('Shape of x', x.shape)

npixels = x.size(1) * x.size(2)

sums = torch.sum(x, dim=(1,2))
print('Sum', sums)

means = sums.view(3,-1) / npixels
print('Mean', means.shape)

x_centered = (x.view(3,-1) - means).view(x.shape)
print(x_centered.shape)

y = x_centered.transpose(0,2).transpose(0,1)
print(y.shape)


# In[25]:


plt.imshow(y)
print(torch.min(y))
print(torch.max(y))


# In[ ]:




