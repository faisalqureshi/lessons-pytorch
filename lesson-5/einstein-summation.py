#!/usr/bin/env python
# coding: utf-8

# # Einstein summation convention
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  

# In[1]:


import numpy as np


# **Scalar (dot) product:**
# 
# $$
# \mathbf{a} . \mathbf{b} = \sum_{i=1}^{3} = a_1 b_1 + a_2 b_2 + a_2 b_2
# $$
# 
# can be expressed using Einstein summation convention as $a_{\theta} b_{\theta}$

# In[2]:


a = np.array([1,2,4])
b = np.array([0,4,5])
print(np.dot(a,b))
print(np.einsum('i,i',a,b))


# **Matrix multiplication:**
# 
# $$
# c_{ij} = \sum_k a_{ik} b_{kj}
# $$
# 
# can be expressed in Einstein summation as $a_{i \theta} b_{\theta j}$

# In[3]:


a = np.array(range(9)).reshape(3,3)
print(a)
b = np.array(range(5,11)).reshape(3,2)
print(b)
print(np.dot(a,b))
print(np.einsum('ik,kj',a,b))


# **Matrix-vector multiplication**
# $$
# c_i = \sum_j a_{ij} b_j
# $$
# can be similarly expressed: $a_{i\theta}b_\theta$

# In[4]:


a = np.array(range(9)).reshape(3,3)
print(a)
b = np.array(range(3)).reshape(3,)
print(b)
print(np.dot(a,b))
print(np.einsum('ij,j',a,b))


# **Vector-matrix multiplication**
# $$
# c_j = \sum_i b_i a_{ij} 
# $$
# can be expressed as: $b_\theta a_{\theta j}$

# In[5]:


a = np.array(range(9)).reshape(3,3)
print(a)
b = np.array(range(3)).reshape(3,)
print(b)
print(np.dot(b,a))
print(np.einsum('i,ij',b,a))


# **Trace of matrix** can be computed using Einstein summation convention as follows:
# 
# $$
# trace(a) = a_{\theta \theta}
# $$

# In[6]:


print(np.trace(a))
print(np.einsum('ii',a))


# Consider the following term that appears in **Multivariate Gaussian**
# $$
# (\mathbf{x} - \mathbf{\mu})^T \Sigma^{-1} (\mathbf{x} - \mathbf{\mu}).
# $$
# Here $\mathbf{x}, \mathbf{\mu} \in \mathbb{R}^d$ and $\Sigma \in \mathbb{R}^{d \times d}$.
# 
# Let $\mathbf{v} = \mathbf{x} - \mathbf{\mu}$ and $\mathbf{M} = \Sigma^{-1}$.
# 
# We can express $\mathbf{M} \mathbf{v}$ using Einstein summation convention as follows: $m_{i \theta} v_{\theta}$.  We can then put it together with $\mathbf{v}^T \mathbf{M} \mathbf{v}$ as follows $v_\alpha m_{\alpha \theta} v_\theta$.

# In[7]:


M = np.array(range(9)).reshape(3,3)
v = np.array(range(5,8))
print(M)
print(v)
print(np.dot(v,np.dot(M,v)))
print(np.einsum('i,ij,j',v.T,M,v))


# Now lets see if we can do it when vector $\mathbf{v} \in \mathbb{R}^{3 \times 1}$.  Again, we are computing $\mathbf{v}^T \mathbf{M} \mathbf{v}$.

# In[8]:


M = np.array(range(9)).reshape(3,3)
v = np.array(range(5,8)).reshape(3,1)
print(M)
print(v)
print(np.dot(v.T,np.dot(M,v)))
print(np.einsum('...i,ij,j...',v.T,M,v))


# Now lets consider the case where $mn$ $\mathbf{v} \in \mathbb{R}^{2}$ vectors are stored in a tensor $\mathbf{t} \in \mathbb{R}^{m \times n \times 2}$.  This is often encountered when we attempt to evaluate a Gaussian over a 2D grid of points. 

# In[9]:


x = np.linspace(-2,2,5)
y = np.linspace(-1,1,3)
xx, yy = np.meshgrid(x, y)
t = np.empty(xx.shape + (2,))
t[:,:,0] = xx
t[:,:,1] = yy
print('t', t.shape)

M = np.array(range(4)).reshape([2,2])
print('M',M.shape)

result2 = np.einsum('...j,jk,...k->...',t, M, t)
print(result2)


# In[ ]:




