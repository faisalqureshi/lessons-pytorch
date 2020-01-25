#!/usr/bin/env python
# coding: utf-8

# # [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) Classification Using AlexNet
# 
# Credits: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py

# ## Lets get the data ready for both training and testing
# 
# We will use torchvision.datasets to download the data.  This tool will only download data if needed.  We are also able to specify our own set of transformations to prepare the data for testing and training.  Below you'll notice that we apply a number of transformations to the data, including converting the downloaded data to torch Tensor and normalizing the data using the numbers provided in the link above. 

# In[1]:


import torch

import torchvision
import torchvision.transforms as transforms


# Training data

# In[2]:


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


# Testing data

# In[3]:


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


# Class labels for CIFAR10

# In[4]:


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# ### Data inspection
# 
# It is always a good idea to inspect your data before you begin to train your network.  Ensure that you understand how data is layed-out.  This is important since most of the networks assume that data is presented in a particular form.  Small things such as flipped channel axes can waste hours, if not days, at end.

# In[5]:


# Lets print the ith item trainset, along with its label.

idx = 0
input1, target = trainset[idx]

print("Input shape: ", input1.shape)

# If you inspect input shape, you will notice that channel information is in the first dimension.  
#  In order for us to display the image, we will have to place channel information in the last 
# dimension.
#img = torch.clone(input1)
img = input1.permute(1,2,0)
print("Image shape: ", img.shape, " and input image: ", input1.shape)

# Remember also that the data is normalized.  We need to rescale it to place pixels values between 0.0 and 1.0.
m = torch.min(img)
r = torch.max(img) - m
img = torch.div(torch.sub(img, m), r)

# The image will appear to change everytime you execute this block.  This simply means that RandomCrop in transforms above 
# is working.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(1,1))
plt.title(classes[target])
plt.imshow(img)


# In[6]:


# Lets print number of items in a single batch
for batch_idx, (inputs, targets) in enumerate(trainloader):
    print(batch_idx, len(inputs), len(targets))
    break


# ## Using a pre-built model
# 
# It is very rare that you'll train your network from scratch.  In a majority of cases you will start with an existing network and gradually add/change it as needed.  The idea is to *transfer* the learning from a pre-trained network to your task.  Indeed one of the reasons why deep learning is so successful is that we are able to repurpose pre-trained networks for new tasks.  E.g., a network trained to recognize objects can be repurposed to do image segmentation.
# 
# PyTorch provides easy access to a number of commonly used pre-trained models via torchvision.

# In[7]:


import torchvision.models as models


# You can find out about which models are available at [https://pytorch.org/docs/stable/torchvision/models.html](https://pytorch.org/docs/stable/torchvision/models.html).
# 
# For the sake of this exercise, we will use AlexNet.  We have two options here: 1) we can choose to create AlexNet with randomly initialized weights, or 2) we can create AlexNet with pre-trained weights.  You'll often pick option 2 above. 

# In[8]:


# Option 1.  AlexNet with randomly initialized weights.
alexnet_random_weights = models.alexnet()


# In[9]:


# Option 2.  AlexNet with pre-trained weights.
# 
# It may take a while to download the weights the first time
# you call this.
alexnet_pretrained = models.alexnet(pretrained=True)


# ### Model inspection
# 
# Lets inspect the model that we just loaded.  This is helpful as it will tell us what sort of information the model expects, what are its outputs, and what is its internal structure.
# 
# Upon cursory examination, we note that AlexNet outputs a 1000-dimensional vector (it was trained on [ImageNet](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).  For our purposes, we only need to output a 10-dimensional vector, since we only have 10 classes.  
# 
# For linear layers, it is straightforward to figure out the input and output sizes; however, the input/output sizes of convolutional layers are not specified *a priori*.  Rather for convolution layers, the output size is related to the input size using the following relationship:
# 
# $$
# O = 1+\frac{I-F+2P}{S},
# $$
# 
# where $I$ is the size of the input, $O$ is the size of the output, $F$ is the size of kernel, $P$ is padding, and $S$ is stride.  Check out [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/) for a clearer explanantion.  Looking at the structure of the AlexNet below, we note that the first Linear layer (Layer 1 in features) expects a 9216-dimensional vector.  This suggests that the output of the last MaxPool2d layer (Layer 12 in classifiers) is somehow reshaped or flattened to a 9216-dimensional vector.  Since the output size of this layer (Layer 12 in classifiers) is related to the input size to the very first layer (layer 0 in features).  Consequently, AlexNet assumes the input to be of a certain size.  Ideally, I would wish some easy way of knowning what input AlexNet expects.  Having said that a quick web search reveals that AlexNet expects inputs of either 227x227x3 or 224x224x3.  We can try both and see which of these two would work.
# 
# It may be that cifar10 images do not match this size.  
# 
# We also need to confirm that the first layer (layer 0 in features) expects channels to be the first dimension.  Since our data loader is returning images as 3x28x28, ie.e., channels take the first dimension.

# In[10]:


print(alexnet_pretrained)


# #### Checking expected input size for AlexNet
# 
# After some sleuth-work, we figure out that the input expected by PyTorch AlexNet is 3x224x224.  I simply tried a few different combinations to figure this out.

# In[11]:


# 0: batch
# 1: channels
# 2: width
# 3: height
test_img = torch.empty([1,3,224,224])

# We can get a forward pass by simply calling the alexnet with
# our test_img
test_output = alexnet_pretrained(test_img)
# If no error, you are good to go.


# #### Using AlexNet with cifar10

# Ok so before we can use AlexNet with Cifar-10 dataset, we need to resize the Cifar-10 images, which are 32x32, to 224x224.
# 
# An easy way to accomplish this is to piggy-back on data loader to resize the images.  We use the transforms.Resize() to resize our images from 32x32 to 224x224.  We implicitly assume that the channels remain untouched.

# In[12]:


transform_train = transforms.Compose([
    transforms.Resize((224,224),2), # Notice that this is the first transform, this is because
                                    # Resize() assumes a PIL image.
    transforms.RandomCrop(224, padding=32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)


# Now lets test it

# In[14]:


idx = 0
input1, target = trainset[idx]

print("Input shape: ", input1.shape)

# If you inspect input shape, you will notice that channel information is in the first dimension.  
#  In order for us to display the image, we will have to place channel information in the last 
# dimension.
#img = torch.clone(input1)
img = input1.permute(1,2,0)
print("Image shape: ", img.shape, " and input image: ", input1.shape)

# Remember also that the data is normalized.  We need to rescale it to place pixels values between 0.0 and 1.0.
m = torch.min(img)
r = torch.max(img) - m
img = torch.div(torch.sub(img, m), r)

# The image will appear to change everytime you execute this block.  This simply means that RandomCrop in transforms above 
# is working.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(1,1))
plt.title(classes[target])
plt.imshow(img)


# Now lets see if this image will pass through the pretrained AlexNet.  Note that AlexNet assumes batch takes the first dimension.

# In[15]:


idx = 0
input1, target = trainset[idx]
print('Size of input1', input1.shape)

input1.unsqueeze_(0)
print('Size of input1 after unsqueeze(0)', input1.shape)

test_output = alexnet_pretrained(test_img)
print('Output of AlexNet', test_output)


# In[ ]:




