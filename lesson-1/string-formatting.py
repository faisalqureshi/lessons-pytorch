#!/usr/bin/env python
# coding: utf-8

# # A note about string formatting
# 
# **[Faisal Z. Qureshi](http://vclab.science.uoit.ca)**  
# 
# Check this out [https://realpython.com/python-f-strings/](https://realpython.com/python-f-strings/) and [https://realpython.com/python-string-formatting/](https://realpython.com/python-string-formatting/) for a more detailed discussion.

# ## Using `%` formatting

# In[1]:


name = 'John'
age = 3

print('My name is %s, and I am %d years old' % (name, age))


# ## Use `str.format()`

# In[2]:


name = 'John'
age = 3

print('My name is {}, and I am {} years old'.format(name, age))


# You can also use indexing as follows, which makes this the preferred, more readable and more scalable option for formatting strings.

# In[3]:


print('I am {1} years old, and my name is {0}'.format(name, age))


# You'll notice that such indexing isn't possible when using `%` formatting.

# ### Using dictionaries
# 
# `str.format()` really shines when using dictionaries.

# In[4]:


person = { 'name':'John', 'age':3}

print('My name is {name}, and I am {age} years old'.format(name=person['name'], age=person['age']))


# Or even better

# In[5]:


print('My name is {name}, and I am {age} years old'.format(**person))


# ## Using f-Strings

# In[6]:


name = 'John'
age = 3

print(f'My name is {name}, and I am {age} years old')


# f-Strings are evaluated at run-time, so you can place valid Python code in them.  f-Strings, for some reason, are also faster than either `%` or `str.format()` formatting.

# In[7]:


def uppercase(x):
    return x.upper()

print(f'My name is {uppercase(name)}, and I am {age} years old')


# In[ ]:




