# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:13:43 2021

@author: VolkanKarakuş
"""

#%% matrices
import numpy as np

# numpy array
array=[[1,2,3],[4,5,6]]
first_array=np.array(array) #2x3 array

print('Array type : {}'.format(type(first_array))) # type
print('Array shape : {}'.format(np.shape(first_array))) # shape
print(first_array)

#%% import pytorch library
import torch

#pytorch array
tensor=torch.Tensor(array)
print('Array Type : {}'.format(tensor.type)) #type
print('Array Shape : {}'.format(tensor.shape)) #shape
print(tensor)

#%% Allocation
# bir tane bos liste yaratip bunu append etmektense belli boyutlarda bir matris yaratiyoruz.

#compare numpy and tensor
    #np.ones()=torch.ones()
    #np.random.rand()=torch.rand()
    
# numpy ones
print('Numpy {} \n'.format(np.ones((2,3))))

#pytorch ones
print(torch.ones((2,3)))

#numpy random
print('Numpy {} \n'.format(np.random.rand((2,3))))

#pytorch random
print(torch.rand(2,3))

#%% from numpy to tensor & from tensor to numpy

# random numpy array
array = np.random.rand(2,2)
print("{} {}\n".format(type(array),array))

# from numpy to tensor
from_numpy_to_tensor = torch.from_numpy(array)
print("{}\n".format(from_numpy_to_tensor))

# from tensor to numpy
tensor = from_numpy_to_tensor
from_tensor_to_numpy = tensor.numpy()
print("{} {}\n".format(type(from_tensor_to_numpy),from_tensor_to_numpy))

#%% Basic Math with Pytorch

# Resize: view()
# a and b are tensor.
# Addition: torch.add(a,b) = a + b
# Subtraction: a.sub(b) = a - b
# Element wise multiplication: torch.mul(a,b) = a * b
# Element wise division: torch.div(a,b) = a / b
# Mean: a.mean()
# Standart Deviation (std): a.std()

# create tensor 
tensor = torch.ones(3,3)
print("\n",tensor)

# Resize
print("{}{}\n".format(tensor.view(9).shape,tensor.view(9)))

# Addition
print("Addition: {}\n".format(torch.add(tensor,tensor)))

# Subtraction
print("Subtraction: {}\n".format(tensor.sub(tensor)))

# Element wise multiplication
print("Element wise multiplication: {}\n".format(torch.mul(tensor,tensor)))

# Element wise division
print("Element wise division: {}\n".format(torch.div(tensor,tensor)))

# Mean
tensor = torch.Tensor([1,2,3,4,5])
print("Mean: {}".format(tensor.mean()))

# Standart deviation (std)
print("std: {}".format(tensor.std()))

#%% Variables
# variable'lar gradient'leri toplayan yapilardir.

