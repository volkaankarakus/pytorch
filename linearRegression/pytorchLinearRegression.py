# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 01:11:41 2021

@author: VolkanKarakuş
"""

#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

#%% 

# y = Ax + B.
# A = slope of curve
# B = bias (point that intersect y-axis)
# For example, we have car company. If the car price is low, we sell more car. 
    # If the car price is high, we sell less car. This is the fact that we know and we have data set about this fact.
# The question is that what will be number of car sell if the car price is 100.

# As a car company we collect this data from previous selling
# lets define car prices
car_prices_array = [3,4,5,6,7,8,9]
car_price_np = np.array(car_prices_array,dtype=np.float32)
car_price_np = car_price_np.reshape(-1,1)

from torch.autograd import Variable

car_price_tensor = Variable(torch.from_numpy(car_price_np))

# lets define number of car sell
number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

# lets visualize our data
import matplotlib.pyplot as plt
plt.scatter(car_prices_array,number_of_car_sell_array)
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Car Price$ VS Number of Car Sell")
plt.show()

#%% LINEAR REGRESSION WITH PYTORCH

# Now this plot is our collected data
# We have a question that is what will be number of car sell if the car price is 100$
# In order to solve this question we need to use linear regression.
# We need to line fit into this data. Aim is fitting line with minimum error.
# Steps of Linear Regression
    # create LinearRegression class
    # define model from this LinearRegression class
    # MSE: Mean squared error
    # Optimization (SGD:stochastic gradient descent)
    # Backpropagation
    # Prediction
     
import torch.nn as nn # neural network

"""
pytorch'da classlari genelde biz yaziyoruz.
"""
# create class
class LinearRegression(nn.Module): # nn.Module'deki herseyi kullanmamiza imkan saglar.
    def __init__(self,input_size,output_size):
        # super function. It inherits from nn.Module and we can access everything in nn.Module
        super(LinearRegression,self).__init__()#nn.Module'deki herseyi cagir.Birazdan kullanicaz.
        # Linear function.
        self.linear = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.linear(x)
    
# define model
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim,output_dim) # input and output size are 1

# MSE
mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02   # how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate) #amac zaten model.parametre'leri guncellemek.

# train model
loss_list = [] # losslari biriktirip zamanla nasil azaldigini gorucez.
iteration_number = 1001
for iteration in range(iteration_number):
        
    # optimization
    optimizer.zero_grad() #optimizer parametrelerini 0'a esitle.(weight ve bias)
    
    # Forward to get output
    results = model(car_price_tensor) # results = predicted_y
    
    # Calculate Loss
    loss = mse(results, number_of_car_sell_tensor) # loss = (real_y-predicted_y)^2
    
    # backward propagation
    loss.backward()  # daha sonra loss'un parametrelere gore turevini aliyorum.Buradan gradientleri hesapliyorum.
    
    # Updating parameters
    optimizer.step() # gradientlerin guncellenmesi .step()
    
    # store loss
    loss_list.append(loss.data) # yaptigimiz islemleri anlamlandirabilmemiz icin depolamamiz gerekiyor.
    
    # print loss
    if(iteration % 50 == 0): #her 50 adimda bir losslari yazdir.
        print('epoch {}, loss {}'.format(iteration, loss.data))

plt.plot(range(iteration_number),loss_list)
plt.xlabel("Number of Iterations")
plt.ylabel("Loss")
plt.show()
    
"""
Number of iteration is 1001.
Loss is almost zero that you can see from plot or loss in epoch number 1000.
Now we have a trained model.
While usign trained model, lets predict car prices.
"""

#%% Predict Our Car Price 
predicted = model(car_price_tensor).data.numpy()
plt.scatter(car_prices_array,number_of_car_sell_array,label = "original data",color ="red")
plt.scatter(car_prices_array,predicted,label = "predicted data",color ="blue")

# predict if car price is 10$, what will be the number of car sell
#predicted_10 = model(torch.from_numpy(np.array([10]))).data.numpy()
#plt.scatter(10,predicted_10.data,label = "car price 10$",color ="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()