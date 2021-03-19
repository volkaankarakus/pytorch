# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 12:14:54 2021

@author: VolkanKarakuş
"""

#%% Logistic Regression

# Linear regression is not good at classification.
# We use logistic regression for classification.
# linear regression + logistic function(softmax) = logistic regression

# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable #gradient hesabi icin
from torch.utils.data import DataLoader #train ve test veri setini pytorch icin kullanilabilir hale gelir.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

#%% Prepare Dataset
# load data
train = pd.read_csv('train.csv',dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values # bu kisim class label(targetlar)
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                             targets_numpy,
                                                                             test_size = 0.2,
                                                                             random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train) #numpy'dan Tensor'e dondu.
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
"""
batch_size, bizim verimizi kaca bolerek train edecegimiz.
    1000 train veri setinden olusan elemani 10'lu gruplarsak herbir grubun icinde 100 tane eleman olur.batch_size=100
    
epoch, kac kere train edildigi.epoch=5 icin ayni veri setini 5 kere train edicem demektir.
    loss azalana kadar train etmeliyiz.
"""
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size) # num_epochs=n_iters/len(group_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain) # elimizdeki datasetini tensor data setine cevirir.
test = torch.utils.data.TensorDataset(featuresTest,targetsTest) #

# data loader (elimizdeki veri seti ile sample'larimizin combine edilmesini saglar ve multiprocess yapmamizi saglar.Bu da sureci hizlandirir.)
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
plt.imshow(features_numpy[10].reshape(28,28))
plt.axis("off")
plt.title(str(targets_numpy[10]))
plt.savefig('graph.png')
plt.show()
    
#%% Create Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        # Linear part
        self.linear = nn.Linear(input_dim, output_dim)
        # There should be logistic function right?
        # However logistic function in pytorch is in loss function
        # So actually we do not forget to put it, it is only at next parts
    
    def forward(self, x):
        out = self.linear(x)
        return out

# Instantiate Model Class
input_dim = 28*28 # size of image px*px
output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9

# create logistic regression model
model = LogisticRegressionModel(input_dim, output_dim)

# Cross Entropy Loss  
error = nn.CrossEntropyLoss()

# SGD Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% Traning the Model
count = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader): # enumerate ilk parametre index sayisi, 2. parametre bir tuple.
        
        # Define variables
        train = Variable(images.view(-1, 28*28)) # .view , pytorch'un reshape'i.
        labels = Variable(labels)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward propagation
        outputs = model(train)
        
        # Calculate softmax and cross entropy loss
        loss = error(outputs, labels)
        
        # Calculate gradients
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        count += 1
        
        # Prediction
        if count % 50 == 0:
            # Calculate Accuracy         
            correct = 0
            total = 0
            # Predict test dataset
            for images, labels in test_loader: 
                test = Variable(images.view(-1, 28*28))
                
                # Forward propagation
                outputs = model(test)
                
                # Get predictions from the maximum value
                predicted = torch.max(outputs.data, 1)[1]
                
                # Total number of labels
                total += len(labels)
                
                # Total correct predictions
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / float(total)
            
            # store loss and iteration
            loss_list.append(loss.data)
            iteration_list.append(count)
        if count % 500 == 0:
            # Print Loss
            print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(count, loss.data, accuracy))
            
# Iteration: 500  Loss: 1.8190536499023438  Accuracy: 66.53571319580078%
# Iteration: 1000  Loss: 1.6197290420532227  Accuracy: 75.0952377319336%
# Iteration: 1500  Loss: 1.2920498847961426  Accuracy: 78.16666412353516%
# Iteration: 2000  Loss: 1.2043055295944214  Accuracy: 79.78571319580078%
# Iteration: 2500  Loss: 1.0402812957763672  Accuracy: 80.95237731933594%
# Iteration: 3000  Loss: 0.9334945678710938  Accuracy: 81.8452377319336%
# Iteration: 3500  Loss: 0.9127282500267029  Accuracy: 82.55952453613281%
# Iteration: 4000  Loss: 0.7477881908416748  Accuracy: 83.02381134033203%
# Iteration: 4500  Loss: 0.9721441864967346  Accuracy: 83.39286041259766%
# Iteration: 5000  Loss: 0.8047115802764893  Accuracy: 83.6547622680664%
# Iteration: 5500  Loss: 0.7546226978302002  Accuracy: 84.07142639160156%
# Iteration: 6000  Loss: 0.8709458708763123  Accuracy: 84.42857360839844%
# Iteration: 6500  Loss: 0.6639580726623535  Accuracy: 84.64286041259766%
# Iteration: 7000  Loss: 0.7140616774559021  Accuracy: 84.89286041259766%
# Iteration: 7500  Loss: 0.6391735672950745  Accuracy: 85.14286041259766%
# Iteration: 8000  Loss: 0.7480819821357727  Accuracy: 85.33333587646484%
# Iteration: 8500  Loss: 0.5484308004379272  Accuracy: 85.54762268066406%
# Iteration: 9000  Loss: 0.6540567874908447  Accuracy: 85.67857360839844%
# Iteration: 9500  Loss: 0.5270421504974365  Accuracy: 85.83333587646484%

#%% Visualization
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Number of iteration")
plt.show()