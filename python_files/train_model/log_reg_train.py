"""
Train a logistic regression model. 
    Input: 
        data: 
    Output: 
        params: parameters of logistic regression model
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import yaml
from pathlib import Path, PurePath
import numpy as np
import streamlit as st

def generate_random_data(num_data_samp, data_dim):
    "Generate some random data for log reg."
    a = np.random.rand(data_dim)
    x_noise  = 0.1*np.random.randn(num_data_samp,1)
    x = 10*np.random.rand(num_data_samp,data_dim) - 5
    b = np.array([-np.dot(a,x[row,...])-x_noise[row,...] for row in range(0,num_data_samp)])
    b = np.exp(b)
    y_float = 1/(1+b)
    y = np.rint(y_float)
    return {"x": x, "y": y}

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def predict(self,x):
        return torch.round(forward(x))


def train(config, train_dict):
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    """train_dict: {"x":ndarray, "y": ndarray}"""
    x, y = torch.Tensor(train_dict["x"]), torch.Tensor(train_dict["y"])
    population = x.shape[0]
    num_features = x.shape[1]
    tensors = torch.utils.data.TensorDataset(x,y)
    train_tensors, test_tensors = torch.utils.data.random_split(tensors, [30,population-30])
    
    train_loader = torch.utils.data.DataLoader(train_tensors,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               )
    
    test_loader = torch.utils.data.DataLoader(test_tensors)

    model = LogisticRegression(num_features,1)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)

    for epoch in range(num_epochs):
        
        x, labels = list(train_loader)[0]
        outputs = model(x) 
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (epoch) % 10 == 0:
            x_test, labels_test = list(test_loader)[0]
            x_test = model(x_test)
            test_loss = torch.nn.functional.binary_cross_entropy(x_test,labels_test)
            st.write(f"epoch: {epoch}/{num_epochs}; loss: {loss}; test_loss: {test_loss}")
    
    return model



if  __name__ == "__main__": #TODO data load
#learning rate schecule?
    #for experimenting to make sure everything works
    data = generate_random_data(5000,10)
    yml_file = PurePath.joinpath(Path.cwd(),"configs.yaml")
    with open(yml_file) as f:
        cfgs = [yaml.load_all(f,Loader = yaml.FullLoader)][0]
    trained_model = train(cfgs,data)
    save_file = PurePath.joinpath(Path.cwd(),cfgs["save_model_file"])
    torch.save(trained_model.state_dict(),save_file)