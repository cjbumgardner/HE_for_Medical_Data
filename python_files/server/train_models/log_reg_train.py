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
import argparse
from datetime import datetime


def generate_random_data(num_data_samp, data_dim):
    "Generate some random data for log reg."
    a = np.random.rand(data_dim)+5
    x_noise  = 0.1*np.random.randn(num_data_samp,1)
    x = 10*np.random.rand(num_data_samp,data_dim) - 5
    b = np.array([-np.dot(a,x[row,...])-x_noise[row,...] for row in range(0,num_data_samp)])
    b = np.exp(b)
    y_float = 1/(1+b)
    y = np.rint(y_float)
    return {"x": x, "y": y}

class logreg(nn.Module):
    def __init__(self, input_size, num_classes):
        super(logreg, self).__init__()
        linear = nn.Linear(input_size, num_classes)
        self.out = torch.nn.utils.weight_norm(linear,name = "weight")
    def forward(self, x):
        return torch.sigmoid(self.out(x))

    def predict(self,x):
        return torch.round(self.forward(x))


def train(config, train_dict):
    """
    Training for log_reg model. 

    config: dict of learning parameters
    train_dict: dict {"x":ndarray, "y": ndarray}
    
    """
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    #TODO change this for better train test split
    x, y = torch.Tensor(train_dict["x"]), torch.Tensor(train_dict["y"])
    population = x.shape[0]
    num_features = x.shape[1]
    tensors = torch.utils.data.TensorDataset(x,y)
    train_tensors, test_tensors = torch.utils.data.random_split(tensors, [30,population-30])
    
    train_loader = torch.utils.data.DataLoader(train_tensors,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               )
    
    test_loader = torch.utils.data.DataLoader(test_tensors,
                                              batch_size = 30,
                                              shuffle = True,
                                              )

    model = logreg(num_features,1)
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)

    for epoch in range(num_epochs):
        
        x, labels = list(train_loader)[0]
        outputs = model(x) 
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    #TODO add checkpoint saving  
        if (epoch) % 50 == 0:
            x_test, labels_test = list(test_loader)[0]
            x_test = model(x_test)
            test_loss = torch.nn.functional.binary_cross_entropy(x_test,labels_test)
            st.write(f"epoch: {epoch}/{num_epochs}; loss: {loss}; test_loss: {test_loss}")
    
    return model

#TODO data load
#TODO normalization and parameter precision cutoff for seal_models
if  __name__ == "__main__": 

    parser = argparse.ArgumentParser(description = "Training for a logistic \
        regression model.")

    parser.add_argument("--modeldir", 
                        metavar = "-M", 
                        type = PurePath,
                        default = "log_reg_mortality",
                        help = "Relative directory name in directory 'model_params' \
                                containing config.yml file for training. This is \
                                also where the model will be saved.",
                        )

    parser.add_argument("--datadir",
                        metavar = "-D",
                        type = PurePath,
                        help = "Path the data directory with train, eval, \
                                and test.",
                        )
#TODO delete arguements 
    args = parser.parse_args(["--modeldir","log_reg_mortality"])
    #args = parser.parse_args()
    serverdir = Path.cwd().parent
    modeldir = serverdir.joinpath("model_params",args.modeldir)
    configs = modeldir.joinpath("configs.yaml")
    if args.datadir != None:
        try:
            datadir = serverdir.joinpath(PurePath("data").joinpath(args.datadir))
            #TODO load data
        except FileNotFoundError as e:
            print("There was a problem finding files in the data directory.")
        except Exception as e:
            print("There was an exception raised when trying to load the data: {e}")
    else:
        data = generate_random_data(10000,50)

    config = modeldir.joinpath(PurePath("config.yml"))
    
    try:
        with open(configs) as f:
            cfgs = yaml.load(f,Loader = yaml.FullLoader)
    except FileNotFoundError as e:
        print("There was a problem finding config.yml.")
    except Exception as e:
        print(f"There was an exception: {e}")
    
    st.write("Training the model...")
    trained_model = train(cfgs, data)
    now = datetime.now().strftime("%d-%m-%Y-%H_%M_%S")
    model_parms = modeldir.joinpath(f"model_{now}")
    st.write(f"Finished training. Saving model parameters to {model_parms}")
    st.write(trained_model.state_dict())
    torch.save(trained_model.state_dict(),model_parms)

