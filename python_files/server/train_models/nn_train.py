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
from torch.nn.utils import weight_norm
from torch.nn.functional import relu
import yaml
import pickle
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

class poly(nn.Module):
    """Polynomial activation function. 
    degreelist: list of powers of the polynomial.
    """
    def __init__(self, degreelist):
        super(poly,self).__init__()
        self.degreelist = degreelist
        p = len(degreelist)
        self.coeff = torch.ones(p,dtype=torch.float32,requires_grad=True)
    def forward(self,x):
        out = [x ** n for n in self.degreelist]
        shape = x.shape
        out = torch.cat([j.reshape(*shape,1) for j in out],dim=-1)
        out = out * self.coeff
        out = out.sum(-1)
        return out


class fully_conn(nn.Module):
    """Creates a fully connected neural network according to specs.
    input_size: features length
    layers: list of how many neurons per layer
    activation: "relu" or "poly" 
    degrees: optional. If choosing activation=poly you must specify degress.
        The activation polynomial will have trainable coefficients but only 
        for the degrees specified. E.g.: [2,3]-> activation=  ax^2 +bx^3. """
    def __init__(self, input_size, layers, activation, degrees = None):
        super(fully_conn, self).__init__()
        network = [weight_norm(nn.Linear(input_size,layers[0]))]
        numlayer = len(layers)
        if activation == "relu":
            network.append(relu())
            for i in range(numlayer-1):
                l = weight_norm(nn.Linear(layers[i],layers[i+1]))
                if i < numlayer-2:
                    network.extend([l,relu()])
                else:
                    network.append(l)
        if activation == "poly":
            network.append(poly(degrees))
            p = len(degrees)
            for i in range(numlayer-1):
                l = weight_norm(nn.Linear(layers[i],layers[i+1]))
                if i < numlayer-2:
                    network.extend([l,poly(degrees)])
                else:
                    network.append(l)
        self.nnet = nn.Sequential(*network)
    def forward(self,x):
        logits = self.nnet(x)
        return logits
    def predict(self,x):
        return nn.functional.sigmoid(self.forward(x))

class logreg(nn.Module):
    def __init__(self, input_size, classes):
        super(logreg, self).__init__()
        linear = nn.Linear(input_size, classes)
        self.logistic_reg = weight_norm(linear,name = "weight")
    def forward(self, x):
        return self.logistic_reg(x)

    def predict(self,x):
        return nn.functional.sigmoid(self.forward(x))

def train(config, train_data, model):
    """
    Training for mortality models. 

    config: dict of learning parameters
    train_dict: dict {"x":ndarray, "y": ndarray}
    
    """
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    lr = config["learning_rate"]

    train_x = train_data["train_x"]
    train_y = train_data["train_y"]
    test_x = train_data["test_x"]
    test_y = train_data["test_y"]

    train_tensors = torch.utils.data.TensorDataset(train_x,train_y)
    train_loader = torch.utils.data.DataLoader(train_tensors,
                                               batch_size = batch_size,
                                               shuffle = True,
                                            )
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    loss_values = []
    for epoch in range(num_epochs):
        round = 0
        for (x,y) in train_loader:
            outputs = model(x) 
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs,y)
            loss.backward()
            optimizer.step()
            if round % 50 == 0:
                pred = model(test_x)
                test_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,test_y)
                print(f"epoch: {epoch}/{num_epochs}; loss: {loss}; test_loss: {test_loss}")
                loss_values.append({"step": round*epoch, "loss": loss, "test_loss": test_loss})

    return model, loss_values

def convert_mortality_data(train_dict):
    new = {}
    trainset = train_dict["train"]
    testset = train_dict["test"]
    new["train_x"] = torch.Tensor(trainset.drop(columns = ["expire"]).values)
    new["train_y"] = torch.Tensor(trainset.expire.values).unsqueeze_(1)
    new["test_x"] = torch.Tensor(testset.drop(columns = ["expire"]).values)
    new["test_y"] = torch.Tensor(testset.expire.values).unsqueeze_(1)
    new["num_features"] = new["train_x"].shape[1]
    return new

if  __name__ == "__main__": 
    #Set up stdin parser
    parser = argparse.ArgumentParser(description = "Training for a logistic \
        regression, or fully connected nn model with optional polynomial or relu\
        acivations.")
    parser.add_argument("--modeldir", 
                        metavar = "-M", 
                        type = str,
                        default = "log_reg_mortality",
                        help = "Relative directory name in directory 'model_params' \
                                containing config.yml file for training and building \
                                the model (in the case of a nn). This is where the \
                                model will be saved.",
                        )
    parser.add_argument("--datadir",
                        metavar = "-D",
                        type = str,
                        default = "mortality_risk",
                        help = "Directory in server/data with train/test dictionary.",
                        )
    
    parser.add_argument("--continuetrain",
                        action = "store_true",
                        help = "Add this flag to pick up training at the last model checkpoint",
                        )
    #Get all parsed arguments
    serverdir = Path.cwd().parent
    args = parser.parse_args()
    modeldir = serverdir.joinpath("model_params",args.modeldir)
    data_pickle = serverdir.joinpath("data",args.datadir,"train_dict.pkl")

    #Load the training configs
    cfgs = modeldir.joinpath("configs.yaml")
    try:
        with open(cfgs) as f:
            configs = yaml.load(f,Loader = yaml.FullLoader)
    except FileNotFoundError as e:
        raise ValueError("There was a problem finding config.yml.")
    except Exception as e:
        raise ValueError(f"There was an exception: {e}")

    #Load the data
    try:
        with open(data_pickle,'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        raise ValueError(f"There was an exception raised when trying to load the data: {e}")

    #Turn data into torch.tensor. For the future: can remove this to processing pipeline.
    try: 
        train_data = convert_mortality_data(data_dict)
    except Exception as e:
        raise ValueError(f"There was an issue with the data format: {e}")
    
    #Put together the model either nn or logreg
    modeltype = configs["modeltype"]
    if modeltype == "nn":
        try: 
            layers = configs["layers"]
            activation = configs["activation"]
            degrees = configs["degrees"]
            input_size = train_data["num_features"]
            model = fully_conn(input_size,
                                layers,
                                activation,
                                degrees=degrees,
                                )
        except Exception as e:
            raise ValueError(f"The model couldn't load: {e}")

    if modeltype == "logreg":
        try: 
            layers = configs["layers"]
            input_size = train_data["num_features"]
            model = logreg(input_size, layers)
        except Exception as e:
            raise ValueError(f"The model couldn't load: {e}")
    
    #Pick up training 
    if args.continuetrain == True:
        list_of_paths = modeldir.glob("*")
        paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
        paths.reverse()
        for path in paths:
            if path.name[0:5] == "model":
                latest_path = path
                break
        model.load_state_dict(torch.load(latest_path))

    #Train the model 
    print("Training the model...")
    print(model)
    trained_model, loss_values = train(configs, train_data, model)
    now = datetime.now().strftime("%d-%m-%Y-%H_%M_%S")
    loss_values_file = modeldir.joinpath(f"loss_values_{now}.pkl")
    with open(loss_values_file, "wb") as f:
        pickle.dump(loss_values,f)
    model_file = modeldir.joinpath(f"model_{now}.pkl")
    print(f"Finished training. Saving model parameters to {model_file}")
    torch.save(trained_model.state_dict(),model_file)

