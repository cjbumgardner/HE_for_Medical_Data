"""
Train a logistic regression model. 
    Input: 
        data: 
    Output: 
        params: parameters of logistic regression model
"""
import os
import pandas as pd
from bokeh.plotting import figure
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import weight_norm
from torch.nn import ReLU
import numpy as np
import streamlit as st
import yaml
import pickle
from pathlib import Path, PurePath
import argparse
from datetime import datetime
from collections import OrderedDict


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
        arr = np.ones(p,dtype=np.float32)
        coeff = torch.nn.Parameter(torch.tensor(arr), requires_grad=True)
        self.register_parameter("coefficients", coeff)
    def forward(self,x):
        out = [torch.pow(x,n) for n in self.degreelist]
        shape = x.shape
        out = torch.cat([j.reshape(*shape,1) for j in out],dim=-1)
        out = out * self.coefficients
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
        network = [("weightedLinear0", weight_norm(nn.Linear(input_size,layers[0])))]
        numlayer = len(layers)
        if activation == "relu":
            Relu = ("relu0", ReLU())
            network.append(Relu)
            for i in range(numlayer-1):
                l = (f"weightedLinear{i+1}", weight_norm(nn.Linear(layers[i],layers[i+1])))
                if i < numlayer-2:
                    Relu = (f"relu{i+1}", ReLU())
                    network.extend([l, Relu])
                else:
                    network.append(l)
        if activation == "poly":
            Poly = (f"poly0", poly(degrees))
            network.append(Poly)
            p = len(degrees)
            for i in range(numlayer-1):
                l = (f"weightedLinear{i+1}",weight_norm(nn.Linear(layers[i],layers[i+1])))
                if i < numlayer-2:
                    Poly = (f"poly{i+1}", poly(degrees))
                    network.extend([l,Poly])
                else:
                    network.append(l)
        self.nnet = nn.Sequential(OrderedDict(network))
        
    def forward(self,x):
        logits = self.nnet(x)
        return logits

    def predict(self,x):
        return torch.sigmoid(self.forward(x))

class logreg(nn.Module):
    def __init__(self, input_size, classes):
        super(logreg, self).__init__()
        linear = nn.Linear(input_size, classes)
        self.logistic_reg = weight_norm(linear,name = "weight")
    def forward(self, x):
        return self.logistic_reg(x)

    def predict(self,x):
        return torch.sigmoid(self.forward(x))

def train(config, train_data, model, optimizer_state=None):
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
    if optimizer_state != None:
        optimizer.load_state_dict(optimizer_state)
    loss_values = []
    pd_loss_value = pd.DataFrame(columns = ["loss", "test_loss","step"])
    round = 0
    placeholderpath = st.empty()
    placeholdergraph = st.empty()
    placeholder = st.empty()
    for epoch in range(num_epochs):
        for (x,y) in train_loader:
            outputs = model(x) 
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs,y)
            loss.backward()
            optimizer.step()
            if round % 50 == 0:
                pred = model(test_x)
                test_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred,test_y)
                print(f"epoch: {epoch}/{num_epochs}; step: {round}; loss: {loss}; test_loss: {test_loss}")
                lossdict = {"epoch": epoch, 
                            "step": round, 
                            "loss": loss.detach().numpy(), 
                            "test_loss": test_loss.detach().numpy(),
                            }
                loss_values.append(lossdict)
                pd_loss_value = pd_loss_value.append(lossdict,ignore_index=True)
                #df = pd_loss_value[["loss","test_loss","step"]].set_index('step')
                p = figure(title="Loss/test loss")
                p.line(pd_loss_value.step,pd_loss_value.loss,line_width=2, color="firebrick", legend="loss")
                p.line(pd_loss_value.step,pd_loss_value.test_loss, line_width=2, legend="test_loss")
                placeholdergraph.bokeh_chart(p)
                placeholder.table(pd_loss_value)
            round+=1
    return model, optimizer, loss_values, placeholderpath

def convert_mortality_data(train_dict, test=False):
    """Converts mortality data dictionary with keys ("train", "test") or just 
    ("test") for testing only when train == False.
    """
    #Hack for now
    if "test_x" in train_dict.keys():
        if test == False:
            train_dict["train_x"] = torch.Tensor(train_dict["train_x"].values)
            train_dict["train_y"] = torch.Tensor(train_dict["train_y"].values)
        train_dict["test_x"] = torch.Tensor(train_dict["test_x"].values)
        train_dict["test_y"] = torch.Tensor(train_dict["test_y"].values)
        
    else:
        if test == False:
            trainset = train_dict.pop("train")
            train_dict["train_x"] = torch.Tensor(trainset.drop(columns = ["expire"]).values)
            train_dict["train_y"] = torch.Tensor(trainset.expire.values).unsqueeze_(1)
        testset = train_dict.pop("test")
        train_dict["test_x"] = torch.Tensor(testset.drop(columns = ["expire"]).values)
        train_dict["test_y"] = torch.Tensor(testset.expire.values).unsqueeze_(1)

    train_dict["num_features"] = train_dict["test_x"].shape[1]
    return train_dict

def main(modeldir = None, datadir = None, continuetrain = None, test = False):
    #Get all parsed arguments
    serverdir = Path(os.path.realpath(__file__)).parent.parent
    modeldir = serverdir.joinpath("model_params",modeldir)
    data_pickle = serverdir.joinpath("data",datadir,"train_dict.pkl")

    #Load the training configs
    cfgs = modeldir.joinpath("configs.yaml")
    try:
        with open(cfgs) as f:
            configs = yaml.load(f,Loader = yaml.FullLoader)
    except FileNotFoundError as e:
        raise ValueError("There was a problem finding configs.yaml.")
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
        train_data = convert_mortality_data(data_dict, test = test)
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
   
    #Initialize model with pretrained params to continue training or test ...
    if continuetrain == True or test == True:
        list_of_paths = modeldir.glob("*")
        paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
        paths.reverse()
        for path in paths:
            if path.name[0:5] == "model":
                latest_path = path
                break
        checkpoint = torch.load(latest_path)
        model_state = checkpoint["model_state_dict"]
        optimizer_state = checkpoint["optimizer_state_dict"]
        model.load_state_dict(model_state)
    else:
        optimizer_state = None
    #Predict only 
    if test == True:
        test_x = train_data["test_x"]
        test_y = train_data["test_y"].squeeze().numpy()
        st.write("Model loaded. Now making predictions...")
        y = model.predict(test_x).squeeze().detach().numpy()
        predictions = np.stack([test_y, y], axis=-1) 
        now = datetime.now().strftime("%d-%m-%Y-%H_%M_%S")
        st.write("Saving predictions alongside true values...")
        file = modeldir/f"predictions_{now}"
        with open(file, "wb") as f:
            pickle.dump(predictions, f)
        st.write(f"Saved to {file}.")
    else:
        #Train the model 
        print("Training the model...")
        trained_model, optimizer, loss_values , placeholderpath = train(configs, 
                                                                train_data,
                                                                model, 
                                                                optimizer_state=optimizer_state,
                                                                )
        now = datetime.now().strftime("%d-%m-%Y-%H_%M_%S")
        loss_values_file = modeldir.joinpath(f"loss_values_{now}.pkl")
        with open(loss_values_file, "wb") as f:
            pickle.dump(loss_values,f)
        model_file = modeldir.joinpath(f"model_{now}.pkl")
        placeholderpath.text(f"Finished training. Saving model parameters to {model_file}")
        d = {"model_state_dict": trained_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            }
        torch.save(d,model_file)

def run():
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
    parser.add_argument("--test",
                        action = "store_true",
                        help = "Add this flag for testing on data under 'test' key in \
                            path input for --datadir"
                        )
    return vars(parser.parse_args())

if  __name__ == "__main__": 
    #Set up stdin parser
    kwargs = run()
    main(**kwargs)
    