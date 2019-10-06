"""This a file of functions to run on encrypted inputs using PySEAL.
They all need:
    encoder: to encode parameters properly
    context: SEAL context object for knowing the multiplication ring.
    model_dir: path to directory with the model parameters, the model file with 
        the most recent date on it is chosen
    
    

All classes should be of this basic structure:
 
class FUNCTION:
   def __init__(parameters, encoder=None, context=None):
   def eval(input) 
"""
import torch
import tensor_ops as tops
import os
from pathlib import Path

#experiments leave this change dir out. 
#path = Path.cwd()/"python_files"/"server"/"model_params"

def most_recent_model(dir_to_models):
    if isinstance(dir_to_models,str):
        dir_to_models = Path(dir_to_models)
    list_of_paths = dir_to_models.glob("*")
    paths = sorted(list_of_paths, key=lambda p: p.stat().st_ctime)
    paths.reverse()
    for path in paths:
        if path.name[0:5] == "model":
            latest_path = path
            break
    return latest_path

class linear_reg_svr(object):
    """
    Methods:
        eval: evaluates logistic regression on input
    Attributes: 
        weight_v: (list) normalized weight from log_reg training
        enc_weight_v: (list) weights encoded with 
    """
    def __init__(self, log_reg_model_dir, encoder = None, context = None):
        """
        log_reg_model_path: path dir with model weights.
        context: SEAL context object for encryption scheme
        """
        
        try:
            latest_path = most_recent_model(log_reg_model_dir)
            modeldict = torch.load(latest_path)
            weight_v = modeldict["logistic_reg.weight_v"].numpy()
            weight_g = modeldict["logistic_reg.weight_g"].numpy()
            bias = modeldict["logistic_reg.bias"].numpy()
        except Exception as e:
            raise ValueError(f"There was a problem with loading log_reg_model_path: {e}.")
        weight = weight_v*weight_g
        self.encoder = encoder
        self.context = context
        try:
            self.enc_weight = tops.vec_encoder(encoder)(weight)
            self.enc_bias = tops.vec_encoder(bias)
        except Exception as e: 
            raise ValueError(f"There was a problem encoding {e}")
    def eval(self,x):
        """
        Inputs:
            x: encrypted inputs for single sample
        Returns: encrypted dot product """

        y = tops.cipher_dot_plain(self.context)(x,self.enc_weight_v)
        
        return y

def nn_svr(object):
    """
    Fully connected NN with poly activations. 
    Properties: 
        context
    """

    def __init__(self, nn_model_path, encoder, context, ):

        coder = tops.vec_encoder(encoder)
        if isinstance(nn_model_path, str):
            nn_model_path = Path(nn_model_path)
        modelstrupath = nn_model_path/"configs"
        recent = most_recent_model(nn_model_path)
        modelparmpath = nn_model_path/recent
        #load model geometry
        cfgs = nn_model_path.joinpath("configs.yaml")
        try:
            with open(cfgs) as f:
                configs = yaml.load(f,Loader = yaml.FullLoader)
        except FileNotFoundError as e:
            raise ValueError("There was a problem finding config.yml.")
        except Exception as e:
            raise ValueError(f"There was an exception: {e}")
        #Get model parameters
        degrees = configs["degrees"]
        layers = configs["layers"]
        l = len(layers)
        #load model parameters
        try:
            modelparm = torch.load(modelparmpath)["model_state_dict"]
        except Exception as e:
            raise ValueError(f"There was a problem with loading log_reg_model_path: {e}.")
        #turn pytorch to numpy tensors
        #due to weight normalization, we need to recombine weight_g weight_v
        #set up pyseal linear op
        self.model = []
        for i in range(len(l)):
            b = modelparm[f"nnet.weightedLinear{i}.bias"].numpy()
            wg = modelparm[f"nnet.weightedLinear{i}.weight_g"].numpy() 
            wv = modelparm[f"nnet.weightedLinear{i}.weight_v"].numpy()
            w = wv * wg
            p = modelparm[f"nnet.poly{i}.coefficients"].numpy()
            bcode = coder(b)
            wcode = coder(w)
            pcode = coder(p)
            self.model.append(tops.PlainLinear(context, wcode, bcode))
            if i < len(l)-1:
                self.model.append(tops.Poly(context, pcode, degrees))

    def eval(self,x):
        """
        Inputs:
            x: encrypted inputs for single sample
        Returns: encrypted dot product """
        for layer in model:
            x = layer(x)
        return x
#%%
modely = most_recent_model(path/"nn_poly12_mortality")

#%%
dict = torch.load(modely)

#%%
for k in dict["model_state_dict"].keys():
    print(k)

#%%
dict.keys()


#%%
