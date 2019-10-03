"""This a file of functions to run on encrypted inputs using PySEAL.
They all need:
    encoder: to encode parameters properly
    context: SEAL context object for knowing the multiplication ring.
    parameters: path to directory with the model parameters, the model file with 
        the most recent date on it is chosen

All classes should be of this basic structure:
 
class FUNCTION:
   def __init__(parameters, encoder=None, context=None):
   def eval(input) 
"""

import torch
import tensor_ops as tops 
import streamlit as st

def most_recent_model(dir_to_models):
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

    def __init__(self, nn_model_path, encoder = None, context = None):
        try:
            model_params_dict = torch.load(nn_model_path)
            for k,v in model_params_dict.items():
                """if k == "out.weight_v":
                    self.weight_v = v.numpy()"""
        except Exception as e:
            raise ValueError(f"There was a problem with loading log_reg_model_path: {e}.")

        self.encoder = encoder
        self.context = context
        try:
            self.enc_weight_v = tops.vec_encoder(encoder)(#Weights)
        except Exception as e: 
            raise ValueError(f"There was a problem encoding {e}")
        #model = build model from model_dict
    def eval(self,x):
        """
        Inputs:
            x: encrypted inputs for single sample
        Returns: encrypted dot product """

        #y = model(x)
        return y